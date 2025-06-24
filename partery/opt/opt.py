from pyhaemo.util import np, _, gauss_quadrature, normalize
from pyhaemo.types import Int
from pyhaemo.spl import NDSpline
from pyhaemo.geo.interp import cubic_hermite_interpolation

import pyvista as pv

import arr

from functools import cached_property
from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jminimize
from jspl import Polynomial, PieceWisePolynomial

from scipy.optimize import minimize
from scipy import sparse


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


class Differentiable(arr.Array):

  @abstractmethod
  def _derivative(self):
    ...

  def derivative(self, n=1):
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._derivative().derivative(n=n-1)


def cubic_hermite_curve(argument: arr.Argument,
                        p0: arr.Array,
                        t0: arr.Array,
                        p1: arr.Array,
                        t1: arr.Array):
  p0, t0, p1, t1 = map(arr.asarray, (p0, t0, p1, t1))
  assert len(p0.shape) == 1
  assert all(elem.shape == p0.shape for elem in (p0, t0, p1, t1))

  weights = arr.stack([p0,
                       t0, -3.0 * p0 - 2.0 * t0 + 3.0 * p1 - t1,
                       2.0 * p0 + t0 - 2.0 * p1 + t1],
                       axis=0)

  return Polynomial(argument, weights)


def optimize(fun: arr.Array, x0: jnp.ndarray, jitfunc=True, **kwargs):
  assert fun.shape in ((), (1,))
  argument, = fun.dependencies
  assert argument.shape == x0.shape

  argname = argument.name

  def func(x):
    ret = fun(**{argname: x})
    return ret

  if not jitfunc:
    optfunc = func
  else:
    optfunc = jax.jit(func)

  jac = jax.grad(optfunc)

  x = minimize(lambda x: float(optfunc(x)), x0, jac=jac, **kwargs)
  return x.x


def _unit_vector_cross(v0, v1):
  v0, v1 = map(jnp.asarray, (v0, v1))
  return (y := jnp.cross(v0, v1)) / jnp.linalg.norm(y)


def rotation_around_axis(argument: arr.Array, v: jnp.ndarray, t: jnp.ndarray) -> arr.Array:
  # t = t / jnp.linalg.norm(t)
  return v * arr.Cos(argument) + \
         arr.Cross(t, v) * arr.Sin(argument) + \
         t * (t * v).sum() * (1 - arr.Cos(argument))


def create_costfunc(sgraph,
                    furcation,
                    mu: float = 1.0):
  """
  Degrees of freedom:
  - Positions on the centerline
  - Rotation of the centerline frame
  - Tangent scalings (tangents are normalized) * 2 on each centerline (left and right)
  - Furcation centerpoint
  - Bottom and top length
  """

  edges = sgraph.get_edges(furcation.indices.indices)

  centerlines = [edge.cline for edge in edges]
  radii = [edge.radius for edge in edges]
  frames = [edge.frame for edge in edges]
  lengths = [edge.length for edge in edges]

  minlength = min(lengths)
  posbounds = [(min(.94, max(0, 1 - minlength / length)), 9.5) for length in lengths]

  centerpoint = furcation.centroid
  normal = furcation.normal

  normalframes = [frame[:, 0] for frame in frames]
  assert all( cline.knotvector == r.knotvector == n.knotvector for
              cline, r, n in zip(centerlines, radii, normalframes) )

  pos = arr.Argument('pos', (3,))
  tscalings = arr.Argument('tscalings', (3,))
  cpoint = arr.Argument('cpoint', (3,))
  h = arr.Argument('h', (1,))
  offsets = arr.Argument('offsets', (3,))  # inlet angle offsets
  evalpos = arr.Argument('evalpos', (3,))  # evaluation position on the hermite curves

  # centerline splines and tangents
  clines = [PieceWisePolynomial.from_NDSpline(mypos, cline)
            for cline, mypos in zip(centerlines, pos)]
  tlines = arr.Stack([cline._derivative() for cline in clines])
  ntangents = tlines / (((tlines ** 2.0).sum(1)**.5)[:, _])  # normalized
  clines = arr.Stack(clines)

  rads = arr.Stack([PieceWisePolynomial.from_NDSpline(mypos, rad[_])
                    for rad, mypos in zip(radii, pos)])

  nframes = arr.Stack([PieceWisePolynomial.from_NDSpline(mypos, frame)
                       for frame, mypos in zip(normalframes, pos)])

  tpoints = [cline + r * rotation_around_axis(offset, n, t)
             for cline, r, n, t, offset in zip(clines, rads, nframes, ntangents, offsets)]

  rpoints = [cline + r * rotation_around_axis(jnp.pi/2 + offset, n, t)
             for cline, r, n, t, offset in zip(clines, rads, nframes, ntangents, offsets)]

  bpoints = [cline + r * rotation_around_axis(jnp.pi + offset, n, t)
             for cline, r, n, t, offset in zip(clines, rads, nframes, ntangents, offsets)]

  lpoints = [cline + r * rotation_around_axis(3 * jnp.pi / 2 + offset, n, t)
             for cline, r, n, t, offset in zip(clines, rads, nframes, ntangents, offsets)]

  qweights, qpoints = map(arr.asarray, gauss_quadrature(0, 1, 5))

  stangents = tscalings[:, _] * ntangents

  ptop = cpoint + h * normal
  pbot = cpoint - h * normal

  # curves
  curves = [ cubic_hermite_curve(pos[_], rp, lt, lp, -rt) for pos, rp, lt, lp, rt in
             zip(evalpos,
                 rpoints,
                 stangents,
                 np.roll(list(lpoints), -1),
                 np.roll(list(stangents), -1)) ]

  # side curves in quadrature points
  scurves = [curve._edit(argument=qpoints) for curve in curves]

  ctangents = [cpoint - cline for cline in clines]

  # curves from the top point of each inlet to the top point of the center
  tcurves = [cubic_hermite_curve(qpoints, tp, st / 2.0, ptop, ct)
             for tp, st, ct in zip(tpoints, stangents, ctangents)]

  bcurves = [cubic_hermite_curve(qpoints, tb, st / 2.0, pbot, ct)
             for tb, st, ct in zip(bpoints, stangents, ctangents)]

  hermite_costs = arr.Add([ (qweights[:, _] * (x.derivative(2)**2)).sum()
                            for x in scurves + tcurves + bcurves ])

  distance_cost = mu * ((pos) ** 2).sum()
  distance_cost += mu * ((cpoint - centerpoint) ** 2).sum()

  cross_edges = arr.Stack([cpoint - curve._edit(argument=pos)
                           for pos, curve in zip(evalpos, curves)])
  cross_lengths = mu * (cross_edges ** 2).sum()

  @jax.jit
  def costfunc(x, static_argnums=(0,)):
    pos, centerpoint, tscalings, h, offsets, evalpos = jnp.array_split(x, [3, 6, 9, 10, 13])
    kwargs = {'pos': pos,
              'cpoint': centerpoint,
              'tscalings': tscalings,
              'h': h,
              'offsets': offsets,
              'evalpos': evalpos}
    ret = hermite_costs(**kwargs)
    ret += distance_cost(**kwargs)
    ret += cross_lengths(**kwargs)
    return ret

  bounds = posbounds + \
           [(c, c) for c in centerpoint] + \
           [(0.02, None)] * 3 + \
           [(0, None)] + [(0, 0)] * 3 + \
           [(.1, .9)] * 3

  c0 = arr.Constant(np.linspace(0, 1, 4))
  crosses = [arr.Cross(curve.derivative(1)._edit(argument=c0),
                       curve.derivative(2)._edit(argument=c0)) for curve, pos in zip(curves, evalpos)]
  constraint_funcs = arr.Concatenate(crosses)

  @jax.jit
  def constraint(x):
    pos, centerpoint, tscalings, h, offsets, evalpos = jnp.array_split(x, [3, 6, 9, 10, 13])
    kwargs = {'pos': pos,
              'cpoint': centerpoint,
              'tscalings': tscalings,
              'h': h,
              'offsets': offsets,
              'evalpos': evalpos}
    return (constraint_funcs(**kwargs) * -normal).sum(1)

  x0 = np.array([.95] * 3 +
                list(centerpoint) +
                [1.0] * 3 +
                [np.mean([r([.95]).ravel() for r in radii])] +
                [0.0] * 3 +
                [.5] * 3)

  return costfunc, bounds, x0, scurves, constraint


def optimize_global(sgraph, **optimizekwargs):
  functions, bounds, x0s, curves, constraints = [], [], [], [], []
  count = 0
  indexdict = {}

  for furc in sgraph.furcations:
    if len(furc.indices) < 3:
      continue
    fun, bound, x0, mycurves, myconstraint = create_costfunc(sgraph, furc, **optimizekwargs)
    functions.append(fun)
    bounds.extend(bound)
    x0s.append(x0)
    curves.append(mycurves)
    constraints.append(myconstraint)
    for i, index in enumerate(furc.indices.indices):
      indexdict.setdefault(abs(index), []).append(count + i)
    count += len(x0)

  indexdict = {k: v for k, v in indexdict.items() if len(v) > 1}

  breaks = np.cumsum([len(x0) for x0 in x0s])[:-1]
  x0 = np.concatenate(x0s)

  def constraint(x):
    x = np.split(x, breaks)
    return np.concatenate([fun(xi) for fun, xi in zip(constraints, x)])

  cons = {'type': 'ineq', 'fun': constraint},

  # @jax.jit
  def costfunc(x):
    x = np.split(x, breaks)
    ret = 0.0
    for fun, xi in zip(functions, x):
      ret += fun(xi)
    print(ret)
    return ret

  if indexdict:
    mat = sparse.lil_matrix((len(indexdict), len(x0)))
    for i, val in enumerate(indexdict.values()):
      for j in val:
        mat[i, j] = 1.0
    mat = mat.tocsr()
    ones = np.ones(mat.shape[0])
    cons += {'type': 'ineq', 'fun': lambda x: mat @ x - ones},  # - 1e-1},

  x = minimize(costfunc, x0, bounds=bounds,
                             options={'disp': True},
                             constraints=cons,
                             method='SLSQP')

  print(x.success)

  xs = np.split(x.x, breaks)

  plotter = sgraph.plot(show=False)

  colors = ['blue', 'red', 'green', 'orange', 'purple']
  xi = arr.Constant(np.linspace(0, 1, 101))
  eta = arr.Constant(np.linspace(0, 1, 3))
  for i, (mycurves, x) in enumerate(zip(curves, xs)):
    kwargs = {
      'pos': x[:3],
      'cpoint': x[3:6],
      'tscalings': x[6:9],
      'h': x[9:10],
      'offsets': x[10:13],
      'evalpos': x[13:]
    }
    for curve in mycurves:
      dx = curve.derivative(1)._edit(argument=eta)(**kwargs)
      ddx = curve.derivative(2)._edit(argument=eta)(**kwargs)
      pc = pv.PolyData(np.asarray(curve._edit(argument=xi)(**kwargs)))
      pc_ = pv.PolyData(np.asarray(curve._edit(argument=eta)(**kwargs)))
      pc_['vectors'] = normalize(np.cross(dx, ddx))
      arrows = pc_.glyph(orient="vectors", scale=0.1, factor=0.05)
      plotter.add_mesh(pc, color=colors[i % len(colors)], point_size=5)
      plotter.add_mesh(arrows, color=colors[i % len(colors)])

  plotter.show()


def test_performance():
  import sys
  sys.path.append('../')
  from artery import VascularGraph, prepare_graph
  graph = prepare_graph(VascularGraph.from_csv('../data/csv/aorta'))

  # indices = graph.nontrivial_furcations[5].indices.indices
  # print(indices)
  sgraph = graph.to_spline_graph(21)  # .take(indices)
  # sgraph = sgraph.take(set([*graph.nontrivial_furcations[0].indices.indices,
  #                           *graph.nontrivial_furcations[1].indices.indices,
  #                           *graph.nontrivial_furcations[3].indices.indices]))

  optimize_global(sgraph, mu=.1)


if __name__ == '__main__':
  test_performance()
