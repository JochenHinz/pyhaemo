from pyhaemo.util import np, normalize, _, gauss_quadrature
from pyhaemo.spl import UnivariateKnotVector, NDSpline
from pyhaemo.log import logger as log
from pyhaemo.mesh import mesh_union

import arr
from jspl import PieceWisePolynomial
from opt import cubic_hermite_curve
from bif import bif_from_matrices

from scipy.spatial import cKDTree
from scipy import optimize
import pyvista as pv
import jax
import vtk


def rotation_around_axis(theta, v, t):
  """
  vectorized in theta
  """
  # t = t / jnp.linalg.norm(t)
  theta = np.asarray(theta)
  shp = theta.shape

  theta = np.atleast_1d(theta)
  v = np.asarray(v)
  t = np.asarray(t)
  return (v * np.cos(theta[:, _]) +
          np.cross(t, v) * np.sin(theta[:, _]) +
          t * (t * v).sum() * (1 - np.cos(theta[:, _]))).reshape(shp + (3,))


def find_inflection_points(sgraph, eps=0.1):
  furcation, = sgraph.nontrivial_furcations
  edges = sgraph.get_edges(furcation.indices.indices)

  centerlines = [edge.cline for edge in edges]
  radii = [edge.radius for edge in edges]
  frames = [edge.frame for edge in edges]

  centerpoint = furcation.centroid
  normal = furcation.normal

  normalframes = [frame[:, 0] for frame in frames]

  batchsize = 31
  xi = np.linspace(0, 1, 201)
  theta = np.linspace(0, 2 * np.pi, batchsize, endpoint=False)

  tangents = [curve(xi, dx=1) for curve in centerlines]
  ntangents = [normalize(t) for t in tangents]
  normals = [normal(xi) for normal in normalframes]
  ps = [curve(xi)[:, _] + r(xi)[:, _, _] * np.array([ rotation_around_axis(theta, n, t)
                                                      for n, t in zip(mynormals, mytangents)])
        for curve, r, mynormals, mytangents in zip(centerlines, radii, normals, ntangents)]

  pR = [curve(xi) + r(xi)[:, _] * np.array([ rotation_around_axis(np.pi/2, n, t)
                                             for n, t in zip(mynormals, mytangents)])
        for curve, r, mynormals, mytangents in zip(centerlines, radii, normals, ntangents)]

  pL = [curve(xi) + r(xi)[:, _] * np.array([ rotation_around_axis(3 * np.pi/2, n, t)
                                             for n, t in zip(mynormals, mytangents)])
        for curve, r, mynormals, mytangents in zip(centerlines, radii, normals, ntangents)]

  ps = [p.reshape(-1, 3) for p in ps]

  trees = [cKDTree(curve) for curve in ps]

  n = len(trees)

  indicesR = [_ for i in range(n)]
  indicesL = [_ for i in range(n)]

  for i, (treeR, treeL) in enumerate(zip(trees, np.roll(trees, -1))):
    mymatch = treeR.sparse_distance_matrix(treeL, eps)

    distances = np.array(list(mymatch.values()))
    myindices = np.array(list(mymatch.keys()))

    I, J = [k // batchsize for k in myindices[np.argmin(distances)]]

    indicesR[i] = I
    indicesL[(i+1) % n] = J

  plotter = sgraph.plot(show=False)
  for i, curve in zip(indicesR, pR):
    pc = pv.PolyData(curve[:i+1])
    plotter.add_mesh(pc, color='red', point_size=5)

  for i, curve in zip(indicesL, pL):
    pc = pv.PolyData(curve[:i+1])
    plotter.add_mesh(pc, color='red', point_size=5)

  plotter.show()


def opt_inflection(sgraph, dx=0.01):
  furcations = sgraph.nontrivial_furcations

  # (i0, side0): (point, i1, (s0, theta0))
  zeros = {}
  pairs = []
  xi = np.linspace(.9, 1, 101)

  for furcation in furcations:
    indices = furcation.indices.indices
    n = len(indices)

    # get all edges ending in the furcation
    edges = sgraph.get_edges(indices)

    # get centerlines, radii, frames
    _centerlines = [edge.cline for edge in edges]
    _radii = [edge.radius for edge in edges]
    _frames = [edge.frame for edge in edges]

    # centerpoint = furcation.centroid
    # normal = furcation.normal

    # get normal and binormal frames
    _normalframes, _binormalframes = [[frame[:, i] for frame in _frames] for i in (0, 1)]

    # (s0, s1, ...), (theta0, theta1, ...)
    s = [arr.Argument(f's{i}', ()) for i in range(n)]
    theta = [arr.Argument(f'theta{i}', ()) for i in range(n)]

    # convert to JAX format
    centerlines = [ PieceWisePolynomial.from_NDSpline(_s, curve) for _s, curve in zip(s, _centerlines) ]
    radii = [ PieceWisePolynomial.from_NDSpline(_s, r) for _s, r in zip(s, _radii) ]
    normalframes = [ PieceWisePolynomial.from_NDSpline(_s, n) for _s, n in zip(s, _normalframes) ]
    binormalframes = [ PieceWisePolynomial.from_NDSpline(_s, b) for _s, b in zip(s, _binormalframes) ]

    # parameterize associated surfaces
    surfaces = [ curve + r * (n * arr.Sin(th) + b * arr.Cos(th))
                 for curve, r, n, b, th in zip(centerlines,
                                               radii,
                                               normalframes,
                                               binormalframes,
                                               theta) ]

    # iterate over the pairs of surfaces and their associated edge indices
    for i, (j0, j1, surf0, surf1) in enumerate(zip(indices,
                                                   np.roll(indices, -1),
                                                   surfaces,
                                                   np.roll(surfaces, -1))):

      # s_i, s_(i+1), theta_i, theta_(i+1)
      key0, key1, key2, key3 = f's{i}', f's{(i+1) % n}', f'theta{i}', f'theta{(i+1) % n}'

      # + 0.1 * ((surf0 - centerpoint) * normal).sum() ** 2
      _func = ((surf0 - surf1) ** 2).sum()  # + (theta[i] ** 2) + (theta[(i+1) % 3] - np.pi) ** 2

      # define intersection point cost function
      @jax.jit
      def costfunc(x):
        s0, s1, theta0, theta1 = x
        kwargs = {
          key0: s0,
          key1: s1,
          key2: theta0,
          key3: theta1
        }
        return _func(**kwargs)

      p0 = _centerlines[i](xi) + _radii[i](xi)[:, _] * _binormalframes[i](xi)
      p1 = _centerlines[(i+1) % n](xi) + _radii[(i+1) % n](xi)[:, _] * _binormalframes[(i+1) % n](xi)

      I, J = np.unravel_index(np.argmin(((p0[:, _] - p1[_])**2).sum(-1)), xi.shape + xi.shape)

      PI16 = np.pi / 16
      x = optimize.minimize(costfunc,
                            [xi[I], xi[J], 0, np.pi],
                            bounds=[(0, 1)] * 2 +
                                   [(-PI16, PI16)] +
                                   [(np.pi - PI16, np.pi + PI16)],
                            method='L-BFGS-B',
                            options={'disp': True})

      log.warning(f"Furcation {indices} pair {j0, j1} optimization success: {x.success}")

      kwargs = dict(zip([key0, key1, key2, key3], x.x))
      point = np.asarray((surf0(**kwargs) + surf1(**kwargs)) / 2)

      zeros[(j0, 'r')] = (point, x.x[::2])
      zeros[(j1, 'l')] = (point, x.x[1::2])

      # ('r', 'l') ordering implied
      pairs.append((j0, j1))

  return zeros, tuple(pairs)


def find_sidecurves(sgraph, opt_result, pairs, eps=10, dx=0.005):
  furcations = sgraph.nontrivial_furcations
  allindices = sorted(set.union(*map(set, (furc.indices for furc in furcations))))
  setindices = set(allindices)
  n = len(allindices)

  bounds0 = [(0.05, min(opt_result[(i, 'r')][1][0],
                        opt_result[(i, 'l')][1][0]) - 0.01) for i in allindices]

  pairs = []
  for i, index in enumerate(allindices):
    if index > 0:
      break
    if -index in setindices:
      pairs.append([i, allindices.index(-index)])

  pairs = np.asarray(pairs, dtype=int)

  cons = ()
  if len(pairs):
    def func(x):
      x = x[:n]
      return (x[pairs[:, 0]] + x[pairs[:, 1]]) - 1.0
    cons += ({'type': 'ineq', 'fun': func},)

  qweights, qpoints = gauss_quadrature(0, 1, 5)
  arg = arr.Argument('dummy', ())
  curves, intersectionpoints = [], []

  cfuncs = []
  for furc in furcations:
    edges = sgraph.get_edges(furc.indices.indices)

    s = [arr.Argument(f's{i}', ()) for i in furc.indices]
    scalings_R = [arr.Argument(f'scaling_R{i}', ()) for i in furc.indices]
    scalings_L = [arr.Argument(f'scaling_L{i}', ()) for i in furc.indices]

    centerlines = \
      [PieceWisePolynomial.from_NDSpline(_s, edge.cline) for _s, edge in zip(s, edges)]

    radii = \
      [PieceWisePolynomial.from_NDSpline(_s, edge.radius) for _s, edge in zip(s, edges)]

    binormals = \
      [PieceWisePolynomial.from_NDSpline(_s, edge.frame[:, 1]) for _s, edge in zip(s, edges)]

    tangents = [cline.derivative(1) for cline in centerlines]
    tangents = [t / ((t**2).sum()**.5) for t in tangents]

    sidecurves_R = [cline + r * binormal for cline, r, binormal in zip(centerlines,
                                                                       radii,
                                                                       binormals)]

    sidecurves_L = [cline - r * binormal for cline, r, binormal in zip(centerlines,
                                                                       radii,
                                                                       binormals)]

    mycurves, mypoints = [], []

    for i, cR, cL, sR, sL, _tR, _tL in zip(furc.indices,
                                           sidecurves_R,
                                           np.roll(sidecurves_L, -1),
                                           scalings_R,
                                           np.roll(scalings_L, -1),
                                           tangents,
                                           np.roll(tangents, -1)):

      mycurves.append( cubic_hermite_curve(arg, cR, sR * _tR, cL, -sL * _tL) )
      mypoints.append( opt_result[(i, 'r')][0] )

    curves.extend(mycurves)
    intersectionpoints.extend(mypoints)

    for curve, point in zip(mycurves, mypoints):
      curvepp = curve.derivative(2)._edit(argument=qpoints)
      smoothness_cost = (qpoints[:, _] * (curvepp**2)).sum()

      evalpoint = curve._edit(argument=arr.Constant(.5))
      distance_cost = ((evalpoint - point) ** 2).sum()

      cfuncs.append( distance_cost + eps * smoothness_cost)

  keys = tuple(f's{i}' for i in allindices) + \
         tuple(f'scaling_R{i}' for i in allindices) + \
         tuple(f'scaling_L{i}' for i in allindices)

  @jax.jit
  def costfunc(x):
    kwargs = dict(zip(keys, x))
    return sum(func(**kwargs) for func in cfuncs)

  bounds = bounds0 + [(.05, None)] * n + [(.05, None)] * n
  x0 = [b[1] for b in bounds0] + [.1] * n + [.1] * n

  x = optimize.minimize(costfunc, x0, bounds=bounds,
                                      constraints=cons,
                                      options={'disp': True})

  log.warning(f"Sidecurve optimization success: {x.success}")

  solvekwargs = dict(zip(keys, x.x))

  xi = np.linspace(0, 1, 201)
  _curves = [curve._edit(argument=xi) for curve in curves]

  plotter = sgraph.plot(show=False, show_normals=False)

  for curve in _curves:
    plotter.add_points(np.asarray(curve(**solvekwargs)),
                       color='green',
                       point_size=5,
                       render_points_as_spheres=True)

  edges = sgraph.edges
  lengths = [edge.length for edge in edges]
  centerlines = [edge.cline for edge in edges]
  radii = [edge.radius for edge in edges]
  frames = [edge.frame for edge in edges]
  theta = [0, np.pi]

  for curve, rad, frame, length in zip(centerlines, radii, frames, lengths):
    npoints = int(length / dx)
    s = np.linspace(0.0, 1, max(2, npoints))
    X, r, f = curve(s), rad(s), frame(s)
    for _X, _r, _f in zip(X, r, f):
      points = _X[_] + _r * (_f[:, 0][_] * np.sin(theta)[:, _] + _f[:, 1][_] * np.cos(theta)[:, _])
      plotter.add_points(points, color='blue', point_size=5, render_points_as_spheres=True)

  plotter.add_points(np.array(intersectionpoints), color='red', point_size=10, render_points_as_spheres=True)

  return solvekwargs, plotter


def make_bifs(sgraph, solvekwargs: dict, plotter=None):

  if plotter is None:
    plotter = sgraph.plot(show=False, show_normals=False)

  bifs = []
  for furc in sgraph.nontrivial_furcations:

    edges = sgraph.get_edges(furc.indices.indices)
    centerlines = [edge.cline for edge in edges]
    radii = [edge.radius for edge in edges]
    frames = [edge.frame for edge in edges]

    normal = furc.normal
    cpoint = furc.centroid

    matrices, centerpoints, scalingsR, scalingsL = [], [], [], []

    myradii = []
    for i, cline, radius, frame in zip(furc.indices, centerlines, radii, frames):
      s = solvekwargs[f's{i}']
      scalingR = solvekwargs[f'scaling_R{i}']
      scalingL = solvekwargs[f'scaling_L{i}']

      myradii.append(radius([s]))

      matrices.append(frame([s]).reshape(3, 3) * radius([s]))

      centerpoints.append(cline([s]).ravel())

      scalingsR.append(scalingR)
      scalingsL.append(scalingL)

    bT = bB = np.mean(myradii)

    bifs.append( bif_from_matrices(matrices,
                                   centerpoints,
                                   scalingsR,
                                   scalingsL,
                                   normal,
                                   cpoint,
                                   bB,
                                   bT) )

  meshes = [ mesh_union(*bif.quick_sample((5, 5, 11)).ravel(), boundary=True)
                                                               for bif in bifs ]

  for mesh in meshes:
    nelems = len(mesh.elements)
    cell_type = { 'point': vtk.VTK_POINTS,
                  'line': vtk.VTK_LINE,
                  'triangle': vtk.VTK_TRIANGLE,
                  'quadrilateral': vtk.VTK_QUAD,
                  'tetrahedron': vtk.VTK_TETRA,
                  'hexahedron': vtk.VTK_HEXAHEDRON }[mesh.element_name]
    points = mesh.points
    elements = np.concatenate([np.full((nelems, 1), mesh.nverts, dtype=int), mesh.pvelements], axis=1).astype(int)

    grid = pv.UnstructuredGrid(elements, np.array([cell_type] * nelems), points)
    plotter.add_mesh(grid, show_edges=True)

  plotter.show()


def test():
  import sys
  sys.path.append('../')
  from artery import VascularGraph, prepare_graph
  graph = prepare_graph(VascularGraph.from_csv('../data/csv/brain'))

  # print(graph.nontrivial_furcations[8].indices.indices)
  # graph = graph.take( graph.nontrivial_furcations[1].indices.indices )
  # graph.plot()

  sgraph = graph.to_spline_graph(21)
  # sgraph.plot()

  opt_result, pairs = opt_inflection(sgraph)

  solvekwargs, plotter = find_sidecurves(sgraph, opt_result, pairs)
  make_bifs(sgraph, solvekwargs, plotter=plotter)


if __name__ == '__main__':
  test()
