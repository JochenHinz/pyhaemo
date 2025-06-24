from pyhaemo.util import np, normalize, _
from pyhaemo.spl import UnivariateKnotVector, NDSpline

import arr
from jspl import PieceWisePolynomial

from scipy.spatial import cKDTree
from scipy import optimize
import pyvista as pv
import jax


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


def opt_inflection(sgraph):
  furcations = sgraph.nontrivial_furcations
  zeros = {}
  pairs = []

  for furcation in furcations:
    indices = furcation.indices.indices
    edges = sgraph.get_edges(indices)

    centerlines = [edge.cline for edge in edges]
    radii = [edge.radius for edge in edges]
    frames = [edge.frame for edge in edges]

    centerpoint = furcation.centroid
    normal = furcation.normal

    normalframes, binormalframes = [[frame[:, i] for frame in frames] for i in (0, 1)]

    s = [arr.Argument(f's{i}', ()) for i in range(3)]
    theta = [arr.Argument(f'theta{i}', ()) for i in range(3)]

    centerlines = [ PieceWisePolynomial.from_NDSpline(_s, curve) for _s, curve in zip(s, centerlines) ]
    radii = [ PieceWisePolynomial.from_NDSpline(_s, r) for _s, r in zip(s, radii) ]
    normalframes = [ PieceWisePolynomial.from_NDSpline(_s, n) for _s, n in zip(s, normalframes) ]
    binormalframes = [ PieceWisePolynomial.from_NDSpline(_s, b) for _s, b in zip(s, binormalframes) ]

    surfaces = [ curve + r * (n * arr.Sin(th) + b * arr.Cos(th))
                 for curve, r, n, b, th in zip(centerlines,
                                               radii,
                                               normalframes,
                                               binormalframes,
                                               theta) ]

    for i, (j0, j1, surf0, surf1) in enumerate(zip(indices,
                                                   np.roll(indices, -1),
                                                   surfaces,
                                                   np.roll(surfaces, -1))):

      key0, key1, key2, key3 = f's{i}', f's{(i+1) % 3}', f'theta{i}', f'theta{(i+1) % 3}'

      # + 0.1 * ((surf0 - centerpoint) * normal).sum() ** 2
      _func = ((surf0 - surf1) ** 2).sum()  # + (theta[i] ** 2) + (theta[(i+1) % 3] - np.pi) ** 2

      @jax.jit
      def costfunc(x):
        s0, s1, theta0, theta1 = x
        kwargs = {
          key0: s0,
          key1: s1,
          key2: theta0,
          key3: theta1
        }
        ret = _func(**kwargs)
        return ret

      x = optimize.minimize(costfunc, [.5, .5, 0, np.pi], bounds=[(0, 1)] * 2 +
                                                                 [(-np.pi/16, np.pi / 16)] +
                                                                 [(np.pi - np.pi/16, np.pi + np.pi/16)],
                            method='L-BFGS-B', options={'disp': True})
      print(x.success)
      kwargs = dict(zip([key0, key1, key2, key3], x.x))
      zeros[j0] = (np.asarray((surf0(**kwargs)) + surf1(**kwargs)) / 2, costfunc(x.x), x.x[::2])
      zeros[j1] = (np.asarray((surf0(**kwargs)) + surf1(**kwargs)) / 2, costfunc(x.x), x.x[1::2])
      pairs.append((j0, j1))

  print(zeros)
  import ipdb
  ipdb.set_trace()

  sidecurves = []
  kv = UnivariateKnotVector(np.linspace(0, 1, 2), degree=2)
  for (j0, j1) in pairs:
    point, fval, (s0, th0) = zeros[j0]
    point, fval, (s1, th1) = zeros[j1]
    edges = sgraph.get_edges([j0, j1])
    clines = [edge.cline for edge in edges]
    radii = [edge.radius for edge in edges]
    binormals = [edge.frame[:, 1] for edge in edges]

    if -j0 in zeros:
      s0 = max(0.9 * s0, 1 - zeros[-j0][2][0])
    else:
      s0 = .9 * s0

    if -j1 in zeros:
      s1 = max(0.9 * s1, 1 - zeros[-j1][2][0])
    else:
      s1 = .9 * s1

    x0 = clines[0]([s0]).ravel() + radii[0]([s0]).ravel() * binormals[0]([s0]).ravel()
    x1 = clines[1]([s1]).ravel() - radii[1]([s1]).ravel() * binormals[1]([s1]).ravel()
    sidecurves.append( NDSpline([kv], np.stack([x0, point, x1])) )

  plotter = sgraph.plot(show=False, show_normals=False)

  for key, val in zeros.items():
    plotter.add_points(val[0][_], color='red', point_size=10, render_points_as_spheres=True)

  s, theta = np.linspace(0, .98, 101), np.array([0, np.pi])  # np.linspace(0, 2 * np.pi, 20, endpoint=False)

  for furc in furcations:
    edges = sgraph.get_edges(furc.indices.indices)
    centerlines = [edge.cline for edge in edges]
    radii = [edge.radius for edge in edges]
    frames = [edge.frame for edge in edges]

    for curve, rad, frame in zip(centerlines, radii, frames):
      X, r, f = curve(s), rad(s), frame(s)
      for _X, _r, _f in zip(X, r, f):
        points = _X[_] + _r * (_f[:, 0][_] * np.sin(theta)[:, _] + _f[:, 1][_] * np.cos(theta)[:, _])
        plotter.add_points(points, color='blue', point_size=5, render_points_as_spheres=True)

  for curve in sidecurves:
    plotter.add_points(curve(s), color='green', point_size=5, render_points_as_spheres=True)

  plotter.show()


def test():
  import sys
  sys.path.append('../')
  from artery import VascularGraph, prepare_graph
  graph = prepare_graph(VascularGraph.from_csv('../data/csv/aorta'))

  # print(graph.nontrivial_furcations[8].indices.indices)
  # graph = graph.take( graph.nontrivial_furcations[1].indices.indices )
  # graph.plot()

  sgraph = graph.to_spline_graph(21)
  # sgraph.plot()

  opt_inflection(sgraph)


if __name__ == '__main__':
  test()
