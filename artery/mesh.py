from pyhaemo.util import np, _, normalize
from pyhaemo.spl import NDSplineArray, NDSpline
from bif import repeated_knot_disc, bif_from_matrices
from pyhaemo.geo.interp import cubic_hermite_interpolation as cubic_hermite_curve

import pyvista as pv

from typing import Sequence

from aux import centerline_length, estimate_normal, compute_frames
from pyhaemo.geo.notwist import compute_notwistframe
from artery import Artery
from solve import fit_constrained

from numpy.typing import NDArray


def plot_frames(frames, radii, points, xC, normal):
  plotter = pv.Plotter()

  plotter.add_points(points, color='red')
  plotter.add_point_labels(points, [str(i) for i in range(len(points))])

  for i, (frame, radius, point) in enumerate(zip(frames, radii, points)):
    for color, vec in zip(['red', 'blue', 'green'], frame.T):
      arrow = pv.Arrow(start=point, direction=vec, scale=float(radius))
      plotter.add_mesh(arrow, color=color, line_width=.5, label="Frame")

  plotter.add_points(xC, color='red')
  n = pv.Arrow(start=xC,
               direction=normal,
               scale=float( np.mean(radii) ))
  plotter.add_mesh(n, color='black', line_width=1, label="Normal")
  plotter.show()
  plotter.close()


def find_butterfly_line(radii, points, frames):

 tangents = frames[:, -1]

 left, right, center, top, bottom = [], [], [], [], []
 n_branches = len(points)

 for i in range(n_branches):
   left.append(points[i] - radii[i] * frames[i][:, 1])
   right.append(points[i] + radii[i] * frames[i][:, 1])
   center.append(points[i])
   top.append(points[i] + radii[i] * frames[i][:, 0])
   bottom.append(points[i] - radii[i] * frames[i][:, 0])

 curves_top = []
 curves_center = []
 curves_bottom = []
 dec_points = []
 dec_tangents = []

 # TODO: make this scaling a parameter specific for each curve.
 alpha = 2
 for i in range(n_branches):

     start_index = i
     end_index = i + 1 if i < (n_branches-1) else 0

     start_top = NDSpline([], [top[start_index]])
     end_top = NDSpline([], [top[end_index]])

     start_center = NDSpline([], [center[start_index]])
     end_center = NDSpline([], [center[end_index]])

     start_bottom = NDSpline([], [bottom[start_index]])
     end_bottom = NDSpline([], [bottom[end_index]])

     start_tangent = -alpha*radii[start_index] * tangents[start_index]
     end_tangent = alpha * radii[end_index] * tangents[end_index]

     # Calculate the curve
     curves_top.append(cubic_hermite_curve(start_top, end_top,
                                           start_tangent, end_tangent))
     curves_bottom.append(cubic_hermite_curve(start_bottom, end_bottom, start_tangent, end_tangent))
     curves_center.append(cubic_hermite_curve(start_center, end_center, start_tangent, end_tangent))

     start = NDSpline([], [[right[start_index]]])
     end = NDSpline([], [[left[end_index]]])

     # Calculate the curve
     curve = cubic_hermite_curve(start, end,
                                 start_tangent, end_tangent)

     dec_points.append((curve([0.5]).reshape(3,)))
     dec_tangents.append(curve([0.5], dx=1))

 X_top = np.mean([curve([0.5]).reshape(3,) for curve in curves_top], axis=0)
 X = np.mean([curve([0.5]).reshape(3,) for curve in curves_center], axis=0)
 X_bottom = np.mean([curve([0.5]).reshape(3,) for curve in curves_bottom], axis=0)

 bT = np.linalg.norm(X_top - X)
 bB = np.linalg.norm(X_bottom - X)
 normal = normalize(X_top - X_bottom)
 assert 1 - 10**-8 < np.abs(np.dot(normalize(X_top - X), normalize(X_bottom - X))) < 1 + 10**-8 , 'X_top, X and X_top must lie on the same line.'
 assert X_top.shape[0] == X.shape[0] == X_bottom.shape[0] == 3, 'All the decomposition points must be 3D'

 return bT, X, bB, normal


def make_bif(points: NDArray,
             radii: NDArray,
             frames: NDArray,
             cross: NDArray) -> NDSplineArray:
  points, radii, frames = map(np.asarray, (points, radii, frames))

  # x_top, xC, x_bot = find_butterfly_line(radii, points, frames)
  # normal = normalize( x_top - x_bot )
  normal = estimate_normal(points)
  bB = bT = np.mean(radii)
  xC = np.mean(points, axis=0)

  # xC = np.sum(points * radii[:, _], axis=0) / np.sum(radii)

  # bT, xC, bB, normal = find_butterfly_line(radii, points, frames)

  # x becomes y and y becomes -x
  matrices = [f @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) for f in frames]

  matrices = [m * r for m, r in zip(matrices, radii)]

  return bif_from_matrices( matrices, points, normal, xC, bB, bT,
                            unitdisc=cross )


def make_mesh(artery: Artery, nelems_cross=4) -> NDSplineArray:
  frames, normals = {}, {}

  endpoints = artery.endpoints()
  endtangents = artery.endpoints(dx=1)
  endpoint_radii = {}
  for i, radius in enumerate(artery.radii, 1):
    endpoint_radii[-i], endpoint_radii[i] = radius(radius.knots[0][[0, -1]]).ravel()

  for bif in artery.nontrivial_bifs:
    mypoints = [endpoints[i] for i in bif]
    mytangents = [endtangents[i] * (1 if i > 0 else -1) for i in bif]

    myframes = compute_frames(mytangents, estimate_normal(mypoints))
    for index, myframe in zip(bif, myframes):
      frames[index] = myframe
      normals[index] = myframe[:, 0]

  cross = repeated_knot_disc(nelems_cross)

  notwists = []
  for index, (cline, radius) in enumerate(zip(artery.centerlines,
                                              artery.radii), 1):
    n0 = normals.get(-index, None)
    nfinal = normals.get(index, None)
    sample_points = cline.knotvector[0].refine().knots
    tangents = cline(sample_points, dx=1)

    Rs = compute_notwistframe(tangents, n0=n0, nfinal=nfinal)
    rs = radius(sample_points)

    notwists.append( fit_constrained(cline.knotvector,
                                     sample_points,
                                     rs[:, _, _] * Rs,
                                     constrain_endpoints=True,
                                     constrain_tangents=False) )

  vessels = []
  for i, (cline, radius, rRs) in enumerate(zip(artery.centerlines,
                                               artery.radii,
                                               notwists)):
    vessel = (cross[:, _] @ rRs[_]).sum(-1) + (cross.unity @ cline)
    vessels.append(vessel)

  bifsplines = []
  for i, bif in enumerate(artery.nontrivial_bifs):
    mypoints = [endpoints[i] for i in bif]
    myradii = [endpoint_radii[i] for i in bif]
    myframes = [frames[i] for i in bif]
    bifsplines.append( make_bif(mypoints, myradii, myframes, cross) )

  return tuple(vessels), tuple(bifsplines)


def plot_mesh(vessels: Sequence[NDSplineArray],
              bifs: Sequence[NDSplineArray],
              nsteps_z: Sequence[int],
              npoints_cross: int = 5):
  from pyhaemo.mesh import mesh_union
  assert len(vessels) == len(nsteps_z)
  assert all(stps >= 2 for stps in nsteps_z), "nsteps_z must be >= 2"

  meshes = []
  for i, (vessel, nsteps) in enumerate(zip(vessels, nsteps_z)):
    meshes.append( mesh_union(*vessel.quick_sample((npoints_cross,
                                                    npoints_cross,
                                                    nsteps)).ravel(), boundary=True) )
  for i, bif in enumerate(bifs):
    meshes.append( mesh_union(*bif.quick_sample((npoints_cross,
                                                 npoints_cross,
                                                 6)).ravel(), boundary=True) )
  mesh = mesh_union(*meshes, boundary=True)
  print('npoints', len(mesh.points))
  mesh.plot()
  # mesh.boundary.interfaces.plot()


if __name__ == '__main__':
  artery = Artery.from_csv("data/csv/coro/") \
                 .chordlength_parameterize() \
                 .create_proximity_tree(0.3) \
                 .reparameterize() \
                 .biftruncate(factor=2)
                 #.delete_small_edges()

  # artery = Artery.from_swc("data/swc_files/BG001.CNG.swc") \
  #                .chordlength_parameterize() \
  #                .create_proximity_tree(0.005) \
  #                .biftruncate(rfac=3) \
  #                .reparameterize()
  print(artery.nontrivial_bifs)
  # artery.plot_pyvista()

  # import ipdb
  # ipdb.set_trace()
  # artery = artery.take(artery.nontrivial_bifs[3])
  artery = artery.orient()
  lengths = [centerline_length(cline) for cline in artery.centerlines]
  dx = 0.05
  nsteps_z = [max(int(length / dx), 2) for length in lengths]
  print(nsteps_z)

  vessels, bifs = make_mesh(artery)
  plot_mesh(vessels, bifs, nsteps_z, npoints_cross=5)
