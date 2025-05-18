from pyhaemo.util import np, _, normalize
from pyhaemo.spl import NDSplineArray, NDSpline
from pyhaemo.geo import ellipse
from pyhaemo.geo.bif import repeated_knot_disc
import pyvista as pv

from typing import Sequence

from aux import centerline_length
from geom import estimate_normal, compute_frames
from notwist import compute_notwistframe
from artery import Artery
from solve import fit_constrained
from bif import bif_from_matrices

from numpy.typing import NDArray
from pyhaeme_nfur import find_butterfly_line


def make_bif_(points: NDArray,
              radii: NDArray,
              frames: NDArray,
              cross: NDArray) -> NDSplineArray:

  # ax
  normal = estimate_normal(points)
  xC = np.mean(points, axis=0)
  bB = bT = np.mean(radii)

  matrices = [f * r for f, r in zip(frames, radii)]
  matrices = [m @ np.array([ [0, 1, 0], [-1, 0, 0], [0, 0, 1] ]) for m in matrices]

  return bif_from_matrices( matrices[::-1], points[::-1], normal, xC, bB, bT,
                            unitdisc=cross )


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


def make_bif(points: NDArray,
             radii: NDArray,
             frames: NDArray,
             cross: NDArray) -> NDSplineArray:
  points, radii, frames = map(np.asarray, (points, radii, frames))
  # ax
  x_top, xC, x_bot = find_butterfly_line(radii, points, frames)
  # normal = normalize( x_top - x_bot )
  normal = estimate_normal(points)
  bB = bT = np.mean(radii)

  # x becomes y and y becomes -x
  matrices = [f @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) for f in frames]
  # plot_frames(matrices, radii, points, np.mean(points, axis=0), normal)

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
    #print(i)
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
  #mesh.plot()
  # mesh.boundary.interfaces.plot()
  mesh.export_gmsh("mesh.vtk")

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
