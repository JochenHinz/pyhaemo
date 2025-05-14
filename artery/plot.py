from pyhaemo.util import np, _
from pyhaemo.mesh import rectilinear, LineMesh, mesh_union
from pyhaemo.geo import ellipse
from aux import compute_bif_input

import pyvista as pv
import vtk


def pyvista_mesh(mesh):

  nelems = len(mesh.elements)
  cell_type = { 'point': vtk.VTK_POINTS,
                'line': vtk.VTK_LINE,
                'triangle': vtk.VTK_TRIANGLE,
                'quadrilateral': vtk.VTK_QUAD,
                'tetrahedron': vtk.VTK_TETRA,
                'hexahedron': vtk.VTK_HEXAHEDRON }.get(mesh.element_name, None)
  if cell_type is None:
    raise NotImplementedError(f"Plotting cells of type {mesh.element_name} "
                               "has not been implementd yet.")

  mesh = mesh.drop_points_and_renumber()

  points = mesh.points
  elements = np.concatenate([np.full((nelems, 1), mesh.nverts, dtype=int), mesh.pvelements], axis=1).astype(int)

  return pv.UnstructuredGrid(elements, np.array([cell_type] * nelems), points)


def line_element(mesh: LineMesh):
  line_mesh = pv.PolyData()
  line_mesh.points = mesh.points
  line_mesh.lines = \
    np.concatenate([ np.full((len(mesh.pvelements), 1), mesh.nverts, dtype=int),
                              mesh.pvelements ], axis=1).ravel()
  return line_mesh


def plot_artery(artery, npoints=1001, plotter=None, show=True):

  if plotter is None:
    plotter = pv.Plotter()

  for i, centerline in enumerate(artery.centerlines):
    mesh = rectilinear([np.linspace(*centerline.knots[0][[0, -1]], npoints)])
    line = centerline.sample_mesh(mesh)

    lmesh = line_element(line)
    plotter.add_mesh(lmesh, color="blue", line_width=3, label=str(i))

    point_cloud = pv.PolyData( centerline(centerline.knots[0][[0, -1]]) )
    plotter.add_mesh(point_cloud, color="red", point_size=10, render_points_as_spheres=True, label="Points")

  xi = np.linspace(0, 1, npoints)

  for ibif, bif in enumerate(artery.bifurcations):
    if len(bif) == 1:
      continue

    points = []
    for index in bif:
      cl = artery.centerlines[abs(index) - 1]
      if index < 0:
        points.append( cl(cl.knots[0][[0]]).ravel() )
      else:
        points.append( cl(cl.knots[0][[-1]]).ravel() )

    cpoint = sum(points) / len(points)

    for point in points:
      lmesh = rectilinear( [npoints] )._edit(points=cpoint[_] * xi[:, _] + point[_] * (1 - xi[:, _]))
      lmesh = line_element(lmesh)
      plotter.add_mesh(lmesh, color="black", line_width=3, label=str(i))

    point_cloud = pv.PolyData([cpoint])
    plotter.add_mesh(point_cloud, color="yellow",
                                  point_size=10,
                                  render_points_as_spheres=True,
                                  label="Bifurcation")

  plotter.add_legend()
  if show:
    plotter.show()

  return plotter


def plot_frames(artery):
  npoints = 4
  frames_ccw, points_ccw, tangents_ccw, bif_ccw = compute_bif_input(artery)

  cross = mesh_union(*ellipse(1, 1, npoints).quick_sample((4, 4)))
  meshes = []

  for bifpoints, bifframe, bifindices in zip(points_ccw, frames_ccw, bif_ccw):
    for point, frame, index in zip(bifpoints, bifframe, bifindices):
      rad = artery.radii[abs(index)-1]
      myradius = rad([rad.knotvector[0].knots[0 if index > 0 else -1]])
      mesh = cross._edit(points=cross.points @ (frame.T * myradius) + point)
      meshes.append(mesh)

  final_mesh = mesh_union(*meshes)
  plotter = plot_artery(artery, npoints=1001, plotter=None, show=False)

  plotter.add_mesh(pyvista_mesh(final_mesh), color='tan', line_width=1, show_edges=True)
  plotter.show()
