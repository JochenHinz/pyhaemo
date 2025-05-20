from pyhaemo.util import np, _, normalize
from pyhaemo.mesh import rectilinear, LineMesh, mesh_union
from pyhaemo.spl import NDSpline, NDSplineArray
from pyhaemo.geo import ellipse
from pyhaemo.log import logger as log
from pyhaemo.geo.notwist import compute_notwistframe
from geom import compute_frames, estimate_normal
from aux import compute_bif_input, centerline_length

import pyvista as pv
import vtk

import matplotlib.pyplot as plt


def make_vessel(centerline: NDSpline,
                radius: NDSpline,
                cross_section: NDSplineArray, **kwargs):

  assert -1 not in centerline.knotvector[0].continuity[1:-1]
  xi, = centerline.knotvector.greville
  tangents = normalize(centerline(xi, dx=1))

  # compute the notwistframe rotation matrices
  Rs = compute_notwistframe(tangents, **kwargs)
  rRs = centerline.knotvector.fit([xi], radius(xi)[:, _, _] * Rs)

  return (cross_section[:, _] @ rRs[_]).sum(-1) + (cross_section.unity @ centerline)


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

  for i, cline in enumerate(artery.centerlines, 1):
    mesh = rectilinear([np.linspace(*cline.knots[0][[0, -1]], npoints)])
    line = cline.sample_mesh(mesh)

    lmesh = line_element(line)
    plotter.add_mesh(lmesh, color="blue", line_width=3, label=str(i))

    point_cloud = pv.PolyData( cline(cline.knots[0][[0, -1]]) )
    plotter.add_mesh(point_cloud, color="red",
                                  point_size=10,
                                  render_points_as_spheres=True,
                                  label=str(i) + ' endpoints')
    centerpoint = cline([np.mean(cline.knots[0][[0, -1]])]).ravel()
    plotter.add_points(centerpoint[_], color='red')
    plotter.add_point_labels(centerpoint[_], [str(i)])

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

    try:
      for point in points:
        lmesh = rectilinear( [npoints] )._edit(points=cpoint[_] * xi[:, _] + point[_] * (1 - xi[:, _]))
        lmesh = line_element(lmesh)
        plotter.add_mesh(lmesh, color="black", line_width=3, label=str(i))

      point_cloud = pv.PolyData([cpoint])
      plotter.add_mesh(point_cloud, color="yellow",
                                    point_size=10,
                                    render_points_as_spheres=True,
                                    label="Bifurcation")
    except Exception as e:
      log.warning(f"Could not plot bifurcation {ibif} with exception {e}")

  plotter.add_legend()
  if show:
    plotter.show()

  return plotter


def plot_frames(artery):
  npoints = 4
  endpoints = artery.endpoints()
  tangents = artery.endpoints(dx=1)

  cross = mesh_union(*ellipse(1, 1, npoints).quick_sample((4, 4)))
  meshes = []

  plotter = plot_artery(artery, npoints=1001, plotter=None, show=False)

  for i, bif in enumerate(artery.nontrivial_bifs):
    if len(bif) < 3:
      continue

    mypoints = [ endpoints[i] for i in bif ]
    mytangents = [ tangents[i] * (1 if i > 0 else -1) for i in bif ]
    mynormal = estimate_normal(mypoints)
    myframes = compute_frames(mytangents, mynormal)

    for j, (point, frame, index) in enumerate(zip(mypoints,
                                                  myframes,
                                                  bif)):
      rad = artery.radii[abs(index)-1]
      cline = artery.centerlines[abs(index)-1]

      myradius = rad([rad.knotvector[0].knots[0 if index > 0 else -1]])
      direction = normalize(cline([cline.knots[0][-1 if index > 0 else 0]], dx=1).ravel())

      arrow = pv.Arrow(start=point,
                       direction=direction,
                       scale=max(float(myradius), 0.1))

      plotter.add_mesh(arrow, color='purple', line_width=1)
      plotter.add_points(point[_], color='red')
      plotter.add_point_labels(point[_], [str(j)])

      mesh = cross._edit(points=cross.points @ (frame.T * myradius) + point)
      meshes.append(mesh)

      for color, vec in zip(['red', 'blue', 'green'], frame.T):
        arrow = pv.Arrow(start=point, direction=vec, scale=float(myradius))
        plotter.add_mesh(arrow, color=color, line_width=1, label="Frame")

    centerpoint = np.mean(mypoints, axis=0)
    plotter.add_points(centerpoint[_], color='red')
    plotter.add_point_labels(centerpoint[_], [str(i)])
    normal = pv.Arrow(start=centerpoint,
                      direction=mynormal,
                      scale=0.1)
    plotter.add_mesh(normal, color='black', line_width=1, label="Normal")

  final_mesh = mesh_union(*meshes)

  plotter.add_mesh(pyvista_mesh(final_mesh), color='tan', line_width=1, show_edges=True)
  plotter.show()


def plot_bifurcation(artery):
  """
  Plot the bifurcation geometry for a given artery and bifurcation configuration.

  Parameters:
  ----------
  artery : object
      An object representing the artery geometry. It should provide the `compute_bif_input` function.
  ax : matplotlib.axes.Axes, optional
      The axes on which to plot the bifurcation. If None, a new figure and axes will be created.

  Returns:
  -------
  ax : matplotlib.axes.Axes
      The axes on which the bifurcation is plotted.
  """

  # Compute the frames, points, tangents, and bifurcation configuration
  all_frames, all_points, _, _, _ = compute_bif_input(artery)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

  # Plot the bifurcation geometry
  for myframes, mypoints in zip(all_frames, all_points):
    for frame, point in zip(myframes, mypoints):

      # Plot the point itself
      ax.scatter(*point, color='red', s=50)

      # Plot the frame at each point using the frame's columns as vectors
      for j, vec in enumerate(frame.T):
        ax.quiver(*point, *vec,
                  color=['blue', 'green', 'orange'][j],  # Different colors for each axis
                  length=1.0, normalize=True)

  plt.show()


def plot_mesh(artery, npoints=5, dx=0.05, plotter=None, show=True, color='tan'):

  bifs = compute_bif_input(artery)
  map_edge_endpoint_normals = {i: [None, None] for i in range(1, artery.nedges + 1)}
  frames = {}

  for bif in bifs:
    for myframe, index in zip(bif.compute_frames(), bif.indices):
      frames[index] = myframe
      if index < 0:
        map_edge_endpoint_normals[abs(index)][0] = myframe[:, 0]
      else:
        map_edge_endpoint_normals[index][1] = myframe[:, 0]

  cross = ellipse(1, 1, npoints)

  if plotter is None:
    plotter = pv.Plotter()

  centers = artery.bifcenters()
  plotter.add_points(centers, color='red')
  plotter.add_point_labels(centers, [str(i) for i in range(len(artery.bifurcations))])

  for i, (centerline, radius) in enumerate(zip(artery.centerlines, artery.radii)):
    nsteps = max(int(centerline_length(centerline) / dx), 2)
    assert nsteps >= 2
    myvessel = mesh_union(*make_vessel(centerline,
                                       radius,
                                       cross,
                                       n0=map_edge_endpoint_normals[i+1][0],
                                       nfinal=map_edge_endpoint_normals[i+1][1]
                                       ).quick_sample((npoints, npoints, nsteps)), boundary=True)
    plotter.add_mesh(pyvista_mesh(myvessel), color=color,
                                             line_width=1,
                                             show_edges=True)

  if show:
    plotter.show()

  return plotter
