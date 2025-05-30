from pyhaemo.util import np, _

import pyvista as pv


def plot_vgraph(graph, plotter=None, show=True, show_normals=True):
  if plotter is None:
    plotter = pv.Plotter()

  for i, cline in enumerate(graph.edges, 1):
    pc = pv.PolyData()
    npoints = cline.points.shape[0]
    pc.points = cline.points[:, :3]
    pc.lines = np.concatenate([[npoints], np.arange(npoints)])
    plotter.add_mesh(pc, color="black",
                         point_size=10,
                         label=str(i),
                         line_width=3)
    centerpoint = cline.points[ (npoints + 1) // 2][:3]

    plotter.add_points(centerpoint[_], color='red')
    plotter.add_point_labels(centerpoint[_], [str(i)])

  for i, furc in enumerate(graph.furcations):
    pc = pv.PolyData([furc.centroid])
    plotter.add_mesh(pc, color="red",
                         point_size=10,
                         render_points_as_spheres=True,
                         label="Bifurcation")

    if show_normals:
      normal = pv.Arrow(start=furc.centroid,
                        direction=furc.normal,
                        scale=0.1)
      plotter.add_mesh(normal, color='black',
                               line_width=1,
                               label=f"Normal{i}")

  plotter.add_legend()
  if show:
    plotter.show()
    plotter.close()

  return plotter
