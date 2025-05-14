"""
This module provides functions to plot a mesh.
We use the `pyvista` library to plot the mesh interactively.
@author: Jochen Hinz
"""

from ..util import np

from typing import TYPE_CHECKING

import vtk
import pyvista as pv


if TYPE_CHECKING:
  from .mesh import Mesh


def plot_mesh(mesh: 'Mesh'):
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

  grid = pv.UnstructuredGrid(elements, np.array([cell_type] * nelems), points)
  grid.plot(show_edges=True, line_width=1, color="tan")


def plot_pointmesh(mesh: 'Mesh'):
  point_cloud = pv.PolyData(mesh.points[mesh.elements.ravel()])
  point_cloud.plot(eye_dome_lighting=True)
