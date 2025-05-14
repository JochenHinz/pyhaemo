from splico.mesh import Mesh
from splico.mesh.element import POINT, LINE, TRIANGLE, QUADRILATERAL, HEXAHEDRON
from splico.log import logger as log

from typing import Optional, Dict, Callable

import numpy as np


VALID_FILEENDINGS = '.msh', '.vtk', '.vtp'


MAP_EL_GMSH_TAG = {
    POINT: 0,
    LINE: 1,
    TRIANGLE: 2,
    QUADRILATERAL: 3,
    HEXAHEDRON: 5
}


def export_gmsh(mesh: Mesh,
                filename: str,
                surface_tags: Optional[Dict[str, Callable]] = None) -> None:
  """
  Export a mesh to a gmsh file.

  Parameters
  ----------
  mesh : :class:`Mesh`
      The mesh to export.
  filename : :class:`str`
      The filename to export the mesh to. Must end with '.msh'.
  surface_tags : :class:`dict`, optional
      A dictionary of surface tags to tag the boundary parts of the mesh.
      The keys are the tags and the values are Callables that take a
      number of points and return a boolean that indicates whether the
      element is part of the tagged boundary or not.
  """

  try:
    import gmsh
  except ImportError:
    raise ImportError('Gmsh is required for exporting meshes.'
                      ' Install it via pip install gmsh.')

  surface_tags = dict(surface_tags or {})
  assert any(filename.endswith(ending) for ending in VALID_FILEENDINGS), \
    "Unsupported file ending. Supported file endings are: " + ', '.join(VALID_FILEENDINGS) \
    + "found " + filename

  try:
    gmsh.initialize()

    model = gmsh.model.add_discrete_entity(mesh.ndims)
    all_vertex_indices = mesh.active_indices
    gmsh.model.mesh.addNodes(mesh.ndims,
                             model,
                             all_vertex_indices+1,
                             mesh.points[all_vertex_indices].ravel())
    gmsh.model.mesh.addElementsByType(model,
                                      MAP_EL_GMSH_TAG[mesh.reference_element],
                                      np.arange(len(mesh.elements)) + 1,
                                      mesh.pvelements.ravel() + 1)
    gmsh.model.add_physical_group(mesh.ndims, [model],
                                              name={0: 'Point',
                                                    1: 'Line',
                                                    2: 'Surface',
                                                    3: 'Volume'}[mesh.ndims])

    nelems = len(mesh.elements)

    if surface_tags:
      dmesh = mesh.boundary

      for tag, selecter in surface_tags.items():
        myboundary = dmesh.take_elements(selecter)
        mynelems = len(myboundary.elements)
        if mynelems == 0:
          log.warning(f'The boundary {tag} is empty.')
          continue

        entity = gmsh.model.add_discrete_entity(dmesh.ndims)
        mytag = MAP_EL_GMSH_TAG[myboundary.reference_element]

        gmsh.model.mesh.addElementsByType(entity,
                                          mytag,
                                          np.arange(nelems, nelems+mynelems)+1,
                                          myboundary.pvelements.ravel() + 1)

        gmsh.model.add_physical_group(myboundary.ndims, [entity], name=tag)
        nelems += mynelems

    gmsh.model.geo.synchronize()

    gmsh.write(filename)

  except Exception as e:
    raise Exception(f'Error exporting mesh to {filename}: {e}')
  finally:
    gmsh.finalize()
