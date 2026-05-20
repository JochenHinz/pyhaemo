from pyhaemo.mesh import Mesh
from pyhaemo.mesh.element import POINT, LINE, TRIANGLE, QUADRILATERAL, HEXAHEDRON
from pyhaemo.log import logger as log

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

  Notes
  -----
  Node coordinates are written once for the volume entity; boundary patches only
  add surface **elements** referencing those node tags — no duplicate vertices are
  introduced for walls or caps.
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

    # Contiguous node tags 1 … n match the number of written coordinates.
    # Without this, sparse tags (max ≫ ncoords) confuse solvers that size DOF
    # maps from the largest tag.
    mesh = mesh.drop_points_and_renumber()
    npoints = len(mesh.points)
    node_tags = np.arange(1, npoints + 1, dtype=np.int64)

    model = gmsh.model.add_discrete_entity(mesh.ndims)
    gmsh.model.mesh.addNodes(mesh.ndims,
                             model,
                             node_tags,
                             mesh.points.ravel())
    gmsh.model.mesh.addElementsByType(model,
                                      MAP_EL_GMSH_TAG[mesh.reference_element],
                                      np.arange(len(mesh.elements)) + 1,
                                      mesh.pvelements.ravel() + 1)
    gmsh.model.add_physical_group(mesh.ndims, [model], name='Volume')

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

    try:
      gmsh.model.mesh.removeDuplicateNodes()
    except Exception as exc:
      log.warning(f'removeDuplicateNodes skipped during export: {exc}')

    gmsh.write(filename)
    log.info(
      'Gmsh export wrote %d nodes and %d volume cells to %s.',
      npoints,
      len(mesh.elements),
      filename,
    )

  except Exception as e:
    raise Exception(f'Error exporting mesh to {filename}: {e}')
  finally:
    gmsh.finalize()
