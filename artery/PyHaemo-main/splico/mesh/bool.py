"""
This module's purpose is performing various boolean operations on meshes.

@author: Jochen Hinz
"""


from ..util import np, _
from ..types import FloatArray, IntArray
from ..err import HasNoSubMeshError
from ..log import logger as log
from ._bool import make_numba_indexmap, _remap_elements, \
                   renumber_elements_from_indexmap as renumber_elements, \
                   _make_matching

from typing import TYPE_CHECKING
from functools import lru_cache
from itertools import product

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


if TYPE_CHECKING:
  from .mesh import Mesh


@lru_cache(maxsize=8)
def _issubmesh(mesh0: 'Mesh', mesh1: 'Mesh') -> bool:
  """
  Check if ``mesh0`` is a submesh of ``mesh1``.
  A submesh is defined as a mesh that contains the same or a subset of the
  other mesh's points and elements. Alternatively, ``mesh0`` is also
  considered a submesh of ``mesh1`` if it is a submesh of ``mesh1.submesh``
  or its submeshes.

  Parameters
  ----------
  mesh0 : :class:`splico.mesh.Mesh`
      The submesh candidate.
  mesh1 : :class:`splico.mesh.Mesh`
      The mesh we check if `mesh0` is a submesh of.

  Returns
  -------
  A boolean indicating whether ``mesh0`` is a submesh of ``mesh1``.
  """

  # mesh0 mesh has more vertices per element than mesh1 => False
  if mesh0.nverts > mesh1.nverts:
    return False

  # both meshes have the same class, check if mesh0's elements
  # are a subset of mesh1's and if the points are the same
  if mesh0.__class__ is mesh1.__class__:

    # mesh0 has more elements => mesh1 can't be a submesh of mesh0
    if not len(mesh0.elements) <= len(mesh1.elements):
      return False

    # get the set of the union of both meshe's elements
    all_unique_elements = np.unique(np.concatenate([mesh0.elements,
                                                    mesh1.elements]), axis=0)

    # the shape differs from mesh1.elements.shape => mesh0 can't be a submesh
    if not all_unique_elements.shape == mesh1.elements.shape:
      return False

    # the shape is the same => make sure that unique_elements and mesh1.elements
    # have the same indices when brought into lexigraphically sorted form
    # XXX: note that two elements can be the same even though the indices appear
    #      in a different order. In the long run, detect this too.
    if (all_unique_elements != np.unique(mesh1.elements, axis=0)).any():
      return False

    # all tests passed => check if the relevant points of mesh0
    # are equal to that of mesh1.
    indices = np.unique(mesh0.elements)
    return (mesh0.points[indices] == mesh1.points[indices]).all()
  # try to take mesh1's submesh and see if mesh0 is a submesh of that one
  try:
    submesh1 = mesh1.submesh
    return _issubmesh(mesh0, submesh1)
  # if mesh1 has no submesh, return False
  except HasNoSubMeshError:
    return False


def mesh_union_depr(*meshes: 'Mesh') -> 'Mesh':
  """
  Take the union of several meshes.
  Duplicate points and elements are detected using a hashmap. Therefore only
  points that match exactly are treated as equal.
  Very sensitive to numerical round-off errors. To reduce sensitivity,
  use in combination with the ``splico.util.global_precision`` context manager.
  """
  log.warning("This function is deprecated and will be removed in the future.")

  assert meshes

  if len(meshes) == 1:
    return meshes[0]

  assert all(mesh.__class__ is meshes[0].__class__ for mesh in meshes)

  # get all unique points
  allpoints = np.unique(np.concatenate([mesh.points
                                        for mesh in meshes]), axis=0)

  # map each unique point to an index
  indexmap = make_numba_indexmap(allpoints)

  # create all new elements (counting different orderings twice)
  # by mapping element indices to new indices
  newelems = np.concatenate([renumber_elements(mesh.elements,
                                               mesh.points,
                                               indexmap) for mesh in meshes])

  # keep get the indices of the unique elements, not counting different
  # orderings twice
  _, unique_indices = np.unique(np.sort(newelems, axis=1), axis=0,
                                                           return_index=True)

  # return new mesh
  return meshes[0].__class__(newelems[unique_indices], allpoints)


def lexsort_meshpoints(mesh: 'Mesh') -> FloatArray:
  """
  Given an instantiation of :class:`Mesh`, sort the :class:`np.ndarray`
  ``mesh_points`` of shape ``(nelems, nverts_per_elem, 3)``, where
  ``mesh_points[i]`` contains the ``(nverts, 3)``-shaped array of points of
  the ``i``-th element, lexicographically along the first axis.

  >>> mesh.points[mesh.elements]
  ... [[[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 2, 0], [0, 0, 2]]]
  >>> lexsort_meshpoints(mesh)
  ... [[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, 2], [0, 2, 0], [1, 0, 0]]]
  """
  mesh_points = mesh.points[mesh.elements]
  shuffle = np.lexsort(tuple(mesh_points[..., i] for i in range(3)))
  return np.take_along_axis(mesh_points, shuffle[..., _], 1)


@lru_cache(maxsize=8)
def _compute_kdtree(mesh: 'Mesh'):
  """
  Compute the KDTree of a mesh's points.
  """
  return cKDTree(mesh.points)


@lru_cache(maxsize=8)
def compute_distance_matrix(mesh0: 'Mesh', mesh1: 'Mesh', eps: float = 1e-6):
  """
  Compute the distance matrix between two meshes.
  """
  tree0 = _compute_kdtree(mesh0)
  tree1 = _compute_kdtree(mesh1)

  mat = tree0.sparse_distance_matrix(tree1, max_distance=eps).tocsr().tocoo().tocsr()

  data1 = mat.data.copy()
  data1[:] = 1
  mat1 = csr_matrix((data1, mat.indices, mat.indptr), shape=mat.shape)

  if (mat1.sum(0) > 1.5).any() or (mat1.sum(1) > 1.5).any():
    raise ValueError("More than one point is closer than the matching tolerance. Reduce the tolerance.")

  return mat.data, mat.indices, mat.indptr


def make_matching(mesh0: 'Mesh', mesh1: 'Mesh', eps: float = 1e-8) -> IntArray:
  """
  Make a matching of two meshes' points based on proximity.
  """
  return _make_matching(*compute_distance_matrix(mesh0, mesh1, eps))


def make_matching_(mesh0: 'Mesh', mesh1: 'Mesh', eps: float = 1e-8) -> IntArray:
  """
  Make a matching of two meshes' points based on proximity.
  """
  tree0 = _compute_kdtree(mesh0)
  tree1 = _compute_kdtree(mesh1)

  mat = tree0.sparse_distance_matrix(tree1, max_distance=eps).tocsr()

  data1 = mat.data.copy()
  data1[:] = 1
  mat1 = csr_matrix((data1, mat.indices, mat.indptr), shape=mat.shape)

  if (mat1.sum(0) > 1.5).any() or (mat1.sum(1) > 1.5).any():
    raise ValueError("More than one point is closer than the matching tolerance. Reduce the tolerance.")
  return np.stack(mat1.tocoo().coords, axis=1)


def match_active(mesh0: 'Mesh', mesh1: 'Mesh', eps: float = 1e-8) -> IntArray:
  """
  Match a matching of two meshs' points based on proximity.
  As opposed to ``make_matching``, only the mesh's active points are matched
  and the global rather than local matching indices are returned.
  """
  act0, act1 = mesh0.active_indices, mesh1.active_indices
  mesh0, mesh1 = mesh0.drop_points_and_renumber(), mesh1.drop_points_and_renumber()
  return np.stack([active[ind] for active, ind
                   in zip((act0, act1), make_matching(mesh0, mesh1, eps).T)],
                   axis=1)


def mesh_union(*meshes: 'Mesh', eps: float = 1e-6,
                                return_matches: bool = False,
                                boundary: bool = False):
  """
  Take the union of several meshes using a KDTree to match points.

  Parameters
  ----------
  meshes : :class:`splico.mesh.Mesh`
      The input meshes. All need to be of the same type.
  eps: :class:`float`
      Matching tolerance that is forwarded to ``_match_active``.
      Determines how close two points need to be to be considered eligible
      for matching.
  return_matches : :class:`bool`
      Whether to return the matched vertex pair indices for re-use in other
      meshes with the same mutual connectivity.
  boundary : :class:`bool`
      Whether to only consider boundary points eligible for matching.
      This is useful for reducing the computational cost of the matching
      in case it is known that the meshes are only connected at the boundary.

  Returns
  -------
  union : :class:`splico.mesh.Mesh`
      The union of all input meshes.
  matches : :class:`np.ndarray`
      The matched vertex pairs. Only returned if ``return_matches`` is
      ``True``.
  """

  assert meshes

  if len(meshes) == 1:
    return meshes[0]

  assert all(mesh.__class__ is (rt := meshes[0].__class__) for mesh in meshes)

  if any( len(mesh.active_indices) != len(mesh.points) for mesh in meshes ):
    log.warning("Warning, inactive points detected in at least one mesh,"
                " they will be removed.")

  meshes = tuple(mesh.drop_points_and_renumber() for mesh in meshes)

  # the local patch index is offset by a certain amount to assign a global
  # index.
  offsets = np.array([0, *map(lambda x: len(x.points), meshes)]).cumsum()

  if boundary:
    _meshes = tuple(mesh.boundary for mesh in meshes)
    fmatch = match_active
  else:
    _meshes = meshes
    fmatch = make_matching

  # make all matchings between differing meshes (i < j)
  # XXX: find a more efficient solution
  all_matches = []
  for (i, msh0), (j, msh1) in product(enumerate(_meshes), enumerate(_meshes)):
    if j <= i: continue
    # add offset to the two columns of the matches to reflect global indexing
    mymatch = fmatch(msh0, msh1, eps) + np.array([offsets[i], offsets[j]])[_]
    all_matches.append(mymatch)

  # concatenate all matches into one array
  all_matches = np.concatenate(all_matches)

  # lexicographically sort all matches
  # XXX: recasting all_matches from list to np.array gives mypy errors
  shuffle = np.lexsort(all_matches.T[::-1])
  all_matches = all_matches[shuffle]

  # concatenate all elements and add offset
  all_elements = np.concatenate([mesh.elements + myoffset
                                 for mesh, myoffset in zip(meshes, offsets)])

  # concatenate all points
  all_points = np.concatenate([mesh.points for mesh in meshes])
  all_points[all_points == -0] = 0
  # unique_points = np.unique(all_points, axis=0)

  # remap the elements from the matches
  elements, points = _remap_elements(all_elements, all_points, all_matches)

  # if not points.shape == unique_points.shape:
  #   raise AssertionError("_remap_elements returned points with wrong shape.")

  # create the new mesh object from the remapped elements and points
  ret = rt(elements, points)

  if return_matches:
    return ret, all_matches

  return ret
