"""
Module defining the abstract base class for all mesh types and some
concrete mesh types.
Based on the `Mesh` class which itself derives from the `Immutable` base class
and a modified version of the `ImmutableMeta` metaclass.

@author: Jochen Hinz
"""

from ..util import np, _, frozen, _round_array, isincreasing, flat_meshgrid, \
                   frozen_cached_property, augment_by_zeros, sorted_tuple
from ..types import Immutable, FloatArray, IntArray, ensure_same_class, Int, \
                    AnyIntSeq
from ..err import MissingVertexError, HasNoSubMeshError, HasNoBoundaryError
from ._refine import refine_structured, _refine_Triangulation
from .pol import eval_mesh_local
from .bool import _issubmesh, mesh_union, lexsort_meshpoints
from .plot import plot_mesh, plot_pointmesh
from .meta import MeshMeta
from .element import ReferenceElement, POINT, LINE, TRIANGLE, QUADRILATERAL, \
                     HEXAHEDRON

from abc import abstractmethod
from typing import Callable, Sequence, Self, Tuple, Dict, List, Any
from functools import cached_property
from itertools import product


# TODO: write another parent class that allows for mixed element types.
#       The meshes that have already been implemented then become special cases
#       with only one element type.
#       To accomplish this, in the long run, move away from a Class-level
#       a attribute structure to a factory pattern structure.
#       This will allow for more flexibility.


# XXX: `np.ndarray` instance attributes are rounded to the environment
#      precision `splico.util.GLOBAL_PRECISION` which can be locally adjusted
#      using the `splico.util.global_precision` context manager.
#      This is to enhance the robustness of point coordinate-based boolean
#      operations. The rounding makes a copy of the array which may not be
#      necessary in all cases but is kept for now or the sake of robustness.
#
#      In the long run, avoid copying the point array whenever possible to
#      prevent unnecessary memory consumption, in particular since the `Mesh`
#      class is structured in a way that not all the mesh's points may be used
#      by its elements. The decision to keep the same point array for different
#      meshes, which aims at reducing memory consumption, currently actually
#      increases memory consumption because a typically larger point array is
#      copied for each mesh instance.
#
#      For instance, taking a subset of the mesh's elements creates a new mesh
#      with a new (smaller) element array but the same (too large) point array.
#      The purpose of this is that `frozen` arrays can be shared by more than
#      one instance.
#
#      To take advantage of this, the point array should not be copied when
#      an operation is performed that only affects the element array.
#
#      One (still non-optimal but better and reasonably safe) way to accomplish
#      this is checking if the array is already rounded to the desired
#      precision using a for-cycle in Numba to avoid making
#      a rounded copy of the array and compare the rounded array to the
#      original. This is also straightforward to parallelize.


class Mesh(Immutable, metaclass=MeshMeta):
  """
  Abstract Base Class for representing various mesh types.

  Parameters
  ----------
  elements : :class:`np.ndarray` of integers or Sequence of integers.
      Element index array of shape (nelems, nverts).
  points : :class:`np.ndarray` or any array-like of type float.
      Point index array of shape (npoints, 3).
      The element index array must satisfy:
          0 <= elements.min() <= elements.max() < len(points).
      Not all ``points in self.points`` may be active. Meaning that the mesh
      can contain points that are not referred-to in `self.elements`.
      This is to minimize the need for copying arrays.

  Attributes
  ----------
  elements : :class:`np.ndarray`, frozen
      The element index array.
  points : :class:`np.ndarray`, frozen
      The point array.
  reference_element : :class:`ReferenceElement`, class attribute
      The type of the mesh's reference element.

  Four class-level attributes are directly inherited from the
  :class:`ReferenceElement` class-level attribute:

  element_name, ndims, nverts, is_affine

  For instance:

  >>> class Triangulation(AffineMesh):
  ...   reference_element = TRIANGLE
  >>> ...
  >>> mesh
  ... Triangulation
  >>> mesh.reference_element.ndims
  ... 2
  >>> mesh.ndims
  ... 2
  """

  # Derived classes that do not implement `reference_element` cannot
  # be instantiated
  reference_element: ReferenceElement

  # inferred from `reference_element` by `MeshMeta`
  element_name: str
  ndims: int
  nverts: int
  is_affine: bool
  _local_ordinances: Callable

  @staticmethod
  def _compute_submesh_elements(mesh: 'Mesh') -> IntArray:
    """
    Compute the mesh's submesh elements without duplicates.
    Requres `self._submesh_indices` to be implemented by the class.
    """
    # XXX: jit-compile the element map

    elements = np.concatenate([ mesh.elements[:, list(slce)]
                                          for slce in mesh._submesh_indices ])

    # iterate over all elements and map the sorted element indices to the element.
    sorted_elements: Dict[Tuple[np.int_], List[Tuple[np.int_, ...]]] = {}
    for elem in map(tuple, elements):
      sorted_elements.setdefault(sorted_tuple(elem), []).append(elem)

    nverts = mesh._submesh_type.nverts

    # for each list of equivalent elements (differing only by a permutation)
    # retain the minimum one. Example [(2, 3, 1), (1, 2, 3)] -> (1, 2, 3)
    # in case of an empty list, return an empty array
    return frozen(sorted(map(min, sorted_elements.values())) or
                  np.zeros((0, nverts), dtype=int))

  def __init__(self, elements: IntArray | Sequence, points: FloatArray):

    self.elements = frozen(elements, dtype=int)
    self.points = frozen(_round_array(points), dtype=float)

    # sanity checks
    assert self.elements.shape[1:] == (self.nverts,)
    if not self.points.shape[1:] == (3,):
      raise NotImplementedError("Meshes are assumed to be manifolds in"
                                " R^3 by default.")

    assert np.unique(self.elements, axis=0).shape == self.elements.shape, \
        "Duplicate element detected."
    assert np.unique(self.points, axis=0).shape == self.points.shape, \
        "Duplicate point detected."

    if self:  # avoid this check for empty meshes
      assert 0 <= self.elements.min() <= self.elements.max() < len(self.points), \
          "Hanging node detected."

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[nelems: {len(self.elements)}, " \
                                    f"npoints: {len(self.points)}]"

  def __bool__(self):
    return bool(self.elements.shape[0])

  @abstractmethod
  def _refine(self) -> Self:
    """
    Refine the entire mesh once.
    We assume that the new elements are ordered such that the element with
    index ``i`` is replaced by
          [nref * i, ..., nref * i + nref - 1],
    where ``nref`` is the proliferation factor of the number of mesh elements
    under a single refinement step.
    """
    # each derived class HAS to implement this method
    return self

  def refine(self, n: int = 1) -> Self:
    """
    Refine the mesh `n` times.
    """
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self and self._refine().refine(n=n-1)

  def issubmesh(self, other: 'Mesh') -> bool:
    """
    Check if self is a submesh of other.
    A submesh is defined as a mesh with a subset or the same vertex indices
    and the same corresponding coordinates.
    If the two meshes are not of the same type, we check if self is
    a submesh of other.submesh and so on.
    """
    return _issubmesh(self, other)

  @frozen_cached_property
  def active_indices(self) -> IntArray:
    """
    The indices of the points in self.points that are used in self.elements.
    """
    return np.unique(self.elements)

  def lexsort_elements(self) -> Self:
    """
    Reorder the elements in lexicographical ordering.
    >>> mesh.elements
        [[3, 2, 1], [1, 2, 0]]
    >>> mesh.lexsort_elements().elements
        [[1, 2, 0], [3, 2, 1]]
    """
    shuffle = np.lexsort(self.elements.T[::-1])
    return self._edit(elements=self.elements[shuffle])

  def get_points(self, vertex_indices: Sequence[int] | IntArray) -> IntArray:
    """
    Same as self.points[vertex_indices] with the difference that it first
    checks if ``vertex_indices`` is a subset of self.elements.
    The rationale behind this is that the self.points array may contain points
    that are not in self.elements.
    """
    vertex_indices = np.asarray(vertex_indices, dtype=int)
    diff = np.setdiff1d(vertex_indices, self.active_indices)
    if len(diff) != 0:
      raise MissingVertexError("Failed to locate the vertices"
                               " with indices '{}'.".format(diff))
    return self.points[vertex_indices]

  def drop_points_and_renumber(self) -> Self:
    """
    Drop all points that are not used by ``self.elements`` and renumber
    ``self.elements`` to reflect the renumbering of the points from
    ``0`` to ``npoints-1``.
    """
    if not self:
      return self._edit(points=np.zeros((0, 3), dtype=float))
    unique_vertices = self.active_indices
    if len(unique_vertices) == unique_vertices[-1] + 1 == len(self.points):
      return self
    points = self.points[unique_vertices]
    elements = np.searchsorted(unique_vertices, self.elements)
    return self._edit(elements=elements, points=points)

  def JK(self, points: FloatArray) -> FloatArray:
    """ Evaluation of the jacobian per element. """
    return np.stack([self.eval_local(points, dx=_dx)
                     for _dx in np.eye(self.ndims, dtype=int)], axis=-1)

  def GK(self, points: FloatArray) -> FloatArray:
    """ Evaluation of the metric tensor per element. """
    JK = self.JK(points)
    return (JK.swapaxes(-1, -2)[..., _] * JK[..., _, :, :]).sum(-2)

  def is_valid(self, order: int = 1, thresh: float = 1e-8) -> bool:
    """
    Check if a mesh is valid.
    These are some standard checks that work for all mesh types.
    Can be overwritten for mesh-type specific validty checks.
    """
    # XXX: this function should eventually get outsourced to a separate module
    #      with more sophisticated validity checks for different mesh types.
    points = self._local_ordinances(order)
    if self.ndims == 3:
      return (np.linalg.det(self.JK(points)) > 0).all()
    if self.ndims == 2:
      JK = self.JK(points)
      # shape (nelems, npoints, 3)
      crosses = np.cross(JK[..., 0], JK[..., 1])
      # first check if there are zeros
      if (np.linalg.norm(crosses, axis=-1) < thresh).any():
        return False
      if self.is_affine:
        # No ? Then check if the orientation of an element changes
        roots = crosses[:, :1]  # (nelems, 1, 3)
        # inner product negative ? => orientation change detected
        return ((crosses[:, 1:] * roots).sum(-1) > 0).all()
      else:
        # XXX: add another validity check for multilinear meshes here
        #      for now just do the metric tensor check
        pass
    # for a 1D mesh it's enough to check if the metric tensor is SPD
    return (np.linalg.det(self.GK(points)) > 0).all()

  def eval_local(self, points: FloatArray, dx=0) -> FloatArray:
    """ Evaluate each element map locally in `points`. """

    # eval_nd_polynomial_local(self, points, dx=dx).shape == (nelems, 3, npoints)
    # we reshape to (nelems, npoints, 3)

    # XXX: restructure to avoid swapaxes and maintain contiguous memory layout
    return eval_mesh_local(self, points, dx=dx).swapaxes(-1, -2)

  @frozen_cached_property
  def _submesh_indices(self) -> Tuple[Tuple[Int, ...], ...]:
    """
    The submesh indices are the columns of `self.elements` that have to be
    extracted to create the mesh's submesh. By default returns a
    `HasNoSubMeshError` but can be overwritten.
    Example: to go from a triangulation to a linmesh, we have to extract the
    columns ([0, 1], [1, 2], [2, 0]) (all edges of each triangle).
    """
    indices = self.reference_element.children_facets
    if not indices:
      raise HasNoSubMeshError(f"A mesh of type '{self.__class__.__name__}'"
                              " has no submesh.")
    return indices

  @property
  def _submesh_type(self):
    raise HasNoSubMeshError(f"A mesh of type '{self.__class__.__name__}'"
                            " has no submesh.")

  @cached_property
  def submesh(self):
    return self._submesh_type(self._compute_submesh_elements(self), self.points)

  @cached_property
  def _boundary_nonboundary_elements(self) -> Tuple[IntArray, IntArray]:
    """
    Split all elements of the submesh into boundary and non-boundary elements.
    """

    # get all facets and reshape
    # (nelems, *_submesh_indices.shape) -> (nelems * si.shape[0], si.shape[1])
    all_facets = self.elements[:, self._submesh_indices]. \
                                    reshape(-1, self._submesh_indices.shape[1])

    # create a unique identifier by sorting the indices
    sorted_facets = np.sort(all_facets, axis=1)

    # get the unique sorted_edges, their corresponding unique indices and the
    # number of occurences of each
    _, unique_indices, counts = np.unique(sorted_facets, return_index=True,
                                                         return_counts=True,
                                                         axis=0)

    # keep the ones that have been counted only once
    one_mask = counts == 1

    # XXX: using a map statement here gives mypy errors
    return ( frozen(all_facets[unique_indices[one_mask]]),
             frozen(all_facets[unique_indices[~one_mask]]) )

  @property
  def boundary(self):
    boundary_elements, _ = self._boundary_nonboundary_elements
    if len(boundary_elements) == 0:
      raise HasNoBoundaryError("The mesh has no boundary.")
    return self._submesh_type(boundary_elements, self.points)

  def subdivide_elements(self):
    """
    This function converts a mesh of one type to a mesh of another type
    through element subdivision. For instance a quadmesh to a triangulation.
    This function should be overwritten, if applicable.
    Note that subdivision may not preserve the geometry exactly.
    """
    raise NotImplementedError

  @ensure_same_class
  def __sub__(self, other: Any) -> Self:
    """
    Subtract ``other`` from ``self`` creating a new (usually smaller) mesh.
    """

    # sort each elements' points lexicographically for both meshes
    all_identifiers = np.concatenate([lexsort_meshpoints(mesh)
                                      for mesh in (self, other)])

    # keep only the unique ones, their first occurence and the times counted
    _, unique_indices, counts = np.unique(all_identifiers, return_index=True,
                                                           return_counts=True,
                                                           axis=0)

    # keep only the first occurences that were counted once
    keep_indices = unique_indices[counts == 1]

    # keep only those indices corresponding to elements in `self`
    return self.take(keep_indices[keep_indices < len(self.elements)])

  @ensure_same_class
  def __or__(self, other):
    """
    Take the union of two meshes.
    """
    return mesh_union(self, other)

  def _and(self, other: Any, return_type=None):
    """
    Take the intersection of two meshes.
    The meshes must be of the same type or the same type must be obtainable
    upon taking submeshes for the intersection to be nonempty.
    If no intersection was found (neither on the mesh itself or its submeshes),
    we return an empty mesh of type `self.__class__`.

    We implement it under the name `_and` because mypy doesn't like the
    additional keyword argument in the `__and__` method.
    """
    if not isinstance(other, Mesh):
      return NotImplemented
    if return_type is None:
      return_type = self.__class__
    kwargs = {'return_type': return_type}
    try:
      if self.ndims > other.ndims:
        return self.submesh._and(other, **kwargs)
      elif self.ndims < other.ndims:
        return self._and(other.submesh, **kwargs)
      if self.__class__ is other.__class__:
        ret = self - (self - other)  # normal union
      # same dimension but different mesh type or empty intersection found
      if self.__class__ is not other.__class__ or not ret:
        return self.submesh._and(other.submesh, **kwargs)
      return ret
    except HasNoSubMeshError:
      return empty_like(self)

  def __and__(self, other: Any):
    """ See ``_and`` """
    return self._and(other)

  @cached_property
  def interfaces(self):
    try:
      dself = self.boundary
    except HasNoBoundaryError:
      return self.submesh
    if dself == self.submesh:
      raise TypeError("The mesh has no interfaces.")
    return self.submesh - dself

  @property
  def pvelements(self):
    """
    PyVista may expect the elements in an order that differs from the chosen
    order. May need to be overwritten in order to properly work with PyVista.
    """
    return self.elements

  def points_iter(self):
    """
    An iterator that returns the vertices of each element.

    Example
    -------
    For a triangulation:

    for (a, b, c) in mesh.points_iter():
      # do stuff with vertices a, b and c

    """
    for indices in self.elements:
      yield self.points[indices]

  def plot(self):
    plot_mesh(self)

  def take(self, elemindices: Sequence[Int] | IntArray):
    return self._edit(elements=self.elements[np.asarray(elemindices, dtype=int)])

  def take_elements(self, selecter: Callable, complement: bool = False) -> Self:
    """
    Keep only those elements for which `selecter` evaluates to `True`.
    If `complement` is `True`, keep only those elements for which `selecter`
    evaluates to `False`.

    Parameters
    ----------
    selecter : :class:`Callable`
        A function that takes the vertices of an element as input and returns
        a boolean. Needs to be vectorized and return a boolean array of shape
        (len(self.elements),). Takes as input the array
        `self.points[self.elements]` of shape (nelems, nverts, 3).
    complement : :class:`bool`
        If `True`, keep only those elements for which `selecter` evaluates to
        `False`.
    """
    if complement:
      return self - self.take_elements(selecter)
    return self.take(elemindices=selecter(self.points[self.elements]))

  def export_gmsh(self, *args, **kwargs):
    from .export import export_gmsh
    export_gmsh(self, *args, **kwargs)


def empty_like(mesh: Mesh) -> Mesh:
  return mesh._edit(elements=np.zeros((0, mesh.nverts), dtype=int))


class MultilinearMesh(Mesh):
  """
  Abstract base class for multilinear mesh types.
  Provides an implementation of the ``_refine`` abstract method
  which is accomplished using the ``_refine_structured`` routine, which works
  for multilinear mesh types of any dimensionality.
  Note that a one-dimensional multilinear mesh is simultaneously affine.
  """

  def _refine(self) -> Self:
    return refine_structured(self)


class AffineMesh(Mesh):
  """
  Derived class for affine mesh types.
  Currently only used to group affine mesh types together. The ``_refine``
  method has to be overwritten by the derived class for it to function as
  intended. In the long run, we are opting for a more general approach to
  affine refinement as in ``MultilinearMesh``.
  """
  # XXX: currently affine meshes require special-tailored refinement methods.
  #      Write a method that can refine any affine mesh type, similar to
  #      ``_refine_structured``. This should be possible by taking the same
  #      approach as in `MultilinearMesh` while restricting the attention
  #      to the (hyper-)plane `x_0 + x_1 + x_2 + ... <= 1`.

  def _refine(self) -> Self:
    # XXX: this function is to be replaced by a general affine refinement
    #      method. For now, we just throw an error if the derived class
    #      doesn't overwrite.
    raise NotImplementedError('Every affine mesh type needs to implement'
                              ' its own refinement method.')


class HexMesh(MultilinearMesh):
  """
  Represents a hexahedral mesh.

              3_______ 7
             /|      /|
           1 _|____ 5 |
           |  |____|__|
           | / 2   | / 6
           |/______|/
           0       4

  ``_refine`` and ``_local_ordinances`` are implemented via inheritance of
  :class:`MultilinearMesh`.
  """

  reference_element = HEXAHEDRON

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 4, 6, 2, 1, 5, 7, 3]]

  @property
  def _submesh_type(self):
    return QuadMesh


class QuadMesh(MultilinearMesh):

  """
  Represents a quadrilateral mesh.

     1 _____ 3
      |     |
      |     |
      |_____|
     0       2
  """

  reference_element = QUADRILATERAL

  @property
  def _submesh_type(self):
    return LineMesh

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 3, 1]]

  def subdivide_elements(self):
    elements = self.elements[:, [[0, 1, 2], [2, 1, 3]]].reshape(-1, 3)
    return Triangulation(elements, self.points)


class Triangulation(AffineMesh):

  """
  Represents a triangulation.

      1
      |\
      |  \
      |    \
      |      \
      |        \
      |__________\
     0            2

  ``_local_ordinances`` is implemented by inheritance of :class:`AffineMesh`.
  """

  reference_element = TRIANGLE

  @classmethod
  def from_polygon(cls, *args, **kwargs):
    """ See the docstring of `triangulation_from_polygon.` """
    from splico.mesh._gmsh import triangulation_from_polygon
    return cls(*triangulation_from_polygon(*args, **kwargs))

  @property
  def _submesh_type(self):
    return LineMesh

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 1]]

  def _refine(self):
    """ See `_refine_Triangulation`. """
    return _refine_Triangulation(self)


# XXX: TetMesh. Leave as an exercise for Fabio.


class LineMesh(AffineMesh):
  """
  Represents a line mesh.
  Is both affine and multilinear.
  Uses `refine_structured` for refinement.
  """

  reference_element = LINE

  def _refine(self):
    return refine_structured(self)

  @property
  def _submesh_type(self):
    return PointMesh


class PointMesh(Mesh):
  """
  Represents a point mesh.
  Is always valid and has no submesh (calling submesh raises an error).
  Refinement returns itself.
  Evaluating it locally returns the points themselves.
  Neither affine nor multilinear.
  """

  reference_element = POINT

  def _refine(self):
    return super()._refine()  # return self

  def plot(self):
    plot_pointmesh(self)

  def is_valid(self, **kwargs):
    return True


def rectilinear(_points: AnyIntSeq) -> LineMesh | QuadMesh | HexMesh:
  """
  Rectilinear mesh in one, two or three dimensions.

  Parameters
  ----------
  _points : Sequence of integers or flat and strictly monotone np.ndarray.
      The dimensionality of the mesh follows from the length of the sequence.
      If an integer is encountered, it is converted to a linspace where the
      integer determines the number of steps.
      Else, it is assumed to be strictly monotone.

  Returns
  -------
  ret : :class:`LineMesh` or :class:`QuadMesh` or :class:`HexMesh`
      A rectilinear mesh whose dimensionality follows from the length
      of `_points`. The mesh vertices follow from a tensor product of all
      values generated by the conversion of `_points`.
  """

  if (0 < (dim := len(_points)) <= 3) is False:
    raise NotImplementedError("Expected between one and three vertex point"
                              f" arrays, found {len(_points)}.")

  # format to linspace if not already and make sure strictly monotone
  points = []
  for elem in _points:

    if isinstance(elem, Int):
      elem = np.linspace(0, 1, elem)

    assert isincreasing((elem := np.asarray(elem, dtype=float)))
    points.append(elem)

  # keep track of the length of each vector of abscissae for later reshape
  lengths = list(map(len, points))

  # convert to meshgrid and augment by zeros if necessary
  all_points = augment_by_zeros(flat_meshgrid(*points, axis=1), axis=1)

  element = np.asarray(list(product(*map(range, (2,)*dim)))).T
  indices = np.arange(len(all_points)).reshape(*lengths)
  ijk = flat_meshgrid(*(np.arange(n-1) for n in lengths), axis=0)

  # 3, nelems, 8
  ijk = ijk[:, :, _] + element[:, _]

  elements = indices[ tuple(ijk) ]

  return {1: LineMesh,
          2: QuadMesh,
          3: HexMesh}[dim](elements, all_points)
