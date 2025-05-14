"""
Reference elements. Used to define various mesh types in the `Mesh` class.
@author: Jochen Hinz
"""

from ..types import Singleton, SingletonMeta, FloatArray, Int
from ..util import freeze, round_result, np, flat_meshgrid

from typing import Tuple
from abc import abstractmethod
from itertools import product


class ReferenceElementMeta(SingletonMeta):
  """
  Metaclass for reference elements.
  The `ReferenceElement` class is a singleton class, i.e. there is only one
  instance of each reference element type. The metaclass ensures that the
  input to the class is in the correct format and that the class is called
  with the correct arguments.
  """

  def __call__(cls, *args, children_facets):
    # sort the children_facets to avoid duplicates resulting from passing
    # the same indices in a different order
    children_facets = tuple(sorted(map(tuple, children_facets)))
    return super().__call__(*args, children_facets=children_facets)


class ReferenceElement(Singleton, metaclass=ReferenceElementMeta):
  """
  Reference element base class.
  Requires the implementation of the `_local_ordinances` abstract method.
  Given `order >= 1`, the local ordinances refer to the nodal points inside
  the mesh's reference element of a Lagrangian basis of order `order`.
  For `order == 1` this should default to the reference element's vertices.

  Parameters
  ----------
  element_name : :class:`str`
      The name of the reference element type.
  ndims : :class:`int`
      The number of dimensions of the reference element.
  nverts : :class:`int`
      Number of vertices of the reference element.
  is_affine : :class:`bool`
      Indicates whether a linear map between an element and the reference
      element is affine or not.
  children_facets : :class:`Tuple[Tuple[int, ...], ...]`
      A tuple of tuples containing integers refering to the vertices of the
      element's facets in the right order.
  """

  # slots are used to reduce memory usage
  __slots__ = 'element_name', 'ndims', 'nverts', 'is_affine', 'children_facets'

  def __init__(self, element_name: str,
                     ndims: int,
                     nverts: int,
                     is_affine: bool,
                     *,
                     children_facets: Tuple[Tuple[Int, ...], ...]):

    # we don't do local type coercion here because the `Singleton` metaclass
    # already uses the input as key in a weakref dictionary and thus already
    # requires the input in the correct form.
    self.element_name = element_name
    self.ndims = ndims
    self.nverts = nverts
    self.is_affine = is_affine
    self.children_facets = children_facets
    assert all( getattr(self, name) >= 0 for name in ('ndims', 'nverts') )

  def __repr__(self) -> str:
    return '{}[name: {}, ndims: {}, nverts: {}]'.format(self.__class__.__name__,
                                                        self.element_name,
                                                        self.ndims,
                                                        self.nverts)

  @abstractmethod
  def _local_ordinances(self, order: int = 1):
    """
    Abstract method to be implemented by subclasses.
    Returns the local ordinances of the reference element
    for a given order of the Lagrangian basis.
    """
    pass


class MultilinearElement(ReferenceElement):
  """
  Multilinear reference element.
  The multilinear reference element is a reference element with a Cartesian
  product structure. The local ordinances are the Cartesian product of the
  local ordinances of the one-dimensional reference element.
  """

  def __init__(self, ndims: int, children_facets: Tuple[Tuple[Int, ...], ...]):
    assert (ndims := int(ndims)) >= 2
    name = {2: 'quadrilateral',
            3: 'hexahedron'}.get(ndims, f'unit_hypercube_{ndims}D')
    super().__init__(name, ndims, 2**ndims, False, children_facets=children_facets)

  @freeze
  @round_result
  def _local_ordinances(self, order: int) -> FloatArray:
    assert (order := int(order)) > 0
    x = np.linspace(0, 1, order+1)
    # for self.ndims == 0, the return shape should be (1, 0)
    return flat_meshgrid(*[x] * self.ndims or [np.array([])], axis=self.ndims and 1)


class SimplexElement(ReferenceElement):
  """
  Simplex reference element.
  The simplex reference element is a reference element with a simplex structure.
  The local ordinances are a subset of the Cartesian product of the local
  ordinances of the one-dimensional reference element. Namely, the local
  ordinances whose coordinates sum up to at most 1.
  """

  def __init__(self, ndims: int, children_facets: Tuple[Tuple[Int, ...], ...]):
    ndims = int(ndims)
    name = {0: 'point',
            1: 'line',
            2: 'triangle',
            3: 'tetrahedron'}.get(ndims, f'simplex_{ndims}D')
    super().__init__(name, ndims, ndims+1, True, children_facets=children_facets)

  @freeze
  @round_result
  def _local_ordinances(self, order: int) -> FloatArray:
    active_indices = \
        [i for i, mi in enumerate(product(*[range(2)]*self.ndims)) if sum(mi) <= 1]
    return MultilinearElement._local_ordinances(self, order)[active_indices]


quad_facets = (0, 1), (0, 2), (1, 3), (2, 3)
hex_facets = (0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), \
             (4, 5, 6, 7), (1, 3, 5, 7), (2, 3, 6, 7)

QUADRILATERAL = MultilinearElement(2, children_facets=quad_facets)
HEXAHEDRON = MultilinearElement(3, children_facets=hex_facets)


point_facets: Tuple[Tuple[Int, ...], ...] = tuple()
line_facets = (0,), (1,)
triangle_facets = (0, 1), (0, 2), (1, 2)

POINT = SimplexElement(0, children_facets=point_facets)
LINE = SimplexElement(1, children_facets=line_facets)
TRIANGLE = SimplexElement(2, children_facets=triangle_facets)


# tetrahedron leave as an exercise for Fabio
