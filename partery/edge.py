from pyhaemo.util import np, frozen, frozen_cached_property
from pyhaemo.types import Immutable, Float, Int, Numeric

from abc import abstractmethod, ABCMeta
from typing import Tuple, Self, Optional, Sequence, Callable
from functools import cached_property, wraps
from collections.abc import Mapping

from numpy.typing import ArrayLike, NDArray

from scipy import interpolate


class Negatable(metaclass=ABCMeta):
  """ABC for objects that implement unary negation via __neg__."""

  @classmethod
  def __subclasshook__(cls, subclass):
    if cls is Negatable:
      return hasattr(subclass, "__neg__")
    return NotImplemented


def sget(mapping: Mapping[Negatable, Negatable], key: Negatable) -> Negatable:
  """
  Get an item from a mapping that supports symmetric negation.
  If the key is not found, it will return the negated value of the negated key.
  """
  try:
    return mapping[key]
  except KeyError:
    return -mapping[-key]


class Orientable(Immutable):

  @abstractmethod
  def _neg(self):
    """
    Returns a new instance of the edge with the orientation negated.
    """
    ...

  def __neg__(self) -> Self:
    if not hasattr(self, '__neg'):
      self.__neg = self._neg()
      self.__neg.__neg = self
    return self.__neg

  def __mul__(self, other: Int) -> Self:
    if other not in (-1, 1):
      return NotImplemented
    return self if other == 1 else -self

  __rmul__ = __mul__


class IntegerTuple(Orientable):

  def __init__(self, indices):
    self.indices = tuple(map(int, indices))

  def __len__(self) -> Int:
    return len(self.indices)

  def __getitem__(self, item) -> Int:
    ret = self.indices[item]
    if isinstance(ret, int):
      return ret
    return self._edit(indices=ret)

  def _neg(self) -> Self:
    return self._edit(indices=self.indices[::-1])

  def __iter__(self):
    yield from self.indices

  def renumber(self, mapping: dict[Int, Int]) -> Self:
    """ Renumber the indices according to the provided mapping.
        If an index is not found in the mapping, it will return the negated value
        of the negated index. """
    return self._edit(indices=[sget(mapping, i) for i in self.indices])


def return_self_if_empty(func: Callable) -> Callable:

  @wraps(func)
  def wrapper(self, *args, **kwargs):
    if not self: return self
    return func(self, *args, **kwargs)

  return wrapper


class VascularEdge(Orientable):

  @classmethod
  def empty(cls) -> Self:
    """ Returns an empty edge with no points. """
    return cls(points=np.empty((0, 4), dtype=float))

  def __init__(self, points: ArrayLike):
    self.points = frozen(points, dtype=float)
    assert len(self.points) >= 2 and self.points.shape[1:] == (4,)

  @property
  def npoints(self):
    return len(self.points)

  @frozen_cached_property
  def dx(self) -> NDArray:
    return ((self.points[1:, :3] - self.points[:-1, :3])**2).sum(-1)**.5

  @cached_property
  def length(self) -> Float:
    return self.dx.sum()

  @frozen_cached_property
  def verts(self) -> NDArray:
    if not self:
      return np.array([], dtype=float)
    return np.concatenate([[0], self.dx.cumsum()]) / (self.length or 1)

  def _neg(self):
    return self._edit(points=self.points[::-1])

  def __len__(self):
    return len(self.points)

  def __bool__(self):
    return bool(len(self))

  def is_empty(self) -> bool:
    """ Returns True if the edge is empty, i.e. has no points. """
    return not self

  def __or__(self, other: Self) -> Self:
    """ Merge two PointSets into one. The connecting vertex will be taken
        as the average of the two vertices at the connection point. """
    if not isinstance(other, VascularEdge):
      return NotImplemented
    return self._edit(points=np.concatenate([self.points[:-1],
                                             (self.points[-1:] + other.points[:1]) / 2,
                                             other.points[1:]]))

  @return_self_if_empty
  def remove_duplicates(self, thresh=1e-12) -> Self:
    mask = np.concatenate([[True], self.dx > thresh])
    return self._edit(points=self.points[mask])

  def split(self, positions: Int | Sequence[Int]) -> Tuple[Self, ...]:
    if isinstance(positions, Int):
      positions = positions,

    positions = sorted(positions)

    if not positions:
      return self,

    positions = [0] + positions + [len(self)]
    assert all( a <= b for a, b in zip(positions, positions[1:]) )

    points = [self.points[pos0: pos1+1] for pos0, pos1 in zip(positions, positions[1:])]

    return tuple( self._edit(points=mypoints) for mypoints in points )

  def nearest_index(self, point: NDArray) -> Int:
    assert (point := np.asarray(point, dtype=float)).shape == (3,)
    return np.argmin( ((self.points[:, :3] - point) ** 2).sum(-1) )

  @return_self_if_empty
  def refit(self, xi: Int | NDArray):

    if isinstance(xi, Int):
      xi = np.linspace(0, 1, xi)

    xi = np.asarray(xi, dtype=float)
    assert 0 <= xi.min() <= xi.max() <= 1, "xi must be in the range [0, 1]."

    spl = interpolate.make_interp_spline(self.verts,
                                         self.points,
                                         k=min(len(self)-1, 3))
    return self._edit(points=spl(xi))

  @return_self_if_empty
  def _refine(self):
    dx = self.dx / self.dx[-1:]
    xi = np.insert(dx, range(1, len(dx)), (dx[:-1] + dx[1:]) / 2.0)
    return self.refit(xi)

  def refine(self, n=1):
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._refine().refine(n-1)

  def truncate(self, p0: Optional[Int] = None,
                     p1: Optional[Int] = None) -> Self:
    """
    Truncate the edge to the segment between points p0 and p1.
    p0 and p1 can be indices or positions along the edge.
    """
    if p0 is None and p1 is None:
      return self

    if p0 is None: p0 = 0
    if p1 is None: p1 = len(self)

    return self._edit(points=self.points[p0: p1])
