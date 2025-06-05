from pyhaemo.util import np, frozen, frozen_cached_property, \
                         normalize as normalized, _, gauss_quadrature
from pyhaemo.types import Immutable, Float, Int, Numeric
from pyhaemo.spl import NDSpline, UnivariateKnotVector

from geo import project_onto_plane
from fit import fit_constrained
from notwist import compute_notwistframe

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


class Truncatable(Orientable):

  @abstractmethod
  def truncate(self, start: Optional[Numeric] = None,
                     end: Optional[Numeric] = None) -> Self:
    """
    Truncate the edge to the segment between points p0 and p1.
    p0 and p1 can be indices or positions along the edge.
    """
    ...

  @property
  @abstractmethod
  def length(self):
    pass


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


class Frame(Orientable):

  @classmethod
  def from_tangents(cls, tangents: ArrayLike, **kwargs):
    return cls(compute_notwistframe(normalized(tangents)), **kwargs)

  def __init__(self, frame: NDArray):
    self.frame = frozen(frame, dtype=float)
    assert len(self) >= 0, \
      NotImplementedError("Frame must have at least one frame.")
    assert self.frame.shape[1:] == (3, 3)

  def __len__(self):
    return len(self.frame)

  def __getitem__(self, item: Int) -> NDArray:
    elem = self.frame[item]
    if elem.ndim == 3:
      return self._edit(frame=elem)
    return elem

  def _neg(self) -> Self:
    return self._edit(frame=self.frame[::-1] * np.array([[1, -1, -1]]))

  def transport_all(self, vectors: ArrayLike, eps=1e-8,
                                              strict=True,
                                              normalize=True) -> NDArray:
    """
    Transport all vectors along the entire frame.
    The vectors should have the same shape as the frame.
    The return shape is (n, m, 3) where n is the number of frames and m
    is the number of vectors. If `vectors` has shape (3,), the return shape
    will be (n, 3).
    """
    shape = (vectors := np.asarray(vectors, dtype=float)).shape
    if len(shape) == 1:
      vectors = vectors[_, :]

    loc = np.linalg.solve(self[0], vectors.T).T

    if strict:
      assert (np.abs(loc[:, -1]) < eps).all(), \
        "Vector must be in the tangent space of the frame."

    # (n, 1, 3, 3) * (1, m, 1, 3) -> (n, m, 3)
    ret = (self.frame[:, _] * loc[_, :, _]).sum(-1).reshape((len(self),) + shape)
    if normalize:
      return normalized(ret)
    return ret

  def transport(self, vectors: ArrayLike, **kwargs) -> NDArray:
    """
    Transport vectors to the end of the frame.
    """
    return self[[0, -1]].transport_all(vectors, **kwargs)[1]

  def project_tangent_plane(self, vectors: ArrayLike, normalize=True) -> NDArray:
    """
    Return shape: (n, m, 3), where n is the number of frames and m
    is the number of vectors.
    If `vectors` has shape (3,), the return shape is (n, 3).
    """
    shape = (vectors := np.asarray(vectors, dtype=float)).shape
    if len(shape) == 1:
      vectors = vectors[_, :]

    # TODO: vectorize
    projections = np.stack([project_onto_plane(vectors, f[:, 2])
                            for f in self.frame], axis=0).reshape(len(self), *shape)

    if normalize:
      return normalized(projections)

    return projections

  def project_and_transport(self, vector: ArrayLike, **kwargs) -> NDArray:
    vector = np.asarray(vector, dtype=float)
    v0 = normalized(project_onto_plane(vector, self.frame[0, :, 2]))
    return self.transport(v0, **kwargs)


class VascularEdge(Truncatable):

  @classmethod
  def empty(cls) -> Self:
    """ Returns an empty edge with no points. """
    return cls(points=np.empty((0, 4), dtype=float))

  def __init__(self, points: ArrayLike, frame=None):
    self.points = frozen(points, dtype=float)
    assert len(self.points) >= 2, \
      NotImplementedError("Edge must have at least two points.")
    assert self.points.shape[1:] == (4,)

    self._frame = frame
    if self._frame is not None:
      self._frame = Frame(self._frame)
      assert len(self._frame) == self.npoints

  @property
  def npoints(self):
    return len(self.points)

  @property
  def frame(self):
    """
    Returns the frame of the edge.
    The frame is computed from the tangents of the edge.
    """
    if self._frame is None:
      spl = interpolate.make_interp_spline(self.verts,
                                           self.cpoints,
                                           k=min(len(self)-1, 3))
      self._frame = Frame(compute_notwistframe(spl(self.verts, nu=1)))

    return self._frame

  @frozen_cached_property
  def tangents(self):
    return self.cpoints[1:] - self.cpoints[:-1]

  @cached_property
  def toscipy(self):
    return interpolate.make_interp_spline(self.verts,
                                          self.points,
                                          k=min(len(self)-1, 3))

  @frozen_cached_property
  def cpoints(self) -> NDArray:
    return self.points[:, :3]

  @frozen_cached_property
  def radii(self) -> NDArray:
    return self.points[:, 3]

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
    return np.clip(np.concatenate([[0], self.dx.cumsum()]) / (self.length or 1), 0, 1)

  def _neg(self):
    return self._edit(points=self.points[::-1], frame=-self.frame)

  def __len__(self):
    return len(self.points)

  def __bool__(self):
    return bool(len(self))

  def is_empty(self) -> bool:
    """ Returns True if the edge is empty, i.e. has no points. """
    return not self

  def __or__(self, other: Self) -> Self:
    """
    Merge two PointSets into one.
    The connecting vertex will be taken as the average of the two vertices
    at the connection point.
    The frame will be recomputed.
    """
    if not isinstance(other, VascularEdge):
      return NotImplemented
    return self._edit(points=np.concatenate([self.points[:-1],
                                             (self.points[-1:] + other.points[:1]) / 2,
                                             other.points[1:]]),
                      frame=None)

  @return_self_if_empty
  def remove_duplicates(self, thresh=1e-12) -> Self:
    mask = np.concatenate([[True], self.dx > thresh])
    return self._edit(points=self.points[mask], frame=self.frame[mask])

  def split(self, positions: Int | Sequence[Int]) -> Tuple[Self, ...]:
    if isinstance(positions, Int):
      positions = positions,

    positions = sorted(positions)
    assert len(positions) == len(set(positions))

    if not positions:
      return self,

    pos = [0] + positions + [len(self)]
    assert all( a < b for a, b in zip(pos, pos[1:]) )

    points = [self.points[pos0: pos1+1] for pos0, pos1 in zip(pos, pos[1:])]
    frames = [self.frame[pos0: pos1+1] for pos0, pos1 in zip(pos, pos[1:])]

    return tuple( self._edit(points=mypoints, frame=myframe)
                  for mypoints, myframe in zip(points, frames) )

  def nearest_index(self, point: NDArray) -> Int:
    assert (point := np.asarray(point, dtype=float)).shape == (3,)
    return np.argmin( ((self.points[:, :3] - point) ** 2).sum(-1) )

  @return_self_if_empty
  def refit(self, xi: Int | NDArray):
    """
    Refits the edge to the given positions xi.
    The frame needs to be recomputed to ensure that all vectors remain
    orthogonal.
    """

    if isinstance(xi, Int):
      xi = np.linspace(0, 1, xi)

    xi = np.asarray(xi, dtype=float)
    assert 0 <= xi.min() <= xi.max() <= 1, "xi must be in the range [0, 1]."

    evalf = interpolate.make_interp_spline(self.verts,
                                           self.points,
                                           k=min(len(self)-1, 3))(xi)
    return self._edit(points=evalf, frame=None)

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

  def truncate(self, start: Optional[Int] = None,
                     end: Optional[Int] = None) -> Self:
    """
    Truncate the edge to the segment between points p0 and p1.
    p0 and p1 can be indices or positions along the edge.
    """
    if start is None and end is None:
      return self

    if start is None: start = 0
    if end is None: end = len(self)

    return self._edit(points=self.points[start: end],
                      frame=self.frame[start: end])

  def to_spline_edge(self, knotvector: Int | UnivariateKnotVector,
                           **fitkwargs) -> 'SplineEdge':
    """
    Converts the VascularEdge to a SplineEdge.
    The knotvector can be specified as an integer (number of knots) or
    as a UnivariateKnotVector object.
    """
    if isinstance(knotvector, Int):
      knotvector = UnivariateKnotVector(np.linspace(0, 1, knotvector+1))

    a, *ignore, b = knotvector.knots
    assert a == 0 and b == 1

    kv = knotvector.to_tensor()

    return SplineEdge(fit_constrained(kv,
                                      self.verts,
                                      self.points,
                                      **fitkwargs),
                      fit_constrained(kv,
                                      self.verts,
                                      self.frame.frame,
                                      **fitkwargs))


def flip_spline(spline: NDSpline):
  assert spline.nvars == 1
  return spline._edit(knotvector=[spline.knotvector[0].flip()],
                      controlpoints=spline.controlpoints[::-1])


class SplineEdge(Truncatable):

  def __init__(self, rcenterline: NDSpline, frame: NDSpline):
    self.rcenterline = NDSpline(rcenterline)
    self.frame = NDSpline(frame)

    assert self.rcenterline.ndim == 1

    assert self.rcenterline.shape == (4,) and \
           self.frame.shape == (3, 3)

    assert self.rcenterline.knotvector == self.frame.knotvector
    self.knotvector = self.rcenterline.knotvector

  def __iter__(self):
    yield from (self.rcenterline, self.frame)

  @property
  def interval(self):
    """
    Returns the interval of the spline edge.
    """
    return self.knotvector.knots[0][0], self.knotvector.knots[0][-1]

  @cached_property
  def cline(self) -> NDSpline:
    return self.rcenterline[:3]

  @cached_property
  def radius(self) -> NDSpline:
    return self.rcenterline[3]

  def _neg(self):
    rcline, frame = self
    return self._edit(rcenterline=flip_spline(rcline),
                      frame=flip_spline(frame) * np.array([1, -1, -1]))

  def split(self, *args, **kwargs):
    return tuple(map(self.__class__,
                     self.rcenterline.split(0, *args, **kwargs),
                     self.frame.split(0, *args, **kwargs)))

  def truncate(self, start: Optional[Float] = None,
                     end: Optional[Float] = None) -> Self:
    if start is None and end is None:
      return self

    if start is None:
      return self.split(xvals=(end,))[0]
    if end is None:
      return self.split(xvals=(start,))[1]

    return self.split(xvals=(start, end))[1]

  def sample(self, xi: Int | ArrayLike) -> VascularEdge:
    if isinstance(xi, Int):
      xi = np.linspace(*self.interval, xi)

    xi = np.asarray(xi, dtype=np.float64)
    return VascularEdge(self.rcenterline(xi), self.frame(xi))

  @cached_property
  def length(self):
    (knots,), (dx,) = self.knotvector.knots, self.knotvector.dx
    _weights, _points = gauss_quadrature(0, 1, self.knotvector.degree[0] + 1)

    weights = (dx[:, _] * _weights[_]).ravel()
    points = knots[:-1, _] + dx[:, _] * _points[_]
    cline = self.rcenterline[:3]

    return (weights * ((cline(points.ravel(), dx=1)**2).sum(1))**.5).sum()
