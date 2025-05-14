"""
Module defining the NDSpline and NDSplineArray classes and related functions.

@author: Jochen Hinz
"""

from ..util import _round_array, np, frozen, augment_by_zeros, _
from ..types import Immutable, FloatArray, NumericArray, Float, \
                    Int, AnyIntSeq, AnyFloatSeq, LockableDict, Numeric, \
                    AnySequence, NumpyIndex, MultiNumpyIndex, ensure_same_class
from ..log import logger as log
from ..kron import KroneckerOperator
from splico.mesh.mesh import Mesh, rectilinear, mesh_union
from ._jit_spl import call
from .kv import UnivariateKnotVector, TensorKnotVector, as_TensorKnotVector, \
                KnotVectorType
from .aux import fit_greville
from .meta import NDSplineMeta

from typing import Sequence, Callable, Tuple, Self, List, Any, TypeVar, \
                   Optional, Dict
from types import GenericAlias, MethodType
from functools import reduce, wraps, cached_property, lru_cache, partial
from itertools import product, chain
from abc import abstractmethod

from scipy import interpolate
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray


sl = slice(_)


IMPLEMENTED_UFUNCS = np.add, np.subtract, np.multiply, np.divide, np.matmul
HANDLED_NDSPLINE_FUNCTIONS = LockableDict()


def try_broadcast_shapes(*shapes: AnyIntSeq) -> Tuple[int, ...]:
  """
  Try to broadcast the shapes of the input arrays. Throws a ValueError if
  broadcasting fails.
  """
  try:
    final_shape = np.broadcast_shapes(*shapes)
  except ValueError as ex:
    raise ValueError("Broadcasting failed with error message '{}'".format(ex))
  return final_shape


class __NDSpline_implementations__:
  """
  Dummy class for implementing various `NDSpline` ufuncs.
  They are added to `HANDLED_NDSPLINE_FUNCTIONS` to be used in the `NDSpline`
  __array_ufunc__ protocol.
  """

  @staticmethod
  def implements(np_function: np.ufunc):
    """ Following the suggestion in the numpy documentation. """
    def wrapper(func):
      HANDLED_NDSPLINE_FUNCTIONS[np_function] = func
      return func
    return wrapper

  @implements(np.add)
  def add(self, other):
    """
    Add two :class:`NDSpline`s. If not yet expressed in the same knotvector,
    will be prolonged to a unified one. Then, the controlpoints will be added.
    """

    final_shape = try_broadcast_shapes(self.shape, other.shape)
    ndims = len(final_shape)
    n, m = map(len, (self.shape, other.shape))

    # prolong to union knotvector
    kv = self.knotvector | other.knotvector
    self, other = self.prolong_to(kv), other.prolong_to(kv)

    # if the knotvectors are the same, simply add the controlpoints together
    return NDSpline(kv, self.controlpoints[(sl,) + (_,) * (ndims - n)]
                        + other.controlpoints[(sl,) + (_,) * (ndims - m)])

  @implements(np.subtract)
  def sub(self, other):
    return self + -other

  @implements(np.multiply)
  def mul(self, other):
    """
    Multiply two :class:`NDSpline`s. Creates a tensor product of the
    knotvector and controlpoints. The number of dependencies of the resulting
    spline is the sum of the dependencies of the input splines.
    """

    # tensor knotvector
    knotvector = self.knotvector * other.knotvector

    n, m = map(len, (self.shape, other.shape))

    # prepend ones to the shorter shaped NDSpline (default numpy behavior)
    myshape = (1,) * (m - n) + self.shape
    othershape = (1,) * (n - m) + other.shape

    final_shape = try_broadcast_shapes(myshape, othershape)

    # take outer product and reshape to (ncontrolpoints,) + shape
    cps = (self.controlpoints.reshape(-1, 1, *myshape) *
           other.controlpoints.reshape(1, -1, *othershape)).reshape(-1, *final_shape)

    return NDSpline(knotvector, cps)

  @implements(np.matmul)
  def matmul(self, other):
    """
    The matmul `@` operator is overloaded to perform true multiplication.
    The idea is to multiply the two spline functions and express the result
    in a knotvector that contains the product of the two.
    """
    knotvector_to = self.knotvector @ other.knotvector

    # Try to broadcast. Will fail if not possible.
    try_broadcast_shapes(self.shape, other.shape)

    n, m = map(len, (self.shape, other.shape))

    def func(*args):
      eval0, eval1 = self(*args, tensor=True), other(*args, tensor=True)
      assert eval0.shape[:1] == eval1.shape[:1]
      return eval0.reshape((-1,) + (1,) * (m - n) + eval0.shape[1:]) *  \
             eval1.reshape((-1,) + (1,) * (n - m) + eval1.shape[1:])

    return NDSpline(knotvector_to, fit_greville(knotvector_to, func))


# lock to prevent unwanted modifications
HANDLED_NDSPLINE_FUNCTIONS.lock()


def canonicalize_sum_args(fn: Callable) -> Callable:

  @wraps(fn)
  def wrapper(self, *args, axis=None):
    if args:
      assert axis is None
      axis = args[0] if len(args) == 1 else args
    if isinstance(axis, Int):
      axis = axis,
    if axis is None:
      axis = tuple(range(self.ndim))
    return fn(self, axis=axis)

  return wrapper


# we add underscore to distinguish from `np.typing.ArrayLike`
class _ArrayLike(Immutable, NDArrayOperatorsMixin):
  """
  Base class for array-like objects.
  Requires at least the following methods to be implemented explicitly:

  - sum
  - __iter__
  - __getitem__
  - __array_ufunc__

  as well as the ``shape`` attribute / property.

  Used for type hinting and duck-typing.
  """

  @property
  @abstractmethod
  def shape(self) -> Tuple[int, ...]:
    ...

  @abstractmethod
  def __getitem__(self, item: NumpyIndex | MultiNumpyIndex):
    ...

  @abstractmethod
  @canonicalize_sum_args
  def sum(self, axis) -> Self:
    ...

  @abstractmethod
  def __iter__(self):
    ...

  @abstractmethod
  def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    ...


class Spline(_ArrayLike):

  @property
  @abstractmethod
  def knotvector(self) -> KnotVectorType | NDArray[np.object_]:
    ...


# register `np.ndarray` as an `ArrayLike` because it is deemed a valid
# substitute for `ArrayLike` in many cases
_ArrayLike.register(np.ndarray)
T = TypeVar('T', bound='_ArrayLike')


def _add_one_axes(*inputs: T) -> Tuple[T, ...]:
  """
  Given two ArraLike objects, add axes to the shorter shaped object
  to make them broadcastable.
  """
  m = max(map(len, (inp.shape for inp in inputs)))
  return tuple( inp[(_,) * (m - len(inp.shape))] for inp in inputs )


def find_first_occurence(inputs: Sequence[_ArrayLike], item) -> int:
  for i, elem in enumerate(inputs):
    if elem is item:
      return i
  else:
    raise ValueError(f"Item {item} not found in inputs.")


class NDSpline(Spline, metaclass=NDSplineMeta):
  """
  Class representing an N-dimensional spline.

  Parameters
  ----------
  knotvector : :class:`splico.spl.kv.TensorKnotVector` or
               :class:`splico.spl.kv.UnivariateKnotVector`
      The (tensorial) knotvector associated with the spline. If not yet
      tensorial, will be converted to tensorial.
  controlpoints : :class:`np.ndarray` or any array-like
      Numpy array or any array-like that can be converted to Numpy array
      representing the controlpoints. Upon conversion, ``self.controlpoints``
      must satisfy ``self.controlpoints.shape[0] == self.knotvector.ndofs``.
      Here, ``self.controlpoints[i]`` denotes the (potentially tensorial)
      control point assigned to the i-th spline function in the basis that
      corresponds to ``self.knotvector``. Is rounded to the current precision
      as set by the `splico.util.global_precision` context manager and then
      frozen. Here, self.controlpoints.shape[1:] can be anything and then
      simply represents a vectorial / tensorial set of splines all with the
      same knotvector.

  Attributes
  ----------
  knotvector: :class:`splico.spl.kv.TensorKnotVector`
      The knotvector.
  controlpoints: :class:`np.ndarray`
      The controlpoints.


  When performing default ``numpy`` arithmetic on an NDSpline, the operations
  are by default performed on ``self.controlpoints[i]`` for all i. Hence, when
  performing an airthmetic operation with a :class:`np.ndarray` ``A``,
  ``A.shape`` has to be broadcastable with ``self.controlpoints.shape[1:]``
  rather than ``self.controlpoints.shape``.

  >>> spline
  ... NDSpline<4, 5, 3>
  >>> spline.controlponts.shape
  ... (10, 4, 5, 3)
  >>> type(A)
  ... np.ndarray
  >>> A.shape
  ... (2, 4, 1, 3)
  >>> (spline + A).shape  # (4, 5, 3) becomes (1, 4, 5, 3)
  ... (2, 4, 5, 3)
  ... # this won't work because (10, 4, 5, 3) + (2, 4, 1, 3) is not broadcastable
  >>> spline.controlpoints + A
  ... ERROR
  """

  _knotvector: TensorKnotVector
  controlpoints: FloatArray

  # methods inherited from `self.knotvector` via the metaclass
  refine: Callable
  ref_by: Callable
  add_knots: Callable
  raise_multiplicities: Callable

  # properties inherited from `self.knotvector`
  km: tuple
  knots: tuple
  degree: tuple
  greville: tuple
  repeated_knots: tuple

  @classmethod
  def one(cls, knotvector, shape=()):
    """
    Constant one function. Helpful when a function of one variable
    has to be made a function of the other variables as well.
    """
    knotvector = as_TensorKnotVector(knotvector)
    return cls(knotvector, np.ones((knotvector.ndofs,) + shape, dtype=float))

  @classmethod
  def from_exact_interpolation(cls, verts: AnyFloatSeq,
                                     data: AnyFloatSeq, **scipyargs):
    """
    Classmethod for creating a an exact interpolation of a verts, data pair
    using scipy. It is then wrapped as an NDSpline.

    Scipy only supports 1D interpolation for now but is quite sophisticated.
    We may add our own interpolation routines in the future.
    """

    # XXX: parameters

    verts = np.asarray(verts, dtype=float)
    data = np.asarray(data, dtype=float)

    # For now only 1D
    assert verts.ndim == data.ndim == 1, NotImplementedError

    k = scipyargs.setdefault('k', 3)
    spline = interpolate.InterpolatedUnivariateSpline(verts, data, **scipyargs)

    coeffs, knots = spline.get_coeffs(), spline.get_knots()
    knotvector = TensorKnotVector([UnivariateKnotVector(knots, degree=k)])
    return cls(knotvector, coeffs)

  @staticmethod
  def join_multiple(splines: Sequence['NDSpline'], direction: Int) -> 'NDSpline':
    assert splines
    assert all(isinstance(spl, splines[0].__class__) for spl in splines)
    return reduce(lambda x, y: x.join(y, direction), splines)

  @staticmethod
  def from_least_squares(knotvector: KnotVectorType, *args, **kwargs):
    """
    Docstring: see `splico.spl.kv.TensorKnotVector.fit`.
    """
    # Contrary to ``from_exact_interpolation`` this one works for any dimension
    # convert to TensorKnotVector if not alreay
    knotvector = as_TensorKnotVector(knotvector)
    # forward to `TensorKnotVector.fit` routine
    return knotvector.fit(*args, **kwargs)

  @classmethod
  def __class_getitem__(cls, dimension):
    assert isinstance(dimension, (int, type(...)))
    return GenericAlias(cls, dimension)

  def __init__(self, knotvector: KnotVectorType, controlpoints: AnyFloatSeq):
    self._knotvector = as_TensorKnotVector(knotvector)
    self.controlpoints = frozen(_round_array(controlpoints), dtype=float)
    assert self.controlpoints.shape[0] == self.knotvector.ndofs

  @property
  def unity(self) -> Self:
    """
    Return the unity function for the current knotvector.
    """
    shape = self.controlpoints.shape[:1] + (1,) * self.ndim
    return self._edit(controlpoints=np.ones(shape, dtype=float))

  @property
  def knotvector(self) -> TensorKnotVector:
    return self._knotvector

  @cached_property
  def controlpoints_T(self) -> FloatArray:
    """
    Transposed controlpoints.
    """
    return np.ascontiguousarray(np.moveaxis(self.controlpoints, 0, -1))

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}<{', '.join(map(str, self.shape))}>"

  @property
  def shape(self) -> Tuple[int, ...]:
    """ The shape of the vectorial spline. """
    return self.controlpoints.shape[1:]

  @property
  def nvars(self) -> int:
    """ Number of dependencies. """
    return self.knotvector.ndim

  @property
  def tcontrolpoints(self) -> FloatArray:
    """
    Represent the controlpoints tensorially.
    >>> spline.controlpoints.shape
    ... (36, 2, 3)
    >>> spline.knotvector.dim
    ... (3, 3, 4)
    >>> spline.tcontrolpoints.shape
    ... (3, 3, 4, 2, 3)
    """
    return self.controlpoints.reshape(self.knotvector.dim + self.shape)

  def transpose_dependencies(self, axes: AnyIntSeq) -> Self:
    """
    Reorder the dependencies of the spline according to the given axes.
    """
    if (axes := tuple(map(int, axes))) == tuple(range(self.nvars)):
      return self

    tail = tuple(range(self.nvars, self.nvars + self.ndim))

    knotvector = TensorKnotVector(np.array(list(self.knotvector))[list(axes)])
    cps = np.transpose(self.tcontrolpoints, axes=axes + tail).reshape(-1, *self.shape)

    return self._edit(knotvector=knotvector, controlpoints=cps)

  def insert_dependencies(self, axes: Int | AnyIntSeq,
                                knotvectors: Self | Sequence[Self]) -> Self:
    """
    Insert new dependencies at specified axes. Here the `axes` refer to
    the knotvectors, not the control points.

    Behaves the same way as `np.insert` applied to a vector containing the
    axis numbers in ascending order.
    """
    if isinstance(axes, Int):
      axes = axes,
    if isinstance(knotvectors, self.__class__):
      knotvectors = knotvectors,

    axes = tuple(axes)
    knotvectors = tuple(knotvectors)

    assert (naxes := len(axes)) == len(knotvectors), \
      f'Number of axes {naxes} must match number of knotvectors {len(knotvectors)}'

    if not axes: return self

    nvars = self.nvars
    allknotvectors = tuple(self.knotvector) + tuple(knotvectors)

    # shape of the tensor control points after inserting the new axes
    # and adding axes of dimension 1 in the new axes' places
    oneshape = tuple(np.insert(self.tcontrolpoints.shape[:self.nvars],
                               np.array(axes, dtype=int),
                               np.ones(naxes)))

    # `indices` represents a shuffling of the knotvectors
    indices = np.insert(np.arange(nvars, dtype=int),
                        np.array(axes, dtype=int),
                        np.arange(nvars, nvars+naxes, dtype=int))

    # shuffle according to `indices`
    knotvector = TensorKnotVector([ allknotvectors[i] for i in indices ])
    tcps = np.broadcast_to(self.tcontrolpoints.reshape(oneshape + self.shape),
                           tuple(kv.dim for kv in knotvector) + self.shape)

    return self._edit(knotvector=knotvector,
                      controlpoints=tcps.reshape(-1, *self.shape))

  def prolong_to(self, knotvector_to: TensorKnotVector, verbose=True) -> Self:
    """
    Prolong the spline to a new knotvector. The prolongation takes place over
    the greville abscissae of the new knotvector unless the new knotvector
    has interior knots with (degree + 1)-fold repetition. In this case the
    fit takes place over the gauss abscissae of the new knotvector.
    """
    if knotvector_to == self.knotvector:
      return self

    if knotvector_to <= self.knotvector and verbose:
      log.warning("Prolongation will be inexact because the knotvectors "
                  "are not nested.")

    cps = fit_greville(knotvector_to, lambda *args: self(*args, tensor=True))
    return self._edit(knotvector=knotvector_to, controlpoints=cps)

  def degree_elevate(self, dp: Int | AnyIntSeq) -> Self:
    if isinstance(dp, Int):
      dp = (dp,) * self.nvars

    if all(_dp == 0 for _dp in dp):
      return self

    tkv = self.knotvector.degree_elevate(..., dp)

    return self.prolong_to(tkv)

  def __call__(self, *positions: FloatArray,
                     tensor: bool = False,
                     dx: Int | AnyIntSeq = ()):
    """
    Evaluate the spline in a set of points.

    Parameters
    ----------
    positions : :class:`tuple` of :class:`np.ndarray` or any array-like
        Positional arguments of the x, y, z, ... coordinates all represented
        by a flat array-like, such as :class:`np.ndarray`.
        Must satisfy all(len(pos) == len(positions[0]) for pos in positions)
        and len(positions) == len(self.knotvector) unless ``tensor`` is True,
        see below.
    tensor : :class:`bool`
        If true, the spline is evaluated over a meshgrid of the positions, i.e.,
        positions is converted to tuple(map(np.ravel, np.meshgrid(*positions))).
    dx : `Int` or sequence of `Int`
        Derivative order of the evaluation. Must satisfy:
        len(dx) == len(self.knotvector).
        If given by a :class:`int` it is repeated len(self.knotvector) times.

    Returns
    -------
    ret : :class:`np.ndarray``
        The evaluation array. Has shape
          (len(positions[0]),) + self.controlpoints.shape[1:]
        if tensor is False and
          (np.prod([len(po) for pos in positions]),) + self.controlpoints.shape[1:]
        else.
    """
    if isinstance(dx, Int):
      assert self.nvars  # 0-D spline doesn't have any derivatives
      dx = (dx,) * self.nvars
    dx = dx or (0,) * self.nvars
    assert len((dx := tuple(map(int, dx)))) == len(positions) == self.nvars

    if not positions:  # 0-D spline
      return self.controlpoints[0].copy()

    positions = tuple(map(lambda x: np.asarray(x, dtype=float), positions))
    assert all( pos.ndim == 1 for pos in positions )

    # if not evaluated tensorially, stack along first axis
    # if length don't match, we get an error here
    if not tensor:
      # reshape to matrix shape, if self.shape == (), np.prod((), dtype=int) == 1
      controlpoints = self.controlpoints_T.reshape(np.prod(self.shape, dtype=int), -1)
      positions = np.stack(positions, axis=1)
      ret = call(positions, self.repeated_knots, self.degree, controlpoints, dx)
    else:
      # tensorial evaluation
      op = KroneckerOperator(self.knotvector.collocate(*positions, dx=dx))
      ret = op.T @ self.controlpoints

    # return in original shape
    return ret.reshape(-1, *self.shape)

  def tensorcall(self, *args, **kwargs):
    """ partial(self, tensor=True) """
    return self(*args, tensor=True, **kwargs)

  def __getitem__(self, item: Any) -> Self:
    """
    Same as np.ndarray.__getitem__ with the difference that the zeroth axis
    is ignored and the broadcasting etc. is applied to all axes with index > 0.
    >>> spline.controlpoints.shape
    ... (54, 2, 3)
    >>> spline[_].controlpoints.shape
    ... (54, 1, 2, 3)
    """
    if not isinstance(item, tuple):  # single integer or slices
      item = item,
    return self._edit(controlpoints=self.controlpoints[(sl,) + item])

  def reshape(self, *shape) -> Self:
    """
    Reshape the NDSpline.
    """
    if len(shape) == 1 and not isinstance(shape[0], Int):
      shape, = shape
    return self._edit(controlpoints=self.controlpoints.reshape(len(self.controlpoints), *shape))

  @property
  def ndim(self):
    """ Dimensionality of the spline (not the number of dependencies). """
    return self.controlpoints.ndim - 1

  def __iter__(self):
    """ Iterate over the each sub-spline the same way as a numpy array. """
    if not self.shape:
      raise TypeError('iteration over 0-d array')
    yield from (self._edit(controlpoints=controlpoints)
                        for controlpoints in self.controlpoints.swapaxes(0, 1))

  def ravel(self) -> Self:
    """ Ravel the tensorial spline. """
    shape = self.controlpoints.shape[:1] + (self.shape and (-1,))
    return self._edit(controlpoints=self.controlpoints.reshape(shape))

  def __neg__(self) -> Self:
    return self._edit(controlpoints=-self.controlpoints)

  @canonicalize_sum_args
  def sum(self, axis) -> Self:
    """
    Same as np.sum but applied to the tail of self.controlpoints.
    >>> type(spl0)
    ... splico.spl.spline.NDSpline
    >>> spl0.shape
    ... (2, 3)
    >>> spl0.controlpoints.shape
    ... (54, 2, 3)
    >>> spl1 = spl0.sum(1)
    >>> spl1.shape
    ... (2,)
    >>> spl1.controlpoints.shape
    ... (54, 2)
    """
    assert all(-self.ndim <= ax < self.ndim for ax in axis)
    assert len(axis) <= self.ndim

    # increment all summation axes by one in order to apply summation to
    # self.controlpoint's tail.
    axis = tuple(ax % self.ndim + 1 for ax in axis)
    return self._edit(controlpoints=np.sum(self.controlpoints, axis=axis))

  @property
  def T(self) -> Self:
    """
    >>> spl0.shape
    ... (2, 3, 4)
    >>> spl0.controlpoints.shape
    ... (54, 2, 3, 4)
    >>> spl1 = spl0.T
    >>> spl1.shape
    ... (4, 3, 2)
    >>> spl1.controlpoints.shape
    ... (54, 4, 3, 2)
    """
    return self._edit(knotvector=self.knotvector,
                      controlpoints=self.controlpoints_T)

  def sample_mesh(self, mesh: Mesh, boundary=True, **mesh_union_kwargs) -> Mesh:
    """
    Sample meshes from `self`.

    Parameters
    ----------
    mesh : :class:`splico.mesh.Mesh`
      The mesh's points serve as the evaluation points to `self` for sampling
      a mesh from a spline with target space R^3. The connectivity, elements
      and element types of the sampled mesh follow directly from `mesh`.
      Must satisfy mesh.ndims == len(self). If mesh.ndims < 3, only the first
      mesh.ndims columns of mesh.points are utilized for sampling and augmented
      by zeros to be a manifold in R^3.
      All meshes in `tail`, where `self.shape = *head, tail` are unified using
      a mesh union.
    boundary : :class:`bool`
      Whether or not the union should be taken over the boundary.
    mesh_union_kwargs: :class:`dict`
      Forwarded to `mesh_union`.

    Returns
    -------
    sampled_mesh: :class:`splico.mesh.Mesh`
        The sampled mesh resulting from taking the union of all meshes sampled
        from the splines in self.shape[:-1].
    """
    assert self.shape[-1:] == (3,)
    assert 0 < self.nvars == mesh.ndims <= 3

    self = self.reshape(-1, 3)

    points = self(*mesh.points.T[:self.nvars])
    meshes = [mesh._edit(points=augment_by_zeros(ps, axis=1))
                                              for ps in points.swapaxes(0, 1)]
    return mesh_union(*meshes, boundary=boundary, **mesh_union_kwargs)

  def quick_sample(self, n: Int | AnyIntSeq = 11, **kwargs) -> Mesh:
    """
    Quickly sample a mesh by passing the number of evaluation points in each
    direction.
    """
    if isinstance(n, Int):
      n = (n,) * self.nvars
    assert len(n) == self.nvars

    sample_mesh = rectilinear(n)

    return self.sample_mesh(sample_mesh, **kwargs)

  def _split(self, direction: Int, position: Int) -> Tuple[Self, Self]:
    """
    Split a spline along a given direction at a given position.
    The split takes place at the knotvalue that corresponds to the given position
    in the knotvector's knotvalues along the given direction.

    Parameters
    ----------
    spl : :class:`NDSplineArray`
        The spline that is to be split.
    direction : :class:`int`
        The direction along which the spline is to be split.
    position : :class:`int`
        The position in the knotvector's knotvalues along the given direction
        at which the spline is to be split.

    Returns
    -------
    :class:`NDSplineArray`
        The two spline arrays that result from the split.
    """
    amount = self.degree[direction] + 1 - self.km[direction][position]

    self = self.raise_multiplicities([direction], [position], [amount])
    mykv = self.knotvector[direction]

    cp_position = mykv.km[:position].sum()

    kv0, kv1 = [ mykv._edit(knotvalues=kv, knotmultiplicities=km) for kv, km
                 in zip((mykv.knots[:position+1], mykv.knots[position:]),
                        (mykv.km[:position+1], mykv.km[position:]))          ]

    shp = self.shape
    tcps = self.tcontrolpoints

    cp0 = tcps[(sl,) * direction + (slice(_, cp_position),)].reshape(-1, *shp)
    cp1 = tcps[(sl,) * direction + (slice(cp_position, _),)].reshape(-1, *shp)

    kvs = self.knotvector.knotvectors

    tkv0, tkv1 = (kvs[:direction] + (kv,) + kvs[direction+1:] for kv in (kv0, kv1))

    return self._edit(knotvector=tkv0, controlpoints=cp0), \
           self._edit(knotvector=tkv1, controlpoints=cp1)

  def split(self, direction: Int, positions: Optional[Int | AnyIntSeq] = None,
                                  xvals: Optional[Float | AnyFloatSeq] = None) -> Tuple[Self, ...]:

    if positions is None:
      assert xvals is not None, \
        "Exactly one of `positions` and `xvals` must be given."

    if xvals is not None:
      self = self.add_knots(direction, knotvalues=[xvals])
      positions = np.searchsorted(self.knots[direction], np.asarray(xvals))
      return self.split(direction, positions=positions)

    assert positions is not None

    if isinstance(positions, Int):
      positions = positions,

    ret: Tuple[Self, ...] = self,
    for pos0, pos1 in zip(chain((0,), positions), positions):
      ret = ret[:-1] + ret[-1]._split(direction, pos1 - pos0)

    return ret

  @ensure_same_class
  def _join(self, other: Self, direction: Int) -> Self:
    """
    Join two splines along a given direction.
    The two splines must have compatible knotvectors along the given direction.

    Parameters
    ----------
    other : :class:`NDSpline`
        The spline that is to be joined with the current spline.
    direction : :class:`int`
        The direction along which the two splines are to be joined.

    Returns
    -------
    :class:`NDSpline`
        The spline that results from the join.
    """
    assert self.shape == other.shape

    kv0, kv1 = self.knotvector[direction], other.knotvector[direction]
    assert kv0.knotvalues[-1] == kv1.knotvalues[0] and kv0.degree == kv1.degree
    assert self.knotvector[:direction] == other.knotvector[:direction] and \
           self.knotvector[direction+1:] == other.knotvector[direction+1:]

    kvn = kv0._edit(knotvalues=kv0.knotvalues + kv1.knotvalues[1:],
                    knotmultiplicities=kv0.knotmultiplicities[:-1]
                                       + (kv0.degree,)
                                       + kv1.knotmultiplicities[1:])
    tkv = TensorKnotVector([kvn if i == direction else kv
                                        for i, kv in enumerate(self.knotvector)])

    tcp0, tcp1 = self.tcontrolpoints, other.tcontrolpoints

    av = (tcp0[(sl,)*direction + (slice(-1, _),)] +
          tcp1[(sl,)*direction + (slice(0, 1),)]) / 2

    cp = np.concatenate([ tcp0[(sl,)*direction + (slice(0, -1),)],
                          av,
                          tcp1[(sl,)*direction + (slice(1, _),)] ],
                          axis=direction).reshape(-1, *self.shape)

    return self._edit(knotvector=tkv, controlpoints=cp)

  def join(self, others: Self | Sequence[Self], direction: Int) -> Self:
    if isinstance(others, self.__class__):
      others = others,
    assert all(other.__class__ is self.__class__ for other in others), \
      "All splines must be of the same class."
    return reduce(lambda x, y: x._join(y, direction), (self,) + tuple(others))

  def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs) -> Self | 'NDSplineArray':
    """
    Numpy arithmetic is handled via the __array_ufunc__ protocol.
    When the inputs are all of type :class:`NDSpline`, the arithmetic operation
    is delegated to the corresponding function in :class:`LockableDict`
    ``HANDLED_NDSPLINE_FUNCTIONS``.

    In case arithmetic between ``self`` and some :class:`np.ndarray` is
    performed, the operation is handled as described in this class's docstring.

    If any of the inputs is of type :class:`SplineCollection`, the operation
    will be handled in largely the same way but the return type will be
    :class:`SplineCollection`, not :class:`NDSpline`.
    """
    # XXX: add support for kwargs. This is tricky because some cannot be
    #      supported by immutable classes.
    if method != '__call__' or kwargs or ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    # Split into instances of the same class and of other classes.
    # If of other class type, needs to be convertable to np.ndarray or
    # of be an instantiation of class `SplineArray`.
    myclass: List[Self] = []
    notmyclass: List[Numeric | NumericArray | 'NDSplineArray'] = []
    for inp in inputs:
      (myclass if isinstance(inp, NDSpline) else notmyclass).append(inp)

    if len(myclass) > 1:  # this block handles interactions between NDSplines
      # since we only handle __call__ for now, the length of `inputs` is
      # two. If both are `NDSpline`, handle them using the implemented protocols.
      if ufunc not in HANDLED_NDSPLINE_FUNCTIONS:
        return NotImplemented
      func = HANDLED_NDSPLINE_FUNCTIONS[ufunc]
      # use a reduce for future handling of more than two inputs
      return reduce(lambda x, y: func(x, y), myclass)

    # What follows handles NDSpline and np.ndarray-like
    # get the ufunc
    func = getattr(ufunc, method)

    # If any of the inputs is of type `SplineArray`, let `NDSplineArray`
    # handle the operation.
    if any(isinstance(inp, NDSplineArray) for inp in inputs):
      return NotImplemented

    # for now only allow the handling of IMPLEMENTED_UFUNCS
    if ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    # at this point there's at most one NDSpline but it need not be the
    # first entry. However to correctly determine the shape of the output
    # the shape of the NDSpline's controlpoints array needs to be known.
    # We find the index `pivot` of the NDSpline in `inputs` and pass it to the
    # `_vectorize_first_axis` function

    pivot = find_first_occurence(inputs, self)

    # operate on `args` to maintain the correct order of the inputs
    args = [inp.controlpoints if isinstance(inp, NDSpline)
                              else np.asarray(inp) for inp in inputs]

    # apply operation with input to controlpoints along the first axis

    controlpoints = _vectorize_first_axis(func,
                                          pivot,
                                          *args,
                                          **kwargs)

    return self._edit(knotvector=self.knotvector, controlpoints=controlpoints)


def _vectorize_first_axis(op: Callable, pivot: Int,
                                        *args: NumericArray, **kwargs):
  """
  Vectorization along the first axis.

  Inputs are broadcast to a form that the operation is performed
  len(controlpoints) times along all axes in controlpoints.shape[1:].

  Here, `controlpoints` refers to the array contained in `args` at
  position `pivot`.
  """
  assert pivot < len(args)

  head, (controlpoints, *tail) = list(args[:pivot]), args[pivot:]

  if not controlpoints.shape:  # 0-D case, no broadcasting required
    return op(*args, **kwargs)

  n, *shape_tail = controlpoints.shape

  # find the output shape by taking the pivot's tail and then the shapes of all
  # other inputs
  output_shape_tail = try_broadcast_shapes(shape_tail,
                                           *map(lambda x: x.shape, head + tail))

  # add artificial axes after the first axis if necessary
  p, q = len(output_shape_tail), len(shape_tail)
  controlpoints = controlpoints[(sl,) + (_,) * (p - q)]
  controlpoints = np.broadcast_to(controlpoints, (n,) + output_shape_tail)

  _args = []
  for i, arg in enumerate(args):
    if i == pivot:
      _args.append(controlpoints)
    else:
      # first broadcast to shape (1,) + output_shape_tail, then to
      # (n,) + output_shape_tail
      _args.append( np.broadcast_to(np.broadcast_to(arg, output_shape_tail)[_],
                                                    (n,) + output_shape_tail) )

  return op(*_args, **kwargs)


@lru_cache
def sample_mesh_from_knotvector(tkv: TensorKnotVector, n: Tuple[Int, ...]) -> Mesh:
  """
  Create a sampling mesh from a knotvector.
  """
  assert len(n) == tkv.ndim
  ranges = [(kv.knotvalues[0], kv.knotvalues[-1]) for kv in tkv]
  return rectilinear([np.linspace(*r, num) for r, num in zip(ranges, n)])


def _object_array(arr: Any) -> NDArray:
  ret = np.empty(len(arr), dtype=object)
  for i, elem in enumerate(arr):
    ret[i] = elem
  return ret


is_objarr = lambda arr: isinstance(arr, np.ndarray) and arr.dtype is np.dtype('O')


class NDSplineArray(Spline):
  """
  Array of splines.
  The knotvectors of all splines may differ but each spline must have the same
  shape (and also the same number of dependencies, but this will change).

  Derives from :class:`np.ndarray` and is intended to be used as a container
  for multiple :class:`NDSpline`s which are all of the same shape.

  The class wraps a :class:`np.ndarray` ``arr`` of :class:`NDSpline`s and
  therefore has an effective shape of ``arr.shape + elemshape`` where
  ``elemshape`` is the shape of each element of the array.

  The class provides a method ``expand`` that expands elements of the
  array into a new instantiation of :class:`NDSplineArray`. So, if the array
  has shape (n0, n1, n2, ...) and the array's elements have shape
  (m0, m1, m2, ...), the expanded array will have shape (n0, n1, n2, ..., m0)
  while the elements will have shape (m1, m2, ...).

  The class additionally provides a method ``_contract`` that performs the inverse
  operation of ``expand``. This operation is only possible if the elements
  self.arr[..., i] all have the same knotvector for all i (they may differ
  between the i's). `_contract` may be alternatively invoked by calling
  `expand` with a negative argument.

  All arithmetic operations that are supported by the :class:`NDSpline` class
  are simultaneously supported by the :class:`NDSplineArray` class.
  This enables passing an instance of :class:`NDSplineArray` to any function
  that expects an instance of :class:`NDSpline`. Alternatively, the function
  can be implemented to expect an instance of :class:`NDSplineArray` and
  an input of type :class:`NDSpline` can be converted to an instance of
  :class:`NDSplineArray` by invoking `NDSplineArray(spline)`.
  Both instances can be used interchangeably in the function thanks to
  duck-typing. However, an input of type :class:`NDSpline` will generally
  be a bit more efficient thanks to more vectorized operations.

  Example
  -------

  >>> B = ellipse(1, 1, 10)  # NDSplineArray of shape (5, 3)
  >>> B.shape
  ... (5, 3)
  >>> B._shape
  ... ()
  >>> B._elemshape
  ... (5, 3)
  >>> B = B.expand()  # absorb the first axis of the elements into the array
  >>> B.shape
  ... (5,)
  >>> B.elements.shape
  ... (3,)
  >>> B.sum().shape  # sum all NDSpline elements
  ... ()
  >>> B.sum(0).shape  # the shape of the elements shouldn't change
  ... (3,)
  >>> B.sum(0)._shape
  ... ()
  >>> B.sum(0)._elemshape
  ... (3,)

  Arithmetic operations can either be performed between instances of
  the NDSplineArray class or between an instance of NDSplineArray and an
  instance of NDSpline. In the latter case, the NDSpline is first coerced into
  an NDSplineArray with a single element.
  Also, operations between NDSplineArray and np.ndarray or types that can be
  converted to np.ndarray are supported. In this case, the operation is
  performed on the controlpoints of the NDSplineArray elements as implemented
  by the __array_ufunc__ protocol of the NDSpline class.

  >>> B.shape
  ... ()
  >>> (B + B).ravel()[0].controlpoints == 2 * B.ravel()[0].controlpoints
  ... [True, True, True, ..., True]

  Parameters
  ----------
  arr : Array-like of `NDSpline`s or just a single `NDSpline`.
      Input array-like containing `NDSpline`s.

  ----------

  NOTE THAT THIS CLASS IS AN EXPERIMENTAL FEATURE AND MAY NOT BEHAVE AS
  EXPECTED IN ALL SITUATIONS. IT IS SUBJECT TO CONTINUOUS DEVELOPMENT AND
  IMPROVEMENT.
  """

  # TODO: in the `contract` and `expand` methods, still a lot of unnecessary
  #       copies are made. We keep this somewhat inefficient solution for now
  #       for the sake of safety. The next milestone is to make the class
  #       more efficient by using array views instead of copies.

  # TODO: introduce a subclass that additionally introduces a topology
  #       between the quadrilateral spline patches plus knotvector
  #       compatibility checks. This would ultimately lead up to a class
  #       that can directly interact with IGA solvers (such as nutils)
  #       for more sophisticated spline operations.

  # methods via `__getattr__` vectorization
  refine: Callable
  ref_by: Callable
  add_knots: Callable
  raise_multiplicities: Callable

  # properties inherited via `__getattr__` vectorization
  km: NDArray
  knots: NDArray
  degree: NDArray
  greville: NDArray
  repeated_knots: NDArray

  def __init__(self, arr: NDArray[np.object_] | NDSpline | Sequence[NDSpline]):
    self.__getattr_cache: Dict[str, Callable | NDArray | Self] = {}
    self.arr = frozen(arr, dtype=NDSpline)
    self._shape = self.arr.shape
    self._elemshape = self.arr.ravel()[0].shape
    self._dtype = self.arr.ravel()[0].__class__
    assert issubclass(self._dtype, NDSpline)
    assert all(isinstance(elem, self._dtype) and elem.shape == self._elemshape
                                                  for elem in self.arr.ravel())
    if any( elem.nvars != self.nvars for elem in self.arr.ravel() ):
      raise NotImplementedError("Currently, all elements must have the same "
                                "number of dependencies.")

  def __repr__(self) -> str:
    shapestr = ', '.join(map(str, self._shape))
    elemstr = '(' + ', '.join(map(str, self._elemshape)) + ')'

    if not self._shape: string = elemstr
    elif not self._elemshape: string = shapestr
    else: string = f"{shapestr}, {elemstr}"

    return f"{self.__class__.__name__}<{string}>"

  @staticmethod
  def _canonicalize_getattr_args(shape, *args, **kwargs):
    args = tuple(arg if is_objarr(arg) else np.asarray(arg, dtype=object) for arg in args)
    kwargs = { k: v if is_objarr(v) else np.asarray(v, dtype=object)
               for k, v in kwargs.items() }
    return tuple( np.broadcast_to(arg, shape) for arg in args ), \
           { k: np.broadcast_to(v, shape) for k, v in kwargs.items() }

  @staticmethod
  def _broadcast_arrshapes(*spls: 'NDSplineArray') -> Tuple['NDSplineArray', ...]:
    """
    Given two NDSplineArrays, broadcast the shape of their `self.arr` attribute
    to a compatible shape. This allows performing an arithmetic operation on
    the entries of the two `self.arr` attributes in pairs.

    For now we only handle two inputs at a time.

    TODO: add example with shapes.
    """
    assert spls
    assert all(isinstance(spl, spls[0].__class__) for spl in spls), \
      f"Cannot broadcast {spls} of different classes."

    if len(spls) == 1:
      return spls[0],

    # for now only two inputs are supported, contract as much as possible
    spls = tuple(map(lambda x: x.contract_all(), _add_one_axes(*spls)))
    m = max(map(lambda spl: spl._ndim, spls))

    # make sure the shapes are compatible
    shp = try_broadcast_shapes(*map(lambda spl: spl.shape, spls))

    # expand the arrays to the same dimension
    return tuple(spl.expand(max(0, m - spl._ndim)) for spl in spls)

  @staticmethod
  def vectorize_NDSpline_func(op: Callable):
    """
    Vectorize a function that operates between :class:`NDSpline` objects to
    make it compatible with :class:`NDSplineArray` objects.

    This is accomplished by broadcasting the input shapes and `self.arr` shapes
    to a common shape and applying the operation to the elements of `self.arr`
    pairwise in a loop.
    """

    def wrapper(*spls: 'NDSplineArray', **kwargs):
      assert all(isinstance(spl, spls[0].__class__) for spl in spls), \
        f"Cannot apply {op.__name__} to {spls} of different classes."

      # Make spline.arr have all the same dimension
      # They may still have one axes.
      spls = NDSplineArray._broadcast_arrshapes(*spls)
      arrshape = try_broadcast_shapes(*map(lambda spl: spl.arr.shape, spls))

      arrs = [ np.broadcast_to(spl.arr, arrshape) for spl in spls ]

      ret = np.empty(arrshape, dtype=object)
      for multiindex in product(*map(range, arrshape)):
        # apply the operation to the elements of the arrays
        ret[multiindex] = op(*(arr[multiindex] for arr in arrs), **kwargs)

      return spls[0]._edit(arr=ret)

    return wrapper

  def _vectorize_method(self, name) -> Callable:
    """
    Vectorize a method of the NDSpline class.
    The method is called on each element of the array and the results are
    returned in an array of dtype object. If the return type is as `self._dtype`,
    the result is wrapped in `self.__class__`.

    The arguments passed to the method are broadcasted to the shape of `self.arr`.
    """

    def func(*args, **kwargs):
      args, kwargs = self._canonicalize_getattr_args(self._shape, *args, **kwargs)
      ret = np.empty(self._shape, dtype=object)
      for mindex in product(*map(range, self._shape)):
        func = getattr(self.arr[mindex], name)
        ret[mindex] = func(*(arg[mindex] for arg in args),
                           **{k: v[mindex] for k, v in kwargs.items()})

      if ret.ravel()[0].__class__ is self._dtype:
        ret = self._edit(arr=ret)

      return frozen(ret)

    return func

  def __getattr__(self, name: str) -> Callable | NDArray | Self:
    try:
      return self.__getattr_cache[name]
    except KeyError:
      try:
        attr = getattr(self.arr.ravel()[0], name)
      except AttributeError:
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

      is_method = isinstance(attr, MethodType) and \
                  attr.__self__ is self.arr.ravel()[0]

      ret: NDArray | Callable | Self

      if is_method:
        ret = self._vectorize_method(name)

      else:
        ret = frozen(_object_array([ getattr(elem, name)
                                     for elem in self.arr.ravel() ])).reshape(self._shape)

        # wrap as self.__class__ if the attribute is self._dtype
        if ret.ravel()[0].__class__ is self._dtype:
          ret = self._edit(arr=ret)

      return self.__getattr_cache.setdefault(name, ret)

  @cached_property
  def knotvector(self) -> NDArray:
    """
    Return an array of :class:`TensorKnotVector` objects.
    """
    return frozen(np.vectorize(lambda x: x.knotvector, otypes=[object])(self.arr))

  @cached_property
  def nvars(self) -> int:
    """ Number of dependencies. """
    return self.arr.ravel()[0].nvars

  @classmethod
  def one(cls, array_of_knotvectors: NDArray | AnySequence[KnotVectorType], shape=None):
    """
    Create a NDSplineArray with all elements being the constant one function.
    """
    array_of_knotvectors = np.array(array_of_knotvectors, dtype=TensorKnotVector)
    if shape is None:
      shape = array_of_knotvectors.shape
    assert array_of_knotvectors.shape == shape[:array_of_knotvectors.ndim]
    shape_tail = shape[array_of_knotvectors.ndim:]
    return cls(np.vectorize(lambda kv: NDSpline.one(kv, shape=shape_tail))(array_of_knotvectors))

  @classmethod
  def one_from_NDSplineArray(cls, arr: 'NDSplineArray'):
    """
    Create a NDSplineArray with all elements being the constant one function
    from a given NDSplineArray.
    """
    return cls.one(arr.knotvector, shape=arr.shape)

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape + self._elemshape

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def _ndim(self) -> int:
    return len(self._shape)

  @property
  def _elemdim(self) -> int:
    return len(self._elemshape)

  @property
  def T(self) -> Self:
    # completely expand to create an array of NDSpline<> and
    # then transpose and contract back up.
    # XXX: this solution is not very efficient
    return self._edit(arr=self.expand_all().arr.T).contract_all()

  def expand(self, n=1) -> Self:
    """ Call self._expand if n > 0 and self._contract if n < 0. """
    if n == 0: return self
    if n > 0: return self._expand(n)
    return self._contract(-n)

  def _expand(self, n=1) -> Self:
    """
    Peel off ``n`` layers of the elements of the NDSplineArray and absorb them
    into the shape of the array. The dimension of the element's target space
    will be reduced by ``n`` while the dimension of ``self.arr`` is increased by
    ``n``.
    """
    assert isinstance(n, Int)
    if n == 0: return self
    if n > self._elemdim:
      raise ValueError("Cannot expand more than the number of dimensions of the elements.")

    elems = np.asarray([ list(iter(elem)) for elem in self.arr.ravel() ]) \
                                                  .reshape(*self._shape, -1)
    return self._edit(arr=elems).expand(n-1)

  def expand_all(self) -> Self:
    """ Expand all layers of the elements of the NDSplineArray. """
    return self.expand(self._elemdim)

  def _contract(self, n=1) -> Self:
    """
    Try to perform the inverse operation of `expand`. This operation is only
    possible if the elements self.arr[..., i] all have the same knotvector
    for all i (they may differ between the i's).
    """
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    if not self._shape:
      raise ValueError("The underlying array must have a shape.")

    contract_elems = self.arr.reshape(-1, *self.arr.shape[-1:])

    if not all( any(el.knotvector != elems[0].knotvector for el in elems) is False
                for elems in contract_elems ):
      raise ValueError("All elements must have the same knotvector.")

    new_elems = \
      np.array([ NDSpline(elems[0].knotvector,
                          np.stack([elem.controlpoints for elem in elems], axis=1))
                          for elems in contract_elems ], dtype=NDSpline)

    return self._edit(arr=new_elems.reshape(self._shape[:-1]))._contract(n-1)

  def contract_all(self) -> Self:
    """
    Contract until the elements no longer have the same knotvector.
    """
    try:
      ret = self.expand(-1)
      if ret is self:
        return ret
    except ValueError:
      return self
    return ret.contract_all()

  def to_ndim(self, ndim: Int) -> Self:
    return self.expand(ndim - self._ndim)

  def to_elemdim(self, elemdim: Int) -> Self:
    return self.expand(self._elemdim - elemdim)

  def prolong_to(self, knotvector_to: TensorKnotVector | NDArray[np.object_]) -> Self:
    if isinstance(knotvector_to, TensorKnotVector):
      knotvector_to = np.asarray(knotvector_to)

    if not knotvector_to.shape == self.knotvector.shape:
      raise NotImplementedError("The shape of `knotvector_to` must exactly"
                                " match the shape of `self.knotvector`. "
                                "Broadcasting has not been implemented, yet.")

    spls = [ spl.prolong_to(kv) for spl, kv in zip(self.arr.ravel(),
                                                   knotvector_to.ravel()) ]
    return self._edit(arr=np.asarray(spls).reshape(knotvector_to.shape))

  def prolong_to_array(self: 'NDSplineArray', spline_to: 'NDSplineArray') -> 'NDSplineArray':
    """
    Prolong all splines in the array to the knotvectors of another NDSplineArray.
    """
    assert self.shape == spline_to.shape, "The shapes of the two arrays must match."

    diff = len(spline_to._shape) - len(self._shape)
    self, other = map(lambda x: x.expand(diff), (self, spline_to))

    return self.prolong_to(spline_to.knotvector)

  def _call(self, *args, tensor=False, **kwargs) -> FloatArray:
    # XXX: np.stack makes a copy. Find better solution.
    evals = [ elem(*args, tensor=tensor, **kwargs) for elem in self.arr.ravel() ]
    shape0, *tail = evals[0].shape
    return np.stack(evals, axis=1).reshape((shape0,) + self.shape)

  def __call__(self, *args, **kwargs) -> FloatArray:
    return self._call(*args, tensor=False, **kwargs)

  def tensorcall(self, *args, **kwargs) -> FloatArray:
    return self._call(*args, tensor=True, **kwargs)

  @canonicalize_sum_args
  def sum(self, axis) -> Self:
    assert all(-self.ndim <= ax < self.ndim for ax in axis)
    axis = tuple(sorted(ax % self.ndim for ax in axis))

    tail = tuple(ax for ax in axis if ax >= self._ndim)
    head = axis[:axis.index(tail[0])] if tail else axis

    arr = self.arr
    if tail:  # skip this step if no tail is present to avoid overhead
      sm = tuple(i - self._ndim for i in tail)
      arr = np.vectorize(lambda el: el.sum(*sm))(arr)  # sum each element

    return self._edit(arr=arr.sum(head))

  @staticmethod
  def canonicalize_getitem_args(item: Any, ndim: Int) -> Tuple[Any, ...]:
    """
    Given a __getitem__ item, return a canonicalized version of the item
    given the number of dimensions of the spline array.
    """
    if not isinstance(item, tuple):
      item = item,

    if (n_ellipsis := item.count(...)) > 1:
      raise IndexError("an index can only have a single ellipsis ('...')")

    if n_ellipsis:
      index = item.index(...)
      nslices = ndim - (len(item) - item.count(_)) + 1
      item = item[:index] + (sl,) * nslices + item[index+1:]

    n_not_ = len(item) - item.count(_)

    return item + (sl,) * (ndim - n_not_)

  def __getitem__(self, item: Any) -> Self:

    # canonicalize the item, removing ellipsis and adding slices if needed
    item = self.canonicalize_getitem_args(item, self.ndim)

    if not item:
      return self

    no_None_indices = [i for i, it in enumerate(item) if it is not _]
    assert len(no_None_indices) == self.ndim

    ihead = (no_None_indices[:self._ndim] or [-1])[-1] + 1

    # ihead is now the index in `item` at which `self._ndim` (or less) entries
    # other than _ have been counted. The first `ihead` entries are forwarded to
    # `self.arr.__getitem__` while the rest are forwarded to the elements'
    # __getitem__ method.

    head, tail = item[:ihead], item[ihead:]
    arr = self.arr
    if tail:
      arr = np.vectorize(lambda el: el[tail])(arr)
    return self._edit(arr=arr[head])

  def __iter__(self):
    assert self.shape, 'iteration over 0-d array'
    if self._shape:
      yield from (self._edit(arr=item) for item in self.arr)
    else:
      yield from self.expand()

  def ravel(self) -> Self:
    return self.__class__(self.expand_all().arr.ravel())

  def __array_ufunc__(self, ufunc: np.ufunc, method: str,
                                             *inputs: Any, **kwargs) -> Self:
    if not len(inputs) == 2:
      raise NotImplementedError

    pivot = find_first_occurence(inputs, self)
    opivot = (pivot + 1) % 2

    op = getattr(ufunc, method)

    if not isinstance((y := inputs[opivot]), self.__class__):
      if isinstance(y, self._dtype):  # coerce to NDSplineArray
        inputs = inputs[:opivot] + (self.__class__(y),) + inputs[opivot+1:]
      else:
        # this block handles operations between NDSplineArray and np.ndarray
        try:
          # try to coerce to np.ndarray and prepend axes if necessary
          _self, y = _add_one_axes(inputs[pivot], np.asarray(y))

          # get shape of the result
          shp = try_broadcast_shapes(_self.shape[:_self._ndim],
                                         y.shape[:_self._ndim])

          # broadcast to the result shape and flatten first _self._ndim axes
          y = np.broadcast_to(y, shp + y.shape[_self._ndim:])
          arr = np.broadcast_to(_self.arr, shp[:_self._ndim])

          # perform operation in a for loop and reshape back to the result shape
          # and contract all possible axes
          newarr = []
          for multi_index in product(*map(range, shp)):
            el0, el1 = (arr[multi_index], y[multi_index]) if pivot == 0 else \
                       (y[multi_index], arr[multi_index])
            newarr.append(op(el0, el1, **kwargs))

          return self._edit(arr=np.array(newarr).reshape(shp)).contract_all()
        except ValueError:  # let other class handle the operation
          return NotImplemented

    # this block handles operations between two NDSplineArrays
    return self.vectorize_NDSpline_func(op)(*inputs, **kwargs)

  def sample_mesh(self, sample_meshes: Mesh | np.ndarray | Sequence) -> NDArray:
    """
    Sample a mesh from each element of the NDSplineArray.
    Requires `self.shape[-1] == 3`.

    If `sample_meshes` is a single mesh, the same mesh is used for all elements.
    If `sample_meshes` is a sequence of meshes, each element is sampled with the
    corresponding mesh. If the shape (after coercion to np.ndarray) of
    ``sample_meshes`` differs from ``self.shape``, the ``sample_meshes`` array
    is broadcast to the shape of ``self.shape[:-1]``.
    """
    if not self._elemshape:
      try:
        self = self.expand(-1)
      except ValueError:
        raise ValueError('Cannot sample a mesh from 0-dimensional splines.')

    assert self._elemshape[-1:] == (3,), 'Mesh export requires the target space to be R^3.'
    sample_meshes = np.broadcast_to(np.asarray(sample_meshes, dtype=Mesh),
                                                          self.shape[:-1])
    _self = self.expand(len(self._elemshape) - 1)

    return _self._vectorize_method('sample_mesh')(sample_meshes)

  def quick_sample(self, n: Int | AnyIntSeq = 11,
                         take_union: bool = False,
                         boundary: bool = False) -> NDArray | Mesh:
    """
    Sample ``n`` points along the knotvector's range in each direction.
    """

    if isinstance(n, Int):
      n = (n,) * self.nvars
    assert len(n) == self.nvars

    sample_meshes = np.broadcast_to(
      np.vectorize(partial(sample_mesh_from_knotvector, n=tuple(n)))(self.knotvector),
      self.shape[:-1])
    ret = self.sample_mesh(sample_meshes)

    if take_union:
      ret = mesh_union(*ret.ravel(), boundary=boundary)

    return ret

  def qplot(self, n=11, boundary=False) -> Mesh:
    """
    Quick plot of the NDSplineArray.
    The shape must be `(..., 3)`. If of matrix of tensor-valued, plot the mesh
    union of all sampled meshes.
    """
    if not self.shape[-1:] == (3,):
      raise ValueError('Quick plotting is only supported for R^3 splines.')

    mesh: Mesh = self.quick_sample(n=n, take_union=True, boundary=boundary)
    mesh.plot()
    return mesh
