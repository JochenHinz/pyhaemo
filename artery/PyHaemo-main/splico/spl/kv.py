"""
Module defining knotvectors. The main objects are :class:`UnivariateKnotVector`
and :class:`TensorKnotVector`. The latter is a vectorized version of the former.

@author: Jochen Hinz
"""

from ..util import _round_array, isincreasing, np, _, \
                   frozen_cached_property, gauss_quadrature, frozen
from ..types import Immutable, ensure_same_class, ensure_same_length, Int, \
                    Numeric, AnyIntSeq, AnyNumericSeq, FloatArray, IntArray, \
                    AnySequence, NumericArray, NumpyIndex
from ..err import EmptyContainerError
from ..kron import freeze_csr, sparse_kron, KroneckerOperator
from ._jit_spl import nonzero_bsplines_deriv_vectorized, _collocation_matrix

from itertools import starmap, chain
from functools import partial, lru_cache, reduce
from typing import Sequence, Self, Any, Optional, Callable, overload, List, \
                   Tuple
from inspect import signature
import operator

from scipy import sparse
from numpy.typing import NDArray


# TODO: In the long run it would be nice to have a more general NDKnotVector
# where we don't make the distinction between univariate and tensorial knotvectors
# anymore. This would also allow for a more general approach to the tensorial
# product of knotvectors.


# XXX: I would like to use functools.total_ordering but it is slightly out of
#      place here because two knotvectors can simultaneously satisfy a < b is
#      False and b < a is False.
#      Couldn't quite get it to work as intended with total_ordering.
class UnivariateKnotVector(Immutable):
  """
  Basic knot-vector object.

  Parameters
  ----------
  knotvalues : Array-like of Int / Float
    Strictly increasing one-dimensional sequence of numbers representing the
    knotvector's knots without repretitions.
  degree : :class:`int`
    Integer value representing the knotvector's polynomial degree.
  knotmultiplicities : Array-like of integers, optional
    Positive sequence of length len(self.knotvalues) representing each knot's
    multiplicity. Must represent an open knotvector, i.e,
        knotmultiplicities[0] == knotmultiplicities[-1] == self.degree + 1.
    If not passed, defaults to an open knotvector without interior knot
    repretitions.
  """
  # XXX: support for periodic knotvectors

  @staticmethod
  def union(*args: 'UnivariateKnotVector') -> 'UnivariateKnotVector':
    assert args
    return reduce(operator.or_, args)

  @staticmethod
  def intersection(*args: 'UnivariateKnotVector') -> 'UnivariateKnotVector':
    assert args
    return reduce(operator.and_, args)

  def __init__(self, knotvalues: AnyNumericSeq,
                     degree: Int = 3,
                     knotmultiplicities: Optional[AnyIntSeq] = None):
    knotvalues = _round_array(knotvalues)
    assert isincreasing(knotvalues), 'The knot sequence needs to be strictly increasing.'

    # XXX: failswitch in case knot sequence is too short for specified degree
    self.knotvalues = tuple(map(float, knotvalues))
    assert len(self.knotvalues) >= 2

    self.degree = int(degree)
    assert self.degree > 0

    if knotmultiplicities is None:
      start = self.degree + 1
      knotmultiplicities = (start,) + (1,)*(len(self.knotvalues) - 2) + (start,)

    self.knotmultiplicities = tuple(map(int, knotmultiplicities))

    assert len(self.knotvalues) == len(self.knotmultiplicities)

    if any( self.knotmultiplicities[i] != self.degree + 1 for i in (0, -1) ):
      raise NotImplementedError('Currently, only open knotvectors are supported.')

    assert (0 < self.km[1:-1]).all() and (self.km[1:-1] <= self.degree + 1).all()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[degree: {self.degree}, nknots: {len(self.knots)}]"

  @frozen_cached_property
  def knots(self) -> FloatArray:
    return np.asarray(self.knotvalues, dtype=float)

  @frozen_cached_property
  def km(self) -> IntArray:
    return np.asarray(self.knotmultiplicities, dtype=int)

  @frozen_cached_property
  def greville(self) -> FloatArray:
    """ Compute the Greville points. """
    knots = self.repeated_knots
    ret = knots[np.arange(self.degree, dtype=int)[_] +
                np.arange(1, self.dim+1, dtype=int)[:, _]].sum(1) / self.degree

    # Check if there are points with discontinuous derivatives
    # If so, we have to slightly move the point away from the discontinuity's
    # knot.
    breaks, = np.where( self.continuity[1:-1] == -1 )
    if len(breaks):
      pass
    return np.clip(ret, self.knots[0], self.knots[-1])

  @frozen_cached_property
  def gauss_abscissae(self) -> FloatArray:
    w, _points = gauss_quadrature(0, 1, self.degree + 1)
    return frozen( self.knots[:-1, _] + self.dx[:, _] * _points[_] ).ravel()

  @frozen_cached_property
  def continuity(self) -> FloatArray:
    return self.degree - self.km

  @property
  def nelems(self) -> int:
    """ Return the number of elements. """
    return len(self.knots) - 1

  @property
  def dim(self) -> int:
    """ Amount of basis functions resulting from knot vector. """
    return np.sum(self.knotmultiplicities[:-1])

  @frozen_cached_property
  def repeated_knots(self) -> FloatArray:
      """ Repeat knots by their knotmultiplicity. """
      return np.repeat(self.knots, self.knotmultiplicities)

  @frozen_cached_property
  def dx(self) -> FloatArray:
    """ Return the distance between distinct knots. """
    return np.diff(self.knots)

  def flip(self) -> Self:
    """
    Flip the knotvector.

    >>> knots = np.array([1.0, 1.2, 1.6, 2.0])
    >>> kv = UnivariateKnotVector(knots)
    >>> kv.knots
    ... np.array([1.0, 1.2, 1.6, 2.0])
    >>> kv.flip().knots
    ... np.array([1.0, 1.4, 1.8, 2.0])
    """
    return self._edit(knotmultiplicities=self.knotmultiplicities[::-1],
                      knotvalues=np.array([0, *self.dx[::-1]]).cumsum() + self.knots[0])

  __neg__ = flip

  def collocate(self, abscissae: AnyNumericSeq, dx: Int = 0) -> sparse.csr_matrix:
    """
    Collocation matrix X over the abscissae ``abscissae``.
    Generates a sparse matrix X such that the solution x of the system
    (X @ X.T) @ x = X @ data contains the control points with respect
    to the basis associated with ``self`` of the least squares problem of
    fitting the abscissae against data.

    Parameters
    ----------
    abscissae : Array-like of ndim == 1
        The fitting abscissae.
    dx : :class:`int`
        The basis derivative associated with the collocation matrix. If
        dx == 0, it's the ordinary collocation matrix.

    Returns
    -------
    X: Collocation matrix of type sparse.csr_matrix.
    """
    abscissae = np.asarray(abscissae, dtype=float)

    a, *ignore, b = self.knots
    assert a <= abscissae.min() <= abscissae.max() <= b

    args = _collocation_matrix(self.repeated_knots,
                               self.degree,
                               abscissae,
                               dx)

    return sparse.coo_matrix(args).tocsr()

  def _refine(self) -> Self:
    """ Uniformly refine the entire knotvector once. """
    nknots = len(self.knots)
    knots = np.insert(self.knots, range(1, nknots), (self.knots[:-1] + self.knots[1:])/2.0)
    knotmultiplicities = np.insert(self.knotmultiplicities, range(1, nknots), 1)
    return self._edit(knotvalues=knots, knotmultiplicities=knotmultiplicities)

  def refine(self, n: Int = 1) -> Self:
    """ Uniformly refine the entire knotvector ``n`` times. """
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._refine().refine(n-1)

  def ref_by(self, indices: Int | AnyIntSeq) -> Self:
    """ Halve elements contained in ``indices``. """
    if isinstance(indices, Int):
      indices = indices,
    indices = np.asarray(indices, dtype=int)
    add = (self.knots[indices + 1] + self.knots[indices]) / 2.0
    return self._edit(knotvalues=np.insert(self.knots, indices+1, add),
                      knotmultiplicities=np.insert(self.km, indices+1, 1))

  def _split(self, position: Int) -> Tuple[Self, ...]:
    """
    Split the knotvector at position `position` into two knotvectors.
    The knotmultiplicity at the new endpoint will be `degree + 1`.
    """
    assert 0 < position <= self.nelems
    km0 = self.knotmultiplicities[:position] + (self.degree + 1,)
    km1 = (self.degree + 1,) + self.knotmultiplicities[position+1:]
    knots0, knots1 = self.knotvalues[:position+1], self.knotvalues[position:]
    return self._edit(knotvalues=knots0, knotmultiplicities=km0), \
           self._edit(knotvalues=knots1, knotmultiplicities=km1)

  def split(self, positions: Int | AnyIntSeq) -> Tuple[Self, ...]:
    if isinstance(positions, Int):
      positions = positions,

    ret: Tuple[Self, ...] = self,
    for pos0, pos1 in zip(chain((0,), positions), positions):
      ret = ret[:-1] + ret[-1]._split(pos1 - pos0)

    return ret

  def add_knots(self, knotvalues: Numeric | AnyNumericSeq) -> Self:
    """
    Add new knots to the knotvector. Adding knots beyond the knotvector's
    limits is currently prohibited.
    """
    if isinstance(knotvalues, Numeric):
      knotvalues = knotvalues,
    knotvalues = np.asarray(knotvalues, dtype=float)

    a, *inner_knots, b = self.knotvalues
    if (a > knotvalues).any() or (knotvalues > b).any():
      raise ValueError("Cannot add knots beyond the knotvector's domain.")

    new_knots = np.unique(np.concatenate([self.knots, knotvalues]))
    if len(new_knots) == len(self.knots):
      return self

    map_kv_km = dict(zip(self.knotvalues, self.knotmultiplicities, strict=True))
    new_km = [map_kv_km.get(val) or 1 for val in new_knots]
    return self._edit(knotvalues=new_knots, knotmultiplicities=new_km)

  def raise_multiplicities(self, positions: Int | AnyIntSeq,
                                 amounts: Int | AnyIntSeq) -> Self:
    """
    Raise the knotmulitplicities corresponding to ``indices`` by ``amount``.
    """
    if isinstance(positions, Int):
      positions = positions,
    positions = np.asarray(positions, dtype=int)

    if isinstance(amounts, Int):
      amounts = (amounts,) * len(positions)

    km = np.asarray(self.knotmultiplicities, dtype=int)
    km[positions] += np.asarray(amounts, dtype=int)
    return self._edit(knotmultiplicities=km)

  def degree_elevate(self, dp: Int, strict=True) -> Self:
    """
    Raise the polynomial degree of the knotvector by ``dp`` while retaining
    the same continuity.

    If `dp` is negative and strict is False, we simply return `self`.
    If strict is true, we raise an assertion error.
    """
    if strict:
      assert dp >= 0
    if dp <= 0:
      return self
    return self._edit(degree=self.degree + dp,
                      knotmultiplicities=self.km + dp)

  def integrate(self, dx: Int = 0):
    """ See ``univariate_integral``. """
    return univariate_integral(self, dx=dx)

  @property
  def M(self):
    """
    Parametric mass matrix.
    """
    return self.integrate(dx=0)

  @property
  def A(self):
    """
    Parametric stiffness matrix.
    """
    return self.integrate(dx=1)

  @property
  def D(self):
    """
    Parametric second-order equivalent of the stiffness matrix.
    """
    return self.integrate(dx=2)

  def __mul__(self, other: Any):
    """
    Multiplying by :class:`UnivariateKnotVector` or :class:`TensorKnotVector`
    yields a :class:`TensorKnotVector`.
    """
    if other.__class__ is self.__class__:
      return TensorKnotVector([self, other])
    if isinstance(other, TensorKnotVector):
      return TensorKnotVector([self, *other])
    return NotImplemented

  def __rmul__(self, other: Any):
    assert isinstance(other, TensorKnotVector)
    return TensorKnotVector([*other, self])

  def __pow__(self, other: Any) -> 'TensorKnotVector':
    if isinstance(other, Int):
      return TensorKnotVector([self] * other)
    return NotImplemented

  def to_tensor(self) -> 'TensorKnotVector':
    return TensorKnotVector([self])

  @ensure_same_class
  def __and__(self, other: Self) -> Self:
    """
    Create a knotvector from the shared knots.
    If a knot is shared, the larger of the two knotmultiplicities is taken.

    We do not support the case where the two knotvectors have different
    polynomial degrees. We require the user to explicitly raise the degree
    of the knotvector with the smaller degree in this case.
    """
    knots = np.intersect1d(self.knots, other.knots)
    if self.degree != other.degree or not len(knots):
      raise EmptyContainerError("Found empty intersection.")
    km0, km1 = map(lambda x: x.km[np.searchsorted(x.knots, knots)], (self, other))
    km = np.maximum(km0, km1)
    return self._edit(knotvalues=knots, knotmultiplicities=km)

  @ensure_same_class
  def intersection_promote(self, other: 'UnivariateKnotVector') -> Self:
    """
    Same as `__and__` but also raises the degree of the knotvector with the
    smaller degree to the larger one.
    """
    self = self.degree_elevate(other.degree - self.degree, strict=False)
    other = other.degree_elevate(self.degree - other.degree, strict=False)
    return self & other

  @ensure_same_class
  def __or__(self, other: Self) -> Self:
    """
    Take the union of two :class:`UnivariateKnotVector`s.
    """
    if not other.degree == self.degree:
      raise NotImplementedError("Found differing polynomial degrees.")

    knots = np.unique(np.concatenate([self.knots, other.knots]))
    km = np.zeros(len(knots), dtype=int)

    for kv in (self, other):
      my_ind = np.searchsorted(knots, kv.knots)
      km[my_ind] = np.maximum(km[my_ind], kv.km)

    return self._edit(knotvalues=knots, knotmultiplicities=km)

  @ensure_same_class
  def union_promote(self, other: 'UnivariateKnotVector') -> Self:
    """
    See ``intersection_promote``.
    """
    self = self.degree_elevate(other.degree - self.degree, strict=False)
    other = other.degree_elevate(self.degree - other.degree, strict=False)
    return self | other

  @ensure_same_class
  def __lt__(self, other: Self) -> bool | np.bool_:
    """
      Check if the basis corresponding to ``self`` is contained in the
      basis of ``other`` but not equal.
    """
    if (dp := other.degree - self.degree) < 0 or len(np.setdiff1d(self.knots, other.knots)) != 0:
      return False
    if self == other:
      return False
    return (self.km + dp <= other.km[np.searchsorted(other.knots, self.knots)]).all()

  @ensure_same_class
  def __gt__(self, other: Self) -> bool | np.bool_:  # see if self contains the basis of other
    return other < self

  def __le__(self, other):
    return self == other or self < other

  def __ge__(self, other):
    return self == other or self > other

  @ensure_same_class
  def __matmul__(self, other: Self) -> Self:
    """
    We overload the `@` operator to compute the knotvector that contains the
    >>product<< of any functional from the linear span of ``self`` and
    ``other``.
    """
    degree = self.degree + other.degree
    all_knots = np.union1d( self.knots, other.knots )
    km = np.zeros(len(all_knots), dtype=int)

    for kv in (self, other):
      my_ind = np.searchsorted(all_knots, kv.knots)
      km[my_ind] = np.maximum(km[my_ind], kv.km + (degree - kv.degree))

    return self._edit(knotvalues=all_knots,
                      knotmultiplicities=km,
                      degree=degree)


@lru_cache(maxsize=32)
@freeze_csr
def univariate_integral(uknotvector: UnivariateKnotVector, dx: Int = 0) -> sparse.csr_matrix:
  r"""
  Compute the matrix with entries M_ij = \int_(a, b) phi_i^(dx) phi_j^(dx) dx,
  where ``uknotvector.knots == a, *ignore, b``.

  Parameters
  ----------

  uknotvector : :class:`UnivariateKnotVector`
    The univariate knotvector over which the phi_i are defined.
  dx : :class:`int`
    The derivative order.
  """

  # XXX: jit-compile with Numba using COO-format instead of lil.

  maxrep = uknotvector.km[1:-1].max()
  assert uknotvector.degree >= dx and maxrep <= uknotvector.degree + 1 - dx

  knots, ext_knots = uknotvector.knots, uknotvector.repeated_knots

  gauss = partial(gauss_quadrature, order=uknotvector.degree + 1)
  M = sparse.lil_matrix((uknotvector.dim,) * 2)

  for i, (weights, points) in enumerate(starmap(gauss, zip(knots, knots[1:]))):
    dofs = np.arange(i, i + uknotvector.degree + 1)
    shapeF = nonzero_bsplines_deriv_vectorized(ext_knots, uknotvector.degree, points, dx)
    M[np.ix_(dofs, dofs)] += (weights[:, _, _] * shapeF[..., _] * shapeF[:, _]).sum(0)

  return M.tocsr()


def _empty_csr(ndofs):
  return sparse.csr_matrix((ndofs, ndofs))


def add_vectorizations(cls):

  def _vectorize_with_indices(name: str) -> Callable:
    """
    Vectorize an operation and apply it to each :class:`UnivariateKnotVector`
    in ``self.knotvectors``.
    Add the vectorized function to the class `cls` with the name `name`.

    Parameters
    ----------
    name: :class:`str`
    The name of the method of :class:`UnivariateKnotVector` that is to
    be applied to all :class:`UnivariateKnotVector`s in `self`.

    ----------

    The resulting function's syntax is the following:
    >>> kv
    ... UnivariateKnotVector[...]
    >>> tkv = kv * kv * kv
    >>> tkv.refine(..., n=[1, 0, 1])

    Here the ``...`` (or None) indicates that the operation should be applied to
    all :class:`UnivariateKnotVector`s in ``self``, where the i-th knotvector
    receives input ``n[i]``.

    The return type is always :class:`self.__class__`.

    Similarly, we may pass the indices explicitly, for instance:
    >>> tkv.refine([0, 2], [1, 1])
    >>> tkv.refine(..., n=[1, 0, 1]) == tkv.refine([0, 2], [1, 1])
    ... True

    Here the knotvector corresponding to the i-th entry in `directions`
    receives ``n[i]`` as input.
    """

    sig = signature(getattr(UnivariateKnotVector, name))

    assert 'directions' not in sig.parameters, \
      "The method to be vectorized must not have an argument 'directions'."

    def wrapper(self, directions, *args, **kwargs):

      if np.isscalar(directions):
        directions = directions,
      elif directions in (Ellipsis, None):
        directions = range(len(self))

      directions = list(directions)

      assert all( -len(self) <= i < len(self) for i in directions )
      directions = [ i % len(self) for i in directions ]

      _self = list(self)
      assert all( len(arg) == len(directions) for arg in args ) and \
             all( len(val) == len(directions) for val in kwargs.values() )

      for j, i in enumerate(directions):
        _self[i] = getattr(_self[i], name)(*(a[j] for a in args),
                                           **{k: v[j] for k, v in kwargs.items()})

      return self.__class__(_self)

    return wrapper

  def _vectorize_operator(name: str, return_type=None) -> Callable:
    """
    Vectorize an operator operation. For instance __and__.
    >>> kv0
    ... UnivariateKnotVector[...]
    >>> kv1
    ... UnivariateKnotVector[...]
    >>> kv2 = kv0 & kv1
    >>> tkv0 = TensorKnotVector([kv0] * 3)
    >>> tkv1 = TensorKnotVector([kv1] * 3)
    >>> (tkv0 & tkv1) == TensorKnotVector([kv2] * 3)
    ... True

    We may optionally pass a return type that differs from ``None``
    in which case the return type defaults to ``self.__class__``.
    """
    op = getattr(operator, name)

    @ensure_same_length
    @ensure_same_class
    def wrapper(self, other):
      rt = return_type or self.__class__
      return rt([op(kv0, kv1) for kv0, kv1 in zip(self, other)])

    return wrapper

  def _vectorize_property(name: str, return_type=tuple) -> Callable:
    """
    Vectorize a (cached) property. Optionally takes a return container-type
    argument which the properties are iterated into.
    For instance, :class:`list` or :class:`tuple`. Defaults to :class:`tuple`.
    """
    @property
    def wrapper(self):
      return return_type([getattr(e, name) for e in self])

    return wrapper

  # add all index vectorized methods
  for name in 'flip', 'refine', 'ref_by', 'add_knots', 'degree_elevate', \
              'raise_multiplicities':
    setattr(cls, name, _vectorize_with_indices(name))

  # add all vectorized properties
  for name in 'dx', 'knots', 'km', 'degree', 'repeated_knots', 'nelems', \
              'dim', 'greville', 'continuity', 'gauss_abscissae':
    setattr(cls, name, _vectorize_property(name))

  # add all operator vectorizations with custom return type
  for name in '__lt__', '__gt__', '__le__', '__ge__':
    setattr(cls, name, _vectorize_operator(name, all))

  # add all operator vectorizations without custom return type
  for name in '__and__', '__or__', '__matmul__':
    setattr(cls, name, _vectorize_operator(name))

  return cls


@add_vectorizations
class TensorKnotVector(Immutable):

  """
  Class representing a tensorial knotvector.
  Takes as input a sequence of :class:`UnivariateKnotVector` and vectorizes
  all relevant properties / methods of its inputs.

  Most of the vectorization is implemented via the :type:`TensorKnotVectorMeta`
  metaclass.
  """

  # XXX: In the long run, UnivariateKnotVector, TensorKnotVector should be
  #      replaced by a more general NDKnotVector class.

  # through metaclass vectorization inferred properties
  dx: tuple
  km: tuple
  dim: tuple
  knots: tuple
  degree: tuple
  nelems: tuple
  greville: tuple
  continuity: tuple
  repeated_knots: tuple
  gauss_abscissae: tuple

  # through metaclass vectorization inferred methods
  flip: Callable
  refine: Callable
  ref_by: Callable
  add_knots: Callable
  degree_elevate: Callable
  raise_multiplicities: Callable

  __or__: Callable
  __and__: Callable

  __lt__: Callable  # all comparisons must hold for ``True``
  __gt__: Callable
  __le__: Callable
  __ge__: Callable

  @staticmethod
  def union(*args: 'TensorKnotVector') -> 'TensorKnotVector':
    assert args
    return reduce(operator.or_, args)

  @staticmethod
  def intersection(*args: 'TensorKnotVector') -> 'TensorKnotVector':
    assert args
    return reduce(operator.and_, args)

  def __init__(self, knotvectors: AnySequence[UnivariateKnotVector] | NDArray):
    self.knotvectors = tuple(map(UnivariateKnotVector, knotvectors))

  def __iter__(self):
    """ By default we iterate over ``self.knotvectors``. """
    yield from self.knotvectors

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[degree: {self.degree}, " \
                                     f"nknots: {tuple(map(len, self.knots))}]"

  def __bool__(self):
    return bool(len(self))

  @overload
  def __getitem__(self, index: Int) -> UnivariateKnotVector:
    """ For integer types the return type is :class:`UnivariateKnotVector`. """
    ...

  @overload
  def __getitem__(self, index: slice | List | NDArray ) -> Self:
    """ For multi-index types it is :class:`Self`. """
    ...

  def __getitem__(self, index: NumpyIndex) -> UnivariateKnotVector | Self:
    # mypy complains about the index type here, but it's correct.
    kvs = np.asarray(self.knotvectors)[index]  # type: ignore
    if isinstance(kvs, np.ndarray):
      return self._edit(knotvectors=tuple(kvs))
    return kvs

  @property
  def ndim(self) -> int:
    """ Number of knotvectors. """
    return len(self.knotvectors)

  @property
  def ndofs(self) -> np.integer:
    """ Total number of DOFs. """
    return np.prod(self.dim, dtype=int)

  def __len__(self) -> int:
    return self.ndim

  def collocate(self, *list_of_abscissae, dx: AnyIntSeq = ()) -> Tuple[sparse.csr_matrix, ...]:
    """
    Tensor-product version of ``UnivariateKnotVector.collocate``.
    """
    if isinstance(dx, Int):
      assert self
      dx = (dx,) * len(self)
    if len(dx) == 0:
      dx = (0,) * len(self)
    assert len( (dx := tuple(dx)) ) == len(list_of_abscissae) == len(self)
    return tuple( kv.collocate(absc, dx=_dx)
                  for kv, absc, _dx, in zip(self, list_of_abscissae, dx) )

  @property
  def M(self):
    """ Tensor-product version of ``UnivariateKnotVector.M`` """
    return sparse_kron(sparse.eye(1), *(kv.M for kv in self))

  @property
  def A(self):
    """ Tensor-product version of ``UnivariateKnotVector.A`` """
    return sum( [sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in np.eye(len(self)).astype(int)],
                start=_empty_csr(self.ndofs) )

  @property
  def D(self):
    """ Tensor-product version of ``UnivariateKnotVector.D`` """
    return sum( [sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in (2 * np.eye(len(self))).astype(int)],
                start=_empty_csr(self.ndofs) )

  def fit(self, list_of_abscissae: Sequence[AnyNumericSeq],
                data: NumericArray,
                lam0: Numeric = 1e-5,
                lam1: Numeric = 0                         ):
    """
    Fit a spline to a set of points and vertices in the least squares sense
    with (optional) added regularisation using ``self`` as knotvector.

    Parameters
    ----------

    list_of_abscissae: Container-like of fitting abscissae in each parametric
                       direction
        The number must match the dimensionality of the knotvector.
        The vertices follow from a tensor product.
    data: Sequence of Array-likes of data points
        The shape must satisfy

        ``self.shape[0] == len(list_of_abscissae[0]) * len(list_of_abscissae[1]) * ...``

        May also be tensorial, i.e., ``self.shape[1:] != ()`` in which case
        each entry is fit individually.
    lam0: :class:`float` or equivalent
        Added least squares first order smoothness regularisation.
        Defaults to 1e-5.
    lam1: :class:`float` or equivalent
        Added least squares second order regularisation. Is ommitted by default.

    Returns
    -------
    ret: NDSpline of shape ``data.shape[1:]`` that follows from a least squares
         fit using ``self`` as a knotvector.

    >>> kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    >>> kv = kv * kv  # two-dimensional tensor knotvector
    >>> absc = np.linspace(0, 1, 21)
    >>> x, y = map(np.ravel, np.meshgrid(*[np.linspace(0, 1, 11)]*2))
    >>> data = np.stack([x, 1 + x + y], axis=1)
    >>> spline = kv.fit([absc, absc], data)
    >>> (np.abs(spline(x, y) - data) < 1e-2).all()
    ... True
    """

    list_of_abscissae = list(map(np.asarray, list_of_abscissae))
    data = np.asarray(data, dtype=float)

    assert data.shape[:1] == (np.multiply.reduce(list(map(len, list_of_abscissae))),)
    assert all(lam >= 0 for lam in (lam0, lam1))

    X = KroneckerOperator(self.collocate(*list_of_abscissae))
    M = X @ X.T

    # when stabilization is desired, we have to let go of the kronecker
    # product structure of the collocation matrix.
    # TODO: find a better way to do this.
    if any( (lam0, lam1) ):
      _M = M.tocsr()
      if lam0 != 0:
        _M += lam0 * self.A
      if lam1 != 0:
        _M += lam1 * self.D
      M = KroneckerOperator([_M])

    rhs = X @ data.reshape((-1,) + (data.shape[1:] and (np.prod(data.shape[1:]),)))

    from .spline import NDSpline
    return NDSpline(self, (M.inv @ rhs).reshape((-1,) + data.shape[1:]))

  def __mul__(self, other: Any):
    """
    Multiplying by a :class:`TensorKnotVector` or a
    :class:`UnivariateKnotVector` simply gives a bigger knotvector.
    """
    if isinstance(other, UnivariateKnotVector):
      knotvectors = self.knotvectors + (other,)
    elif isinstance(other, TensorKnotVector):
      knotvectors = self.knotvectors + other.knotvectors
    else:
      return NotImplemented
    return self._edit(knotvectors=knotvectors)


KnotVectorType = UnivariateKnotVector | \
                 TensorKnotVector | \
                 Sequence[UnivariateKnotVector]


def as_TensorKnotVector(kv: KnotVectorType) -> TensorKnotVector:
  """
  Convert a knotvector to a :class:`TensorKnotVector` if not already.
  """
  if isinstance(kv, TensorKnotVector):
    return kv
  if isinstance(kv, UnivariateKnotVector):
    kv = kv,
  return TensorKnotVector(tuple(kv))
