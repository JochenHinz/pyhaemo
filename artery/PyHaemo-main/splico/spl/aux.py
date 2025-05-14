"""
Module providing helper functionality for use in ``kv.py`` and ``spline.py``,
such as prolonging to a refined knotvector.

@author: Jochen Hinz
"""

from ..util import np
from ..kron import sparse_kron, freeze_csr
from ._jit_spl import _univariate_prolongation_matrix, _pseudo_inverse
from ..types import Int
from ..kron import KroneckerOperator

from itertools import starmap
from functools import lru_cache
from typing import List, TYPE_CHECKING, Any, Callable

from scipy import sparse
from numpy.typing import NDArray


if TYPE_CHECKING:
  from .kv import UnivariateKnotVector, TensorKnotVector


GREVILLE_COLLOCATION_MATRIX_CACHE_SIZE = 32


# The following routines are deprecated and should be removed in the future.


@lru_cache(maxsize=8)
@freeze_csr
def univariate_prolongation_matrix(kvold: 'UnivariateKnotVector',
                                   kvnew: 'UnivariateKnotVector') -> sparse.csr_matrix:
  """
  Compute the prolongation matrix from a coarser to a finer knotvector.
  Alternatively the restriction matrix from a finer to a coarser knotvector
  is computed. The restriction matrix is a pseudo-inverse of the prolongation
  matrix which employs the entries of the prolongation matrix to compute the
  entries of the restriction matrix.
  Either `kvold <= kvnew` or `kvnew <= kvold` must hold.
  """
  if kvold == kvnew:
    return sparse.eye(kvold.dim, format='csr')

  coarsen = False
  if not kvold <= kvnew:
    kvold, kvnew = kvnew, kvold
    coarsen = True

  assert kvold <= kvnew

  T = _univariate_prolongation_matrix(kvold.repeated_knots,
                                      kvnew.repeated_knots, kvold.degree)
  ret = sparse.csr_matrix(T)
  if not coarsen:
    return ret

  ret = ret.tocoo()
  data = ret.data
  coords = ret.coords

  return sparse.coo_matrix(_pseudo_inverse(data, *coords),
                                    shape=ret.shape[::-1]).tocsr()


@lru_cache(maxsize=8)
@freeze_csr
def tensorial_prolongation_matrix(kvold: 'TensorKnotVector',
                                  kvnew: 'TensorKnotVector') -> sparse.csr_matrix:
  # TODO: implement a variant that never explicitly carries out the kronecker product
  return sparse_kron(*starmap(univariate_prolongation_matrix, zip(kvold, kvnew)))


#  Fitting to Greville points


@lru_cache(maxsize=GREVILLE_COLLOCATION_MATRIX_CACHE_SIZE)
@freeze_csr
def greville_collocation_matrix(kv: 'UnivariateKnotVector') -> sparse.csr_matrix:
  """
  This function cached the collocation matrix of a :class:`UnivariateKnotVector`
  object in the Greville points. We cache across knot vector instances because
  this matrix is extensively used in the prolongation of splines to richer
  knot vectors.
  While more efficient prologation methods exist for special cases, prolongation
  via the collocation matrix is a good general purpose method that works in all
  cases.

  There is one catch which occurs when there is an interior knot with
  multiplicity `degree + 1`, see below.
  """
  # When `kv` has a `p+1`-fold repeated knot, one of the Greville points
  # will be repeated twice. When finding the associated knot span, he will
  # erroneously evaluate it from one side twice which will lead to a singular
  # matrix X @ X.T.

  # If there are no C^-1 continuity knots, immediately return the matrix
  # There are at least two knots so this will always work.
  breakpoints = np.where(kv.continuity == -1)[0][1:-1]

  if len(breakpoints) == 0:
    return kv.collocate(kv.greville)

  # Else, find the positions where the continuity is C^-1 (excl. the endpoints)
  kvs = kv.split(breakpoints)
  return sparse.block_diag( [greville_collocation_matrix(_kv) for _kv in kvs] ).tocsr()


@lru_cache(maxsize=GREVILLE_COLLOCATION_MATRIX_CACHE_SIZE)
@freeze_csr
def gauss_collocation_matrix(kv: 'UnivariateKnotVector') -> sparse.csr_matrix:
  """
  Same as `greville_collocation_matrix` but uses the Gauss points instead of the
  Greville points.
  """
  return kv.collocate(kv.gauss_abscissae)


def colmat_and_abscissae(kv: 'UnivariateKnotVector') -> tuple[sparse.csr_matrix, NDArray]:
  """
  Return the greville collocation matrix and the associated abscissae if the
  knotvector is at least C^0 continuous, else return the Gauss
  collocation matrix and the associated abscissae.
  """
  if -1 in kv.continuity[1:-1]:
    return gauss_collocation_matrix(kv), kv.gauss_abscissae
  return greville_collocation_matrix(kv), kv.greville


def fit_greville(kv: 'TensorKnotVector', func: Callable) -> NDArray:
  """
  Special method for fitting an NDSpline to a function over the Greville
  absissae. We return only the spline coefficients to avoid cicular imports.

  We make this a static, dedicated method to make use of caching in the
  `greville_collocation_matrix` method.

  As before, this method uses the least squares method to fit the
  function in case the knotvector has global C^-1 continuity.

  The function has to take a tensorial vector input and output a tensor whose
  first dimension equals the product of the lengths of the tensorial inputs.

  To fit a NDSpline to a new knot vector, we can pass
                    `func = lambda *args: spl(*args, tensor=True)`.
  """
  mats, absc = [], []
  for mat, myabsc in map(colmat_and_abscissae, kv):
    mats.append(mat)
    absc.append(myabsc)

  X = KroneckerOperator(mats)
  data = func(*absc)

  assert data.shape[:1] == X.shape[1:]

  if all( x.issquare for x in X ):
    # if all sub-matrices are square, we can directly invert
    cps = X.T.inv @ data
  else:
    # else we solve in the least squares sense
    cps = (X @ X.T).inv @ (X @ data)

  return cps


def denest_objarr(arr: NDArray):
  """
  Recursively flattens a numpy object array.
  Only flattens along entries that are themselves object arrays.

  Parameters
  ----------
  arr : np.ndarray
      A numpy array of dtype object.

  Returns
  -------
  np.ndarray
      A 1D numpy object array containing all non-array objects.
  """

  if not isinstance(arr, np.ndarray) or arr.dtype != object:
    raise TypeError("Input must be a numpy object array.")

  shape: List[Int] = []

  elem = arr
  while isinstance(elem, np.ndarray) and elem.dtype == object:
    shape.extend(elem.shape)
    elem = elem.ravel()[0]

  ret: List[Any] = []

  def recurse(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
      for item in x.flat:
        recurse(item)
    else:
      ret.append(x)

  recurse(arr)

  return np.array(ret, dtype=object).reshape(shape)
