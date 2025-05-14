"""
This module contains the definition of the `LinearOperator` class and its
subclasses. The `LinearOperator` class is a base class for representing
linear operators in a matrix-free way. It provides methods for matrix-vector
multiplication, matrix-matrix multiplication, and computing the transpose
and inverse of the operator.

An additional feature is the `KroneckerOperator` class, which represents a
Kronecker product of linear operators without explicitly carrying out the product.
This class allows for efficient matrix-vector multiplication and matrix-matrix
multiplication with Kronecker products and has many applications in
tensor-product spline spaces.

@author: Jochen Hinz
"""

from splico.util import frozen, _round_array
from splico.types import Int, Immutable

from functools import lru_cache, reduce, wraps
from collections.abc import Sequence, Hashable
from abc import abstractmethod
from typing import Self, Any, Tuple, Union, List

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


def frozen_csr(arr: sparse.csr_matrix) -> sparse.csr_matrix:
  for item in map(lambda x: getattr(arr, x), ('data', 'indices', 'indptr')):
    frozen(item)  # convert the array to read-only to avoid accidental overwrites
  return arr


def freeze_csr(fn):
  """
  Decorator for freezing a :class:`sparse.csr_matrix` before it is returned
  to avoid accidental overwrites, for instance when returning from cached
  functions.

  If a mutable instance of the matrix is desired, the matrix has to be copied.
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    ret = fn(*args, **kwargs)
    assert isinstance(ret, sparse.csr_matrix)
    return frozen_csr(ret)
  return wrapper


def sparse_kron(*_mats: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
  """
  Take the Kronecker product of several sparse matrices.

  If not matrix is passed, we return the identity matrix of size 1.
  If only one matrix is passed, we return the matrix itself.

  In all other cases, we compute the Kronecker product of all matrices, as
  expected.
  """
  mats: List[sparse.spmatrix] = list(map(sparse.csr_matrix, _mats))
  if not mats:
    return sparse.eye(1).tocsr()
  if len(mats) == 1:
    return mats[0]
  return reduce(lambda x, y: sparse.kron(x, y, format='csr'), mats)


@lru_cache(maxsize=32)
def _linop_T(A: 'LinearOperator') -> 'LinearOperator':
  return A._make_T()


@lru_cache(maxsize=32)
def _linop_inv(A: 'LinearOperator') -> 'LinearOperator':
  return A._make_inv()


class LinearOperator(Immutable):

  def __init__(self, shape: Sequence[Int]):
    shape = tuple(shape)
    assert len(shape) == 2
    self.shape: Tuple[int, int] = (int(shape[0]), int(shape[1]))

  @abstractmethod
  def _matvec(self, b: NDArray) -> NDArray:
    pass

  @property
  def issquare(self):
    return self.shape[0] == self.shape[1]

  @property
  def T(self):
    if not hasattr(self, '_T'):
      self._T = _linop_T(self)
      if not hasattr(self._T, '_T'):
        self._T._T = self
    return self._T

  def _matmat(self, other: 'LinearOperator') -> 'LinearOperator':
    return MatMul((self, other))

  @abstractmethod
  def _make_inv(self) -> 'LinearOperator':
    pass

  @abstractmethod
  def _make_T(self) -> 'LinearOperator':
    pass

  @abstractmethod
  def tocsr(self) -> sparse.csr_matrix:
    pass

  def toarray(self) -> NDArray:
    return self.tocsr().toarray()

  @property
  def inv(self) -> 'LinearOperator':
    if not self.issquare:
      raise ValueError("Operator is not square")

    if not hasattr(self, '_inv'):
      self._inv = _linop_inv(self)
      if not hasattr(self._inv, '_inv'):
        self._inv._inv = self

    return self._inv

  def __matmul__(self, other: Any) -> Union['LinearOperator', NDArray]:
    if isinstance(other, LinearOperator):
      return self._matmat(other)
    elif isinstance(other, np.ndarray):
      return self._matvec(other)
    else:
      return NotImplemented

  def simplify(self):
    """
    Simplify the operator. This is for future use.
    For now just return self.
    """
    return self


def unpack_csr(mat: sparse.csr_matrix) -> tuple[NDArray,
                                                NDArray,
                                                NDArray,
                                                Tuple[Int, Int]]:
  return mat.data, mat.indices, mat.indptr, mat.shape


def pack_csr(data: NDArray,
             indices: NDArray,
             indptr: NDArray,
             shape: Tuple[Int, Int]) -> sparse.csr_matrix:
  return sparse.csr_matrix((data, indices, indptr), shape=shape)


class Matrix(LinearOperator):
  """
  Class representing a sparse matrix. By default we store it in CSR format.

  We have to implement the `_field_names` attribute excplicitly because it
  does not follow from the `__init__` introspection because we allow the class
  to be initialized both my a `sparse.spmatrix` object or by the individual
  fields `data`, `indices`, `indptr` and `shape` that characterize a CSR matrix.
  """

  _field_names = 'data', 'indices', 'indptr', 'shape'

  def __init__(self, *args, **kwargs):

    # Check if the first argument is a sparse matrix
    if len(args) == 1 and isinstance(args[0], sparse.spmatrix):
      assert not kwargs  # kwargs should be empty
      data, indices, indptr, shape = unpack_csr(args[0].tocsr())
    else:
      # Otherwise we assume that the arguments are data, indices, indptr, shape
      # either all passed as positional arguments or as keyword arguments
      if len(args) == 0:
        args = tuple( kwargs[name] for name in self._field_names )
      data, indices, indptr, shape = args

    self.data = frozen(_round_array(data), dtype=np.float64)
    self.indices = frozen(indices, dtype=np.int64)
    self.indptr = frozen(indptr, dtype=np.int64)

    super().__init__(shape)

    self.mat = frozen_csr(pack_csr(self.data,
                                   self.indices,
                                   self.indptr,
                                   self.shape))

  def _matvec(self, b: NDArray) -> NDArray:
    return self.mat @ b

  def _matmat(self, other: 'LinearOperator') -> 'LinearOperator':
    if isinstance(other, Matrix):
      return self.__class__(self.mat @ other.mat)
    return super()._matmat(other)

  def _make_inv(self) -> 'LU':
    return LU(sparse.linalg.splu(self.mat.tocsc()))

  def _make_T(self) -> Self:
    return self.__class__(self.mat.T)

  def tocsr(self) -> NDArray:
    return self.mat


class LU(LinearOperator):
  """
  Class representing the inverse of a sparse matrix using LU factorization.
  Matrix multiplication is overloaded as solving the linear system Ax = b
  using the LU factorization.

  The class comes with a few caveats in the context of the `Immutable` class,
  namely that there is not default hash for sparse.linalg.SuperLU objects.
  We circumvent this by hashing the matrix that the LU factorizes.

  We have to overwrite the `_tobytes` and `__setstate__` methods to ensure
  proper functionality of the `Immutable` class.
  """

  def __init__(self, lu: sparse.linalg.SuperLU):
    assert isinstance(lu, sparse.linalg.SuperLU)
    self.lu = lu
    super().__init__(lu.shape)

  def _matvec(self, b: NDArray) -> NDArray:
    return self.lu.solve(b)

  def _make_inv(self) -> Matrix:
    n = self.shape[0]
    Pr = sparse.csc_array((np.ones(n), (self.lu.perm_r, np.arange(n))))
    Pc = sparse.csc_array((np.ones(n), (np.arange(n), self.lu.perm_c)))
    return Matrix(Pr.T @ (self.lu.L @ self.lu.U) @ Pc.T)

  def _make_T(self) -> Self:
    return self.inv.T.inv

  def toarray(self) -> NDArray:
    return self._matvec(np.eye(self.shape[0]))

  def tocsr(self) -> NDArray:
    return sparse.csr_matrix(self.toarray())

  @property
  def _tobytes(self) -> Tuple[Hashable, ...]:
    # we hash using the matrix that the LU factorizes
    return self.inv._tobytes

  def __setstate__(self, state):
    """
    For unpickling
    We first unpickle the stored matrix and then recompute the LU.
    """
    mat = Matrix(*state).mat
    self.__init__(sparse.linalg.splu(mat.tocsc()))


def as_LinearOperator(mat: Any) -> LinearOperator:
  if isinstance(mat, LinearOperator):
    return mat
  return Matrix(mat)


class MatMul(LinearOperator):

  def __init__(self, operators: Sequence[LinearOperator | sparse.spmatrix]):

    ops = []
    for op in map(as_LinearOperator, operators):
      if not isinstance(op, self.__class__):
        ops.append(op)
      else:
        ops.extend(op.operators)

    self.operators: Tuple[LinearOperator, ...] = tuple(ops)
    assert self.operators

    assert all( a.shape[1] == b.shape[0]
                for a, b in zip(self.operators, self.operators[1:]) ), \
            "Operators must be compatible for multiplication"

    shape = (self.operators[0].shape[0], self.operators[-1].shape[1])

    super().__init__(shape)

  def _matvec(self, b: NDArray) -> NDArray:
    for op in reversed(self.operators):
      b = op._matvec(b)
    return b

  def _matmat(self, other: 'LinearOperator') -> 'LinearOperator':
    if isinstance(other, self.__class__):
      return self.__class__(self.operators + other.operators)
    return self.__class__(self.operators + (other,))

  def _make_inv(self) -> Self:
    assert all(op.issquare for op in self.operators)
    return self.__class__(list(map(lambda x: x.inv, reversed(self.operators))))

  def _make_T(self) -> Self:
    return self.__class__(list(map(lambda x: x.T, reversed(self.operators))))

  def tocsr(self) -> NDArray:
    return reduce(lambda x, y: x @ y, (a.tocsr() for a in self.operators))


def splu_solve(mat: sparse.linalg.SuperLU, X: np.ndarray):
  assert X.shape[:1] == mat.shape[1:]
  return mat.solve(X.reshape(mat.shape[1], -1)).reshape(-1, *X.shape[1:])


class KroneckerOperator(Immutable):
  """
  Class that represents a Kronecker product of matrices that is never
  explicitly computed.
  """

  def __init__(self, operators: Sequence[LinearOperator | sparse.spmatrix]):
    self.operators = tuple(map(as_LinearOperator, operators))
    self.n, self.m = (tuple(map(lambda x: x.shape[i], self.operators)) for i in range(2))
    self.shape = (np.prod(self.n, dtype=int), np.prod(self.m, dtype=int))

  def __getitem__(self, item):
    ans = self.operators[item]
    if isinstance(ans, tuple):
      return self.__class__(ans)
    return ans

  def __len__(self):
    return len(self.operators)

  def __iter__(self):
    yield from self.operators

  def __bool__(self):
    return bool(len(self))

  def _matvec(self, b):
    """
    Function for performing a matrix-vector multiplication with a Kronecker
    product of matrices.

    The routine never carries out the Kronecker product explicitly, but rather
    applies the operators in the correct order.

    The input vector needs to have shape `self.shape[1]`. The output vector
    will have shape `self.shape[0] + b.shape[1:]` which means that any
    additional dimension of the input vector will just be vectorized.
    """
    b = np.asarray(b)
    assert b.shape[:1] == self.shape[1:]

    # input vector tensorial shape
    shape_in = [op.shape[1] for op in self]

    # output vector tensorial shape
    shape_out = [op.shape[0] for op in self]

    tail_shape = b.shape[1:]  # the tail shape of the vector for vectorial operations

    X = b.reshape(*shape_in, -1)

    for axis, op in enumerate(self):
      # Get current shape
      shapenow = X.shape

      # Apply operation along axis
      # Move axis to front, solve, then move back

      # Move to front and flatten all remaining axes
      X = np.moveaxis(X, axis, 0).reshape(shape_in[axis], -1)

      # We apply the operation to the zeroth axis, then we bring all remaining
      # axes to tensor-product shape
      X = (op @ X).reshape(shape_out[axis],
                           *(shapenow[:axis] + shapenow[axis + 1:]),
                           -1)

      # Move back to original axis
      X = np.moveaxis(X, 0, axis)

    return X.reshape(-1, *tail_shape)

  def _matmat(self, other):
    assert isinstance(other, KroneckerOperator) and len(other) == len(self)
    return self.__class__((a @ b for a, b in zip(self, other)))

  def __matmul__(self, other):
    if isinstance(other, KroneckerOperator):
      return self._matmat(other)
    elif isinstance(other, np.ndarray):
      return self._matvec(other)
    else:
      return NotImplemented

  @property
  def inv(self):
    return self.__class__([op.inv for op in self])

  @property
  def T(self):
    return self.__class__([op.T for op in self])

  def tocsr(self):
    return sparse_kron(*map(lambda x: x.tocsr(), self))

  def toarray(self):
    return self.tocsr().toarray()
