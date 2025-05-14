from splico.util import np, _, frozen

from typing import Tuple
from itertools import permutations, product
from functools import lru_cache

from numpy.typing import NDArray, ArrayLike


def tuple_convert(arr: ArrayLike) -> Tuple | ArrayLike:
  """
  Recursively convert an array to a tuple.
  """
  try:
    return tuple(map(tuple_convert, np.asarray(arr)))
  except Exception:
    return arr


def rotation_matrices_iter(n: int):
  """
  All orientation preserving rotation matrices in n dimensions.
  """
  A = np.eye(n)
  for B in map(A.__getitem__, map(list, permutations(range(n)))):
    for item in map(list, product(*([(1, -1)]*n))):
      ret = B * np.array(item, dtype=int)[_]
      if np.linalg.det(ret) == 1:
        yield frozen(ret)


def all_permutations_iter(n: int):
  """
  Given a flat array representing the indices of points of a hypercube
  of length (2 ** n), return all permutations of the points that are
  orientation preserving.
  For example, 2D:

  [0, 1, 2, 3]
  [2, 0, 3, 1]
  [3, 2, 1, 0]
  [1, 3, 0, 2]

  """
  assert n > 0
  if n == 1:
    yield np.array([0, 1])
    return

  points = np.array(list(product([-1, 1], repeat=n)), dtype=int)

  for mypoints in map(points.__matmul__, rotation_matrices_iter(n)):
    yield np.lexsort(mypoints[:, ::-1].T)


@lru_cache
def all_permutations(n: int) -> Tuple[NDArray, ...]:
  return tuple(map(frozen, all_permutations_iter(n)))
