""" Numba implementations of various utility functions. """


from .util import np

from typing import Sequence
from functools import partial
import math

from numpy.typing import NDArray
from numba import njit, vectorize
from numba.types import int32, int64, float32, float64


# cache enabled by default
njit = partial(njit, cache=True)


""" itertools-equivalent Numba implementations. """


@njit
def _product(arr0: np.ndarray, list_of_linspaces: Sequence[np.ndarray]):
  """
  Given :class:`np.ndarray` `arr0` and :class:`list` of :class:`np.ndarray`s
  ``list_of_linspaces``,  create a column tensor product with ``arr0`` and
  all arrays in ``list_of_linspaces``.
  The input ``arr0`` is assumed to be two-dimensional, i.e., in the case of a
  single array, the shape needs to be ``(npoints, 1)``.
  """
  while True:
    lin0, list_of_linspaces = list_of_linspaces[0], list_of_linspaces[1:]
    n, m = len(lin0), len(arr0)
    m, ndims = arr0.shape[0], arr0.shape[1]

    ret = np.empty((len(arr0) * n, ndims + 1), dtype=arr0.dtype)
    for i in range(n):
      ret[i * m: (i+1) * m, -ndims:] = arr0

    counter = 0
    for i in range(n):
      myval = lin0[i]
      for j in range(arr0.shape[0]):
        ret[counter, 0] = myval
        counter += 1
    if len(list_of_linspaces) == 0:
      return ret
    arr0 = ret


@njit
def product(list_of_arrays):
  """
  Numba equivalent of the ``itertools.product`` iterator with the difference
  that it can be used inside of Numba, works only with array inputs and
  creates all products at once.
  >>> linspaces = [np.linspace(0, 1, i) for i in (4, 5, 6)]
  >>> X = np.stack(list(map(np.ravel, np,meshgrid(*linspaces))), axis=1)
  >>> Y = _product(linspaces)
  >>> np.allclose(X, Y)
      True
  """
  # XXX: This is a workaround for the fact that Numba does not support
  #      itertools.product. This implementation creates all products at once.
  #      This is not a problem for small arrays, but can be for
  #      large ones. In the long run, we should implement a more efficient
  #      version of this function. For now, this is a good enough solution.

  assert len(list_of_arrays) >= 1
  # If there is only one array, we need to add a dimension to it
  if len(list_of_arrays) == 1:
    return list_of_arrays[0][:, None]
  # Reverse the list of arrays to get the correct order
  list_of_arrays = list_of_arrays[::-1]
  # Call the recursive function
  return _product(list_of_arrays[0][:, None], list_of_arrays[1:])


@njit
def linspace_product(array_of_steps):
  """
  Convenience function for creating a product of linspaces
  over [0, 1] from an array of integers representing the steps.
  """
  list_of_arrays = []
  for i in array_of_steps:
    list_of_arrays.append(np.linspace(0, 1, i))
  return product(list_of_arrays)


@njit
def arange_product(array_of_integers):
  """
  Convenience function for creating a product of aranges
  from [0, i) from an array of integers containing the `i`.
  """
  list_of_arrays = []
  for i in array_of_integers:
    list_of_arrays.append(np.arange(i).astype(np.int64))
  return product(list_of_arrays)


"""
Formatting to strings for homogeneous string-based Numba hashing.
"""


@njit
def cut_trail(f_str):
  """
  Given a string representation of a float, remove trailing zeros.

  Used in the float2str function.
  """
  cut = 0
  for c in f_str[::-1]:
    if c == "0":
      cut += 1
    else:
      break
  if cut == 0:
    for c in f_str[::-1]:
      if c == "9":
        cut += 1
      else:
        cut -= 1
        break
  if cut > 0:
    f_str = f_str[:-cut]
  if f_str == "":
    f_str = "0"
  return f_str


@njit
def float2str(value):
  """
  Numba implementation of a float to string conversion.
  Credit to norok2 for the original implementation.

  Numba does not provide a built-in way to convert floats to strings, hence
  this (not so pretty) custom implementation.
  """
  if math.isnan(value):
    return "nan"
  elif value == 0.0:
    return "0.0"
  elif value < 0.0:
    return "-" + float2str(-value)
  elif math.isinf(value):
    return "inf"
  else:
    max_digits = 16
    min_digits = -4
    e10 = math.floor(math.log10(value)) if value != 0.0 else 0
    if min_digits < e10 < max_digits:
      i_part = math.floor(value)
      f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
      i_str = str(i_part)
      f_str = cut_trail(str(f_part)[1:max_digits - e10])
      return i_str + "." + f_str
    else:
      m10 = value / 10.0 ** e10
      i_part = math.floor(m10)
      f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
      i_str = str(i_part)
      f_str = cut_trail(str(f_part)[1:max_digits])
      e_str = str(e10)
      if e10 >= 0:
        e_str = "+" + e_str
      return i_str + "." + f_str + "e" + e_str


""" np.ufunc equivalents """


@njit
def mul_reduce(tpl):
  """
  Numba equivalent of np.multiply.reduce.

  Deprecated in favor of the `multiply` ufunc.
  """
  # it seems that initializing the value to 1 works fine for any (relevant)
  # array type contained in `tpl`.
  ret = 1
  for fac in tpl:
    ret *= fac
  return ret


# numba throws an experimental feature warning when using multiply.reduce
# but everything seems to work fine in practice
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)], nopython=True, cache=True)
def multiply(x, y):
  return x * y


@njit
def add_reduce(tpl):
  """
  Numba equivalent of np.add.reduce.

  Deprecated in favor of the `add` ufunc.
  """
  ret = 0
  for item in tpl:
    ret += item
  return ret


# Same warning as for multiply.reduce. Works fine in practice.
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)], nopython=True, cache=True)
def add(x, y):
  return x + y


"""
Various custom implementations of numpy functions not yet supported in Numba
"""


@njit
def ravel_multi_index(multi_index, dims):
  """
  Numba implementation of np.ravel_multi_index.
  """
  flat_index = 0
  stride = 1

  # Loop through dimensions in reverse order to calculate the flat index
  for i in range(len(dims) - 1, -1, -1):
    flat_index += multi_index[i] * stride
    stride *= dims[i]

  return flat_index


@njit
def unravel_multi_index(flat_index, dims):
  """
  Numba implementation of np.unravel_multi_index.

  Parameters
  ----------
  flat_index : int
      The flattened index to convert.
  dims : np.ndarray
      The shape of the multi-dimensional array.

  Returns
  -------
  multi_index : tuple of ints
      The multi-dimensional indices corresponding to the flat index.
  """
  multi_index = np.empty(len(dims), dtype=np.int64)

  for i in range(len(dims) - 1, -1, -1):
      multi_index[i] = flat_index % dims[i]  # Get the remainder
      flat_index //= dims[i]  # Update the flat index for the next dimension

  return multi_index


@njit
def _apply_pairs(indices: NDArray[np.integer], pairs: NDArray[np.integer]):
  while True:
    indices_new = indices.copy()
    for pair in pairs:
      indices_new[pair] = indices_new[pair].min()
    if (indices == indices_new).all():
      return indices
    indices = indices_new
