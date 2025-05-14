"""
Various utility functions.
@author: Jochen Hinz
"""

import sys
from functools import wraps, cached_property
from collections import namedtuple
from typing import Tuple, Sequence, Optional, Callable, Any, Literal
import contextlib

import numpy as np
from scipy import sparse


# alias for vectorisation.
# >>> arr = np.ones((10,), dtype=int)
# >>> print(arr.shape)
#     (10,)
# >>> print(arr[:, np.newaxis].shape)
#     (10, 1)
# >>> print(arr[:, _].shape)
#     (10, 1)
_ = np.newaxis


# points are rounded to this number of significant digits
GLOBAL_PRECISION = 12
BAR_LENGTH = 100


def _round_array(arr: np.ndarray | Sequence[Any], precision: Optional[Any] = None) -> np.ndarray:
  """
  Round an array to ``precision`` which defaults to ``GLOBAL_PRECISION``
  if not passed.
  """
  if precision is None:
    precision = GLOBAL_PRECISION
  return np.round(arr, precision)


def round_result(fn: Callable) -> Callable:
  """
  Decorator that rounds the returned array to the global precision.
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return _round_array(fn(*args, **kwargs))
  return wrapper


@contextlib.contextmanager
def global_precision(precision: int):
  """
  Context manager for locally adjusting the machine precision.
  Applies to all attributes / arrays that are truncated using the
  ``_round_array`` method.

  Example
  -------

  >>> class MyClass:
  ... def __init__(self, a: np.ndarray):
  ...   self.a = frozen(_round_array(a))  # round and freeze array

  >>> A = MyClass(np.linspace(0, 1, 4))
  >>> A.a
  ... [0, 0.333333333333, 0.666666666667, 1]

  >>> with global_precision(4):
  >>>   A = MyClass(np.linspace(0, 1, 4))
  >>> A.a
  ... [0, 0.3333, 0.6667, 1]
  """
  global GLOBAL_PRECISION
  old_precision = int(GLOBAL_PRECISION)
  assert 0 < (precision := int(precision)) < 16
  GLOBAL_PRECISION = precision
  try:
    yield
  finally:
    GLOBAL_PRECISION = old_precision


def frozen(array: np.ndarray | Sequence[Any] | Any, dtype=None) -> np.ndarray:
  """
  Freeze a vector inplace and return it.

  Example
  -------

  >>> arr = np.zeros((10,), dtype=int)
  >>> print(arr[0])
  ... 0
  >>> arr[0] = 1
  >>> print(arr[0])
  ... 1
  >>> arr = np.zeros((10,), dtype=int)
  >>> arr = frozen(arr)
  >>> arr[0] = 1
  ... ERROR

  If the input is and array of the desired dtype alread, Both in and out of
  place will work.
  >>> arr = np.zeros((10,), dtype=int)
  >>> frozen(arr)
  >>> arr[0] = 1
  ... ERROR
  """
  array = np.asarray(array, dtype=dtype)
  array.flags.writeable = False
  return array


def freeze(fn: Callable, dtype=None) -> Callable:
  """
  Decorator that freezes the returned array inplace.

  Example
  -------

  def multiply(arr, val):
    return val * arr

  >>> arr = np.ones((5,), dtype=int)
  >>> new_arr = multiply(arr, 2)
  >>> print(new_arr)
  ... [2, 2, 2, 2, 2]
  >>> new_arr[0] = 10
  >>> print(new_arr)
  ... [10, 2, 2, 2, 2]

  @freeze
  def multiply(arr, val):
    return val * arr

  >>> arr = np.ones((5,), dtype=int)
  >>> new_arr = multiply(arr, 2)
  >>> print(new_arr)
  ... [2, 2, 2, 2, 2]
  >>> new_arr[0] = 10
  ... ERROR
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return frozen(fn(*args, **kwargs), dtype=dtype)
  return wrapper


def frozen_cached_property(fn: Callable) -> cached_property:
  """
  Combined decorator for `freeze` and `cached_property`.
  Equivalent to:

  >>> @cached_property
  >>> @freeze
  >>> def myfunc(self):
  ...   return np.arange(self.n)
  """
  return cached_property(freeze(fn))


# named tuple with fields corresponding to a serialized array (for hashing)
serialized_array = namedtuple('serialized_array', ('shape', 'dtype', 'bytes'))
serialized_objarr = namedtuple('serialized_objarr', ('shape', 'data'))


# serialize an array for hashing
def serialize_array(arr: np.ndarray) -> serialized_array:
  return serialized_array(arr.shape, arr.dtype.str.encode(), arr.tobytes())


# convert serialized array back to ordinary np.ndarray
def deserialize_array(serial: serialized_array) -> np.ndarray:
  shape, dtype, _bytes = serial
  return np.frombuffer(_bytes, dtype=np.dtype(dtype.decode())).reshape(shape)


# serialize an object array for hashing
def serialize_objarr(arr: np.ndarray) -> serialized_objarr:
  return serialized_objarr(arr.shape, tuple(arr.ravel()))


# convert serialized object array back to ordinary np.ndarray
def deserialize_objarr(serial: serialized_objarr) -> np.ndarray:
  shape, data = serial
  ret = np.empty(len(data), dtype=object)
  for i, elem in enumerate(data):
    ret[i] = elem
  return ret.reshape(shape)


def serialize_input(fn: Callable) -> Callable:
  """
  Serialize the input array before passing it to the function.
  """
  @wraps(fn)
  def wrapper(arr, *args, **kwargs):
    return fn(serialize_array(arr), *args, **kwargs)
  return wrapper


class ProgressBar:
  """ For monitoring the progress of a loop. """

  def __init__(self, prefix='Task', suffix='completed'):
    self.prefix = str(prefix).strip()
    self.suffix = str(suffix).strip()

  def __call__(self, factor):
    filled_up_Length = int(round(BAR_LENGTH * factor))
    percentage = round(100.0 * factor, 1)
    bar = '=' * filled_up_Length + '-' * (BAR_LENGTH - filled_up_Length)
    sys.stdout.write('%s [%s] %s%s %s.\r' % (self.prefix, bar, percentage, '%', self.suffix))
    sys.stdout.flush()

  def __del__(self):
    sys.stdout.write('\n')


def isincreasing(arr: Sequence | np.ndarray) -> np.bool_:
  """
  Return True if an array-like is strictly increasing.
  Else return False.
  """
  # XXX: add axis argument
  arr = np.asarray(arr)
  assert arr.ndim == 1
  return (np.diff(arr) > 0).all()


def gauss_quadrature(a: int | float, b: int | float, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
  """
  Given the element boundaries ``(a, b)``, return the weights and evaluation points
  corresponding to a gaussian quadrature scheme of order ``order``.

  Parameters
  ----------
  a : :class:`float`
    The left boundary of the element.
  b : :class:`float`
    The right boundary of the element.
  order : :class:`int`
    The order of the Gaussian quadrature scheme.

  Returns
  -------
  weights : :class:`np.ndarray`
    The weights of the quadrature scheme.
  points : :class:`np.ndarray`
    The points (abscissae) over (a, b).
  """
  assert b > a
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  return (b - a) / 2 * weights, a + points * (b - a)


def clparam(points: Sequence | np.ndarray) -> np.ndarray:
  """
  Compute the cumulative relative length parameter of a curve defined by the
  points. Returns the parameter as a 1D array from 0 to 1.
  """
  points = np.asarray(points)
  assert points.ndim
  if points.ndim == 1:
    points = points[:, _]

  ret = np.array([0, *np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()])
  return ret / ret[-1]


def normalize(array: np.ndarray, axis=-1) -> np.ndarray:
  """
  Normalize an array of vectors.
  """
  array = np.asarray(array)
  return array / np.linalg.norm(array, axis=axis, ord=2, keepdims=True)


def flat_meshgrid(*arrays, indexing: Literal['xy', 'ij'] = 'ij',
                           axis: int = 0) -> np.ndarray:
  """
  Create a meshgrid and flatten it along the specified axis.
  """
  meshgrid = tuple(np.meshgrid(*arrays, indexing=indexing))
  return np.stack(list(map(np.ravel, meshgrid)), axis=axis)


def augment_by_zeros(points: np.ndarray, axis_target=3, axis=1):
  """
  Augment the points with zeros along the specified axis to reach the target.
  """
  n = (points := np.asarray(points)).shape[axis]
  if n > axis_target:
    raise AssertionError('Cannot augment zeros to axis whose length'
                         ' exceeds `axis_target`.')
  if n == axis_target:
    return points
  zeros_shape = tuple(dim if i != axis else axis_target - n
                      for i, dim in enumerate(points.shape))
  return np.concatenate([points, np.zeros(zeros_shape, dtype=points.dtype)], axis=axis)


def sorted_tuple(tpl):
  """
  Return a tuple with sorted elements.
  """
  return tuple(sorted(tpl))


""" rotation matrices """


def rot_mat(angle: float, axis: np.ndarray):
  assert axis.ndim == 1

  axis = normalize(axis)
  n = len(axis)

  if n == 2:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  elif n == 3:
    return np.cos(angle) * np.eye(3) + \
           np.sin(angle) * np.cross(np.eye(3), axis) + \
           (1 - np.cos(angle)) * np.outer(axis, axis)
  else:
    raise NotImplementedError('Only 2D and 3D rotations are supported.')


def nutils_to_scipy(mat) -> sparse.csr_matrix:
  data = mat.export('csr')
  return sparse.csr_matrix(data, dtype=float)
