"""
Auxiliary functionality for the geo module.

@author: Jochen Hinz
"""

from splico.spl import NDSpline, NDSplineArray


from typing import Callable
from itertools import chain
from functools import wraps


Spline = NDSpline | NDSplineArray


to_NDSplineArray = lambda x: NDSplineArray(x) if isinstance(x, NDSpline) else x


def spline_or_array(f: Callable) -> Callable:
  """
  Decorator for functions that can either take :class:`splico.spl.NDSpline`
  or :class:`splico.spl.NDSplineArray` inputs. The function is then called
  with the input converted to a :class:`splico.spl.NDSplineArray` and the output
  is converted back to :class:`NDSpline` if all inputs were of this type.
  """

  @wraps(f)
  def wrapper(*args, **kwargs):
    # check if there are no NDSplineArrays
    allsplines = len([ a for a in chain(args, kwargs.values())
                               if isinstance(a, NDSplineArray) ]) == 0

    ret = f(*map(to_NDSplineArray, args),
            **{k: to_NDSplineArray(v) for k, v in kwargs.items()})

    # convert back to NDSpline if applicable
    if allsplines:
      assert ret._shape == ()
      return ret.arr.ravel()[0]

    return ret

  return wrapper
