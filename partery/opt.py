from pyhaemo.spl import NDSpline

from functools import lru_cache

import jax


@lru_cache
def NDSpline_to_jax(spline: NDSpline, dx=0):
  """
  Convert a NDSpline object to a JAX-compatible function.
  """

  assert spline.nvars == 1, \
    NotImplementedError("Only 1D splines are supported for JAX conversion.")

  shape = spline.shape

  @jax.custom_vjp
  def splfunc(x):
    return spline([x], dx=dx).reshape(shape)

  def splfunc_fwd(x):
    return splfunc(x), x

  def splfunc_bwd(x, g):
    return spline([x], dx=dx+1).reshape(shape) * g,

  splfunc.defvjp(splfunc_fwd, splfunc_bwd)

  return splfunc
