"""
Module for spline interpolation. Either linear or higher order interpolation.

@author: Jochen Hinz
"""

from splico.spl import NDSplineArray, UnivariateKnotVector
from splico.types import Int
from ..util import np
from .aux import spline_or_array


@spline_or_array
def linear_interpolation(spl0: NDSplineArray,
                         spl1: NDSplineArray,
                         kvz: Int | UnivariateKnotVector = 1) -> NDSplineArray:
  """
  Perform a linear interpolation between two splines.
  This introduces a new parametric dependency which is appended at the end,
  i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

  Parameters
  ----------
  spl0 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The first input spline.
  spl1 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The second input spline.
  kvz : :class:`int` or class:`UnivariateKnotVector`
      The spline basis order in the xn+1-direction. Must be >= 1.
      Alternatively a :class:`UnivariateKnotVector` can be provided.
      Must be of degree >= 1.

  Returns
  -------
  spl : :class:`splico.spl.NDSpline`
      The resulting linearly interpolated spline satisfying
      `spl.nvars == spl0.nvars + 1`.
  """
  if isinstance(kvz, Int):
    kvz = UnivariateKnotVector(np.linspace(0, 1, 2), degree=kvz)

  assert kvz.degree >= 1
  assert spl0.shape == spl1.shape
  assert (spl0.contract_all().knotvector == spl1.contract_all().knotvector).all()

  # get the greville points
  z = kvz.greville

  # we compute the hermite interpolation functions by fitting againts the
  # stable greville points. We need not stabilize and the fit should be exact.
  f0, f1 = kvz.to_tensor().fit([z], np.stack([1 - z, z], axis=1), lam0=0)

  return spl0 * f0 + spl1 * f1


@spline_or_array
def cubic_hermite_interpolation(spl0: NDSplineArray,
                                spl1: NDSplineArray,
                                t0: np.ndarray,
                                t1: np.ndarray,
                                kvz: UnivariateKnotVector | Int = 3) -> NDSplineArray:
  """
  Perform a cubic hermite interpolation between two splines.
  This introduces a new parametric dependency which is appended at the end,
  i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

  Parameters
  ----------
  spl0 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The first input spline.
  spl1 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The second input spline.
  t0 : :class:`np.ndarray` or :class:`NDSplineArray`
      The desired tangent at xn+1 == 0.
  t1 : :class:`np.ndarray` or :class:`NDSplineArray`
      The desired tangent at xn+1 == 1.
  kvz : :class:`int` or class:`UnivariateKnotVector`
      The spline basis order in the xn+1-direction. Must be >= 3.
      Alternatively a :class:`UnivariateKnotVector` can be provided.
      Must be of degree >= 3.

  Returns
  -------
  spl : :class:`splico.spl.NDSpline`
      The resulting cubic Hermite interpolation spline satisfying
      `spl.nvars == spl0.nvars + 1`.
  """
  if isinstance(kvz, Int):
    kvz = UnivariateKnotVector(np.linspace(0, 1, 2), degree=kvz)

  assert kvz.degree >= 3
  assert spl0.shape == spl1.shape
  assert (spl0.contract_all().knotvector == spl1.contract_all().knotvector).all()

  one = spl0.one(spl0.knotvector)

  if not isinstance(t0, NDSplineArray):
    t0 = one * t0
  if not isinstance(t1, NDSplineArray):
    t1 = one * t1

  if t0.shape[-1:] != (3,) or t1.shape[-1:] != (3,):
    raise NotImplementedError

  z = kvz.greville

  # as in `linear_interpolation` but this time we fit the cubic Hermite functions
  f0, f1, f2, f3 = kvz.to_tensor().fit([z], np.stack([2 * z**3 - 3 * z**2 + 1,
                                                      z**3 - 2 * z**2 + z,
                                                      -2 * z**3 + 3 * z**2,
                                                      z**3 - z**2], axis=1), lam0=0)

  # Interpolation along the last axis.
  return spl0 * f0 + t0 * f1 + spl1 * f2 + t1 * f3
