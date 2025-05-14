from splico.util import np, normalize, isincreasing
from splico.spl import NDSpline

from typing import Optional, Sequence

from numba import njit
import treelog as log


@njit(cache=True)
def normalize_vector(vec):
  return vec / np.sqrt((vec**2).sum())


@njit(cache=True)
def _compute_notwistframe(T, n0, zero_thresh):
  """
    Compute a discrete no-twist-frame.

    Parameters
    ----------
    T : :class:`np.ndarray`
        Array of shape (n, 3) containing n tangents. Assumed to be normalized.
    n0 : :class:`np.ndarray`
        The initial normal vector of shape (3,).
    zero_thresh : :class:`float`
        A threshold value that dictates whether two consecutive tangents
        should be considered equal.
        They're considered equal if |t0 - t1|_2 < zero_thresh.

    Returns
    -------
    Rs : :class:`np.ndarray`
        Array of shape (n, 3, 3) containing rotation matrices.
        Rs[i][:, j] equals the i-th normal for j=0, binormal for j=1 and
        tangent for j=2, respectively.
  """

  t0 = T[0]
  n0 = normalize_vector(n0)
  binormal0 = normalize_vector(np.cross(t0, n0))

  Rs = np.empty((len(T), 3, 3), dtype=np.float64)
  Rs[0] = np.stack((n0, binormal0, t0), axis=1)

  for i, (told, tnew) in enumerate(zip(T, T[1:]), 1):
    omega = np.cross(told, tnew)
    norm = (omega**2).sum()**.5
    if norm <= zero_thresh:
      binormal = Rs[i-1][:, 1]
      normal = Rs[i-1][:, 0]
    else:
      omega = omega / norm
      phi = np.arccos(np.dot(told, tnew))

      binormal_p = Rs[i-1][:, 1]
      binormal = binormal_p * np.cos(phi) + \
                 np.cross(omega, binormal_p) * np.sin(phi) + \
                 omega * np.dot(omega, binormal_p) * (1 - np.cos(phi))

      normal_p = Rs[i-1][:, 0]
      normal = normal_p * np.cos(phi) + \
               np.cross(omega, normal_p) * np.sin(phi) + \
               omega * np.dot(omega, normal_p) * (1 - np.cos(phi))

    Rs[i][:, 0] = normalize_vector(normal)
    Rs[i][:, 1] = normalize_vector(binormal)
    Rs[i][:, 2] = normalize_vector(T[i])

  return Rs


def compute_notwistframe(tangents: np.ndarray,
                         n0: Optional[np.ndarray | Sequence[float]] = None,
                         zero_thresh: float = 1e-8) -> np.ndarray:
  """
    Compute a discrete no-twist-frame.
    The docstring is the same as for `_compute_notwistframe` but `n0` and
    `zero_thresh` receive reasonable default values if not passed.
  """

  assert (tangents := normalize(tangents)).shape[1:] == (3,)
  assert zero_thresh >= 0

  if n0 is None:
    n0 = np.array([-tangents[0, 2], 0, tangents[0, 0]])

  assert (n0 := np.asarray(n0)).shape == (3,)

  return _compute_notwistframe(tangents, n0, zero_thresh)


def compute_notwistframe_from_spline(spline: NDSpline,
                                     abscissae: np.ndarray | Sequence[float] | int,
                                     refit=False,
                                     framekwargs=None,
                                     refitkwargs=None) -> np.ndarray | NDSpline:
  """
    Compute a discrete no-twist-frame from an `NDSpline` input.
    The spline's tangent is computed in `abscissae` and the result is forwarded
    to `compute_notwistframe`. If `refit` is `True`, a spline with the same
    knotvector as `spline` is fit to the discrete no-twist-frame.

    Parameters
    ----------
    spline : :class:`splico.spl.NDSpline`
        The input spline. Must satisfy spline.shape == (3,) and spline.nvars == 1.
    abscissae : :class:`np.ndarray` or Sequence[float] or :class:`int`
        A strictly monotone sequence of parametric values. If passed as an integer,
        it will be converted to a linspace from a to b in `abscissae` steps, where
        a and b are the first and last knot for the spline's knotvector.
        Input must be contained in the interval [a, b].
    refit : :class:`bool`
        Whether or not the result should be refit.
    framekwargs : :class:`dict`, optional
        Keyword arguments that are forwarded to `compute_notwistframe`.
    refitkwargs : :class:`dict`, optional
        Keyword arguments that are forwarded to `splico.spl.TensorKnotVector.fit`.

    Returns
    -------

    Rs : :class:`np.ndarray` or :class:`splico.spl.NDSpline`
        The discrete no-twist-frame or a spline fit of it if `refit` is `True`.
  """

  assert spline.nvars == 1 and spline.shape == (3,)
  if refit is False and refitkwargs:
    log.warning("Received refit keyword arguments but refit is set to false."
                " They will be ignored.")

  (a, *ignore, b), = spline.knotvector.knots

  if np.isscalar(abscissae):
    assert isinstance(abscissae, (int, np.int_))
    abscissae = np.linspace(a, b, abscissae)

  abscissae = np.asarray(abscissae, dtype=float)

  assert a <= abscissae.min() < abscissae.max() <= b
  assert isincreasing(abscissae)

  tangents = normalize(spline(abscissae, dx=1))

  Rs = compute_notwistframe(tangents, **framekwargs or {})

  if refit:
    Rs = spline.knotvector.fit([abscissae], Rs, **refitkwargs)

  return Rs
