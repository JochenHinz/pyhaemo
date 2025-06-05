from pyhaemo.util import np, normalize, isincreasing
from pyhaemo.spl import NDSpline

from typing import Optional, Sequence

from numba import njit
import numba.np.extensions as ext
import treelog as log


@njit(cache=True)
def normalize_vector(vec):
  return vec / np.sqrt((vec**2).sum())


@njit
def clip(value, a, b):
  return min(max(value, a), b)


@njit(cache=True)
def rotate_vector(v, u, theta):
  """Eulero Rodrigues formula for rotating a vector v
     around an axis u of an angle theta.
  """

  u = normalize_vector(u)
  v_rot = v * np.cos(theta) + \
          np.cross(u, v) * np.sin(theta) + \
          u * np.dot(u, v) * (1 - np.cos(theta))

  return v_rot


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
      phi = np.arccos(clip(np.dot(told, tnew), -1.0, 1.0))

      binormal = rotate_vector(Rs[i-1][:, 1], omega, phi)
      normal = rotate_vector(Rs[i-1][:, 0], omega, phi)

    Rs[i][:, 0] = normalize_vector(normal)
    Rs[i][:, 1] = normalize_vector(binormal)
    Rs[i][:, 2] = normalize_vector(T[i])

  return Rs


@njit(cache=True)
def _compute_notwistframe_end_to_end(T, n0, nfinal, zero_thresh):
  """
    Notwistframe where both ends are fixed.
  """

  n0 = normalize_vector(n0)
  nfinal = normalize_vector(nfinal)

  # compute the forward notwistframe
  Rs_forward = _compute_notwistframe(T, n0, zero_thresh)

  # express the final normal in the local coordinate system
  nfinal_loc = np.linalg.solve(Rs_forward[-1], nfinal)

  assert np.allclose(np.abs(nfinal_loc[-1]), 0)

  # compute the angle between the two local normals on the last section
  # between (-pi, pi]
  angle0 = np.arctan2(ext.cross2d(np.array([1, 0]), nfinal_loc[:2]),
                      np.array([1, 0], dtype=np.float64) @ nfinal_loc[:2])

  # check if one of the options is closer to 0
  # options = angle0 + np.linspace(-np.pi, np.pi, 5)[:-1]

  # keep it
  # myangle = options[np.argmin(np.abs(options))]
  myangle = angle0

  ans = np.empty_like(Rs_forward)
  for i, R in enumerate(Rs_forward):
    binormal = R @ rotate_vector(np.array([1.0, 0.0, 0.0], dtype=np.float64),
                                 np.array([0.0, 0.0, 1.0], dtype=np.float64),
                                 myangle * i / (len(Rs_forward) - 1))

    normal = R @ rotate_vector(np.array([0.0, 1.0, 0.0], dtype=np.float64),
                               np.array([0.0, 0.0, 1.0], dtype=np.float64),
                               myangle * i / (len(Rs_forward) - 1))

    ans[i] = np.stack((binormal, normal, T[i]), axis=1)

  return ans


def compute_notwistframe(tangents: np.ndarray,
                         n0: Optional[np.ndarray | Sequence[float]] = None,
                         nfinal: Optional[np.ndarray | Sequence[float]] = None,
                         zero_thresh: float = 1e-8) -> np.ndarray:
  """
    Compute a discrete no-twist-frame.
    The docstring is the same as for `_compute_notwistframe` but `n0` and
    `zero_thresh` receive reasonable default values if not passed.
  """
  assert (tangents := normalize(tangents)).shape[1:] == (3,)
  assert zero_thresh >= 0

  if n0 is None and nfinal is None:
    n0 = np.array([-tangents[0, 2], 0, tangents[0, 0]])
    return _compute_notwistframe(tangents, n0, zero_thresh)

  elif n0 is None:
    return _compute_notwistframe(tangents[::-1], nfinal, zero_thresh)[::-1]
  elif nfinal is None:
    return _compute_notwistframe(tangents, n0, zero_thresh)
  else:
    assert (nfinal := np.asarray(nfinal)).shape == (3,)
    return _compute_notwistframe_end_to_end(tangents, n0, nfinal, zero_thresh)


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
    spline : :class:`pyhaemo.spl.NDSpline`
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
        Keyword arguments that are forwarded to `pyhaemo.spl.TensorKnotVector.fit`.

    Returns
    -------

    Rs : :class:`np.ndarray` or :class:`pyhaemo.spl.NDSpline`
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
