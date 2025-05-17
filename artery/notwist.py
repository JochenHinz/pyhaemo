from pyhaemo.util import np, normalize, isincreasing
from pyhaemo.spl import NDSpline

from typing import Optional, Sequence

from numba import njit
import treelog as log


@njit(cache=True)
def rotate_vector(v, u, theta):

    u = u / np.linalg.norm(u)  # Normalize the axis vector
    cross_product = np.cross(u, v)  # Compute the cross product
    dot_product = np.dot(u, v)  # Compute the dot product
    v_rot = v * np.cos(theta) + cross_product * np.sin(theta) + u * dot_product * (1 - np.cos(theta))

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
  n0 = n0/np.linalg.norm(n0)
  binormal0 = np.cross(t0, n0) / np.linalg.norm(np.cross(t0, n0))

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
      phi = np.arccos(max(-1, min(np.dot(told, tnew), 1)))

      binormal = rotate_vector(Rs[i-1][:, 1], omega, phi)
      normal = rotate_vector(Rs[i-1][:, 0], omega, phi)

    Rs[i][:, 0] = normal/np.linalg.norm(normal)
    Rs[i][:, 1] = binormal/np.linalg.norm(binormal)
    Rs[i][:, 2] = T[i]/np.linalg.norm(T[i])

  return Rs


# @njit(cache=True)
def _compute_notwistframe_bif_to_bif(T, n0, nfinal, zero_thresh):
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

  Rs_twist = _compute_notwistframe(T, n0, zero_thresh)
  nfinal_twisted = Rs_twist[-1][:, 0]

  t0 = T[0]
  n0 = n0/np.linalg.norm(n0)
  binormal0 = np.cross(t0, n0) / np.linalg.norm(np.cross(t0, n0))

  nfinal = nfinal/np.linalg.norm(nfinal)

  absolute_angle = np.arccos(np.dot(nfinal_twisted, nfinal))

  ref_angle = np.array(([0, np.pi/2, np.pi, 3/2*np.pi]))

  close_to = np.argmin(np.abs(ref_angle - absolute_angle))
  angle = absolute_angle - ref_angle[close_to]

  tolerance = 1e-10

  condition_1 = np.abs(np.dot(nfinal, rotate_vector(Rs_twist[-1][:, 0], T[-1], angle))) < tolerance
  condition_2 = np.isclose(np.abs(np.dot(nfinal, rotate_vector(Rs_twist[-1][:, 0], T[-1], angle))), 1, atol=tolerance)

  if condition_1 or condition_2:
    pass
  else:
    angle = -angle

  n_of_slice = len(T)

  Rs_final = np.empty((len(T), 3, 3), dtype=np.float64)
  Rs_final[0] = np.stack((n0, binormal0, t0), axis=1)

  for i in range(len(Rs_twist)):

      rotation = angle * i/(n_of_slice-1)
      normal = rotate_vector(Rs_twist[i][:, 0], T[i], rotation)
      binormal = rotate_vector(Rs_twist[i][:, 1], T[i], rotation)

      Rs_final[i][:, 0] = (normal)/np.linalg.norm(normal)
      Rs_final[i][:, 1] = binormal/np.linalg.norm(binormal)
      Rs_final[i][:, 2] = T[i]/np.linalg.norm(T[i])

  return Rs_final


def _compute_notwistframe_end_to_end(T, n0, nfinal, zero_thresh):
  """
  Notwistframe where both ends are fixed.
  """
  n0 = normalize(n0)
  nfinal = normalize(nfinal)

  # compute the forward notwistframe
  Rs_forward = _compute_notwistframe(T, n0, zero_thresh)

  # express the final normal in the local coordinate system
  nfinal_loc = np.linalg.solve(Rs_forward[-1], nfinal)

  assert np.allclose(np.abs(nfinal_loc[-1]), 0)

  # compute the angle between the two local normals on the last section
  # between (-pi, pi]
  angle0 = np.arctan2(np.cross(np.array([1, 0]), nfinal_loc[:2]),
                      np.array([1, 0]) @ nfinal_loc[:2])

  # check if one of the options is closer to 0
  options = angle0 + np.linspace(-np.pi, np.pi, 5)[:-1]

  # keep it
  myangle = options[np.argmin(np.abs(options))]

  ans = []
  for i, R in enumerate(Rs_forward):
    rotate = lambda v: rotate_vector(v,
                                     np.array([0, 0, 1.0]),
                                     myangle * i / (len(Rs_forward) - 1))

    # rotate the local vectors by the partial angle and put back
    # in the global frame
    ans.append( np.stack([ R @ rotate(np.array([1.0, 0, 0])),
                           R @ rotate(np.array([0, 1.0, 0])),
                                                        T[i] ], axis=1) )

  return np.stack(ans, axis=0)


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
  else:  # bif to bif tract
    assert (nfinal := np.asarray(nfinal)).shape == (3,)
    # return _compute_notwistframe_bif_to_bif(tangents, n0, nfinal, zero_thresh)
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


if __name__ == '__main__':
  from splico.spl import UnivariateKnotVector
  from splico.geo import ellipse
  xi = np.linspace(0, 1, 101)

  kv = UnivariateKnotVector(np.linspace(0, 1, 11)).to_tensor()
  data = np.stack([xi, np.zeros(101), np.zeros(101)], axis=1)

  spl = kv.fit([xi], data, lam0=0)

  n0 = np.array([0, 1, 0]).astype(float)
  nfinal = normalize(np.array([0, .5, .5]).astype(float))

  frame = _compute_notwistframe_end_to_end(spl(xi, dx=1),
                                           n0,
                                           nfinal,
                                           1e-8)

  rRs = kv.fit([xi], frame, lam0=0)
  ell = ellipse(1, 1, 4)

  vessel = (ell[:, None] @ rRs).sum(-1) + ell.unity @ spl

  vessel.qplot()

  import ipdb
  ipdb.set_trace()
