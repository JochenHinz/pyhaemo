from pyhaemo.util import np, _
from pyhaemo.types import Int, Numeric
from pyhaemo.util import normalize
from pyhaemo.spl import NDSplineArray, NDSpline
from pyhaemo.geo.bif import repeated_knot_disc, hermite_pols_interval, \
                            quarter_disc, join_quarters, hermite_nonconstant_tangent
from pyhaemo.geo.interp import cubic_hermite_interpolation

from typing import Sequence


def bif_from_matrices(matrices: Sequence[np.ndarray],
                      centerpoints: Sequence[np.ndarray],
                      ax: np.ndarray,
                      xC: np.ndarray,
                      bB: Numeric,
                      bT: Numeric,
                      unitdisc: NDSplineArray | Int = 4) -> NDSplineArray:
  """
  Create a bifurcation from a set of matrices and centerpoints.
  The matrices represent the rotation and stretch of the unit disc on the three
  input vessel cross sections leading up to the bifurcation.
  The centerpoints are the center points of the input vessels.
  The unit disc is the disc that is used to create the roof of the bifurcation.

  Parameters
  ----------
  matrices: :class:`ArrayLike`
    Array-like containing :class:`np.ndarray` objects of shape (3, 3).
    There must be at least three.
  centerpoints: :class:`ArrayLike`
    The corresponding centerpoints.
    The number must match the number of matrices.
  ax: :class:`np.ndarray`
    The local z-axis of the bifurcation.
  xC: :class:`np.ndarray`
    The center point of the bifurcation.
  bB: :class:`Float`
    The bottom height of the bifurcation.
  bT: :class:`Float`
    The top height of the bifurcation.
  unitdisc: :class:`NDSplineArray` or :class:`Int`
    The unit disc that is used to create the roof of the bifurcation.
    If :class:`Int`, a new unit disc is created with the given number of elements.

  There are not `cevalpoints` in this function, as the input curves are not
  defined by a set of evaluation points, but by the matrices.
  The curves are created automatically and the evaluation points are
  simply given by 0.5.
  """

  if isinstance(unitdisc, Int):
    unitdisc = repeated_knot_disc(unitdisc, reparam=True)

  radii = [ np.mean( np.linalg.norm(m[:2], axis=0) ) for m in matrices ]

  unitdisc = unitdisc.to_ndim(1)

  evalL = [.5], [1]
  evalR = [1], [.5]

  xL = unitdisc[0](*evalL).ravel()
  xR = unitdisc[4](*evalR).ravel()
  ax = normalize(np.asarray(ax))

  discs = list(map(lambda x, c: (c + (x * unitdisc[:, _]).sum(-1)).to_ndim(1),
                   matrices, centerpoints))

  spls = []
  for r0, r1, mat0, mat1, center0, center1, disc0, disc1 in \
                                          zip(radii,
                                              np.roll(radii, -1),
                                              matrices,
                                              np.roll(matrices, -1, axis=0),
                                              centerpoints,
                                              np.roll(centerpoints, -1, axis=0),
                                              discs,
                                              np.roll(discs, -1)):

    t0, t1 = map(normalize, (mat0[:, 2], mat1[:, 2]))
    p0 = disc0[0](*evalL).ravel()
    p1 = disc1[4](*evalR).ravel()

    # dist = np.linalg.norm(center1 - center0)

    spls.append( cubic_hermite_interpolation(NDSpline([], p0[_]),
                                             NDSpline([], p1[_]),
                                             4 * r0 * t0,
                                             -4 * r1 * t1) )

  wings = []
  for splR, splL, disc, center in zip(np.roll(spls, 1),
                                      spls,
                                      discs,
                                      centerpoints):

    # TODO: reuse the `wing` function from above

    xR = splR([.5]).ravel()  # in ccw order this one comes first
    xL = splL([.5]).ravel()
    tR = -splR([.5], dx=1).ravel() / 2
    tL = splL([.5], dx=1).ravel() / 2

    aR = np.linalg.norm(xC - xR)
    aL = np.linalg.norm(xC - xL)

    # Right arm rotation matrix (normalized in x and y, not necessarily in z)
    RR = np.stack([(xR - xC) / aR, ax, tR], axis=1)

    # Left arm rotation matrix
    RL = np.stack([(xC - xL) / aL, ax, tL], axis=1)

    t0f = lambda geom: splL([0], dx=1).ravel() / 2

    def t1f(geom):
      x, *ignore = geom

      # we split the disc into two parts, separated by x = 0.
      H00, H10, *ignore = hermite_pols_interval(-1, 0)
      H01, H11, *ignore = hermite_pols_interval(0, 1)

      # The tangent is the center point minus the center of the inlet.
      tcomb = xC - center

      # This is essentially to say that at x = -1, the tangent is `tL`
      # at x = 1, the tangent is `tR` and at x = 0, the tangent is `tcomb`.
      # The fact that the functions Hi2 and Hi3 are zero means that the tangent
      # does not change at x = -1, x = 0 and x = 1 in the direction of x.
      return tL * H00(x) + tcomb * (H10(x) + H01(x)) + tR * H11(x)

    A = quarter_disc(unitdisc)[0] * np.array([aL, bT, 0])
    B = quarter_disc(unitdisc)[1] * np.array([aL, bB, 0])
    C = quarter_disc(unitdisc)[2] * np.array([aR, bB, 0])
    D = quarter_disc(unitdisc)[3] * np.array([aR, bT, 0])

    # rotate and translate
    A = (RL * A[..., _, :]).sum(-1) + xC
    B = (RL * B[..., _, :]).sum(-1) + xC
    C = (RR * C[..., _, :]).sum(-1) + xC
    D = (RR * D[..., _, :]).sum(-1) + xC

    roof = join_quarters(A, B, C, D).to_ndim(1)

    wings.append(
      hermite_nonconstant_tangent(unitdisc, disc, roof, t0f, t1f, 3).arr)

  return NDSplineArray(wings)
