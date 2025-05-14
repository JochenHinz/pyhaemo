from splico.util import np, _
from splico.types import Int, Float, Numeric
from splico.util import normalize
from splico.spl import NDSplineArray, NDSpline, UnivariateKnotVector, \
                       TensorKnotVector
from splico.geo.disc import PATCHES, ellipse
from splico.geo.interp import cubic_hermite_interpolation
from splico.geo.aux import spline_or_array
from splico.nutils import NutilsInterface

from functools import lru_cache
from collections import namedtuple
from typing import Tuple, Callable, Sequence, Optional

from nutils import function
from numpy.typing import ArrayLike


J = function.J


KnotVector = UnivariateKnotVector | TensorKnotVector | Int
Spline = NDSpline | NDSplineArray


@lru_cache
def quarter_disc(spl: NDSplineArray) -> Tuple[NDSplineArray, ...]:
  r"""
    1 - - - - - 7
    | \   D   / |
    |  3 - - 5  |
    |A |  C  | E|
    |  |     |  |
    |  2-----4  |
    | /   B   \ |
    0 - - - - - 6

    PATCHES = (2, 0, 3, 1),
              (2, 4, 0, 6),
              (2, 3, 4, 5),
              (3, 1, 5, 7),
              (4, 5, 6, 7)

    becomes

        o - - o - - o
        | \ C | C / |
     0  |  o -o- o  |  3
        |A |B | B| A|
        o--o--o--o--o
        |A |B | B| A|
     1  |  o--o--o  |  2
        | / C | C \ |
        o - - o - - o


    The splitting is chosen in a way that the patch indices that require
    coupling upon putting the patches together are the same as in the
    neighbouring patches.


  Parameters
  ----------
  spl : :class:`NDSplineArray`
      The spline that is to be split. Needs to have shape ``(5, 3)``.

  Returns
  -------
  :class:`NDSplineArray`, :class:`NDSplineArray`, \
  :class:`NDSplineArray`, :class:`NDSplineArray`
      The four spline arrays that result from the split.
  """

  # TODO: rewrite this as a topology-independent operation

  spl = spl.expand(spl._elemdim - 1)
  assert spl.shape == (5, 3) and spl._elemshape == (3,)
  assert all(all(.5 in knots for knots in kv.knots) for kv in spl.knotvector.ravel()), \
    "Each knotvector must be splittable at .5, i.e., .5 must be a knot."

  # find the index .5
  pos = lambda kv: np.searchsorted(kv, .5)

  spl0, spl1, spl2, spl3 = [[_ for _ in range(3)] for i in range(4)]

  # patch A direction 0
  spl1[0], spl0[0] = (y := spl.arr[0]).split(0, pos(y.knotvector[0].knots))

  # patch D direction 0
  spl0[2], spl3[2] = (y := spl.arr[3]).split(0, pos(y.knotvector[0].knots))

  # patch B direction 1
  spl1[2], spl2[2] = (y := spl.arr[1]).split(1, pos(y.knotvector[1].knots))

  # patch E direction 1
  spl2[0], spl3[0] = (y := spl.arr[4]).split(1, pos(y.knotvector[1].knots))

  # patch C direction 0 and 1
  y0, y1 = (y := spl.arr[2]).split(0, pos(y.knotvector[0].knots))

  spl1[1], spl0[1] = y0.split(1, pos(y0.knotvector[1].knots))
  spl2[1], spl3[1] = y1.split(1, pos(y1.knotvector[1].knots))

  return tuple(map(NDSplineArray, (spl0, spl1, spl2, spl3)))


@lru_cache
def join_quarters(*allspls: NDSplineArray) -> NDSplineArray:

  """
  Inverse operation of :func:`quarter_disc`. Puts the four quarters together
  to form a single disc spline. The input splines need to be ordered in the
  same way as returned by :func:`quarter_disc`.
  """

  spl0, spl1, spl2, spl3 = map(lambda x: x.expand(len(x._elemshape) - 1), allspls)
  assert all(spl.shape == (3, 3) and spl._elemshape == (3,) for spl in allspls)

  spls = [_ for i in range(5)]

  # patch A
  spls[0] = spl1.arr[0].join(spl0.arr[0], 0)

  # patch B
  spls[1] = spl1.arr[2].join(spl2.arr[2], 1)

  # patch D
  spls[3] = spl0.arr[2].join(spl3.arr[2], 0)

  # patch E
  spls[4] = spl2.arr[0].join(spl3.arr[0], 1)

  # patch C
  spls[2] = (spl1.arr[1].join(spl0.arr[1], 1)). \
                         join(spl2.arr[1].join(spl3.arr[1], 1), 0)

  return NDSplineArray(np.asarray(spls, dtype=object)).contract_all()


@spline_or_array
def hermite_nonconstant_tangent(unitdisc: NDSplineArray,
                                spl0: NDSplineArray,
                                spl1: NDSplineArray,
                                t0: Callable[[function.Array], function.Array],
                                t1: Callable[[function.Array], function.Array],
                                zdegree: Int | UnivariateKnotVector = 3) -> NDSplineArray:
  """
  Given two splines ``spl0`` and ``spl1`` that are defined on the same knotvectors
  and a pair of tangent functions ``t0`` and ``t1`` that are defined on the same
  ``cross.controlmap``, this function computes a Hermite spline that interpolates
  the splines and the tangent functions. Here, the nonconstant tangent functions
  are projected onto the basis associated with ``cross`` to ensure that the
  result is a spline in the space of the basis functions.
  Here, ``kvz`` corresponds to the knotvector in the interpolation direction
  and needs to be at least cubic.

  Parameters
  ----------
  spl0 : :class:`NDSplineArray`
      The first spline that is to be interpolated. Needs to be defined on the
      same knotvectors as ``cross.knotvectors`` and have shape ``(5, 3)``.
  spl1 : :class:`NDSplineArray`
      The second spline that is to be interpolated. Needs to be defined on the
      same knotvectors as ``cross.knotvectors`` and have shape ``(5, 3)``.
  t0 : Callable
      The tangent function associated with ``spl0``.
      Needs to be defined over the controlmap of ``cross``.
  t1 : Callable
      The tangent function associated with ``spl1``.
      Needs to be defined over the controlmap of ``cross``.
  zdegree : :class:`Int`
      The knotvector in the interpolation direction. Needs to be at least cubic.
      If integer, a knotvector without interior knots and order `kvz` is created.

  Returns
  -------
  :class:`NDSplineArray`
      The Hermite spline that interpolates the splines and the tangent functions,
      expressed in the desired basis.
  """

  assert (unitdisc.knotvector == spl0.contract_all().knotvector).all() and \
         (unitdisc.knotvector == spl1.contract_all().knotvector).all()

  intf = NutilsInterface(unitdisc.contract_all(), PATCHES)

  t0_ = t0(intf.geom)
  t1_ = t1(intf.geom)

  assert t0_.shape == t1_.shape == (3,)

  rhss = \
    map(lambda x: x.export('dense'),
        intf.domain.integrate([intf.basis[:, _] * t[_] * J(intf.geom)
                               for t in (t0_, t1_)], degree=intf.degree))

  t0x, t1x = [ intf.break_apart(intf.CM.solve(rhs), split=True) for rhs in rhss]

  t0 = NDSplineArray([NDSpline(kv, t)
                              for kv, t in zip(spl0.knotvector.ravel(), t0x)])
  t1 = NDSplineArray([NDSpline(kv, t)
                               for kv, t in zip(spl1.knotvector.ravel(), t1x)])

  return cubic_hermite_interpolation(spl0, spl1, t0, t1, zdegree)


@lru_cache
def repeated_knot_disc(nelems: Int,
                       degree: Int = 3,
                       reparam: bool = True) -> NDSplineArray:
  """
  Has a repeated knot at x = 0.5 for kv0 and kv1.
  """
  assert nelems % 2 == 0
  kv0 = UnivariateKnotVector(np.linspace(0, 1, nelems+1), degree=degree)
  kv0 = kv0.raise_multiplicities(kv0.knotvalues.index(.5), degree-1)
  kv1 = kv0

  kv2 = UnivariateKnotVector(np.linspace(0, 1, nelems+1), degree=degree)

  return ellipse(1, 1, kv0, kv1, kv2, reparam=reparam)


WingData = namedtuple('WingData', ['ax', 'xC', 'xL',
                                   'xR', 'bB', 'bT',
                                   'xIn', 'rIn', 'tInL',
                                   'tInR', 'tL', 'tR'   ])


def rotate_wing_data(data: WingData, mat: np.ndarray):
  ax, xC, *tail = data
  return WingData(ax, xC,
                  *((mat @ (arr - xC)) + xC if isinstance(arr, np.ndarray)
                                                      else arr for arr in tail))


def hermite_pols_interval(a, b):
  """
  The canonical cubic Hermite polynomials over the interval (a, b)
  """
  assert a < b

  ba = b - a
  H0 = lambda x: 1 - 3 * ((x - a) / ba)**2 + 2 * ((x - a) / ba)**3
  H1 = lambda x: 3 * ((x - a) / ba)**2 - 2 * ((x - a) / ba)**3
  H2 = lambda x: (x - a) / ba * (1 - 2 * ((x - a) / ba) + ((x - a) / ba)**2)
  H3 = lambda x: (x - a) / ba * (-(x - a) / ba + ((x - a) / ba)**2)

  return lambda x: function.piecewise(x, [a, b], 0, H0(x), 0), \
         lambda x: function.piecewise(x, [a, b], 0, H1(x), 0), \
         lambda x: function.piecewise(x, [a, b], 0, H2(x), 0), \
         lambda x: function.piecewise(x, [a, b], 0, H3(x), 0)


hat_functions2 = lambda x: [ .5 * (1 - x), .5 * (1 + x) ]


def wing(wingdata: WingData, unitdisc: NDSplineArray | Int = 4):
  """
                  xC + ax * bT
                      |
                      |
            A         |         D
                      |
                      |
      xL -- aL ----  xC ---- aR ----- xR
                      |
                      |
            B         |         C
                      |
                      |
                  xC - ax * bB

  XXX: more detailed docstring
  """

  # TODO: this function should be rewritten to use a rotation matrix
  # instead of vectors and float values

  if isinstance(unitdisc, Int):
    assert unitdisc % 2 == 0
    unitdisc = repeated_knot_disc(unitdisc, reparam=True)

  ax, xC, xL, xR, bB, bT, xIn, rIn, tInL, tInR, tL, tR = wingdata

  # Normalize to get the direction only
  ax = normalize(ax)
  tin = normalize(xC - xIn)

  # roof left and right arm radii
  aL = np.linalg.norm(xC - xL)
  aR = np.linalg.norm(xC - xR)

  # left arm roof rotation matrix
  xin = normalize(np.cross(xC - xL, ax))
  RL = np.stack([(xC - xL) / aL, ax, xin], axis=1)

  # right arm roof rotation matrix
  xin = normalize(np.cross(xR - xC, ax))
  RR = np.stack([(xR - xC) / aR, ax, xin], axis=1)

  # input disc rotation matrix
  xnew = normalize(np.cross(ax, tin))
  Rin = np.stack([xnew, ax, tin], axis=1)

  def t0f(geom):
    # Inlet tangent just interpolates the left tangent to the right.
    f0, f1 = hat_functions2(geom[0])
    return tInL * f0 + tInR * f1

  def t1f(geom):
    x, *ignore = geom

    # we split the disc into two parts, separated by x = 0.
    H00, H10, *ignore = hermite_pols_interval(-1, 0)
    H01, H11, *ignore = hermite_pols_interval(0, 1)

    # The tangent is the center point minus the center of the inlet.
    tcomb = xC - xIn

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

  roof = join_quarters(A, B, C, D)

  inlet = (Rin * (unitdisc * np.array([rIn, rIn, 0]))
                 .prolong_to_array(roof)[..., _, :]).sum(-1) + xIn

  result = hermite_nonconstant_tangent(unitdisc, inlet, roof, t0f, t1f, 3)

  return result


def bif_from_curves(curves: NDSplineArray | Sequence[NDSplineArray],
                    ax: ArrayLike,
                    xC: ArrayLike,
                    bB: Float | Int,
                    bT: Float | Int,
                    unitdisc: NDSplineArray | Int = 4,
                    cevalpoints: Optional[ArrayLike] = None) -> NDSplineArray:
  r"""
         ------                 ---<--
                \  cruves[1]  /
                  \         /
         -->-       \ _ _ /       -->-
              \                 /
                \     xC      x   cevalpoints[0]
                  \         /
       curves[2]    \     /    cruves[0]
                     |   |
                     |   |

  Create a bifurcation from a set of input curves, see above drawing.
  The input curves represent the side curves of the bi- (or n-) furcation.

  Parameters
  ----------
  curves: :class:`NDSpline` or :class:`NDSplineArray` or :class:`Sequence`
    The univariate input curves all of shape (3,). There must be at least 3.
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
  cevalpoints: :class:`Sequence[Float]`, optional.
    The evaluation points of the input curves to create the butterfly structure.
    If not passed, defaults to all (a + b) / 2, where a and b are the
    knotvector's endpoints.
  """

  curves = list(map(NDSplineArray, curves))
  assert len(curves) >= 3
  assert all(curve.shape == (3,)
             and curve.knotvector == curves[0].knotvector for curve in curves)

  if cevalpoints is None:
    cevalpoints = [(c.knotvector.ravel()[0].knots[0][0] +
                    c.knotvector.ravel()[0].knots[0][-1]) / 2 for c in curves]

  cevalpoints = np.asarray(cevalpoints, dtype=float)

  if isinstance(unitdisc, Int):
    unitdisc = repeated_knot_disc(unitdisc, reparam=True)

  assert len(cevalpoints) == len(curves), \
    "Number of evaluation points must match the number of curves."

  spls = []
  for splR, splL, evalR, evalL in zip(np.roll(curves, 1),
                                      curves,
                                      np.roll(cevalpoints, 1),
                                      np.asarray(cevalpoints)):

    x1, x0 = splR( np.array([1]) ).ravel(), splL( np.array([0]) ).ravel()
    xR, xL = splR( np.array([evalR]) ).ravel(), splL( np.array([evalL]) ).ravel()

    x01 = (x0 + x1) / 2
    r = np.linalg.norm(x0 - x1) / 2

    tInR, tR = splR(np.array([1]), dx=1).ravel() * (evalR - 1), \
               splR(np.array([evalR]), dx=1).ravel() * (evalR - 1)

    tInL, tL = splL(np.array([0]), dx=1).ravel() * evalL, \
               splL(np.array([evalL]), dx=1).ravel() * evalL

    # this assumes that tInR and tInL are parallel
    # bif z axis, center point, left and right arms of the roof,
    # stretch in the negative and positive z direction, center input disc,
    # radius of input disc, left and right tangents of input disc,
    # and left and right tangents of the roof
    wingdata = WingData(ax, xC, xL, xR, bB, bT, x01, r, tInL, tInR, tL, tR)
    spls.append(wing(wingdata, unitdisc))

  return NDSplineArray(np.stack([spl.arr for spl in spls], axis=0))


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

  unitdisc = unitdisc.to_ndim(1)

  evalL = [.5], [1]
  evalR = [1], [.5]

  xL = unitdisc[0](*evalL).ravel()
  xR = unitdisc[4](*evalR).ravel()
  ax = normalize(np.asarray(ax))

  discs = list(map(lambda x, c: c + (x * unitdisc[:, _]).sum(-1),
                   matrices, centerpoints))

  spls = []
  for mat0, mat1, center0, center1, disc0, disc1 in \
                                        zip(matrices,
                                            np.roll(matrices, -1, axis=0),
                                            centerpoints,
                                            np.roll(centerpoints, -1, axis=0),
                                            discs,
                                            np.roll(discs, -1)):

    t0, t1 = map(normalize, (mat0[:, 2], mat1[:, 2]))
    p0 = disc0[0](*evalL).ravel()
    p1 = disc1[4](*evalR).ravel()

    dist = np.linalg.norm(center1 - center0)

    spls.append( cubic_hermite_interpolation(NDSpline([], p0[_]),
                                             NDSpline([], p1[_]),
                                             dist * t0,
                                             -dist * t1) )

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

    roof = join_quarters(A, B, C, D)

    wings.append(
      hermite_nonconstant_tangent(unitdisc, disc, roof, t0f, t1f, 3).arr)

  return NDSplineArray(wings)
