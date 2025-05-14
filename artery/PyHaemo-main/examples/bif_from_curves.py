from splico.util import np, _
from splico.spl import NDSpline, NDSplineArray
from splico.geo import bif_from_curves, repeated_knot_disc
from splico.geo.interp import cubic_hermite_interpolation


def rotmat(theta):
  return np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])


def main(nbranches: int = 4):
  """
  Simple example of how to create a bifurcation from a set of curves.
  """
  assert nbranches in (3, 4)

  # create unit disc with a repeated knot that halves it
  unitdisc = repeated_knot_disc(4)

  p0 = NDSpline([], np.array([0, -1, 0])[_])
  t0 = NDSpline([], np.array([0, 2, 0])[_])
  rm = rotmat(2 * np.pi / nbranches)

  p1, t1 = (rm * p0[_]).sum(1), -(rm * t0[_]).sum(1)

  # create the cubic hermite spline for the input curve
  c0 = NDSplineArray(cubic_hermite_interpolation(p0, p1, t0, t1)) + \
                     np.array([.4, -.4, 0])

  # the others follow from a rotation
  curves = [(rotmat(i * 2 * np.pi / nbranches) * c0[_]).sum(1)
                                                      for i in range(nbranches)]

  ax = np.array([0, 0, 1])
  xC = np.array([0, 0, 0])
  bB, bT = .5, .5

  bif = bif_from_curves(curves, ax, xC, bB, bT, unitdisc)

  bif.qplot()

  print(f"The mesh is valid ? {bif.quick_sample((5, 5, 11), take_union=True).is_valid()}")


if __name__ == '__main__':
  main()
