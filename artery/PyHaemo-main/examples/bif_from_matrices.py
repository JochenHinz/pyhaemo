from splico.geo.interp import cubic_hermite_interpolation
from splico.geo import compute_notwistframe
from splico.types import Int
from splico.mesh import mesh_union
from splico.spl import UnivariateKnotVector, TensorKnotVector, NDSplineArray, NDSpline
from splico.util import np, _, clparam, normalize
from splico.geo.bif import repeated_knot_disc, bif_from_matrices

from typing import Sequence
from functools import partial

from numpy.typing import NDArray


def make_vessel(disc: NDSplineArray,
                centerline_curve: NDSplineArray,
                rRs: NDSplineArray) -> NDSplineArray:
  """
  Create a vessel from centerline, cross section and rotation / stretch
  splines.
  """

  kv = centerline_curve.knotvector
  assert kv.ndim == 1

  return (disc[:, _] * rRs[_]).sum(-1) + (disc.unity * centerline_curve)


def main(nelems_centerline: Int,
         nelems_cross_section: Int,
         allpoints: Sequence[NDArray],
         radii: Sequence[NDArray]):

  # create a cross section disc that has a repeated knot at .5
  # to properly resolve the butterfly structure
  disc = repeated_knot_disc(nelems_cross_section)

  ncurves = len(allpoints)
  assert len(radii) == ncurves and ncurves >= 3

  # all chord length parameterizations
  Xis = tuple(map(clparam, allpoints))

  # knotvector in `t` direction
  kv: TensorKnotVector = \
    UnivariateKnotVector(np.linspace(0, 1, nelems_centerline + 1)).to_tensor()

  # the centerline curves for each vessel that is attached to the bif
  vesselcurves = [ kv.fit([xi], points) for xi, points in zip(Xis, allpoints) ]

  # From the bifurcation inward oriented
  tangents = normalize(np.stack([curve(np.array([1]), dx=1).ravel() for curve in vesselcurves],
                                axis=0))

  # We put the disc's local y-axis onto the z-axis while z becomes t.
  # The local x-axis then becomes the cross product of y (z) and z (t).
  r"""
        ^ x (n)
        |
        |-----> z (t)
        |
        |
        o y (z)
  """

  # compute the initial frame for the bifurcation vessels
  bifframe = compute_notwistframe(tangents,
                                  n0=normalize(np.cross([0, 0, 1], tangents[0])))

  vessels, matrices = [], []
  for xi, points, curve, rs, R in zip(Xis, allpoints, vesselcurves, radii, bifframe):
    n0 = R[:, 0]
    tangents = curve(xi, dx=1)
    Rs = compute_notwistframe(tangents[::-1], n0=n0)[::-1]

    # TODO add option to fit first data point exactly
    rRs = kv.fit([xi], rs[:, _, _] * Rs)
    matrices.append(rRs.controlpoints[-1])

    vessels.append(make_vessel(disc, curve, rRs))

  endpoints = np.stack([curve(np.array([1])).ravel() for curve in vesselcurves],
                       axis=0)

  # Take the center as the center of all endpoints
  xC = sum(endpoints) / len(endpoints)

  roll = partial(np.roll, endpoints, axis=0)

  normals = []
  for x0, x1, x2 in zip(*(roll(-i) for i in range(3))):
    normals.append( normalize(np.cross(x1 - x0, x2 - x0)) )

  ax = normalize(np.asarray(sum(normals) / len(normals)))
  bT, bB = (sum(r[0] for r in radii) / len(radii),)*2

  bif = bif_from_matrices(matrices, endpoints, ax, xC, bB, bT, disc)

  npoints = 5
  mesh_union(*bif.quick_sample((npoints, npoints, 5)).ravel(),
             *(ves.quick_sample((npoints, npoints, 101), take_union=True)
                                for ves in vessels)).interfaces.boundary.plot()

  mesh_union(*bif.quick_sample((npoints, npoints, 5)).ravel(),
             *(ves.quick_sample((npoints, npoints, 101), take_union=True)
                                for ves in vessels)).plot()


if __name__ == '__main__':
  kv = UnivariateKnotVector(np.linspace(0, 1, 12)).to_tensor()
  eps = .1
  npoints = 1001
  xi = np.linspace(0, 1, npoints)

  points0 = cubic_hermite_interpolation(NDSpline([], np.array([0, -eps, 0])[_]),
                                        NDSpline([], np.array([-1, -1, 0])[_]),
                                        np.array([0, -1, 0]),
                                        np.array([-1, 0, 0]))(xi)[::-1]

  angle = 2 / 3 * np.pi
  rotmat = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

  points1 = (rotmat @ points0.T).T
  points2 = (rotmat @ points1.T).T

  main(30, 4, [points0, points1, points2], [.5 * eps * np.ones(npoints)]*3)
