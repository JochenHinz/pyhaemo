from splico.util import np, _
from splico.geo import interp, ellipse
from splico.spl import NDSpline, NDSplineArray
from splico.mesh import rectilinear, mesh_union

from numpy.typing import ArrayLike


Spline = NDSpline | NDSplineArray


rotmat = lambda phi: np.array([
                                [np.cos(phi), -np.sin(phi), 0],
                                [np.sin(phi), np.cos(phi), 0],
                                [0, 0, 1]
                              ])


def main(spl0: Spline, spl1: Spline, t0: ArrayLike, t1=None) -> None:
  """
    Perform a linear interpolation and a cubic hermite interpolation between
    `spl0` and `spl1`.

    Parameters
    ----------
    spl0 : :class:`splico.spl.NDSpline`
        The first input spline.
    spl1 : :class:`splico.spl.NDSpline`
        The second input spline.
    t0 : :class:`np.ndarray`
        The desired tangent at xn+1 == 0.
    t1 : :class:`np.ndarray`
        The desired tangent at xn+1 == 1. If not passed, defaults to `t0`.
  """
  spl0, spl1 = map(NDSplineArray, (spl0, spl1))
  assert spl0.nvars == spl1.nvars == 2

  if t1 is None:
    t1 = t0
  t0, t1 = map(np.asarray, (t0, t1))

  sample_mesh = rectilinear([np.linspace(0, 1, 21)] * 3)

  linear_interp = interp.linear_interpolation(spl0, spl1, kvz=1)
  mesh_union(*linear_interp.sample_mesh(sample_mesh).ravel(), boundary=True).plot()

  hermite_interp = interp.cubic_hermite_interpolation(spl0, spl1, t0, t1)
  mesh_union(*hermite_interp.sample_mesh(sample_mesh).ravel(), boundary=True).plot()


if __name__ == '__main__':
  spl0 = ellipse(1, 1, 4)
  spl1 = (rotmat(1) * ellipse(1, 1, 4)[:, _]).sum(-1) + np.array([[0, 0, 2]])

  main(spl0, spl1, [0, 1, 1])
