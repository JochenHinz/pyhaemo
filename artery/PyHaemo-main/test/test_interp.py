from splico.geo import ellipse
from splico.geo.interp import linear_interpolation, cubic_hermite_interpolation
from splico.mesh import rectilinear, mesh_union
from splico.util import np
from splico.nutils import NutilsInterface
from splico.geo.disc import PATCHES
from splico.spl import NDSplineArray, NDSpline

import unittest
from itertools import starmap

from nutils import function


class TestInterpolation(unittest.TestCase):

  def test_interpolate(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    spl0 = disc * np.array([.5, .5, 0])
    spl1 = disc * np.array([2, 2, 0]) + np.array([0, 0, 3])

    sample_mesh = rectilinear([np.linspace(0, 1, 8),
                               np.linspace(0, 1, 8),
                               np.linspace(0, 1, 101)])

    with self.subTest('Linear interpolation.'):
      test = linear_interpolation(spl0, spl1)
      mesh = mesh_union(*test.sample_mesh(sample_mesh), boundary=True)
      mesh.plot()

    with self.subTest('Cubic Hermite interpolation.'):
      e3 = np.array([0, 0, 1])
      test = cubic_hermite_interpolation(spl0, spl1, 3 * e3, 3 * e3)
      mesh = mesh_union(*test.sample_mesh(sample_mesh), boundary=True)
      mesh.plot()

    with self.subTest('Cubic Hermite interpolation with nonconstant tangent.'):
      intf = NutilsInterface(disc, PATCHES)

      t0 = intf.break_apart(
        intf.domain.project( function.normalized(intf.geom) + 3 * e3,
                             geometry=intf.geom,
                             onto=intf.basis.vector(3),
                             degree=intf.degree ).reshape(-1, 3), split=True)
      t1 = intf.break_apart(
        intf.domain.project( -function.normalized(intf.geom) + 3 * e3,
                             geometry=intf.geom,
                             onto=intf.basis.vector(3),
                             degree=intf.degree ).reshape(-1, 3), split=True)

      kv = disc.knotvector.ravel()
      t0 = NDSplineArray(list(starmap(NDSpline, zip(kv, t0))))
      t1 = NDSplineArray(list(starmap(NDSpline, zip(kv, t1))))

      test = cubic_hermite_interpolation(spl0, spl1, t0, t1)

      mesh = mesh_union(*test.sample_mesh(sample_mesh), boundary=True)
      mesh.plot()


if __name__ == '__main__':
  unittest.main()
