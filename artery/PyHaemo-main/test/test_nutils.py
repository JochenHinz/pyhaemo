from splico.nutils.interface import NutilsInterface
from splico.topo import Topology
from splico.geo.disc import ellipse, PATCHES
from splico.spl import NDSplineArray
from splico.geo.interp import linear_interpolation
from splico.err import DuplicateOrientationError

import unittest

import numpy as np


def add_zeros(ndarr: NDSplineArray, n=1):
  """ Extend last axis by zeros """
  if n == 0:
    return ndarr
  ndarr = ndarr.to_elemdim(1)
  newarr = []
  for _a in ndarr.arr.ravel():
    cps = _a.controlpoints
    ncps = np.concatenate([cps, np.zeros(cps.shape[:-1] + (n,))], axis=-1)
    newarr.append(_a._edit(controlpoints=ncps))
  return ndarr._edit(arr=np.asarray(newarr, dtype=object).reshape(ndarr.arr.shape))


class TestNutilsInterface(unittest.TestCase):

  def test2D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    intf = NutilsInterface(disc, PATCHES)
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .to_ndim(1)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))

    arr = disc.to_ndim(1).arr.ravel().copy()
    arr[2] = arr[2].refine(...)

    disc = disc._edit(arr=arr).contract_all()
    self.assertRaises(DuplicateOrientationError,
                      lambda: NutilsInterface(disc, PATCHES))

  def test3D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    disc = linear_interpolation(disc, disc + np.array([0, 0, 1]), kvz=1)
    topo = Topology(PATCHES).extrude()

    intf = NutilsInterface(disc, topo)
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))

    arr = disc.to_ndim(1).arr.ravel().copy()
    arr[2] = arr[2].refine(...)

    disc = disc._edit(arr=arr).contract_all()
    self.assertRaises(DuplicateOrientationError,
                      lambda: NutilsInterface(disc, topo))

  def test4D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    disc = add_zeros(linear_interpolation(disc,
                                          disc + np.array([0, 0, 1]),
                                          kvz=1))
    disc = linear_interpolation(disc, disc + np.array([0, 0, 0, 1]), kvz=1)
    topo = Topology(PATCHES).extrude().extrude()
    intf = NutilsInterface(disc, topo)

    print("Solving a 4D problem, this may take a while...")
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))


if __name__ == '__main__':
  unittest.main()
