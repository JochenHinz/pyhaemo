from pyhaemo.util import np
from pyhaemo.spl import UnivariateKnotVector, NDSpline


from artery_ import Artery

import unittest


def make_random_network(ncurves=10):
  kv = UnivariateKnotVector(np.linspace(0, 1, 11), 3)
  centerlines = [ NDSpline([kv], np.random.rand(kv.dim, 3)) for i in range(10) ]
  radii = [ NDSpline([kv], np.random.rand(kv.dim)) for i in range(10) ]

  return Artery.from_centerlines_radii(centerlines, radii)


class TestArtery(unittest.TestCase):

  def test_init_(self):
    artery = make_random_network()
    nedges = artery.nedges
    self.assertTrue( set(artery.bifurcations) ==
                     (set( (i,) for i in range(1, nedges+1) ) |
                      set( (-i,) for i in range(1, nedges+1) )) )

  def test_bif_merge(self):
    artery = make_random_network()

    (i0, bif0), (i1, bif1) = artery.get_bif(1, return_index=True), \
                             artery.get_bif(4, return_index=True)

    artery = artery.merge_bifs(i0, i1)

    self.assertTrue( set(artery.bifurcations) ==
                      (set( (i,) for i in range(1, artery.nedges+1) if i not in (1, 4) ) |
                        set( (-i,) for i in range(1, artery.nedges+1) )) | {(1, 4)} )

    (i0, bif0), (i1, bif1) = artery.get_bif(7, return_index=True), \
                             artery.get_bif(-4, return_index=True)

    artery = artery.merge_bifs(i0, i1)

    self.assertTrue( set(artery.bifurcations) ==
                      (set( (i,) for i in range(1, artery.nedges+1) if i not in (1, 4, 7) ) |
                       set( (-i,) for i in range(1, artery.nedges+1) if i != 4 )) | {(1, 4), (7, -4)} )

    (i0, bif0), (i1, bif) = artery.get_bif(1, return_index=True), \
                            artery.get_bif(-4, return_index=True)

    artery = artery.merge_bifs(i0, i1)
    self.assertTrue( 4 not in artery.indices )


if __name__ == '__main__':
  unittest.main()
