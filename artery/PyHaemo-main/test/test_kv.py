from splico.spl import UnivariateKnotVector, TensorKnotVector, NDSpline

import unittest

import numpy as np

_ = np.newaxis


class TestUnivariateKnotVector(unittest.TestCase):

  def test_flip(self):
    knotvalues = np.array([0.0, 0.1, 0.25, 0.5, 0.95, 1.0]) + 2
    knotvector = UnivariateKnotVector(knotvalues)
    self.assertTrue( np.allclose(np.array([0.0, 0.05, 0.5, 0.75, 0.9, 1.0]) + 2, knotvector.flip().knots) )

  def test_multiply(self):
    knotvalues = np.linspace(0, 2, 21)
    knotvector = UnivariateKnotVector(knotvalues)
    tknotvector = knotvector * knotvector
    self.assertTrue( len(tknotvector) == 2 )
    self.assertTrue( isinstance(tknotvector, TensorKnotVector) )
    self.assertTrue( isinstance(knotvector * tknotvector, TensorKnotVector) )
    self.assertTrue( isinstance(tknotvector * knotvector, TensorKnotVector) )

  def test_functionality(self):
    knotvector = UnivariateKnotVector(np.linspace(-1, 1, 21))
    with self.subTest('Test refine'):
      self.assertTrue(np.allclose(knotvector.refine().knots,
                                  np.linspace(-1, 1, 41)))
    with self.subTest('Test ref_by'):
      self.assertTrue(np.allclose(knotvector.ref_by(0).knots,
                                  np.insert(knotvector.knots, 1, -0.95)))
      self.assertTrue(np.allclose(knotvector.ref_by([0, 1, 2]).knots,
                                  np.insert(knotvector.knots, [1, 2, 3], [-0.95, -.85, -.75])))
      # refining beyond the last element should require indexing into
      # knotvector.knots[21] which should raise an IndexError
      self.assertRaises(IndexError, lambda: knotvector.ref_by(20))

    with self.subTest('Test add_knots'):
      knots = knotvector.knots
      # adding beyond the knotvector's bounds is prohibited and should raise a ValueError
      self.assertRaises(ValueError, lambda: knotvector.add_knots(-3))
      self.assertTrue( np.allclose(knotvector.refine().knots,
                                   np.unique([*knots, *(knots[1:] + knots[:-1])/2])))

    with self.subTest('Test raise_multiplicities'):
      self.assertTrue( np.allclose(
          knotvector.raise_multiplicities(np.arange(1, len(knots) - 1), 3).km,
          4 * np.ones(len(knots), dtype=int) ))

    with self.subTest('Test arithmetic'):
      kv = UnivariateKnotVector(np.linspace(0, 1, 11))
      self.assertTrue( (knotvector | kv) == knotvector.raise_multiplicities(10, 3) )

      self.assertFalse( kv < knotvector )
      self.assertFalse( kv > knotvector )
      self.assertFalse( kv >= knotvector )
      self.assertFalse( kv <= knotvector )

      # if we repeat kv.knots[0] in `knotvector`, kv should be a strict subset of knotvector
      self.assertTrue( kv < knotvector.raise_multiplicities(10, 3) )

      self.assertFalse( knotvector > knotvector )
      self.assertFalse( knotvector < knotvector )
      self.assertTrue( knotvector >= knotvector )
      self.assertTrue( knotvector <= knotvector )

      kv = UnivariateKnotVector(np.linspace(0, 1, 21))
      self.assertTrue( np.allclose((knotvector & kv).knots, np.linspace(0, 1, 11)) )

    def test_greville_collocation_matrix(self):
      kv0 = UnivariateKnotVector(np.linspace(0, 1, 11))
      kv1 = UnivariateKnotVector(np.linspace(0, 1, 11))

      mat = kv0.greville_collocation_matrix(kv1)

      # the greville collocation matrix is cached across knot vector instances
      self.assertTrue(mat is kv1.greville_collocation_matrix)


class TestTensorKnotvector(unittest.TestCase):

  def test_vectorization(self):
    kv0 = UnivariateKnotVector(np.linspace(-1, 1, 21))

    with self.subTest('Test arithmetic'):
      kv1 = UnivariateKnotVector(np.linspace(0, 1, 11))
      tkv0 = kv0 * kv0
      tkv1 = kv1 ** 2

      self.assertTrue( (tkv0 | tkv1) == tkv0.raise_multiplicities(..., [10]*2, [3]*2) )

      self.assertFalse( tkv1 < tkv0 )
      self.assertFalse( tkv1 > tkv0 )
      self.assertFalse( tkv0 >= tkv1 )
      self.assertFalse( tkv0 <= tkv1 )

  def test_fit(self):
    kvs = [UnivariateKnotVector(np.linspace(0, 1, i)) for i in [5, 6, 7] ]
    # kvs = [kv.raise_multiplicities(3, 2).to_tensor() for kv in kvs]
    kvs = [kv.to_tensor() for kv in kvs]

    for i in range(1, 4):
      kv = np.prod(kvs[:i])
      spl = NDSpline(kv, np.random.randn(kv.ndofs))

      kvp = spl.knotvector.refine(...)
      # kvp = spl.knotvector.degree_elevate(..., [1]*len(kv))

      absc = kvp.gauss_abscissae
      xi = [np.linspace(0, 1, 11)] * len(kv)

      data = spl(*map(np.ravel, np.meshgrid(*absc, indexing='ij')))

      splp = kvp.fit(absc, data, lam0=0)

      self.assertTrue(
          np.allclose(splp(*xi), spl(*xi))
      )


if __name__ == '__main__':
  unittest.main()
