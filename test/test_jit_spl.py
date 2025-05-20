from pyhaemo.spl import UnivariateKnotVector, TensorKnotVector
from pyhaemo.spl import NDSpline
from pyhaemo.spl._jit_spl import position_in_knotvector, position_in_knotvector_trunc

import unittest

import numpy as np


class TestJIT(unittest.TestCase):
  """
  Test the JIT compiled helper routines from the `pyhaemo.spl._jit_spl` module.
  """
  def test_position_in_knotvector(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11), 3)

    with self.subTest('Test position in Knotvector normal'):
      # position equal to the first knot should give the degree
      self.assertTrue(position_in_knotvector(kv.repeated_knots, kv.degree, np.array([0.0]))[0] == kv.degree)

      # position slightly to the left of the first knot should still give -1
      self.assertTrue(position_in_knotvector(kv.repeated_knots, kv.degree, np.array([-0.0001]))[0] == -1)

      # position slightly to the right of the last knot should give -1 (not found)
      self.assertTrue(position_in_knotvector(kv.repeated_knots, kv.degree, kv.repeated_knots[-1:] + 0.00001)[0] == -1)

      # position equal to the last knot should give ndofs - 1
      self.assertTrue(position_in_knotvector(kv.repeated_knots, kv.degree, kv.repeated_knots[-1:])[0] == kv.dim - 1)

    with self.subTest('Test position in Knotvector truncated'):
      # position slightly to the left of the first knot should give -1
      self.assertTrue(position_in_knotvector_trunc(kv.repeated_knots,
                                                   kv.degree,
                                                   np.array([-0.0001]))[0] == kv.degree)

      # position slightly to the right of the last knot should give ndofs - 1
      self.assertTrue(position_in_knotvector_trunc(kv.repeated_knots,
                                                   kv.degree,
                                                   kv.repeated_knots[-1:] + 0.00001)[0] == kv.dim - 1)

  def test_against_scipy_1D(self):
    from scipy.interpolate import splev

    # no repeated knots
    tkv = TensorKnotVector([UnivariateKnotVector(np.linspace(0, 1, 21), 3)])

    controlpoints = np.random.randn(tkv.ndofs)
    spl = NDSpline(tkv, controlpoints)

    xi = np.linspace(0, 1, 1001)
    self.assertTrue(np.allclose(spl(xi), splev(xi, (tkv.repeated_knots, controlpoints, 3))))
    self.assertTrue(np.allclose(spl(xi, dx=1), splev(xi, (tkv.repeated_knots, controlpoints, 3), der=1)))
    self.assertTrue(np.allclose(spl(xi, dx=2), splev(xi, (tkv.repeated_knots, controlpoints, 3), der=2)))

    # repeated knots
    tkv = TensorKnotVector([UnivariateKnotVector(np.linspace(0, 1, 21), 3)]). \
                            raise_multiplicities(0, [5], [3])

    controlpoints = np.random.randn(tkv.ndofs)
    spl = NDSpline(tkv, controlpoints)

    xi = np.linspace(0, 1, 1001)
    self.assertTrue(np.allclose(spl(xi), splev(xi, (tkv.repeated_knots, controlpoints, 3))))

  def test_against_scipy_2D(self):
    from scipy.interpolate import bisplev

    # no repeated knots
    tkv = TensorKnotVector([UnivariateKnotVector(np.linspace(0, 1, i), 3) for i in (11, 21)])

    controlpoints = np.random.randn(tkv.ndofs)
    spl = NDSpline(tkv, controlpoints)

    tck = (*tkv.repeated_knots, spl.controlpoints.ravel(), *tkv.degree)

    xi = [np.linspace(0, 1, 101)] * 2
    Xi = list(map(np.ravel, np.meshgrid(*xi, indexing='ij')))

    # test tcall
    self.assertTrue(np.allclose(spl(*xi, tensor=True), bisplev(*xi, tck, dx=0, dy=0).ravel()))
    self.assertTrue(np.allclose(spl(*xi, tensor=True, dx=(1, 1)), bisplev(*xi, tck, dx=1, dy=1).ravel()))
    self.assertTrue(np.allclose(spl(*xi, tensor=True, dx=(2, 2)), bisplev(*xi, tck, dx=2, dy=2).ravel()))

    # test normal call
    self.assertTrue(np.allclose(spl(*Xi), bisplev(*xi, tck, dx=0, dy=0).ravel()))
    self.assertTrue(np.allclose(spl(*Xi, dx=(1, 1)), bisplev(*xi, tck, dx=1, dy=1).ravel()))
    self.assertTrue(np.allclose(spl(*Xi, dx=(2, 2)), bisplev(*xi, tck, dx=2, dy=2).ravel()))

    # repeated knots
    tkv = TensorKnotVector([UnivariateKnotVector(np.linspace(0, 1, 21), 3)]*2)
    tkv = tkv.raise_multiplicities([0, 1], [4, 4], [3, 3])

    controlpoints = np.random.randn(tkv.ndofs)
    spl = NDSpline(tkv, controlpoints)

    tck = (*tkv.repeated_knots, spl.controlpoints.ravel(), *tkv.degree)

    xi = [np.linspace(0, 1, 9)] * 2
    Xi = list(map(np.ravel, np.meshgrid(*xi, indexing='ij')))

    X = spl(*map(np.ravel, np.meshgrid(*xi, indexing='ij')))
    X_ = bisplev(*xi, tck, dx=0, dy=0).ravel()

    self.assertTrue(np.allclose(X, X_))

  def test_extrapolation(self):
    # this module tests if the behavior of the JIT compiled functions
    # is as expected when we evaluate the spline outside the knot vector range

    tkv = TensorKnotVector([UnivariateKnotVector(np.linspace(-1, 1, i), 3) for i in (11, 21)])

    xi = np.linspace(-2, -1.00001, 1001)
    eta = np.linspace(-1, 1, 1001)

    controlpoints = np.random.randn(tkv.ndofs)
    spl = NDSpline(tkv, controlpoints)

    for dx in (0, 1, 2):
      self.assertTrue( np.allclose(spl(xi, eta, tensor=True, dx=dx), 0) )

    self.assertRaises(Exception, lambda: spl(xi, eta, oob=1))


if __name__ == '__main__':
  unittest.main()
