"""
Regression tests for find_intersection.py.

Run with: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import matplotlib
matplotlib.use('Agg')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import find_intersection as fd


class TestPolynomialCurve(unittest.TestCase):

    def test_cubic_hermite_endpoints(self):
        p0 = np.array([0.0, 0.0, 0.0])
        t0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 2.0, 0.0])
        t1 = np.array([0.0, 0.0, 0.0])

        curve = fd.cubic_hermite_curve(p0, t0, p1, t1)

        np.testing.assert_allclose(curve(0.0)[0], p0, atol=1e-10)
        np.testing.assert_allclose(curve(1.0)[0], p1, atol=1e-10)


class TestOptimizeSmoothnessMethodValidation(unittest.TestCase):
    """
    Regression test for the `assert method in ('smooth, length')` bug: the
    missing comma turned the tuple into a single string, so the assert
    checked substring membership instead of exact match, letting almost any
    short string through.
    """

    def _call_with_method(self, method):
        p0 = np.array([-1.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.5, 1.0, 0.0])
        t0 = [0.5, 1.0, 0.5]
        t1 = [-0.5, 1.0, 0.5]
        t2 = [0.0, 1.0, 1.0]
        v0 = (p0 + p1 + p2) / 3

        return fd.optimize_smoothness(p0, t0, p1, t1, p2, t2, v0=v0, method=method, show=False)

    def test_valid_methods_are_accepted(self):
        self._call_with_method('smooth')
        self._call_with_method('length')

    def test_bogus_substring_is_rejected(self):
        # before the fix, 'smooth, length'[i] substrings such as 's' or
        # 'oo' would silently pass this assert.
        with self.assertRaises(AssertionError):
            self._call_with_method('s')
        with self.assertRaises(AssertionError):
            self._call_with_method('oo')
        with self.assertRaises(AssertionError):
            self._call_with_method('bogus')


if __name__ == '__main__':
    unittest.main()
