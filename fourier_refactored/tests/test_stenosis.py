"""
Regression tests for stenosis.py.

Run with: python -m unittest discover -s tests

These double as a regression check for the hardcoded, machine-specific
sys.path.append() that used to make this whole module fail to import
(ModuleNotFoundError: No module named 'fit') on any machine/path layout
other than the original author's.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStenosisImport(unittest.TestCase):

    def test_module_imports(self):
        import stenosis  # noqa: F401 -- import success is the assertion


class TestCreateStenosis(unittest.TestCase):

    def setUp(self):
        from stenosis import create_stenosis
        self.create_stenosis = create_stenosis

    def test_stenosis_reduces_radius_at_target_region(self):
        # create_stenosis expects radii shaped (n_points, n_samples), like
        # TimeData.radii -- it transposes internally to iterate per sample.
        n_points, n_samples = 40, 3
        radii = np.full((n_points, n_samples), 1.0)

        stenotic = self.create_stenosis(radii=radii,
                                        position='middle bottom',
                                        severity='high',
                                        n_points=n_points)

        self.assertEqual(stenotic.shape, (n_samples, n_points))

        start = int(n_points / 2)
        end = int(n_points / 2 + n_points / 3)
        mean = int(start / 2 + end / 2)

        # the stenosis should narrow the vessel at the constricted point
        self.assertLess(stenotic[0, mean], radii[mean, 0])

        # untouched region (well outside [start, end]) must be unchanged
        np.testing.assert_allclose(stenotic[:, :start - 5], radii[:start - 5, :].T)

    def test_invalid_position_raises(self):
        radii = np.full((20, 2), 1.0)
        with self.assertRaises(AssertionError):
            self.create_stenosis(radii=radii, position='bogus', severity='high', n_points=20)

    def test_invalid_severity_raises(self):
        radii = np.full((20, 2), 1.0)
        with self.assertRaises(AssertionError):
            self.create_stenosis(radii=radii, position='middle up', severity='bogus', n_points=20)


if __name__ == '__main__':
    unittest.main()
