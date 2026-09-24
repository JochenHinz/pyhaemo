"""
Regression tests for DataTime.py (TimeData/Motion + verify_decomposition).

Run with: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataTime import TimeData, Motion, verify_decomposition


def make_time_data(n_points=4, n_samples=6, sampling_rate=1 / 6, seed=0):
    rng = np.random.default_rng(seed)
    centerlines = rng.normal(size=(n_points, n_samples, 3))
    radii = np.abs(rng.normal(size=(n_points, n_samples))) + 1.0
    return TimeData(centerlines=centerlines, radii=radii, sampling_rate=sampling_rate)


class TestTimeDataValidation(unittest.TestCase):

    def test_mismatched_point_count_raises(self):
        centerlines = np.zeros((4, 6, 3))
        radii = np.zeros((3, 6))
        with self.assertRaises(AssertionError):
            TimeData(centerlines=centerlines, radii=radii)

    def test_mismatched_sample_count_raises(self):
        centerlines = np.zeros((4, 6, 3))
        radii = np.zeros((4, 5))
        with self.assertRaises(AssertionError):
            TimeData(centerlines=centerlines, radii=radii)

    def test_add_concatenates_along_samples(self):
        td1 = make_time_data(n_samples=3, seed=1)
        td2 = make_time_data(n_samples=4, seed=2)
        combined = td1 + td2
        self.assertEqual(combined.n_samples, 7)
        np.testing.assert_allclose(combined.centerlines[:, :3], td1.centerlines)
        np.testing.assert_allclose(combined.centerlines[:, 3:], td2.centerlines)


class TestSubSample(unittest.TestCase):

    def test_sub_sample_rescales_sampling_rate(self):
        td = make_time_data(n_samples=6, sampling_rate=1 / 6)
        sub = td.sub_sample(sample_rate=2)

        self.assertEqual(sub.n_samples, 3)
        self.assertAlmostEqual(sub.sampling_rate, td.sampling_rate * 2)

        n_heartbeat_before = int(td.sampling_rate * td.n_samples)
        n_heartbeat_after = int(sub.sampling_rate * sub.n_samples)
        self.assertEqual(n_heartbeat_before, n_heartbeat_after)


class TestCleanNoise(unittest.TestCase):

    def test_interior_outlier_is_averaged(self):
        n_points, n_samples = 3, 6
        centerlines = np.zeros((n_points, n_samples, 3))
        for t in range(n_samples):
            centerlines[:, t, 0] = t  # smooth drift in x

        # inject a large outlier at t=3 for every point
        centerlines[:, 3, :] += 100.0

        radii = np.ones((n_points, n_samples))
        td = TimeData(centerlines=centerlines, radii=radii)

        cleaned = td.clean_noise(factor=2.5)

        expected = (cleaned.centerlines[:, 2] + cleaned.centerlines[:, 4]) / 2
        np.testing.assert_allclose(cleaned.centerlines[:, 3], expected)

    def test_trailing_outliers_with_no_recovery_raises(self):
        n_points, n_samples = 3, 5
        centerlines = np.zeros((n_points, n_samples, 3))
        for t in range(n_samples):
            centerlines[:, t, 0] = t

        # blow up only the very last sample, with nothing "clean" after it
        centerlines[:, 4, :] += 1000.0

        radii = np.ones((n_points, n_samples))
        td = TimeData(centerlines=centerlines, radii=radii)

        with self.assertRaises(AssertionError):
            td.clean_noise(factor=2.5)


class TestMotionValidation(unittest.TestCase):

    def test_rejects_non_time_data(self):
        with self.assertRaises(AssertionError):
            Motion("not a TimeData instance")


class TestMotionDecomposition(unittest.TestCase):

    def setUp(self):
        self.td = make_time_data(n_points=5, n_samples=8, sampling_rate=1 / 8)
        self.motion = Motion(self.td)

    def test_verify_decomposition_passes(self):
        # Motion.__init__ already runs this; call it again explicitly too.
        verify_decomposition(self.motion)

    def test_rigid_radii_are_frozen_to_t0(self):
        rigid = self.motion.rigid_data()
        np.testing.assert_allclose(rigid.radii, self.td.radii[:, 0:1] * np.ones_like(self.td.radii))

    def test_stretching_radii_are_frozen_to_t0(self):
        stretching = self.motion.stretching_data()
        np.testing.assert_allclose(stretching.radii, self.td.radii[:, 0:1] * np.ones_like(self.td.radii))

    def test_fsi_centerlines_are_frozen_to_t0(self):
        fsi = self.motion.fsi_data()
        for t in range(fsi.n_samples):
            np.testing.assert_allclose(fsi.centerlines[:, t], self.td.centerlines[:, 0])

    def test_fsi_radii_are_untouched(self):
        fsi = self.motion.fsi_data()
        np.testing.assert_allclose(fsi.radii, self.td.radii)

    def test_rigid_plus_stretching_reconstructs_full_motion(self):
        rigid = self.motion.rigid_data()
        stretching = self.motion.stretching_data()

        disp_full = self.td.centerlines - self.td.centerlines[:, 0:1]
        disp_rigid = rigid.centerlines - rigid.centerlines[:, 0:1]
        disp_stretching = stretching.centerlines - stretching.centerlines[:, 0:1]

        np.testing.assert_allclose(disp_rigid + disp_stretching, disp_full, atol=1e-8)

    def test_rigid_displacement_is_same_for_every_point(self):
        # a pure rigid translation must move every centerline point by the
        # same amount at each timestep.
        rigid = self.motion.rigid_data()
        disp = rigid.centerlines - rigid.centerlines[:, 0:1]
        for p in range(1, rigid.n_points):
            np.testing.assert_allclose(disp[p], disp[0])


if __name__ == '__main__':
    unittest.main()
