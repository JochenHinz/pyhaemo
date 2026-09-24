"""
Regression tests for TimeInterpolation.py (Spline/Fourier).

Run with: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataTime import TimeData, Motion
from TimeInterpolation import Spline, Fourier
from fourier import HEARTBEAT_PERIOD


def make_time_data(n_points=6, n_samples=15, n_heartbeat=1, seed=0):
    rng = np.random.default_rng(seed)
    centerlines = rng.normal(size=(n_points, n_samples, 3)).cumsum(axis=1)
    radii = np.abs(rng.normal(size=(n_points, n_samples))) + 1.0
    sampling_rate = n_heartbeat / n_samples
    return TimeData(centerlines=centerlines, radii=radii, sampling_rate=sampling_rate)


class TestTimeInterpolationValidation(unittest.TestCase):

    def test_rejects_non_time_data(self):
        with self.assertRaises(TypeError):
            Spline(Data="not TimeData", timestep=0.1)

    def test_rejects_non_float_timestep(self):
        td = make_time_data()
        with self.assertRaises(TypeError):
            Spline(Data=td, timestep=1)  # int, not float

    def test_rejects_timestep_ge_1(self):
        td = make_time_data()
        with self.assertRaises(ValueError):
            Spline(Data=td, timestep=1.5)

    def test_fourier_rejects_out_of_range_percentage(self):
        td = make_time_data()
        with self.assertRaises(ValueError):
            Fourier(Data=td, timestep=0.1, frequency_cutoff_percentage=-5)
        with self.assertRaises(ValueError):
            Fourier(Data=td, timestep=0.1, frequency_cutoff_percentage=101)

    def test_fourier_accepts_boundary_percentages(self):
        td = make_time_data()
        Fourier(Data=td, timestep=0.1, frequency_cutoff_percentage=0)
        Fourier(Data=td, timestep=0.1, frequency_cutoff_percentage=100)


class TestPeriodScaling(unittest.TestCase):
    """
    Regression test for the bug where Fourier silently used a period that
    did not scale with the number of heartbeats in Data, corrupting the
    frequency axis for any multi-heartbeat input.
    """

    def test_period_scales_with_n_heartbeat(self):
        for n_heartbeat in [1, 2, 4]:
            td = make_time_data(n_samples=15 * n_heartbeat, n_heartbeat=n_heartbeat)
            spline = Spline(Data=td, timestep=0.1, smoothing_factor=0)
            fourier = Fourier(Data=td, timestep=0.1, frequency_cutoff_percentage=50)

            self.assertEqual(spline.n_heartbeat, n_heartbeat)
            self.assertEqual(fourier.n_heartbeat, n_heartbeat)
            self.assertAlmostEqual(spline.period, HEARTBEAT_PERIOD * n_heartbeat)
            self.assertAlmostEqual(fourier.period, HEARTBEAT_PERIOD * n_heartbeat)

    def test_spline_and_fourier_agree_on_reconstruction_length(self):
        td = make_time_data(n_samples=45, n_heartbeat=3)
        timestep = 0.1
        spline = Spline(Data=td, timestep=timestep, smoothing_factor=0)
        fourier = Fourier(Data=td, timestep=timestep, frequency_cutoff_percentage=50)

        expected_T = int(spline.period / timestep)
        self.assertEqual(spline.interpolated_centerlines.shape[1], expected_T)
        self.assertEqual(fourier.interpolated_centerlines.shape[1], expected_T)

    def test_sub_sample_preserves_n_heartbeat(self):
        # sub_sample must rescale sampling_rate so n_heartbeat survives.
        td = make_time_data(n_samples=40, n_heartbeat=4)
        sub = td.sub_sample(sample_rate=2)

        spline = Spline(Data=sub, timestep=0.1, smoothing_factor=0)
        self.assertEqual(spline.n_heartbeat, 4)


class TestAsTimeData(unittest.TestCase):

    def test_round_trip_shape_and_motion_compatible(self):
        td = make_time_data(n_points=5, n_samples=20, n_heartbeat=2)
        spline = Spline(Data=td, timestep=0.1, smoothing_factor=0)

        interp_td = spline.as_time_data()
        self.assertIsInstance(interp_td, TimeData)
        self.assertEqual(interp_td.n_points, td.n_points)

        n_heartbeat_recovered = int(interp_td.sampling_rate * interp_td.n_samples)
        self.assertEqual(n_heartbeat_recovered, spline.n_heartbeat)

        # Motion() must accept it and its internal invariants must hold
        # (verify_decomposition runs automatically inside Motion.__init__).
        motion = Motion(interp_td)
        rigid = motion.rigid_data()
        self.assertEqual(rigid.n_samples, interp_td.n_samples)


if __name__ == '__main__':
    unittest.main()
