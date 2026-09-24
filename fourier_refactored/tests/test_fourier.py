"""
Regression tests for fourier.py.

Run with: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourier import fourier_1D, fourier_3D, HEARTBEAT_PERIOD


class TestFourier1D(unittest.TestCase):

    def test_no_cutoff_reconstructs_a_sampled_sine(self):
        period = 1.0
        M = 40
        t_samples = np.linspace(0, period, M, endpoint=False)
        signal = np.sin(2 * np.pi * 3 * t_samples / period)

        timestep = period / M
        reconstructed = fourier_1D(signal, timestep=timestep, freq_cut=0, period=period)

        # with no frequencies cut, resampling on the same grid should
        # reproduce the original samples closely.
        np.testing.assert_allclose(reconstructed, signal, atol=1e-8)

    def test_full_cutoff_collapses_to_mean(self):
        period = 1.0
        M = 40
        t_samples = np.linspace(0, period, M, endpoint=False)
        signal = np.sin(2 * np.pi * 3 * t_samples / period) + 5.0

        timestep = period / M
        reconstructed = fourier_1D(signal, timestep=timestep, freq_cut=M // 2, period=period)

        np.testing.assert_allclose(reconstructed, np.mean(signal) * np.ones_like(reconstructed), atol=1e-8)

    def test_reconstruction_length_matches_period_over_timestep(self):
        period = 2.0
        timestep = 0.05
        signal = np.random.default_rng(0).normal(size=20)

        reconstructed = fourier_1D(signal, timestep=timestep, freq_cut=2, period=period)
        self.assertEqual(len(reconstructed), int(period / timestep))

    def test_period_scaling_changes_output_length(self):
        # this is the behaviour TimeInterpolation.Fourier relies on to
        # correctly scale with multi-heartbeat data.
        signal = np.random.default_rng(0).normal(size=20)
        timestep = 0.01

        short = fourier_1D(signal, timestep=timestep, freq_cut=2, period=HEARTBEAT_PERIOD)
        long = fourier_1D(signal, timestep=timestep, freq_cut=2, period=HEARTBEAT_PERIOD * 4)

        self.assertEqual(len(long), 4 * len(short))


class TestFourier3D(unittest.TestCase):

    def test_shape_matches_input_points_and_period(self):
        period = 1.0
        timestep = 0.05
        array = np.random.default_rng(0).normal(size=(30, 3))

        reconstructed = fourier_3D(array, timestep=timestep, freq_cut=3, period=period)
        self.assertEqual(reconstructed.shape, (int(period / timestep), 3))

    def test_independent_axes(self):
        # x/y/z should be reconstructed independently: a constant y,z with
        # a varying x must not leak into the other axes.
        period = 1.0
        M = 30
        t_samples = np.linspace(0, period, M, endpoint=False)
        array = np.zeros((M, 3))
        array[:, 0] = np.sin(2 * np.pi * 2 * t_samples / period)

        timestep = period / M
        reconstructed = fourier_3D(array, timestep=timestep, freq_cut=0, period=period)

        np.testing.assert_allclose(reconstructed[:, 1], 0, atol=1e-8)
        np.testing.assert_allclose(reconstructed[:, 2], 0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
