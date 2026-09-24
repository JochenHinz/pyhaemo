"""
Shared data-loading/reconstruction pipeline used by the run_*.py mesh-export
scripts, so the LAD/CX/RCA convergence and multi-vessel-motion studies don't
each re-implement the same load -> concatenate heartbeats -> clean -> spline
sequence.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline

from DataTime import TimeData, Motion
from TimeInterpolation import Spline, Fourier
import paths as pt


def load_full_motion(vessel):
    """
    Load `vessel`'s raw reconstruction (paths.VESSEL_DATA_PATHS[vessel]) --
    every frame, with no heartbeat splitting/truncation (how many heartbeats
    it represents is unknown up front, see TimeData.from_npy_all_frames) --
    and clean outlier centerlines. Returns the resulting raw, discrete
    TimeData -- the "full motion" (both centerline and radius vary).
    """

    if vessel not in pt.VESSEL_DATA_PATHS:
        raise ValueError(f'Unknown vessel {vessel!r}, expected one of {list(pt.VESSEL_DATA_PATHS)}')

    full_motion = TimeData.from_npy_all_frames(pt.VESSEL_DATA_PATHS[vessel])

    return full_motion.clean_noise()


def build_spline(sub_sampled, timestep=0.001, smoothing_factor=1):
    """ Best-compromise spline reconstruction (see TimeInterpolation.py). """
    return Spline(Data=sub_sampled, timestep=timestep, smoothing_factor=smoothing_factor)


def spline_motion(data, timestep, t_end=None, smoothing_factor=1):
    """
    Same smoothing spline as TimeInterpolation.Spline (fit over the sample
    index, lam=smoothing_factor), but evaluated at exactly t_k = k * timestep,
    k = 0 .. floor(t_end / timestep), in seconds. t_end defaults to the last
    sample time, (n_samples - 1) / ACQUISITION_FPS.

    Spline spreads int(period / timestep) points over the samples instead, so
    its actual step is not `timestep` (e.g. 0.000955s for CX at 0.001) and
    differs between timesteps -- grids for different timesteps don't nest,
    which breaks time convergence studies. Returns (centerlines, radii) in the
    physical_centerline/physical_radii layout: (n_steps, n_points, 3) and
    (n_steps, n_points).
    """

    last_sample_time = (data.n_samples - 1) / pt.ACQUISITION_FPS
    t_end = last_sample_time if t_end is None else t_end
    if t_end > last_sample_time + 1e-12:
        raise ValueError(f't_end={t_end}s is past the last sample ({last_sample_time:.4f}s)')

    n_steps = int(np.floor(t_end / timestep + 1e-9)) + 1
    sample_index = np.arange(n_steps) * timestep * pt.ACQUISITION_FPS
    x = np.arange(data.n_samples)

    centerlines = np.stack([make_smoothing_spline(x, cen, lam=smoothing_factor)(sample_index)
                            for cen in data.centerlines])
    radii = np.stack([make_smoothing_spline(x, rad, lam=smoothing_factor)(sample_index)
                      for rad in data.radii])

    return centerlines.transpose(1, 0, 2), radii.transpose(1, 0)


def build_fourier(sub_sampled, timestep=0.001, frequency_cutoff_percentage=40):
    """ Best-compromise Fourier reconstruction (see TimeInterpolation.py). """
    return Fourier(Data=sub_sampled, timestep=timestep,
                   frequency_cutoff_percentage=frequency_cutoff_percentage)


def motion_centerlines_radii(spline):
    """
    Build the physical_centerline/physical_radii pair (as expected by
    export.compute_incremental_displacement) for each of the 4 motion types,
    from a Spline reconstruction of the full motion.
    """

    motion = Motion(spline.as_time_data())
    rigid = motion.rigid_data()
    fsi = motion.fsi_data()
    stretching = motion.stretching_data()

    return {
        'full': (spline.physical_centerline, spline.physical_radii),
        'rigid': (rigid.centerlines.transpose(1, 0, 2), rigid.radii.transpose(1, 0)),
        'fsi': (fsi.centerlines.transpose(1, 0, 2), fsi.radii.transpose(1, 0)),
        'stretching': (stretching.centerlines.transpose(1, 0, 2), stretching.radii.transpose(1, 0)),
    }
