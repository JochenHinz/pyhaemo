"""
Shared data-loading/reconstruction pipeline used by the run_*.py mesh-export
scripts, so the LAD/CX/RCA convergence and multi-vessel-motion studies don't
each re-implement the same load -> concatenate heartbeats -> clean -> spline
sequence.
"""
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
