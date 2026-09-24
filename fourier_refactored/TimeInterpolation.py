from DataTime import TimeData, Motion
from pyhaemo.types import Immutable
from pyhaemo.util import np, _, clparam
from fourier import fourier_1D, fourier_3D
from paths import INPUT_data_path, ACQUISITION_FPS
from plots import plot_spline_vs_samples, plot_fourier_frequencies, plot_spline_vs_fourier, \
    plot_signed_reference_disp, plot_relative_motion, plot_radius
from abc import ABC, abstractmethod
from scipy.interpolate import make_smoothing_spline


# this class should perform on 1 hearbeat.
class TimeInterpolation(Immutable, ABC):

    def __init__(self,
                 Data,
                 timestep):

        if not isinstance(Data, TimeData):
            raise TypeError("Data must be an instance of Datatime class.")

        if not isinstance(timestep, (float, np.floating)):
            raise TypeError("timestep must be a float.")

        if timestep >= 1:
            raise ValueError("timestep must be less than 1.")

        self.Data = Data
        self.timestep = timestep

        # total physical duration spanned by Data, in seconds -- from the
        # actual acquisition frame rate, not an assumed heartbeat count
        # (which may be unknown, e.g. Data built with
        # TimeData.from_npy_all_frames()).
        self.period = self.Data.n_samples / ACQUISITION_FPS

    @property
    @abstractmethod
    def interpolated_centerlines(self):
        """
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def interpolated_radii(self):
        """
        Abstract property to create time-radii data.
        Must be implemented by subclasses.
        """
        pass

    @property
    def physical_centerline(self):
        ret = self.interpolated_centerlines.transpose(1, 0, 2)
        return ret


    @property
    def physical_radii(self):
        ret = self.interpolated_radii.transpose(1, 0)
        return ret

    def as_time_data(self):
        """
        Wrap the interpolated (dense) centerlines/radii back into a TimeData
        instance, so Motion() (rigid_data/fsi_data/stretching_data) can be
        applied directly to the interpolated data instead of the original
        samples.
        """

        return TimeData(centerlines=self.interpolated_centerlines,
                        radii=self.interpolated_radii,
                        sampling_rate=None)

# this class should perform on 1 hearbeat.
class Fourier(TimeInterpolation):

    def __init__(self,
                 Data,
                 timestep,
                 frequency_cutoff_percentage):

        super().__init__(Data, timestep)

        if not isinstance(frequency_cutoff_percentage, int):
            raise TypeError("timestep must be an integer.")

        if not (0 <= frequency_cutoff_percentage <= 100):
            raise ValueError("percentage has to be between 0 and 100 %")

        # frequencies to cut for each heartbeat
        self.freq_cut = int(
            (frequency_cutoff_percentage / 100)
            * (self.Data.n_samples / 2)
            )

    @property
    def interpolated_centerlines(self):
        ret = []
        for my_cen in self.Data.centerlines:
            fourier_cen = fourier_3D(my_cen,
                                     self.timestep,
                                     self.freq_cut,
                                     period=self.period)
            ret.append(fourier_cen)

        ret = np.array(ret)
        return ret

    @property
    def interpolated_radii(self):
        ret = []
        for my_rad in self.Data.radii:
            fourier_rad = fourier_1D(my_rad,
                                     self.timestep,
                                     self.freq_cut,
                                     period=self.period)
            ret.append(fourier_rad)

        ret = np.array(ret)
        return ret


class Spline(TimeInterpolation):
    def __init__(self,
                 Data,
                 timestep,
                 smoothing_factor=0):

        super().__init__(Data, timestep)

        if smoothing_factor < 0:
            raise ValueError("smothing factor can not be negative")

        self.smoothing_factor = smoothing_factor
        self.interp_points = int(self.period / self.timestep)


    @property
    def interpolated_centerlines(self):
        ret = []
        for my_cen in self.Data.centerlines:
            # Parameter along the centerline
            t = np.arange(self.Data.n_samples)

            # Fit univariate splines for each coordinate
            spline_x = make_smoothing_spline(t, my_cen[:, 0], lam=self.smoothing_factor)
            spline_y = make_smoothing_spline(t, my_cen[:, 1], lam=self.smoothing_factor)
            spline_z = make_smoothing_spline(t, my_cen[:, 2], lam=self.smoothing_factor)

            # Dense interpolation grid
            t_interp = np.linspace(t.min(), t.max(), self.interp_points)

            x_interp = spline_x(t_interp)
            y_interp = spline_y(t_interp)
            z_interp = spline_z(t_interp)

            # Stack into (N, 3)
            new_data = np.stack((x_interp, y_interp, z_interp), axis=1)

            ret.append(new_data)

        ret = np.array(ret)
        return ret

    @property
    def interpolated_radii(self):
        ret = []
        for my_rad in self.Data.radii:
            # Parameter along the centerline
            t = np.arange(self.Data.n_samples)

            # Fit univariate splines for each coordinate
            spline_rad = make_smoothing_spline(t, my_rad, lam=self.smoothing_factor)

            # Dense interpolation grid
            t_interp = np.linspace(t.min(), t.max(), self.interp_points)

            rad_interp = spline_rad(t_interp)

            ret.append(rad_interp)

        ret = np.array(ret)
        return ret


if __name__ == '__main__':

    # every frame in the recording, no heartbeat splitting/truncation --
    # how many heartbeats it represents is unknown up front (see
    # DataTime.py's __main__).
    time_data_healthy = TimeData.from_npy_all_frames(INPUT_data_path)

    cleaned = time_data_healthy.clean_noise()  #time_data_healthy.clean_noise(factor=2.5)
    sub_sampled = cleaned.sub_sample(1)

    # sweep the Fourier frequency cutoff to compare against the raw samples
    freqs = [0, 20, 40, 60, 80]
    fourier = [Fourier(Data=sub_sampled,
                       timestep=0.01,
                       frequency_cutoff_percentage=myfreq)
              for myfreq in freqs]

    plot_fourier_frequencies(data_sample=sub_sampled,
                             data_fourier=fourier,
                             list_percetage_freq=freqs,
                             title='Fourier frequency cutoff sweep')

    # best compromise found from the sweeps above: Fourier 60% cutoff keeps
    # the main peaks while denoising, spline smoothing_factor=1 avoids both
    # the raw-spline oscillation (factor=0) and the over-smoothing of large
    # factors.
    best_spline = Spline(Data=sub_sampled, timestep=0.001, smoothing_factor=1)
    best_fourier = Fourier(Data=sub_sampled, timestep=0.001, frequency_cutoff_percentage=40)

    plot_spline_vs_fourier(sub_sampled,
                           best_spline,
                           best_fourier,
                           title='Best compromise: spline vs Fourier')

    # Motion decomposition (rigid / fsi / stretching) applied directly to
    # the interpolated (dense, smooth) data instead of the raw samples.
    fourier_motion = Motion(best_fourier.as_time_data())
    fourier_rigid = fourier_motion.rigid_data()
    fourier_fsi = fourier_motion.fsi_data()
    fourier_stretching = fourier_motion.stretching_data()

    spline_motion = Motion(best_spline.as_time_data())
    spline_rigid = spline_motion.rigid_data()
    spline_fsi = spline_motion.fsi_data()
    spline_stretching = spline_motion.stretching_data()

    # the same decomposition applied to the raw (non-interpolated) samples,
    # to overlay the true discrete data points on top of the smooth curves.
    raw_motion = Motion(sub_sampled)
    raw_rigid = raw_motion.rigid_data()
    raw_fsi = raw_motion.fsi_data()
    raw_stretching = raw_motion.stretching_data()

    # Generate meshes + incremental displacement (for LifeX) for the full
    # spline motion (best_spline), the interpolated (dense, smooth) source
    # of centerlines/radii computed above.
    from export import compute_incremental_displacement
    import paths as pt

    NELEMS_CENTERLINE = 30
    NELEMS_CROSS_SECTION = 5
    NELEMS_Z = 301
    BOUNDARY_LAYER_EXPONENT = 3.  # 1. = uniform cross-section grid, no boundary layer

    compute_incremental_displacement(
        best_fourier.physical_centerline,
        best_fourier.physical_radii,
        timestep=0.001,
        meshes_dir=str(pt.OUTPUT_meshes_path / 'full'),
        displacement_dir=str(pt.OUTPUT_displacement_path / 'full'),
        nelems_centerline=NELEMS_CENTERLINE,
        nelems_cross_section=NELEMS_CROSS_SECTION,
        nelems_z=NELEMS_Z,
        boundary_layer_exponent=BOUNDARY_LAYER_EXPONENT,
    )


    for label, motion_data, samples, period in [
            ('Fourier rigid', fourier_rigid, raw_rigid, best_fourier.period),
            ('Fourier FSI', fourier_fsi, raw_fsi, best_fourier.period),
            ('Fourier stretching', fourier_stretching, raw_stretching, best_fourier.period),
            ('Spline rigid', spline_rigid, raw_rigid, best_spline.period),
            ('Spline FSI', spline_fsi, raw_fsi, best_spline.period),
            ('Spline stretching', spline_stretching, raw_stretching, best_spline.period)]:

        plot_signed_reference_disp(motion_data, data_samples=samples, period=period,
                                   title=f'{label} - signed reference displacement')
        plot_relative_motion(motion_data, data_samples=samples, period=period,
                             title=f'{label} - relative motion')
        plot_radius(motion_data, data_samples=samples, period=period,
                   title=f'{label} - radius')

    '''
    # Generate meshes + incremental displacement (for LifeX) for all 4 motion
    # types, using the interpolated (spline, smoothing_factor=1) data as the
    # source of centerlines/radii. This is a heavy, long-running batch job
    # (mesh generation takes seconds per timestep, and the interpolated data
    # has thousands of timesteps), so it is left commented out on purpose:
    # uncomment and run manually (e.g. in the background) when ready.
    from export import compute_incremental_displacement
    import paths as pt

    # spline_rigid/spline_fsi/spline_stretching are already the interpolated
    # (dense, smooth) decompositions computed above, sharing the same
    # (n_points, n_samples[, 3]) layout as TimeInterpolation's own arrays, so
    # they only need the same transpose physical_centerline/physical_radii apply.
    motions = {
        'full': (best_spline.physical_centerline, best_spline.physical_radii),
        'rigid': (spline_rigid.centerlines.transpose(1, 0, 2), spline_rigid.radii.transpose(1, 0)),
        'fsi': (spline_fsi.centerlines.transpose(1, 0, 2), spline_fsi.radii.transpose(1, 0)),
        'stretching': (spline_stretching.centerlines.transpose(1, 0, 2), spline_stretching.radii.transpose(1, 0)),
    }

    for label, (centerlines, radii) in motions.items():
        compute_incremental_displacement(
            centerlines,
            radii,
            timestep=0.001,
            meshes_dir=str(pt.OUTPUT_meshes_path / label),
            displacement_dir=str(pt.OUTPUT_displacement_path / label),
        )
    '''

    '''
    # Time-step convergence study on the FULL motion: re-interpolate
    # sub_sampled with the spline (smoothing_factor=1) at each candidate
    # timestep and export its meshes/displacements to their own
    # subdirectory, so the resulting meshes can be compared across dt.
    # Each dt is a separate, heavy, long-running mesh-generation job, so
    # this is left commented out on purpose: uncomment and run manually.
    from export import compute_incremental_displacement
    import paths as pt

    convergence_timesteps = [0.01, 0.005, 0.001, 0.0005]

    for dt in convergence_timesteps:
        spline_dt = Spline(Data=sub_sampled, timestep=dt, smoothing_factor=1)
        label = f'full_dt_{dt:.4f}'

        compute_incremental_displacement(
            spline_dt.physical_centerline,
            spline_dt.physical_radii,
            timestep=dt,
            meshes_dir=str(pt.OUTPUT_meshes_path / label),
            displacement_dir=str(pt.OUTPUT_displacement_path / label),
        )
    '''
