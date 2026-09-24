import numpy as np
from pyhaemo.types import Immutable
from typing import Literal
from plots import plot_signed_reference_disp, plot_clean_vs_no_clean, plot_sampled_vs_no_sampled, plot_relative_motion, plot_radius, plot_pc_loop
from paths import INPUT_data_path, ACQUISITION_FPS

# the class handles 1 hearbeat
# each heartbeat has its own TimeData
class TimeData(Immutable):

    @classmethod
    def from_npy(cls,
                 filename,
                 sampling_rate=1/15):

        # Load the Data
        recon = np.load(filename, allow_pickle=True)

        # Extract the time snapshots
        centerlines = []
        radii = []
        for i in range(len(recon)):
            centerlines.append(recon[i]['points3D']['centreLine'])
            radii.append(recon[i]['points3D']['radius'])

        centerlines = np.array(centerlines).transpose(1, 0, 2)  # shape (n_points, n_samples, 3)
        radii = np.array(radii).transpose(1, 0)  # shape (n_points, n_samples)

        # suppose the heartbeat period is 1s
        n_heartbeats = int(centerlines.shape[1] * sampling_rate)

        # if the n_bearts is a float, we approximate to the lowest integer
        sample_until = int(n_heartbeats*(1/sampling_rate))

        centerlines = centerlines[:, :sample_until, :]
        radii = radii[:, :sample_until]

        # Split by heartbeats
        # TODO: from the second heartbeat, we should consider as
        # a first centerline the last one of the one before. use list instead of array.
        n_points = centerlines.shape[0]
        samples_per_beat = int(centerlines.shape[1] // n_heartbeats)
        centerlines_by_beat = centerlines.reshape(
            n_points, n_heartbeats, samples_per_beat, 3
        )

        radii_by_beat = radii.reshape(
            n_points, n_heartbeats, samples_per_beat
        )

        # Return a list of TimeData, one per heartbeat
        return [
            cls(centerlines_by_beat[:, i, :, :], radii_by_beat[:, i, :], sampling_rate)
            for i in range(n_heartbeats)
        ]

    @classmethod
    def from_npy_all_frames(cls, filename):
        """
        Load every frame in the recording as a single TimeData, with no
        heartbeat splitting/truncation (from_npy() discards trailing frames
        that don't fill a whole assumed heartbeat, which requires already
        knowing how many heartbeats are present). Use this when that count
        is unknown and has to be read off the plotted motion instead.
        """

        recon = np.load(filename, allow_pickle=True)

        centerlines = []
        radii = []
        for i in range(len(recon)):
            centerlines.append(recon[i]['points3D']['centreLine'])
            radii.append(recon[i]['points3D']['radius'])

        centerlines = np.array(centerlines).transpose(1, 0, 2)  # shape (n_points, n_samples, 3)
        radii = np.array(radii).transpose(1, 0)  # shape (n_points, n_samples)

        # number of heartbeats is unknown here, unlike the sampling_rate
        # from from_npy(); heartbeat-boundary markers are skipped for data
        # built this way (see plots._heartbeat_boundaries).
        return cls(centerlines, radii, sampling_rate=None)

    def __init__(self,
                 centerlines,
                 radii,
                 sampling_rate=1/15):

        """
        4D data of the centerline and radius.
        """

        assert centerlines.shape[0] == radii.shape[0], \
            'snapshots of centerlines and radii has to be the same'

        assert centerlines.shape[1] == radii.shape[1], \
            'points of centerlines and radii has to be the same'

        self.centerlines = centerlines
        self.radii = radii
        self.sampling_rate = sampling_rate

    @property
    def n_points(self):
        """
        Number of points of each sampled centerline.
        """
        ret = self.centerlines.shape[0]
        return ret

    @property
    def n_samples(self):
        """
        Number of sampled centerline.
        """
        ret = self.centerlines.shape[1]
        return ret

    def __add__(self, other):
        """
        Concatenate the centerlines and radii of two TimeData instances along the sample axis.
        This operation returns a new TimeData instance with the combined data.

        Parameters
        ----------
        other : TimeData
            Another TimeData instance to concatenate with this one.

        Returns
        -------
        TimeData
            A new TimeData instance with concatenated centerlines and radii.
        """
        if not isinstance(other, TimeData):
            return NotImplemented

        # Element-wise addition of centerlines and radii
        new_centerlines = np.append(self.centerlines, other.centerlines, axis=1)
        new_radii = np.append(self.radii, other.radii, axis=1)

        return self._edit(centerlines=new_centerlines,
                          radii=new_radii)

    def add_stenosis(self, position: Literal['middle up', 'middle bottom'],
                           severity: Literal['medium', 'high']):

        """
        Method for modifing the radius data to add a stenosis.
        """

        from stenosis import create_stenosis
        stenotic_radii = create_stenosis(radii=self.radii,
                                         position=position,
                                         severity=severity,
                                         n_points=self.n_points)

        return self._edit(radii=stenotic_radii.transpose(1, 0))

    def clean_noise(self, factor=2.5):
        """
        Method for detect the outlier centerlines and substituting them with
        the average of the closest not-outlier centerlines.
        """

        mean_point = np.mean(self.centerlines, axis=0)
        disp_from_reference = np.linalg.norm(mean_point[0:] - mean_point[0],
                                             axis=1)

        mask = disp_from_reference > factor * np.mean(disp_from_reference)

        cleaned_centerlines = np.copy(self.centerlines)
        cleaned_radii = np.copy(self.radii)

        for i, my_mask in enumerate(mask):
            if my_mask == True:
                assert i > 0, \
                    'first sample is an outlier, cannot average with the previous sample'

                not_outlier = np.where(mask[i:] == False)[0]
                assert len(not_outlier) > 0, \
                    'no non-outlier sample found after index {} to interpolate with'.format(i)

                index = not_outlier[0] + i
                cleaned_centerlines[:, i] = (cleaned_centerlines[:, i-1] +
                                             cleaned_centerlines[:, index]) / 2

        return self._edit(centerlines=cleaned_centerlines, radii=cleaned_radii)

    def sub_sample(self, sample_rate=2):

        sampled_centerlines = self.centerlines[:, 0::sample_rate]
        sampled_radii = self.radii[:, 0::sample_rate]

        # fewer samples now represent the same heartbeat(s), so the
        # heartbeats-per-sample rate must grow proportionally, otherwise
        # downstream heartbeat counts would be computed from a stale sample
        # count. When the heartbeat count is unknown (sampling_rate=None,
        # e.g. from_npy_all_frames()), it stays unknown.
        new_sampling_rate = (None if self.sampling_rate is None
                             else self.sampling_rate * sample_rate)

        return self._edit(centerlines=sampled_centerlines,
                          radii=sampled_radii,
                          sampling_rate=new_sampling_rate)

    def averaging(self):
        pass


class Motion(Immutable):

    def __init__(self,
                 time_data):

        assert isinstance(time_data, TimeData), \
            'Motion must be initialized with a TimeData instance'

        self.centerlines = time_data.centerlines
        self.radii = time_data.radii
        self.sampling_rate = time_data.sampling_rate

        verify_decomposition(self)

    @property
    def mean_centerlines(self):
        ret = np.mean(self.centerlines, axis=0)
        return ret

    @property
    def mean_radii(self):
        ret = np.mean(self.radii, axis=0)
        return ret

    @property
    def disp_rigid(self):
        ret = self.mean_centerlines[0:]-self.mean_centerlines[0]
        return ret

    def rigid_data(self):
        centerlines = []

        for myrigid in self.disp_rigid:
            centerlines.append(self.centerlines[:, 0] + myrigid)

        centerlines = np.array(centerlines).transpose(1, 0, 2)
        radii = self.radii[:, 0].reshape(-1, 1)*np.ones_like(self.radii)

        # TODO: understand why #self._edit(centerlines=centerlines, radii=radii)
        return TimeData(centerlines=centerlines, radii=radii, sampling_rate=self.sampling_rate)

    def fsi_data(self):
        # centerline is keep fixed, while only the radius changes.
        # This is a FSI surrgoate, no streatching and rigid motion considered.
        # Only the change in the radius data is considered.
        centerlines = self.centerlines[:, 0].reshape(-1, 1, 3) * \
                      np.ones_like(self.centerlines)

        return TimeData(centerlines=centerlines, radii=self.radii, sampling_rate=self.sampling_rate)

    def stretching_data(self):
        # centerline keeps only its non-rigid (shape) deformation, while the
        # radius is kept fixed to its t=0 value: this isolates the local
        # stretching of the centerline shape, complementing rigid_data().
        disp_total = self.centerlines.transpose(1, 0, 2) - \
                          self.centerlines.transpose(1, 0, 2)[0]

        disp_stretching = disp_total - self.disp_rigid.reshape(-1, 1, 3)
        centerlines = []

        for mydisp in disp_stretching:
           centerlines.append(self.centerlines[:, 0] + mydisp)

        centerlines = np.array(centerlines).transpose(1, 0, 2)
        radii = self.radii[:, 0].reshape(-1, 1)*np.ones_like(self.radii)

        return TimeData(centerlines=centerlines, radii=radii, sampling_rate=self.sampling_rate)


def verify_decomposition(motion: Motion, atol=1e-8):
    """
    Checks the algebraic invariants that the rigid/stretching/fsi
    decomposition of a Motion instance must satisfy by construction.
    Raises an AssertionError with a descriptive message if any check fails.
    """

    rigid = motion.rigid_data()
    stretching = motion.stretching_data()
    fsi = motion.fsi_data()

    # rigid displacement + stretching displacement == full centerline displacement
    disp_full = motion.centerlines - motion.centerlines[:, 0:1]
    disp_rigid = rigid.centerlines - rigid.centerlines[:, 0:1]
    disp_stretching = stretching.centerlines - stretching.centerlines[:, 0:1]

    assert np.allclose(disp_rigid + disp_stretching, disp_full, atol=atol), \
        'rigid_data() + stretching_data() centerline displacement does not ' \
        'reconstruct the full centerline displacement'

    # rigid_data() and stretching_data() must have radii frozen to t=0
    assert np.allclose(rigid.radii, motion.radii[:, 0:1], atol=atol), \
        'rigid_data() radii are not constant and equal to radii at t=0'

    assert np.allclose(stretching.radii, motion.radii[:, 0:1], atol=atol), \
        'stretching_data() radii are not constant and equal to radii at t=0'

    # fsi_data() must have centerlines frozen to t=0 and untouched radii
    assert np.allclose(fsi.centerlines, motion.centerlines[:, 0:1], atol=atol), \
        'fsi_data() centerlines are not constant and equal to centerlines at t=0'

    assert np.allclose(fsi.radii, motion.radii, atol=atol), \
        'fsi_data() radii were modified from the original radii'


if __name__ == '__main__':

    ######################FULL motion###################
    # Every frame in the recording, with no heartbeat splitting/truncation
    # -- how many heartbeats it represents is exactly what's unknown, so it
    # can't be used to pick a `sampling_rate` up front. The x-axis is real
    # elapsed time (frame_count / fps), so the number of heartbeats can be
    # read off the periodicity of the signed-distance curve instead.
    full_motion = TimeData.from_npy_all_frames(INPUT_data_path)
    full_motion_period = full_motion.n_samples / ACQUISITION_FPS

    #time_data_stenosis = full_motion.add_stenosis(position='middle bottom',
    #                                            severity='high')

    # plot the (signed) reference displacement
    plot_signed_reference_disp(full_motion, period=full_motion_period,
                                title='Full motion (all frames)')
    plot_pc_loop(full_motion, period=full_motion_period,
                 title='Full motion - PC1 vs PC2 loop')
    plot_relative_motion(full_motion, period=full_motion_period, title='Full motion - relative motion')
    plot_radius(full_motion, period=full_motion_period, title='Full motion - radius')

    ######################FSI motion###################
    fsi = Motion(full_motion).fsi_data()

    plot_signed_reference_disp(fsi, title='FSI motion - signed reference displacement')
    plot_relative_motion(fsi, title='FSI motion - relative motion')
    plot_radius(fsi, title='FSI motion - radius')


    ######################RIGID motion###################
    # Motion Separation
    rigid = Motion(full_motion).rigid_data()

    # quantify the stretching of the centerlines
    plot_signed_reference_disp(rigid, title='Rigid motion - signed reference displacement')
    plot_relative_motion(rigid, title='Rigid motion - relative motion')
    plot_radius(rigid, title='Rigid motion - radius')


    ######################STRETCHING motion###################

    # Motion Separation
    stretching = Motion(full_motion).stretching_data()

    # quantify the stretching of the centerlines
    plot_signed_reference_disp(stretching, title='Stretching motion - signed reference displacement')
    plot_relative_motion(stretching, title='Stretching motion - relative motion')
    plot_radius(stretching, title='Stretching motion - radius')
