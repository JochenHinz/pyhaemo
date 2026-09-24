import matplotlib.pyplot as plt
import numpy as np

import fourier


FONTSIZE_LABEL = 28
FONTSIZE_TICK = 30
FONTSIZE_LEGEND = 14
DISP_YLABEL = "Displacements from reference (mm)"
SIGNED_DISP_YLABEL = "Signed reference displacement (mm)"


def reference_displacement(data):
    return np.linalg.norm(data[0:] - data[0], axis=1)


def dominant_direction(disp):
    """
    1st principal component (unit vector) of a (n_samples, 3) displacement
    trajectory, sign-fixed so that the largest excursion counts as positive
    (SVD only determines the direction up to a sign).
    """
    magnitude = np.linalg.norm(disp, axis=1)
    centered = disp - disp.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    peak = np.argmax(magnitude)
    if np.dot(disp[peak], direction) < 0:
        direction = -direction

    return direction


def signed_reference_displacement(data, direction=None):
    # Signed-distance version of reference_displacement(): keeps the full
    # 3D magnitude, but attaches a sign based on which side of the t=0
    # position the point is on, relative to `direction`. If no shared
    # `direction` is given, it falls back to this point's own dominant
    # motion direction.
    disp = data[0:] - data[0]
    magnitude = np.linalg.norm(disp, axis=1)

    if direction is None:
        direction = dominant_direction(disp)
    import ipdb
    ipdb.set_trace()
    sign = np.sign(disp @ direction)
    return magnitude * sign


def _time_axis(n_samples, period=1, endpoint=False):
    return np.linspace(0, period, n_samples, endpoint=endpoint)


def _heartbeat_boundaries(data, period):
    # data.sampling_rate is heartbeats-per-sample (see TimeData), so
    # sampling_rate * n_samples gives how many heartbeats are concatenated
    # in `data`. Objects without a sampling_rate, or with it set to None
    # (e.g. TimeData.from_npy_all_frames(), where the heartbeat count is
    # exactly what's unknown), have nothing to mark.
    if getattr(data, 'sampling_rate', None) is None:
        return []

    n_heartbeats = int(round(data.sampling_rate * data.n_samples))
    if n_heartbeats <= 1:
        return []

    return [k * period / n_heartbeats for k in range(1, n_heartbeats)]


def _plot_heartbeat_boundaries(data, period):
    boundaries = _heartbeat_boundaries(data, period)
    for i, t in enumerate(boundaries):
        plt.axvline(t, color='black', linestyle='--', linewidth=1,
                    label='Heartbeat boundary' if i == 0 else None)

    if boundaries:
        ax = plt.gca()
        ax.set_xticks(boundaries, minor=True)
        ax.tick_params(axis='x', which='minor', length=12, width=1.5, color='black')


def _finalize_plot(ylabel, xlabel="Time", title=None, save_path=None):
    plt.xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    plt.ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    if title is not None:
        plt.title(title, fontsize=FONTSIZE_LABEL)
    plt.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_reference_disp(data, data_samples=None, period=1, title=None, save_path=None):

    # data_samples lets the true (discrete) samples be scattered on top of a
    # smooth/interpolated `data` curve; defaults to `data` itself so plain
    # discrete data (e.g. from DataTime) still scatters its own points.
    samples = data if data_samples is None else data_samples

    disp_first_point = reference_displacement(data.centerlines[0])

    middle = len(data.centerlines) // 2
    disp_mean_point = reference_displacement(data.centerlines[middle])

    disp_last_point = reference_displacement(data.centerlines[-1])

    time = _time_axis(data.n_samples, period=period)

    disp_first_samples = reference_displacement(samples.centerlines[0])
    disp_mean_samples = reference_displacement(samples.centerlines[len(samples.centerlines) // 2])
    disp_last_samples = reference_displacement(samples.centerlines[-1])
    time_samples = _time_axis(samples.n_samples, period=period)

    plt.figure()
    plt.plot(time, disp_first_point, color='orange')
    plt.scatter(time_samples, disp_first_samples, label='Samples first point', color='orange')
    plt.plot(time, disp_mean_point, color='blue')
    plt.scatter(time_samples, disp_mean_samples, label='Samples mean point', color='blue')
    plt.plot(time, disp_last_point, color='green')
    plt.scatter(time_samples, disp_last_samples, label='Samples last point', color='green')
    _plot_heartbeat_boundaries(data, period)
    _finalize_plot(ylabel=DISP_YLABEL, title=title, save_path=save_path)


def plot_signed_reference_disp(data, data_samples=None, period=1, title=None, save_path=None):

    # Same as plot_reference_disp, but keeping the sign of the displacement
    # instead of an unsigned magnitude, so a curve above zero means the
    # point moved to one side of its t=0 position, below zero the other.
    # All points/curves share a single sign direction -- the dominant
    # motion direction of the mean centerline -- so +/- means the same
    # physical direction everywhere in the plot, not just within one curve.
    samples = data if data_samples is None else data_samples

    mean_centerline = np.mean(data.centerlines, axis=0)
    direction = dominant_direction(mean_centerline - mean_centerline[0])

    disp_first_point = signed_reference_displacement(data.centerlines[0], direction)

    middle = len(data.centerlines) // 2
    disp_mean_point = signed_reference_displacement(data.centerlines[middle], direction)

    disp_last_point = signed_reference_displacement(data.centerlines[-1], direction)

    time = _time_axis(data.n_samples, period=period)

    disp_first_samples = signed_reference_displacement(samples.centerlines[0], direction)
    disp_mean_samples = signed_reference_displacement(samples.centerlines[len(samples.centerlines) // 2], direction)
    disp_last_samples = signed_reference_displacement(samples.centerlines[-1], direction)
    time_samples = _time_axis(samples.n_samples, period=period)

    plt.figure()
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(time, disp_first_point, color='orange')
    plt.scatter(time_samples, disp_first_samples, label='Samples first point', color='orange')
    plt.plot(time, disp_mean_point, color='blue')
    plt.scatter(time_samples, disp_mean_samples, label='Samples mean point', color='blue')
    plt.plot(time, disp_last_point, color='green')
    plt.scatter(time_samples, disp_last_samples, label='Samples last point', color='green')
    _plot_heartbeat_boundaries(data, period)
    _finalize_plot(ylabel=SIGNED_DISP_YLABEL, title=title, save_path=save_path)


def plot_pc_loop(data, period=1, title=None, save_path=None):
    # Checks how much of plot_signed_reference_disp's PC1-only assumption
    # holds: projects the mean centerline's displacement trajectory onto its
    # top-2 principal components and plots it as a loop over one cycle,
    # colored by time. If the motion really is ~1D along PC1, the loop
    # collapses to (near) a straight line; a fat/open loop means real motion
    # happens off-axis too, which the signed-displacement plot can't show.
    mean_centerline = np.mean(data.centerlines, axis=0)
    disp = mean_centerline - mean_centerline[0]
    centered = disp - disp.mean(axis=0)

    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pct_variance = 100 * s**2 / (s**2).sum()

    pc = centered @ vt[:2].T
    time = _time_axis(data.n_samples, period=period)

    plt.figure()
    plt.plot(pc[:, 0], pc[:, 1], color='gray', linewidth=1, alpha=0.5)
    points = plt.scatter(pc[:, 0], pc[:, 1], c=time, cmap='viridis', s=40, zorder=3)
    plt.scatter(pc[0, 0], pc[0, 1], color='black', marker='x', s=100, label='t=0', zorder=5)
    cbar = plt.colorbar(points)
    cbar.set_label('Time', fontsize=FONTSIZE_LABEL - 8)
    plt.xlabel(f'PC1 (mm, {pct_variance[0]:.0f}% variance)', fontsize=FONTSIZE_LABEL)
    plt.ylabel(f'PC2 (mm, {pct_variance[1]:.0f}% variance)', fontsize=FONTSIZE_LABEL)
    if title is not None:
        plt.title(title, fontsize=FONTSIZE_LABEL)
    plt.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    plt.axis('equal')
    plt.grid()
    plt.legend(fontsize=FONTSIZE_LEGEND)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_radius(data, data_samples=None, period=1, title=None):

    samples = data if data_samples is None else data_samples

    # plotted as change from t=0, like reference_displacement() for
    # centerlines: points at different vessel locations have very different
    # absolute radii, so a shared y-axis on absolute values would hide the
    # (much smaller) pulsation of each individual point.
    radius_first_point = data.radii[0] - data.radii[0, 0]

    middle = len(data.radii) // 2
    radius_mean_point = data.radii[middle] - data.radii[middle, 0]

    radius_last_point = data.radii[-1] - data.radii[-1, 0]

    time = _time_axis(data.n_samples, period=period)

    radius_first_samples = samples.radii[0] - samples.radii[0, 0]
    middle_samples = len(samples.radii) // 2
    radius_mean_samples = samples.radii[middle_samples] - samples.radii[middle_samples, 0]
    radius_last_samples = samples.radii[-1] - samples.radii[-1, 0]
    time_samples = _time_axis(samples.n_samples, period=period)

    plt.figure()
    plt.plot(time, radius_first_point, color='orange')
    plt.scatter(time_samples, radius_first_samples, label='Samples first point', color='orange')
    plt.plot(time, radius_mean_point, color='blue')
    plt.scatter(time_samples, radius_mean_samples, label='Samples mean point', color='blue')
    plt.plot(time, radius_last_point, color='green')
    plt.scatter(time_samples, radius_last_samples, label='Samples last point', color='green')
    _plot_heartbeat_boundaries(data, period)
    _finalize_plot(ylabel="Radius change from t=0 (mm)", title=title)


def plot_relative_motion(data, data_samples=None, period=1, title=None):

    samples = data if data_samples is None else data_samples

    middle = len(data.centerlines) // 2
    disp_rel_first_point = np.linalg.norm(data.centerlines[0] - data.centerlines[middle], axis=1)
    disp_rel_last_point = np.linalg.norm(data.centerlines[-1] - data.centerlines[middle], axis=1)

    time = _time_axis(data.n_samples, period=period)

    middle_samples = len(samples.centerlines) // 2
    disp_rel_first_samples = np.linalg.norm(samples.centerlines[0] - samples.centerlines[middle_samples], axis=1)
    disp_rel_last_samples = np.linalg.norm(samples.centerlines[-1] - samples.centerlines[middle_samples], axis=1)
    time_samples = _time_axis(samples.n_samples, period=period)

    plt.figure()
    plt.plot(time, disp_rel_first_point, color='orange')
    plt.scatter(time_samples, disp_rel_first_samples, label='Relative first point', color='orange')
    plt.plot(time, disp_rel_last_point, color='green')
    plt.scatter(time_samples, disp_rel_last_samples, label='Relative last point', color='green')
    _plot_heartbeat_boundaries(data, period)
    _finalize_plot(ylabel="Distance from centerpoint (mm)", title=title)


def plot_clean_vs_no_clean(data_clean, data_no_clean, title=None):

    time_clean = _time_axis(data_clean.n_samples)
    disp_mean_clean = reference_displacement(np.mean(data_clean.centerlines, axis=0))

    time_no_clean = _time_axis(data_no_clean.n_samples)
    disp_mean_original = reference_displacement(np.mean(data_no_clean.centerlines, axis=0))

    plt.figure()
    plt.plot(time_clean, disp_mean_clean, color='blue', label='Cleaned motion')
    plt.plot(time_no_clean, disp_mean_original, color='red', label='Original motion')
    _finalize_plot(ylabel=DISP_YLABEL, title=title)


def plot_sampled_vs_no_sampled(data_original, data_clean, data_sampled_clean, title=None):

    time_original = _time_axis(data_original.n_samples)
    disp_mean_original = reference_displacement(np.mean(data_original.centerlines, axis=0))

    time_clean = _time_axis(data_clean.n_samples)
    disp_mean_clean = reference_displacement(np.mean(data_clean.centerlines, axis=0))

    time_sampled_clean = _time_axis(data_sampled_clean.n_samples)
    disp_mean_sampled_clean = reference_displacement(np.mean(data_sampled_clean.centerlines, axis=0))

    plt.figure()
    plt.plot(time_original, disp_mean_original, color='blue', label='Original motion')
    plt.plot(time_clean, disp_mean_clean, color='green', label='Cleaned motion')
    plt.plot(time_sampled_clean, disp_mean_sampled_clean, color='red', label='Cleaned and sub-sampled motion')
    _finalize_plot(ylabel=DISP_YLABEL, title=title)


def plot_spline_vs_samples(data_sample, data_spline, fourier=True, title=None):

    time_for_sample = _time_axis(data_sample.n_samples, period=data_spline.period, endpoint=not fourier)

    mean_point_sample = np.mean(data_sample.centerlines, axis=0)
    disp_mean_sample = reference_displacement(mean_point_sample)

    time_for_spline = _time_axis(data_spline.interpolated_centerlines.shape[1], period=data_spline.period, endpoint=True)

    mean_point_spline = np.mean(data_spline.interpolated_centerlines, axis=0)
    disp_mean_spline = reference_displacement(mean_point_spline)

    plt.figure()
    plt.plot(time_for_spline, disp_mean_spline, color='blue', label='mean spline motion')
    plt.plot(time_for_sample, disp_mean_sample, color='red', label='mean sampled motion')
    plt.scatter(time_for_sample, disp_mean_sample, color='red', label='mean samples')
    _finalize_plot(ylabel=DISP_YLABEL, title=title)


def plot_spline_vs_fourier(data_sample, data_spline, data_fourier, title=None):

    time_for_sample = _time_axis(data_sample.n_samples, period=data_spline.period)
    disp_mean_sample = reference_displacement(np.mean(data_sample.centerlines, axis=0))

    time_for_spline = _time_axis(data_spline.interpolated_centerlines.shape[1], period=data_spline.period, endpoint=True)
    disp_mean_spline = reference_displacement(np.mean(data_spline.interpolated_centerlines, axis=0))

    time_for_fourier = _time_axis(data_fourier.interpolated_centerlines.shape[1], period=data_fourier.period, endpoint=True)
    disp_mean_fourier = reference_displacement(np.mean(data_fourier.interpolated_centerlines, axis=0))

    plt.figure()
    plt.scatter(time_for_sample, disp_mean_sample, color='gray', label='samples')
    plt.plot(time_for_spline, disp_mean_spline, color='blue', linewidth=2,
            label=f'spline (smoothing_factor={data_spline.smoothing_factor})')
    plt.plot(time_for_fourier, disp_mean_fourier, color='red', linewidth=2,
            label=f'fourier ({100*data_fourier.freq_cut/(data_fourier.Data.n_samples/2):.0f}% cut)')
    _finalize_plot(ylabel=DISP_YLABEL, title=title)


def plot_fourier_frequencies(data_sample, data_fourier: list, list_percetage_freq, title=None):

    period = data_fourier[0].period
    time_for_sample = _time_axis(data_sample.n_samples, period=period)
    mean_point_sample = np.mean(data_sample.centerlines, axis=0)
    disp_mean_sample = reference_displacement(mean_point_sample)

    plt.figure()
    plt.scatter(time_for_sample, disp_mean_sample, label='samples', s=200)

    for mydata, myfreq in zip(data_fourier, list_percetage_freq):

        mean_point_fourier = np.mean(mydata.interpolated_centerlines, axis=0)
        disp_mean_fourier = reference_displacement(mean_point_fourier)
        time_for_fourier = _time_axis(mydata.interpolated_centerlines.shape[1], period=mydata.period, endpoint=True)
        plt.plot(time_for_fourier, disp_mean_fourier, label=f'{myfreq}% cut', linewidth=2)

    _finalize_plot(ylabel=DISP_YLABEL, title=title)


