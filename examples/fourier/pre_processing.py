import numpy as np
from fourier import fourier_1D, fourier_3D, plot_fourier

def load_data_frames():
    """
    load of the 66 frames of the reconstruction of the LAD.
    """

    recon = np.load('/home/fabio/reconstructions-LAD.npy', allow_pickle=True)

    centerlines = []
    radii = []
    for i in range(len(recon)):
        centerlines.append(recon[i]['points3D']['centreLine'])
        radii.append(recon[i]['points3D']['radius'])

    # suppose that the 66 frames are equispaced in time.
    # The 66 frames cover 3 hearbeat, so 22 for each hearbeat.
    # we consider the first hearbeat
    centerlines = centerlines[:22]
    radii = radii[:22]

    return centerlines, radii


def generate_4D_data(centerlines, radii):
    """
    from the centerlines and radii frame, I generate the 4D data
    """

    n_snapshot = len(centerlines)
    n_points = len(centerlines[0])

    cen_4D = []
    rad_4D = []
    for i in range(n_points):
        centerlines_time = []
        radii_time = []
        # for each point in the centerline, I collect its position at each time
        for j in range(n_snapshot):
            centerlines_time.append(centerlines[j][i])
            radii_time.append(radii[j][i])

        cen_4D.append(np.array(centerlines_time))
        rad_4D.append(np.array(radii_time))

    return cen_4D, rad_4D


def generate_fourier_interpolant(cen_4D, rad_4D, freq_cut=0, plot=False, save_csv=False):
    """
    from the 4D data, I generate the fourier interpolant
    """

    cleaned_4D_cen = []
    cleaned_4D_rad = []

    for cen, rad in zip(cen_4D, rad_4D):
        # fourier transform
        fourier_cen = fourier_3D(cen, freq_cut)
        fourier_rad = fourier_1D(rad, freq_cut)
        cleaned_4D_cen.append(fourier_cen)
        cleaned_4D_rad.append(fourier_rad)

    # let's plot how cutting frequencies affects the motion of the x-coordinate
    # of the first point of the centerline
    if plot:
        freqs_to_cut = [0]
        original_signal = cen_4D[0][:,0]
        cleaned_signals = [fourier_1D(original_signal, fc) for fc in freqs_to_cut]
        plot_fourier(original_signal, cleaned_signals, freqs_to_cut)

    if save_csv:
            np.savetxt(f"cleaned_centerline_x_first_point.csv", cleaned_signals[0], delimiter=",")
            print(f"saved cleaned_centerline_x_first_point.csv")

    return cleaned_4D_cen, cleaned_4D_rad
