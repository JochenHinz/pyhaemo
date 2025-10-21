import numpy as np
from fourier import fourier_1D, fourier_3D

def load_data_frames():
    """
    load of the 66 frames of the reconstruction of the LAD.
    """

    recon = np.load('reconstructions-LAD.npy', allow_pickle=True)

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


def generate_fourier_interpolant(cen_4D: list, 
                                 rad_4D: list,
                                 timestep: float,
                                 freq_cut: int,
                                 plot=True):
    """
    from the 4D data, I generate the fourier interpolant
    """

    cleaned_4D_cen = []
    cleaned_4D_rad = []

    for cen, rad in zip(cen_4D, rad_4D):
        # fourier transform
        fourier_cen = fourier_3D(cen, timestep=timestep, freq_cut=freq_cut)
        fourier_rad = fourier_1D(rad, timestep=timestep, freq_cut=freq_cut)
        cleaned_4D_cen.append(fourier_cen)
        cleaned_4D_rad.append(fourier_rad)

    if plot:
        plot_fourier_interpolant(cen_4D, timestep=timestep, freq_cut_plot= [0, 3, 5, 7, 10])

    return cleaned_4D_cen, cleaned_4D_rad


def plot_fourier_interpolant(cen_4D, timestep, freq_cut_plot):
    import matplotlib.pyplot as plt
    plt.figure()

    for freq_cut in freq_cut_plot:
        cleaned_4D_cen = []
        for cen in cen_4D:
            # fourier transform
            fourier_cen = fourier_3D(cen, timestep=timestep, freq_cut=freq_cut)
            cleaned_4D_cen.append(fourier_cen)
            disp_form_ref = np.linalg.norm(cleaned_4D_cen[0][1:] - cleaned_4D_cen[0][0], axis=1)
            plt.plot(disp_form_ref, label=f'freq cut {freq_cut}', linewidth=10)
            break   # only the first point

    plt.title("Reference Displacement vs frequency cut", fontsize=28)
    plt.xlabel("Time", fontsize=28)
    plt.ylabel("Displacement from reference", fontsize=28)
    plt.tick_params(axis='both', labelsize=30)
    plt.legend(fontsize=28)
    plt.grid()
    plt.show()
