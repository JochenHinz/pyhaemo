from pre_processing import load_data_frames, generate_4D_data, generate_fourier_interpolant
from post_processing import create_centerline, compute_incremental_displacement

def run(timestep:float, freq_cut:int):

    # loadthe 66 frames of the reconstruction of the LAD.
    centerlines, radii = load_data_frames()

    # create the 4D data
    cen_4D, rad_4D = generate_4D_data(centerlines, radii)

    # fourier transform of the 4D data
    cleaned_4D_cen, cleaned_4D_rad = generate_fourier_interpolant(cen_4D,
                                                                  rad_4D,
                                                                  timestep = timestep,
                                                                  freq_cut= freq_cut)

    # create the physical centerline and radii
    physical_cen, physical_rad = create_centerline(cleaned_4D_cen,
                                                   cleaned_4D_rad)

    # compute the incremental displacement to feed LifeX
    compute_incremental_displacement(physical_cen,
                                     physical_rad,
                                     timestep=timestep)



if __name__ == "__main__":

    run(timestep=0.00005, freq_cut=7)
