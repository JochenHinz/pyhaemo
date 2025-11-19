import h5py
import numpy as np
from mesh import vessel
import matplotlib.pyplot as plt
import find_intersection as fd


def create_centerline(cleaned_4D_cen, cleaned_4D_rad):
    """
    Samples should be equal to the number of timestep of the
    fluid dynamics simulations.
    """

    n_snapshot = len(cleaned_4D_cen[0])
    n_points = len(cleaned_4D_cen)

    physical_cen = []
    physical_rad = []
    for i in range(n_snapshot):
        centerlines = []
        radii = []
        for j in range(n_points):
            centerlines.append(cleaned_4D_cen[j][i])
            radii.append(cleaned_4D_rad[j][i])

        physical_cen.append(np.array(centerlines))
        physical_rad.append(np.array(radii))

    return physical_cen, physical_rad

def create_mesh(centerline: np.array,
                radii: np.array,
                filename: str,
                export=False):

    """
       Create a mesh for the given centerline and radii.
       The mesh is already scaled in meters, therefore is ready for LifeX
    """

    mesh = vessel(nelems_centerline=30,
                  nelems_cross_section=5,
                  nelems_z=301, 
                  radii=radii * 10**-3,
                  centerline_points=centerline * 10**-3,
                  normal=None,
                  export=export,
                  filename=filename)

    return mesh

def compute_scalings(until_, timestep):
    steps_to_smooth = int(until_/timestep)
    t0 = np.array([0,0,0])
    p0 = np.array([0,0,0])

    p1 = np.array([1,1,0])
    t1 = np.array([0,0,0])

    hermite_ramp = fd.cubic_hermite_curve(p0, t0, p1, t1)
    
    scalings = hermite_ramp(np.linspace(0, 1, steps_to_smooth))[:,0]
    
    return scalings


def compute_incremental_displacement(centerlines: list, 
                                     radii: list, 
                                     timestep:float,
                                     save_hdf5=True,
                                     ramp_disp_until=0.1):
    """
       Compute the incremental displacements between consecutive time steps.
       The incremental displacement at time step i is computed as the difference
       between the mesh at time step i and the mesh at time step i-1.
    """
    
    scalings = compute_scalings(until_=ramp_disp_until, timestep=timestep)
    print(scalings.shape[0])
    
    # reference mesh for lifex
    mesh_ref = create_mesh(radii=radii[0],
                           centerline=centerlines[0],
                           export=True,
                           filename=f'/home/marcinno/pyhaemo/fourier/meshes/vessel_mesh_cut_7_{0:05d}.msh')

    ref_disp = []
    for i in range(1,len(centerlines)):

        mesh_new = create_mesh(radii=radii[i],
                               centerline=centerlines[i],
                               export=True,
                               filename=f'/home/marcinno/pyhaemo/fourier/meshes/vessel_mesh_{i+1:05d}.msh')

        # If i want to scale the displacement, I have to take the displacement w.r.t. the reference mesh, not w.r.t the one before
        # otherwise this create a different motion.
        disp = (mesh_new.points - mesh_ref.points) * scalings[i] if i < scalings.shape[0] else (mesh_new.points - mesh_ref.points)
        
        ref_disp.append(disp)
        
        print(f'mesh n {i} / {len(centerlines)}')
        #if save_hdf5:
        #    filename = f"/home/marcinno/pyhaemo/fourier/displacement_1e-3/clean_vector_{i+1:05d}.hdf5"
        #    incremental_displacement.append(disp)
        #    with h5py.File(filename, 'w') as hf:
        #        hf.create_dataset('vector', data=incremental_displacement[-1].flatten())
        #        print(f"Saved vector to {filename}")

        #mesh_old.points = mesh_new.points + disp
    
    # compute the incremental displacement, mesh^(n+1)-mesh^(n)
    incremental_disp = [ref_disp[0]] + [ref_disp[i] - ref_disp[i - 1] for i in range(1, len(ref_disp))]
    
    save_to_hdf5(incremental_disp)
    
    print('Displacement is computed!')
    
    
    
def save_to_hdf5(displacement: list):

    print("Saving vectors to HDF5 files...")

    for i, vector in enumerate(displacement):
        vec = vector.flatten()
        filename = f"/home/marcinno/pyhaemo/fourier/displacement_1e-3/clean_vector_{i+1:05d}.hdf5"  # Generate filenames like vector_0000.h5, vector_0001.h5, etc.
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('vector', data=vec)
            print(f"Saved vector to {filename}")




# XXX: deprecated
def compute_reference_displacement(centerlines: list, radii: list, save_hdf5=True):
    """
       Compute the reference displacement between the first and the last time step.
       The reference displacement is computed as the difference between the mesh at
       the last time step and the mesh at the first time step.
    """

    ref_displacement = []

    # mesh for lifex
    ref_mesh = create_mesh(radii=radii[0],
                           centerline=centerlines[0],
                           export=True,
                           filename=f'/home/marcinno/pyhaemo/fourier/vessel_mesh_{0:05d}.msh')

    for i in range(len(centerlines)-1):
        mesh_new = create_mesh(radii=radii[-i],
                           centerline=centerlines[i],
                           export=False,
                           filename=f'/home/marcinno/pyhaemo/fourier/vessel_mesh_{i:05d}.vtk')

        ref_displacement = mesh_new.points - ref_mesh.points
        if save_hdf5:
            filename = f"/home/marcinno/pyhaemo/fourier/clean_vector_{i+1:05d}.hdf5"
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('vector', data=ref_displacement)
                print(f"Saved vector to {filename}")

    return ref_displacement
