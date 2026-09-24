from pathlib import Path

import h5py
import numpy as np
from mesh import vessel
import find_intersection as fd
import paths as pt


def create_mesh(centerline: np.array,
                radii: np.array,
                filename: str,
                export=False,
                nelems_centerline=30,
                nelems_cross_section=5,
                nelems_z=301,
                boundary_layer_exponent=1.):

    """
       Create a mesh for the given centerline and radii.
       The mesh is already scaled in meters, therefore is ready for LifeX
    """

    mesh = vessel(nelems_centerline=nelems_centerline,
                  nelems_cross_section=nelems_cross_section,
                  nelems_z=nelems_z,
                  radii=radii * 10**-3,
                  centerline_points=centerline * 10**-3,
                  normal=None,
                  export=export,
                  filename=filename,
                  boundary_layer_exponent=boundary_layer_exponent)

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

def compute_incremental_displacement(centerlines: np.array,
                                     radii: np.array,
                                     timestep: float,
                                     ramp_disp_until=0.1,
                                     meshes_dir=None,
                                     displacement_dir=None,
                                     nelems_centerline=30,
                                     nelems_cross_section=5,
                                     nelems_z=301,
                                     boundary_layer_exponent=1.):
    """
       Compute the incremental displacements between consecutive time steps.
       The incremental displacement at time step i is computed as the difference
       between the mesh at time step i and the mesh at time step i-1.

       meshes_dir/displacement_dir default to paths.OUTPUT_meshes_path and
       paths.OUTPUT_displacement_path; pass a dedicated subdirectory per call
       (e.g. per motion type) to avoid different exports overwriting each
       other's files. nelems_centerline/nelems_cross_section/nelems_z control
       the spatial mesh resolution (see mesh.vessel), for space convergence
       studies. boundary_layer_exponent controls the cross-section boundary
       layer clustering (see mesh.vessel / pyhaemo.geo.ellipse); 1. = uniform.
    """

    meshes_dir = pt.OUTPUT_meshes_path if meshes_dir is None else Path(meshes_dir)
    displacement_dir = pt.OUTPUT_displacement_path if displacement_dir is None else Path(displacement_dir)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    displacement_dir.mkdir(parents=True, exist_ok=True)

    scalings = compute_scalings(until_=ramp_disp_until, timestep=timestep)
    # reference mesh for lifex
    mesh_ref = create_mesh(radii=radii[0],
                           centerline=centerlines[0],
                           export=True,
                           filename=str(meshes_dir)
                           + f'/vessel_mesh_separated_{0:05d}.msh',
                           nelems_centerline=nelems_centerline,
                           nelems_cross_section=nelems_cross_section,
                           nelems_z=nelems_z,
                           boundary_layer_exponent=boundary_layer_exponent)

    print('Starting computing the incremental displacement...')

    ref_disp = []
    for i in range(1,len(centerlines)):

        mesh_new = create_mesh(radii=radii[i],
                               centerline=centerlines[i],
                               export=True,
                               filename=str(meshes_dir)
                               + f'/vessel_mesh_{i:05d}.vtk',
                               nelems_centerline=nelems_centerline,
                               nelems_cross_section=nelems_cross_section,
                               nelems_z=nelems_z,
                               boundary_layer_exponent=boundary_layer_exponent)

        # If i want to scale the displacement, I have to take the displacement w.r.t. the reference mesh, not w.r.t the one before
        # otherwise this create a different motion.
        disp = (mesh_new.points - mesh_ref.points) * scalings[i] \
            if i < scalings.shape[0] else (mesh_new.points - mesh_ref.points)

        ref_disp.append(disp)

        print(f'mesh n {i} / {len(centerlines)}')

    # compute the incremental displacement, mesh^(n+1)-mesh^(n)
    incremental_disp = [ref_disp[0]] + [ref_disp[i] - ref_disp[i - 1] for i in range(1, len(ref_disp))]

    save_to_hdf5(incremental_disp, output_dir=displacement_dir)

    print('Done computing the incremental displacement')


def save_to_hdf5(displacement: list, output_dir=None):

    output_dir = pt.OUTPUT_displacement_path if output_dir is None else Path(output_dir)

    print("Saving vectors to HDF5 files...")

    for i, vector in enumerate(displacement):
        vec = vector.flatten()
        # Generate filenames like vector_0000.h5, vector_0001.h5, etc.
        filename = str(output_dir) + f'/clean_vector_{i+1:05d}.hdf5'
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('vector', data=vec)
            print(f"Saved vector to {filename}")
