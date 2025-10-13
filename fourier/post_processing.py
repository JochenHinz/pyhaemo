import h5py
import numpy as np
from mesh import vessel
import matplotlib.pyplot as plt

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
                  radii=radii * 10**-3,
                  centerline_points=centerline * 10**-3,
                  normal=None,
                  export=export,
                  filename=filename)

    return mesh


def incremental_displacement(centerlines: list, radii: list):
    """
       Compute the incremental displacements between consecutive time steps.
       The incremental displacement at time step i is computed as the difference
       between the mesh at time step i and the mesh at time step i-1.
    """

    incremental_displacement = []

    # mesh for lifex
    ref_mesh = create_mesh(radii=radii[0],
                           centerline=centerlines[0],
                           export=True,
                           filename=f'/home/marcinno/pyhaemo/fourier/meshes/vessel_mesh_{0:05d}.msh')

    for i in range(len(centerlines)-1):

        mesh_old = create_mesh(radii=radii[i],
                               centerline=centerlines[i],
                               export=False,
                               filename=f'/home/marcinno/pyhaemo/fourier/meshes/vessel_mesh_{i:05d}.vtk')

        mesh_new = create_mesh(radii=radii[i+1],
                               centerline=centerlines[i+1],
                               export=False,
                               filename=f'/home/marcinno/pyhaemo/fourier/meshes/vessel_mesh_{i+1:05d}.vtk')


        filename = f"/home/marcinno/pyhaemo/fourier/displacement_4e-5/clean_vector_{i+1:05d}.hdf5"
        incremental_displacement.append(mesh_new.points - mesh_old.points)
	#filename = f"/home/marcinno/pyhaemo/fourier/displacement_5e-5/clean_vector_{i+1:05d}.hdf5" # Generate filenames like vector_0000.h5, vector_0001.h5, etc.
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('vector', data=incremental_displacement[-1])
            print(f"Saved vector to {filename}")

    return incremental_displacement


def save_to_hdf5(displacement: list):

    print("Saving vectors to HDF5 files...")

    for i, vector in enumerate(displacement):
        vec = vector.flatten()
        filename = f"/home/marcinno/pyhaemo/fourier/displacement_4e-5_last/clean_vector_{i+1:05d}.hdf5"  # Generate filenames like vector_0000.h5, vector_0001.h5, etc.
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('vector', data=vec)
            print(f"Saved vector to {filename}")
            
            
            
def plot_WSS():
    """
    This function read the WSS results from LifeX and plot them against the displacement
    """
    plot_pressure()
    wss_rigid = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/rigid_D-N/wss_rigid.csv', delimiter=',', skip_header=1)[:,1]
    wss_motion = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/real_motion_D-N/wss_motion.csv', delimiter=',', skip_header=1)[:,24]
    wss_3 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/validations_eddy_D-N/_10_3/wss_-3.csv', delimiter=',', skip_header=1)[:,24]
    wss_2 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/validations_eddy_D-N/_10_2/wss_-2.csv', delimiter=',', skip_header=1)[:,1]
    wss_1 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/validations_eddy_D-N/_10_1/wss_-1.csv', delimiter=',', skip_header=1)[:,1]
    wss_05 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/validations_eddy_D-N/_10_05/wss_-05.csv', delimiter=',', skip_header=1)[:,24]
    #disp = np.genfromtxt('/home/fabio/pyhaemo/examples/fourier/cleaned_centerline_x_first_point.csv', delimiter=',', skip_header=1)[0::2]
    disp = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/displ.csv', delimiter=',', skip_header=1)[:,12]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    time = np.linspace(0,0.8,800)[0::2]
    ax1.plot(time, wss_motion, label='WSS Motion',linewidth=7)
    ax1.plot(time,wss_05,label='WSS -05',linewidth=7)
    #ax1.plot(time,wss_1,label='WSS -1',linewidth=7)
    ax1.plot(time,wss_2,label='WSS -2',linewidth=7)
    #ax1.plot(time,wss_3, label='WSS -3',linewidth=7)

    ax1.plot(time,wss_rigid,label='WSS Rigid',linewidth=7)
   
    ax1.set_xlabel('time(s)',fontsize=28)
    ax1.set_ylabel('WSS Value',fontsize=28)
    plt.tick_params(axis='both', labelsize=30)  

    ax1.legend(loc='upper left', fontsize=28)
    ax1.grid(True)

    ax2 = ax1.twinx()
    if disp.ndim == 1:
        ax2.plot(time,disp, color='black', label='Displacement x', linewidth=7)
    else:
        for col in range(disp.shape[1]):
            ax2.plot(disp[:, col], label=f'Displacement', linestyle='--')
    ax2.set_ylabel('Displacement',fontsize=28)
    ax2.legend(loc='upper right',fontsize=28)

    plt.title('Wall Shear Stress (left) and Displacement (right)', fontsize=28)
    plt.tight_layout()
    plt.show()
    

def plot_pressure():
    
    #p_rigid = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/rigid_D-N/pressure_rigid.csv', delimiter=',', skip_header=1)[:,5]
    p_rigid_5_e_4 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/pressure_rigid_5e-4.txt', delimiter=',', skip_header=1)[:,5]
    p_motion_e_3 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/real_motion_D-N/pressure_motion.csv', delimiter=',', skip_header=1)[:,9]
    p_motion_5_e_4 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/5_e-4_moving/pressure_5e-4.csv', delimiter=',', skip_header=1)[:,9]
    p_motion_e_4 = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/1_e-4_moving/pressure_1e-4.csv', delimiter=',', skip_header=1)[:,9]
    p = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/1_e-4_moving_gmres_-14/pressure.csv', delimiter=',', skip_header=1)[:,9]
    disp = np.genfromtxt('/home/fabio/lifex-dev/newbuild/tests/fluid_dynamics_cylinder/displ.csv', delimiter=',', skip_header=1)[:,12]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    time_800 = np.linspace(0,0.8,800)
    time_8000 = np.linspace(0,0.8,8000)
    time_1600 = np.linspace(0,0.8,1602)
    
    ax1.plot(time_800, p_motion_e_3, label='Pressure motion 1e-3',linewidth=7)
    ax1.plot(time_1600,p_motion_5_e_4,label='Pressure motion 5e-4',linewidth=7)
    ax1.plot(time_8000,p_motion_e_4,label='Pressure motion 1e-4',linewidth=7)
    #ax1.plot(time_800, p_rigid[0:800],label='Pressure No-motion 1e-3',linewidth=7)
    ax1.plot(time_1600, p_rigid_5_e_4,label='Pressure No-motion 5e-4',linewidth=7)
    ax1.plot(time_8000, p[0:8000],label='Pressure No-motion 1e-4',linewidth=7)
    ax1.set_xlabel('time(s)',fontsize=28)
    ax1.set_ylabel('Pressure (Pa)',fontsize=28)
    plt.tick_params(axis='both', labelsize=30)  

    ax1.legend(loc='upper left', fontsize=28)
    ax1.grid(True)

    time_400 = np.linspace(0,0.8,800)[0::2]
    ax2 = ax1.twinx()
    if disp.ndim == 1:
        ax2.plot(time_400,disp, color='black', label='Displacement x', linewidth=10)
    else:
        for col in range(disp.shape[1]):
            ax2.plot(disp[:, col], label=f'Displacement', linestyle='--')
    ax2.set_ylabel('Displacement',fontsize=28)
    ax2.legend(loc='upper right',fontsize=28)

    plt.title('Pressure (left) and Displacement (right)', fontsize=28)
    plt.tight_layout()
    plt.show()

    
    
