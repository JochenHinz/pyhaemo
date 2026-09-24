"""
Time convergence study: full LAD motion, fixed spatial mesh resolution,
sweeping the reconstruction timestep dt.

This is a long-running batch job (dt=0.0005 alone reconstructs ~8000
timesteps). Run with:

    python3 run_time_convergence.py
"""
import paths as pt
from pipeline import load_full_motion, build_spline
from export import compute_incremental_displacement

VESSEL = 'CX'
SMOOTHING_FACTOR = 1

# fixed spatial resolution, matching export.create_mesh's default ('medium'
# in run_space_convergence.py).
NELEMS_CENTERLINE = 30
NELEMS_CROSS_SECTION = 5
NELEMS_Z = 301
BOUNDARY_LAYER_EXPONENT = 2  # 1. = uniform cross-section grid, no boundary layer

TIMESTEPS = [0.005, 0.001, 0.0005]


def main():
    sub_sampled = load_full_motion(VESSEL)

    for dt in TIMESTEPS:
        print(f'--- time convergence: dt={dt} ---')

        spline = build_spline(sub_sampled, timestep=dt, smoothing_factor=SMOOTHING_FACTOR)
        label = f'{VESSEL}/dt_{dt:.4f}'

        compute_incremental_displacement(
            spline.physical_centerline,
            spline.physical_radii,
            timestep=dt,
            meshes_dir=str(pt.OUTPUT_meshes_path / 'time_convergence' / label),
            displacement_dir=str(pt.OUTPUT_displacement_path / 'time_convergence' / label),
            nelems_centerline=NELEMS_CENTERLINE,
            nelems_cross_section=NELEMS_CROSS_SECTION,
            nelems_z=NELEMS_Z,
            boundary_layer_exponent=BOUNDARY_LAYER_EXPONENT,
        )


if __name__ == '__main__':
    main()
