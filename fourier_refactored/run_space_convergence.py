"""
Space convergence study: full LAD motion, fixed timestep, sweeping the
spatial mesh resolution.

Each resolution level is specified as a single target element size dh (mm),
not as a raw (nelems_cross_section, nelems_z) tuple: vessel length varies a
lot between LAD/CX/RCA (see resolution_for_dh below), so a fixed nelems_z
means a different physical element size for each vessel. dh is converted
into (nelems_cross_section, nelems_z) from the vessel's own geometry, so the
same DH_LEVELS give comparable resolution regardless of which VESSEL is set.

nelems_centerline is held FIXED across the sweep (see NELEMS_CENTERLINE
below): it sets the knot vector that the centerline spline is *fit* to
(mesh.vessel -> UnivariateKnotVector), so changing it changes the fitted
geometry itself, not just the numerical mesh resolution. Sweeping it would
confound geometric change with numerical convergence.

This is a long-running batch job (each resolution level generates ~4000
meshes at TIMESTEP=0.001s over the reconstructed heartbeats). Run with:

    python3 run_space_convergence.py

Before committing to that, check report_hmin() (bottom of this file): it
builds just one mesh per resolution level (cheap) and prints h_min and mesh
validity, e.g. to pick a CFL-safe solver timestep.
"""
import numpy as np

import paths as pt
from pipeline import load_full_motion, build_spline, build_fourier
from export import compute_incremental_displacement, create_mesh
from mesh import min_edge_length

VESSEL = 'CX'
TIMESTEP = 0.001
SMOOTHING_FACTOR = 1
BOUNDARY_LAYER_EXPONENT = 1.8 # 1. = uniform cross-section grid, no boundary layer

# fixed across every resolution level -- see module docstring.
NELEMS_CENTERLINE = 50

# target average element size (mm) per resolution level -- converted to
# (nelems_cross_section, nelems_z) by resolution_for_dh(), from the actual
# vessel geometry (VESSEL above).
DH_LEVELS = {
    'coarse': 0.4,
    'medium': 0.2,  # current default resolution
    'fine': 0.1,
}


def vessel_length(centerline):
    """ Physical arc length (mm) of a single frame's centerline. """
    return float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())


def resolution_for_dh(dh, centerline, radii):
    """
    (nelems_cross_section, nelems_z) giving ~dh (mm) average element size,
    from one representative frame's centerline/radii: nelems_z from the
    vessel's physical arc length, nelems_cross_section from its mean radius.

    This is only an *average* target: mesh.vessel scales the cross-section
    locally by each frame's own radii (so the true local element size still
    varies along the vessel and over time), and boundary_layer_exponent
    clusters points toward the boundary rather than spacing them uniformly.

    nelems_cross_section is floored at 3: empirically (checked against this
    pipeline's mesh), 1 produces a degenerate mesh (HasNoBoundaryError) and
    2 is valid but has essentially no margin, so a small dh/large-radius
    combination doesn't silently hand back a degenerate cross-section.
    """
    length = vessel_length(centerline)
    mean_radius = float(radii.mean())

    nelems_cross_section = max(3, round(mean_radius / dh))
    nelems_z = max(1, round(length / dh))

    return nelems_cross_section, nelems_z


def main():
    sub_sampled = load_full_motion(VESSEL)
    spline = build_fourier(sub_sampled, timestep=TIMESTEP, frequency_cutoff_percentage=40)

    centerlines = spline.physical_centerline
    radii = spline.physical_radii

    for label, dh in DH_LEVELS.items():
        nelems_cross_section, nelems_z = resolution_for_dh(dh, centerlines[0], radii[0])

        print(f'--- space convergence: {label} (dh={dh}mm -> '
             f'nelems_centerline={NELEMS_CENTERLINE}, '
             f'nelems_cross_section={nelems_cross_section}, '
             f'nelems_z={nelems_z}) ---')

        out_label = f'{VESSEL}/{label}'
        compute_incremental_displacement(
            centerlines,
            radii,
            timestep=TIMESTEP,
            meshes_dir=str(pt.OUTPUT_meshes_path / 'space_convergence' / out_label),
            displacement_dir=str(pt.OUTPUT_displacement_path / 'space_convergence' / out_label),
            nelems_centerline=NELEMS_CENTERLINE,
            nelems_cross_section=nelems_cross_section,
            nelems_z=nelems_z,
            boundary_layer_exponent=BOUNDARY_LAYER_EXPONENT,
        )


def report_hmin():
    """
    Cheaply report h_min (minimum mesh edge length, in meters) and mesh
    validity for every resolution level in DH_LEVELS, from a single mesh
    (the first frame) instead of running the full compute_incremental_
    displacement export over every frame -- useful e.g. to pick a CFL-safe
    solver timestep before running the expensive full study.
    """
    sub_sampled = load_full_motion(VESSEL)
    spline = build_fourier(sub_sampled, timestep=TIMESTEP, frequency_cutoff_percentage=40)

    centerline = spline.physical_centerline[0]
    radii = spline.physical_radii[0]

    for label, dh in DH_LEVELS.items():
        nelems_cross_section, nelems_z = resolution_for_dh(dh, centerline, radii)

        mesh = create_mesh(
            centerline=centerline,
            radii=radii,
            filename='',
            export=False,
            nelems_centerline=NELEMS_CENTERLINE,
            nelems_cross_section=nelems_cross_section,
            nelems_z=nelems_z,
            boundary_layer_exponent=BOUNDARY_LAYER_EXPONENT,
        )

        valid = mesh.is_valid()
        h_min = min_edge_length(mesh) if valid else float('nan')

        print(f'{label:>8}  dh={dh:>5}mm  nelems_cross_section={nelems_cross_section:<3} nelems_z={nelems_z:<4} '
             f'valid={valid!s:<5} h_min={h_min:.3e} m')


if __name__ == '__main__':
    main()
