"""
Space convergence study: full CX motion, fixed timestep, sweeping the
spatial mesh resolution.

The motion is the smoothing spline (SMOOTHING_FACTOR) of the reconstruction,
evaluated at exactly t = k * TIMESTEP up to T_END (pipeline.spline_motion),
so the same instants can be reused by the time convergence study.

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

This is a long-running batch job (each resolution level generates
T_END / TIMESTEP + 1 meshes). Run every level, or only the given ones (e.g.
one level per cluster job), with:

    python3 run_space_convergence.py [coarse] [medium] [fine]

Each level first checks mesh validity at a few instants (check_validity) and
stops before the long export if any mesh is invalid. report_hmin() (bottom of
this file) builds just one mesh per resolution level (cheap) and prints h_min
and mesh validity, e.g. to pick a CFL-safe solver timestep.
"""
import sys
import time

import numpy as np

import paths as pt
from pipeline import load_full_motion, spline_motion
from export import compute_incremental_displacement, create_mesh
from mesh import min_edge_length

VESSEL = 'CX'
TIMESTEP = 0.001
T_END = 1.0  # seconds of motion to export, from t=0
SMOOTHING_FACTOR = 1
BOUNDARY_LAYER_EXPONENT = 2.  # 1. = uniform cross-section grid, no boundary layer

# fixed across every resolution level -- see module docstring.
NELEMS_CENTERLINE = 50

# target average element size (mm) per resolution level -- converted to
# (nelems_cross_section, nelems_z) by resolution_for_dh(), from the actual
# vessel geometry (VESSEL above).
DH_LEVELS = {
    'coarse': 0.3,
    'medium': 0.15,
    'fine': 0.075,
}

# instants (s) meshed by check_validity() before each level's full export.
CHECK_TIMES = np.linspace(0, T_END, 11)


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


def check_validity(centerlines, radii, nelems_cross_section, nelems_z):
    """
    Build (without exporting) the mesh at each of CHECK_TIMES and raise if
    any is invalid, so an invalid configuration fails in minutes instead of
    after the whole export. Also prints the time per mesh.
    """
    for t in CHECK_TIMES:
        k = int(round(t / TIMESTEP))
        start = time.time()
        mesh = create_mesh(centerline=centerlines[k],
                           radii=radii[k],
                           filename='',
                           export=False,
                           nelems_centerline=NELEMS_CENTERLINE,
                           nelems_cross_section=nelems_cross_section,
                           nelems_z=nelems_z,
                           boundary_layer_exponent=BOUNDARY_LAYER_EXPONENT)
        valid = mesh.is_valid()
        print(f'check t={t:.2f}s (step {k}): valid={valid}, '
              f'{len(mesh.elements)} elements, built in {time.time() - start:.1f}s', flush=True)
        if not valid:
            raise RuntimeError(f'invalid mesh at t={t:.2f}s (step {k}), not exporting')


def main(levels):
    sub_sampled = load_full_motion(VESSEL)
    centerlines, radii = spline_motion(sub_sampled, TIMESTEP, t_end=T_END,
                                       smoothing_factor=SMOOTHING_FACTOR)

    for label in levels:
        dh = DH_LEVELS[label]
        nelems_cross_section, nelems_z = resolution_for_dh(dh, centerlines[0], radii[0])

        print(f'--- space convergence: {label} (dh={dh}mm -> '
             f'nelems_centerline={NELEMS_CENTERLINE}, '
             f'nelems_cross_section={nelems_cross_section}, '
             f'nelems_z={nelems_z}, '
             f'{len(centerlines)} steps of {TIMESTEP}s) ---', flush=True)

        check_validity(centerlines, radii, nelems_cross_section, nelems_z)

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
    centerlines, all_radii = spline_motion(sub_sampled, TIMESTEP, t_end=T_END,
                                           smoothing_factor=SMOOTHING_FACTOR)

    centerline = centerlines[0]
    radii = all_radii[0]

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
    levels = sys.argv[1:] or list(DH_LEVELS)
    unknown = [label for label in levels if label not in DH_LEVELS]
    if unknown:
        raise SystemExit(f'unknown level(s) {unknown}, expected any of {list(DH_LEVELS)}')
    main(levels)
