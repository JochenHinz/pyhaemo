from pyhaemo.geo import ellipse, compute_notwistframe_from_spline
from pyhaemo.geo.disc import (
    PATCHES,
    _exterior_undirected_edges,
    _patch_side_edges,
    _scalar_expansion,
    _apply_discrete_bl_expansion,
    _get_side_line,
    _set_side_line,
    _orient_line_like,
)
from pyhaemo.mesh import rectilinear, mesh_union
from pyhaemo.spl import UnivariateKnotVector
from pyhaemo.topo import Topology
from pyhaemo.util import np, _, clparam

import os
import unittest

_SKIP_VISUAL = bool(os.environ.get('PYHAEMO_SKIP_VISUAL'))


class TestDisc(unittest.TestCase):

  def test_eval(self):
    sol = ellipse(4, 4, 4)

    for n in (3, 5, 7, 11, 13, 17, 19):
      with self.subTest('Evaluating in {n} x {n} points'):
        rmesh = rectilinear([n, n])

        mesh = mesh_union(*sol.sample_mesh(rmesh), boundary=True)

        # 5 faces, 12 edges, 8 vertices
        npoints = 5 * (n-2)**2 + 12 * (n-2) + 8

        self.assertEqual(len(mesh.points), npoints)


# ---------------------------------------------------------------------------
# _scalar_expansion: one-dimensional expansion r ↦ 1 − (1 − r)^exponent on [0, 1]
#                    r ↦ r                     for r > 1  (identity)
# ---------------------------------------------------------------------------

class TestScalarExpansion(unittest.TestCase):

  def test_identity_at_exponent_1(self):
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
      with self.subTest(r=r):
        self.assertAlmostEqual(_scalar_expansion(r, 1), r, places=12)

  def test_zero_is_fixed_point(self):
    for exp in (1, 2, 3, 10):
      with self.subTest(exp=exp):
        self.assertAlmostEqual(_scalar_expansion(0.0, exp), 0.0, places=12)

  def test_one_is_fixed_point(self):
    for exp in (1, 2, 3, 10):
      with self.subTest(exp=exp):
        self.assertAlmostEqual(_scalar_expansion(1.0, exp), 1.0, places=12)

  def test_pushes_interior_toward_one_for_exp_gt_1(self):
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
      for exp in (2, 3, 10):
        with self.subTest(r=r, exp=exp):
          self.assertGreater(_scalar_expansion(r, exp), r)

  def test_known_values(self):
    # 1 - (1 - 0.5)^2 = 0.75
    self.assertAlmostEqual(_scalar_expansion(0.5, 2), 0.75, places=12)
    # 1 - (1 - 0.5)^3 = 0.875
    self.assertAlmostEqual(_scalar_expansion(0.5, 3), 0.875, places=12)

  def test_identity_above_one(self):
    # Identity branch for r > 1 (no clamp-at-1): required so that
    # slightly-overshooting outer control points (|p| > 1, a standard
    # cubic-spline artefact when approximating a circle) are left in
    # place instead of being pulled inward along non-radial chords.
    for r, exp in ((1.5, 3), (2.0, 5), (1.001, 4)):
      with self.subTest(r=r, exp=exp):
        self.assertAlmostEqual(_scalar_expansion(r, exp), r, places=12)

  def test_monotone_increasing_in_r(self):
    rs = np.linspace(0.0, 1.0, 20)
    expanded = [_scalar_expansion(r, 4) for r in rs]
    self.assertTrue(all(a <= b for a, b in zip(expanded, expanded[1:])))

  def test_larger_exponent_gives_larger_output_for_fixed_interior_r(self):
    r = 0.5
    prev = _scalar_expansion(r, 1)
    for exp in (2, 3, 5, 10):
      curr = _scalar_expansion(r, exp)
      self.assertGreaterEqual(curr, prev)
      prev = curr


# ---------------------------------------------------------------------------
# _patch_side_edges
# ---------------------------------------------------------------------------

class TestPatchSideEdges(unittest.TestCase):

  def test_all_four_sides_present_for_every_patch(self):
    for patch in PATCHES:
      edges = _patch_side_edges(patch)
      self.assertSetEqual(set(edges.keys()), {'left', 'right', 'bottom', 'top'})

  def test_each_edge_is_a_sorted_pair(self):
    for patch in PATCHES:
      for side, edge in _patch_side_edges(patch).items():
        with self.subTest(patch=patch, side=side):
          self.assertEqual(len(edge), 2)
          self.assertLessEqual(edge[0], edge[1])

  def test_known_center_patch_edges(self):
    # PATCHES[2] = (2, 3, 4, 5) — center patch
    edges = _patch_side_edges(PATCHES[2])
    self.assertEqual(edges['left'],   (2, 3))
    self.assertEqual(edges['right'],  (4, 5))
    self.assertEqual(edges['bottom'], (2, 4))
    self.assertEqual(edges['top'],    (3, 5))


# ---------------------------------------------------------------------------
# _exterior_undirected_edges
# ---------------------------------------------------------------------------

class TestExteriorEdges(unittest.TestCase):

  def setUp(self):
    self.topo = Topology(PATCHES)
    self.exterior = _exterior_undirected_edges(self.topo)

  def test_disc_topology_has_exactly_four_exterior_edges(self):
    self.assertEqual(len(self.exterior), 4)

  def test_known_exterior_edges_are_outer_square_boundary(self):
    # Outer corner vertices: 0, 1, 6, 7.
    expected = {(0, 1), (0, 6), (1, 7), (6, 7)}
    self.assertSetEqual(self.exterior, expected)

  def test_center_patch_shares_no_exterior_edges(self):
    center_edges = set(_patch_side_edges(PATCHES[2]).values())
    self.assertTrue(center_edges.isdisjoint(self.exterior))

  def test_each_outer_patch_has_exactly_one_exterior_side(self):
    for i, patch in enumerate(PATCHES):
      outer = [s for s, e in _patch_side_edges(patch).items()
               if e in self.exterior]
      if i == 2:   # center patch
        self.assertEqual(len(outer), 0,
                         msg=f'patch {i} (center) should have 0 exterior sides')
      else:
        self.assertEqual(len(outer), 1,
                         msg=f'patch {i} should have exactly 1 exterior side')


# ---------------------------------------------------------------------------
# _get_side_line / _set_side_line
# ---------------------------------------------------------------------------

class TestSideLineOps(unittest.TestCase):

  def _make_cps(self, nxi=4, neta=5):
    rng = np.random.default_rng(42)
    return rng.standard_normal((nxi, neta, 2))

  def test_get_left_returns_first_i_slice(self):
    cps = self._make_cps()
    np.testing.assert_array_equal(_get_side_line(cps, 'left'), cps[0, :, :])

  def test_get_right_returns_last_i_slice(self):
    cps = self._make_cps()
    np.testing.assert_array_equal(_get_side_line(cps, 'right'), cps[-1, :, :])

  def test_get_bottom_returns_first_j_slice(self):
    cps = self._make_cps()
    np.testing.assert_array_equal(_get_side_line(cps, 'bottom'), cps[:, 0, :])

  def test_get_top_returns_last_j_slice(self):
    cps = self._make_cps()
    np.testing.assert_array_equal(_get_side_line(cps, 'top'), cps[:, -1, :])

  def test_get_returns_a_copy(self):
    cps = self._make_cps()
    line = _get_side_line(cps, 'left')
    line[:] = 9999.0
    self.assertFalse(np.any(cps == 9999.0))

  def test_set_get_roundtrip_for_all_sides(self):
    cps = self._make_cps(nxi=3, neta=4)
    for side in ('left', 'right', 'bottom', 'top'):
      with self.subTest(side=side):
        new_line = np.full_like(_get_side_line(cps, side), fill_value=7.0)
        _set_side_line(cps, side, new_line)
        np.testing.assert_array_almost_equal(_get_side_line(cps, side), new_line)


# ---------------------------------------------------------------------------
# _orient_line_like
# ---------------------------------------------------------------------------

class TestOrientLineLike(unittest.TestCase):

  def setUp(self):
    self.ref = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

  def test_same_orientation_is_returned_unchanged(self):
    line = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = _orient_line_like(line, self.ref)
    np.testing.assert_array_equal(result, line)

  def test_reversed_orientation_is_flipped(self):
    line = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    result = _orient_line_like(line, self.ref)
    np.testing.assert_array_equal(result, self.ref)

  def test_slightly_perturbed_same_orientation_not_flipped(self):
    line = self.ref + 1e-8
    result = _orient_line_like(line, self.ref)
    np.testing.assert_array_almost_equal(result[0], line[0])

  def test_single_point_line(self):
    ref_1pt  = np.array([[0.5, 0.5]])
    line_1pt = np.array([[0.5, 0.5]])
    result = _orient_line_like(line_1pt, ref_1pt)
    np.testing.assert_array_equal(result, line_1pt)


# ---------------------------------------------------------------------------
# _apply_discrete_bl_expansion
# ---------------------------------------------------------------------------

class TestApplyDiscreteBLExpansion(unittest.TestCase):
  """Tests for the per-patch discrete boundary-layer expansion step."""

  def _radial_fiber_patch(self, angles, n):
    """cps (nxi, n, 2): each row i is a radial fiber from origin to unit circle."""
    nxi = len(angles)
    cps = np.zeros((nxi, n, 2))
    for i, theta in enumerate(angles):
      for j in range(n):
        t = j / (n - 1)
        cps[i, j, :] = [t * np.cos(theta), t * np.sin(theta)]
    return cps

  def test_exponent_1_is_identity_on_radial_fibers(self):
    """For exponent=1 and radial fibers from the origin, positions do not move."""
    cps = self._radial_fiber_patch([0.0, np.pi / 4, np.pi / 2], n=5)
    original = np.array(cps, copy=True)
    _apply_discrete_bl_expansion(cps, 'top', 1)
    np.testing.assert_array_almost_equal(cps, original, decimal=10)

  def test_origin_inner_endpoint_is_a_fixed_point(self):
    """Fiber from the origin: the inner endpoint (r=0) is not moved."""
    n = 5
    cps = np.zeros((1, n, 2))
    for j in range(n):
      cps[0, j, :] = [j / (n - 1), 0.0]
    for exp in (2, 3, 5):
      with self.subTest(exp=exp):
        cp = np.array(cps, copy=True)
        _apply_discrete_bl_expansion(cp, 'top', exp)
        np.testing.assert_array_almost_equal(cp[0, 0, :], [0.0, 0.0], decimal=10)

  def test_unit_circle_outer_endpoint_is_a_fixed_point(self):
    """Fiber to the unit circle: the outer endpoint (r=1) is not moved."""
    n = 5
    cps = np.zeros((1, n, 2))
    for j in range(n):
      cps[0, j, :] = [j / (n - 1), 0.0]
    for exp in (2, 3, 5):
      with self.subTest(exp=exp):
        cp = np.array(cps, copy=True)
        _apply_discrete_bl_expansion(cp, 'top', exp)
        np.testing.assert_array_almost_equal(cp[0, -1, :], [1.0, 0.0], decimal=10)

  def test_exponent_gt_1_increases_interior_point_radii(self):
    """For exponent > 1, interior radii must be strictly larger after expansion."""
    n = 5
    cps = np.zeros((1, n, 2))
    for j in range(n):
      cps[0, j, :] = [j / (n - 1), 0.0]
    for exp in (2, 3, 5):
      with self.subTest(exp=exp):
        cp = np.array(cps, copy=True)
        _apply_discrete_bl_expansion(cp, 'top', exp)
        for j in range(1, n - 1):
          r_before = float(np.hypot(*cps[0, j, :]))
          r_after  = float(np.hypot(*cp[0, j, :]))
          self.assertGreater(r_after, r_before,
                             msg=f'exp={exp}, j={j}: radius did not increase')

  def test_larger_exponent_pushes_points_further_toward_boundary(self):
    """Larger exponent should yield a larger mean radius."""
    n = 7
    cps_base = np.zeros((1, n, 2))
    for j in range(n):
      cps_base[0, j, :] = [j / (n - 1), 0.0]
    means = []
    for exp in (2, 5, 10):
      cp = np.array(cps_base, copy=True)
      _apply_discrete_bl_expansion(cp, 'top', exp)
      means.append(np.mean(np.hypot(cp[:, :, 0], cp[:, :, 1])))
    self.assertLess(means[0], means[1])
    self.assertLess(means[1], means[2])

  def test_expanded_points_remain_on_straight_fiber(self):
    """After expansion every point must lie on the straight fiber (chord i_new–e)."""
    # Diagonal fiber (45°) makes the cross-product test non-trivial.
    n = 6
    cps = np.zeros((1, n, 2))
    for j in range(n):
      t = j / (n - 1)
      cps[0, j, :] = [t / np.sqrt(2), t / np.sqrt(2)]
    s_ref = np.array(cps[0, 0, :])
    e_ref = np.array(cps[0, -1, :])
    _apply_discrete_bl_expansion(cps, 'top', 3)
    chord = e_ref - s_ref
    for j in range(n):
      p = cps[0, j, :]
      cross = (p[0] - s_ref[0]) * chord[1] - (p[1] - s_ref[1]) * chord[0]
      self.assertAlmostEqual(cross, 0.0, places=10,
                             msg=f'j={j}: point not on fiber chord')

  def test_side_right_inner_endpoint_is_fixed(self):
    """For side='right', inner endpoint at i=0 with r=0 is unchanged."""
    ni, nj = 5, 3
    cps = np.zeros((ni, nj, 2))
    for i in range(ni):
      for j in range(nj):
        cps[i, j, :] = [i / (ni - 1), 0.0]
    cp = np.array(cps, copy=True)
    _apply_discrete_bl_expansion(cp, 'right', 3)
    np.testing.assert_array_almost_equal(cp[0, :, :], cps[0, :, :], decimal=10)

  def test_degenerate_zero_fiber_is_left_unchanged(self):
    """Zero-span fibers (all zeros) are skipped and not modified."""
    cps = np.zeros((3, 4, 2))
    original = np.array(cps, copy=True)
    _apply_discrete_bl_expansion(cps, 'top', 5)
    np.testing.assert_array_equal(cps, original)


# ---------------------------------------------------------------------------
# Integration tests via ellipse() — require Nutils
# ---------------------------------------------------------------------------

class TestEllipseBoundaryLayer(unittest.TestCase):
  """End-to-end tests for ellipse() with boundary_layer_exponent > 1.

  KV = 3 keeps the Nutils solve fast.
  """

  KV = 3

  def _control_points(self, disc):
    """Return a list of (nxi, neta, 2) xy arrays, one per patch."""
    result = []
    for p in range(5):
      block = disc[p]
      spl = np.asarray(block.arr, dtype=object).ravel()
      result.append(np.asarray(spl[0].tcontrolpoints[..., :2], dtype=float))
    return result

  def test_returns_five_patches(self):
    d = ellipse(1, 1, self.KV, boundary_layer_exponent=3)
    self.assertEqual(int(d.shape[0]), 5)

  def test_all_control_points_are_finite(self):
    for exp in (1, 2, 5, 10):
      with self.subTest(exp=exp):
        d = ellipse(1, 1, self.KV, boundary_layer_exponent=exp)
        for p, xy in enumerate(self._control_points(d)):
          self.assertTrue(np.all(np.isfinite(xy)),
                          msg=f'exp={exp}, patch {p}: non-finite control points')

  def test_bl_shifts_mean_radius_toward_boundary(self):
    """With BL active the mean radius of control points exceeds the uniform case."""
    d_uniform = ellipse(1, 1, self.KV, boundary_layer_exponent=1)
    d_bl      = ellipse(1, 1, self.KV, boundary_layer_exponent=5)
    all_u = np.concatenate([xy.reshape(-1, 2)
                            for xy in self._control_points(d_uniform)])
    all_b = np.concatenate([xy.reshape(-1, 2)
                            for xy in self._control_points(d_bl)])
    self.assertGreater(np.mean(np.hypot(all_b[:, 0], all_b[:, 1])),
                       np.mean(np.hypot(all_u[:, 0], all_u[:, 1])))

  def test_larger_exponent_gives_larger_mean_radius(self):
    """Increasing boundary_layer_exponent monotonically increases mean radius."""
    means = []
    for exp in (1, 2, 5):
      d = ellipse(1, 1, self.KV, boundary_layer_exponent=exp)
      cps = np.concatenate([xy.reshape(-1, 2) for xy in self._control_points(d)])
      means.append(np.mean(np.hypot(cps[:, 0], cps[:, 1])))
    self.assertLess(means[0], means[1],
                    msg='exp=2 should have larger mean radius than exp=1')
    self.assertLess(means[1], means[2],
                    msg='exp=5 should have larger mean radius than exp=2')

  def test_ellipse_x_and_y_scale_independently(self):
    """ellipse(a, b, …) must scale x by a and y by b relative to ellipse(1,1,…)."""
    a, b = 2.0, 3.0
    d_unit   = ellipse(1, 1, self.KV, boundary_layer_exponent=2)
    d_scaled = ellipse(a, b, self.KV, boundary_layer_exponent=2)
    for p in range(5):
      xy_u = self._control_points(d_unit)[p]
      xy_s = self._control_points(d_scaled)[p]
      np.testing.assert_array_almost_equal(
          xy_s[..., 0], xy_u[..., 0] * a, decimal=8,
          err_msg=f'x-scaling mismatch on patch {p}')
      np.testing.assert_array_almost_equal(
          xy_s[..., 1], xy_u[..., 1] * b, decimal=8,
          err_msg=f'y-scaling mismatch on patch {p}')

  def test_center_patch_interface_matches_neighbor_no_bl(self):
    """For exponent=1 the center-patch boundary lines match adjacent patches."""
    self._check_center_interface_continuity(exponent=1)

  def test_center_patch_interface_matches_neighbor_with_bl(self):
    """After discrete boundary-layer expansion, shared interface control points still agree."""
    self._check_center_interface_continuity(exponent=4)

  def _check_center_interface_continuity(self, exponent):
    """Compare center-patch boundary lines with their outer-patch counterparts.

    Shared edges (from PATCHES connectivity):
      center left   (2,3) ↔ patch 0 bottom  (cps[0][:,0,:])
      center bottom (2,4) ↔ patch 1 left    (cps[1][0,:,:])
      center top    (3,5) ↔ patch 3 bottom  (cps[3][:,0,:])
      center right  (4,5) ↔ patch 4 left    (cps[4][0,:,:])
    """
    d = ellipse(1, 1, self.KV, boundary_layer_exponent=exponent)
    cps = self._control_points(d)
    center = cps[2]

    pairs = [
        ('center-left',   center[0,  :, :], cps[0][:,  0, :]),
        ('center-bottom', center[:,  0, :], cps[1][0,  :, :]),
        ('center-top',    center[:, -1, :], cps[3][:,  0, :]),
        ('center-right',  center[-1, :, :], cps[4][0,  :, :]),
    ]
    for label, c_line, n_line in pairs:
      oriented = _orient_line_like(n_line, c_line)
      np.testing.assert_array_almost_equal(
          c_line, oriented, decimal=6,
          err_msg=f'exp={exponent}: {label} vs neighbour mismatch')


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _make_cylinder(boundary_layer_exponent, nelems_cs=3, nelems_cl=5, nz=21):
  """Return a hex mesh for a straight unit cylinder along the z-axis."""
  xi = np.linspace(0, 1, 41)
  centerline = np.stack([np.zeros_like(xi),
                         np.zeros_like(xi),
                         xi], axis=1)
  radii = np.ones(len(xi))
  kv    = UnivariateKnotVector(np.linspace(0, 1, nelems_cl + 1)).to_tensor()
  xi_cl = clparam(centerline)
  X     = kv.fit([xi_cl], centerline)
  Rs    = compute_notwistframe_from_spline(X, xi_cl)
  disc  = ellipse(1, 1, nelems_cs, boundary_layer_exponent=boundary_layer_exponent)
  rRs   = kv.fit([xi_cl], radii[:, _, _] * Rs)
  vessel = (disc[:, _] @ rRs[_]).sum(-1) + (disc.unity @ X)
  eval_mesh = rectilinear([2 * nelems_cs + 1, 2 * nelems_cs + 1, nz])
  return mesh_union(*vessel.sample_mesh(eval_mesh), boundary=True)


def _to_pv_grid(mesh, cell_type):
  import pyvista as pv
  nelems = len(mesh.elements)
  cells  = np.concatenate(
      [np.full((nelems, 1), mesh.nverts, dtype=int), mesh.pvelements],
      axis=1).astype(int)
  return pv.UnstructuredGrid(cells, [cell_type] * nelems, mesh.points)


def _assert_valid(mesh, label):
  """Raise AssertionError if the mesh is not valid."""
  assert mesh.is_valid(), f'{label}: mesh.is_valid() returned False'


def _assert_circular_boundary(mesh, label, atol=5e-3):
  """Check that all outer boundary points are within atol of unit radius.

  drop_points_and_renumber() is called on the boundary sub-mesh so that
  only points actually referenced by boundary edges are checked (the parent
  mesh shares its full point array with the sub-mesh, which would otherwise
  include interior points at r ≈ 0).

  atol=5e-3 covers the harmonic map's own projection error; discrete expansion
  keeps the outer control ring fixed so the boundary stays close to the unit
  circle without widening tolerance with the exponent.
  """
  bnd_pts = mesh.boundary.drop_points_and_renumber().points
  radii   = np.hypot(bnd_pts[:, 0], bnd_pts[:, 1])
  max_err = float(np.max(np.abs(radii - 1.0)))
  print(f'  {label}: boundary max|r−1| = {max_err:.4e}  '
        f'(min_r={radii.min():.4f}, max_r={radii.max():.4f})')
  assert max_err <= atol, (
      f'{label}: boundary not circular — max |r − 1| = {max_err:.2e} > {atol}')


def _assert_cylinder_radius(mesh, label, atol=5e-3):
  """Check the outer cylindrical wall is present and close to radius 1.

  A tolerance of 5e-3 is used because degree-3 B-spline evaluation can
  overshoot the convex hull of control points by a small amount.
  """
  radii_xy = np.hypot(mesh.points[:, 0], mesh.points[:, 1])
  assert radii_xy.max() <= 1.0 + atol, (
      f'{label}: point outside unit cylinder — max r = {radii_xy.max():.6f}')
  assert np.isclose(radii_xy.max(), 1.0, atol=atol), (
      f'{label}: outer cylinder wall not found — max r = {radii_xy.max():.6f}')


def plot_cross_section_bl():
  """Side-by-side 2-D disc meshes for exponents 1 / 2 / 3 / 4."""
  import pyvista as pv
  import vtk

  pl = pv.Plotter(shape=(1, 4), title='BL cross-section: exponent 1 / 2 / 3 / 4')
  for col, exp in enumerate((1, 2, 3, 4)):
    disc  = ellipse(1, 1, 4, boundary_layer_exponent=exp)
    m     = mesh_union(*disc.sample_mesh(rectilinear([11, 11])), boundary=True)
    m     = m.drop_points_and_renumber()
    label = f'cross-section exp={exp}'

    _assert_valid(m, label)
    _assert_circular_boundary(m, label)

    pl.subplot(0, col)
    pl.add_mesh(_to_pv_grid(m, vtk.VTK_QUAD), show_edges=True, color='tan')
    pl.add_title(f'exponent = {exp}')
    print(f'[cross-section] exponent={exp}  valid=True  boundary on circle=True')
  pl.show()


def plot_cylinder_bl():
  """Side-by-side 3-D cylinder meshes for exponents 1 / 2 / 3 / 4."""
  import pyvista as pv
  import vtk

  pl = pv.Plotter(shape=(1, 4), title='BL cylinder: exponent 1 / 2 / 3 / 4')
  for col, exp in enumerate((1, 2, 3, 4)):
    m     = _make_cylinder(boundary_layer_exponent=exp)
    m     = m.drop_points_and_renumber()
    label = f'cylinder exp={exp}'

    _assert_valid(m, label)
    _assert_cylinder_radius(m, label)

    pl.subplot(0, col)
    pl.add_mesh(_to_pv_grid(m, vtk.VTK_HEXAHEDRON), show_edges=True, color='lightblue')
    pl.add_title(f'exponent = {exp}')
    print(f'[cylinder]       exponent={exp}  valid=True  radius check=True')
  pl.show()


# ---------------------------------------------------------------------------
# Visual test class — always runs, opens PyVista windows
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP_VISUAL, 'PYHAEMO_SKIP_VISUAL set')
class TestVisualBoundaryLayer(unittest.TestCase):

  def test_cross_section_bl(self):
    """2-D disc: four panels (exponent 1 / 2 / 3 / 4)."""
    plot_cross_section_bl()

  def test_cylinder_bl(self):
    """3-D cylinder: four panels (exponent 1 / 2 / 3 / 4)."""
    plot_cylinder_bl()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  unittest.main()
