import os
import sys
import unittest

# partery/ has no __init__.py -- its modules (gui, edge, network, ...) use
# bare sibling imports and expect partery/ itself on sys.path. Append
# (not insert(0, ...)) so nothing in partery/ can shadow a same-named
# stdlib/third-party module for the rest of this test run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "partery"))

from pyhaemo.util import np

from edge import VascularEdge
from fix_self_intersection import curvature_mask, refit_until_valid


def make_arc_edge(R=1.0, tube_r=0.1, n=200, angle=np.pi / 2):
  """A circular arc of curvature radius ``R`` with constant tube radius
  ``tube_r`` -- the canonical max-radius-condition test shape: the swept
  tube self-intersects iff ``tube_r > R``.
  """
  theta = np.linspace(0, angle, n)
  x = R * np.cos(theta)
  y = R * np.sin(theta)
  z = np.zeros_like(theta)
  r = np.full(n, tube_r)
  points = np.stack([x, y, z, r], axis=1)
  return VascularEdge(points=points)


class TestCurvatureMask(unittest.TestCase):

  def test_curvature_matches_known_circle_radius(self):
    R = 2.0
    edge = make_arc_edge(R=R, tube_r=0.01, angle=np.pi / 2)
    spline_edge = edge.to_spline_edge(30, lam0=1e-5, lam1=0)
    xi = np.linspace(0.1, 0.9, 20)
    v = spline_edge.cline(xi, dx=1)
    a = spline_edge.cline(xi, dx=2)
    curvature = np.linalg.norm(np.cross(v, a), axis=1) \
              / np.linalg.norm(v, axis=1) ** 3
    np.testing.assert_allclose(curvature, 1.0 / R, rtol=0.05)

  def test_mask_flags_tube_radius_exceeding_bend_radius(self):
    edge = make_arc_edge(R=1.0, tube_r=1.3, angle=np.pi / 2)
    spline_edge = edge.to_spline_edge(30, lam0=1e-5, lam1=0)
    mask = curvature_mask(spline_edge.cline, spline_edge.radius, n=100)
    self.assertTrue(mask.any())

  def test_mask_clear_for_gentle_bend(self):
    edge = make_arc_edge(R=5.0, tube_r=0.05, angle=np.pi / 2)
    spline_edge = edge.to_spline_edge(30, lam0=1e-5, lam1=0)
    mask = curvature_mask(spline_edge.cline, spline_edge.radius, n=100)
    self.assertFalse(mask.any())


class TestRefitUntilValid(unittest.TestCase):

  def test_gentle_bend_is_valid_original(self):
    edge = make_arc_edge(R=5.0, tube_r=0.05, angle=np.pi / 2)
    result = refit_until_valid(edge, n=30, max_iter=8)
    self.assertTrue(result.initial_valid)
    self.assertTrue(result.final_valid)
    self.assertEqual(result.n_iterations, 0)
    self.assertIs(result.edge, edge)

  def test_sharp_bend_gets_auto_smoothed(self):
    edge = make_arc_edge(R=1.0, tube_r=1.3, angle=np.pi / 2)
    result = refit_until_valid(edge, n=30, max_iter=8)
    self.assertFalse(result.initial_valid)
    self.assertTrue(result.final_valid)
    self.assertGreater(result.n_iterations, 0)
    self.assertIsNot(result.edge, edge)
    # The corrected polyline itself (independent of any further
    # regularization) must satisfy the condition it was fit for.
    resmoothed = result.edge.to_spline_edge(30, lam0=1e-5, lam1=0)
    checked = curvature_mask(resmoothed.cline, resmoothed.radius, n=100)
    self.assertFalse(checked.any())

  def test_max_iter_zero_disables_correction(self):
    edge = make_arc_edge(R=1.0, tube_r=1.3, angle=np.pi / 2)
    result = refit_until_valid(edge, n=30, max_iter=0)
    self.assertFalse(result.initial_valid)
    self.assertFalse(result.final_valid)
    self.assertEqual(result.n_iterations, 0)
    self.assertIs(result.edge, edge)


if __name__ == '__main__':
  unittest.main()
