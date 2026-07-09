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

from artery import VascularGraph
from edge import VascularEdge
from network import NFurcation
from gui import _find_split_sibling_pair


def _make_edge(start, end, n=20, radius=0.05):
  pts = np.linspace(start, end, n)
  return VascularEdge(points=np.concatenate([pts, np.full((n, 1), radius)], axis=1))


def make_split_and_branch_graph(branch_offset):
  """Two collinear edges (1: far-left -> J, 2: J -> far-right) that together
  form ONE physically-continuous vessel artificially split at J (sharing
  the *exact* same point, as ``split_edges_proximity`` would produce), plus
  a third edge (3) touching near J but offset by ``branch_offset`` -- a
  separately-digitized vessel merely proximity-merged in, not a split
  sibling.
  """
  e1 = _make_edge([-2, 0, 0], [0, 0, 0])
  e2 = _make_edge([0, 0, 0], [2, 0, 0])
  e3 = _make_edge(np.array([0, 0, 0]) + branch_offset, [0, -2, 0])

  g = VascularGraph.from_edges([e1.points, e2.points, e3.points])
  at_junction = {1, -2, -3}
  far_furcs = [f for f in g.furcations
               if not (set(map(int, f.indices)) & at_junction)]
  junction_furc = NFurcation(list(at_junction), np.zeros(3))
  return g._edit(furcations=far_furcs + [junction_furc]), junction_furc


def _apply_removal(g, furc):
  """Mirror build_graph()'s removed_keys handling for a single furcation."""
  def _own_endpoint_furc(i):
    return NFurcation([int(i)], g.get_edges([int(i)])[0].cpoints[-1])

  pair = _find_split_sibling_pair(g, furc)
  if pair is None:
    new_furcs = [f for f in g.furcations if f is not furc]
    new_furcs.extend(_own_endpoint_furc(i) for i in furc.indices)
    return g._edit(furcations=new_furcs)
  ia, ib = pair
  other = [int(i) for i in furc.indices if int(i) not in (ia, ib)]
  new_furcs = [f for f in g.furcations if f is not furc]
  new_furcs.append(NFurcation([ia, ib], furc.centroid))
  new_furcs.extend(_own_endpoint_furc(i) for i in other)
  return g._edit(furcations=new_furcs).delete_twofurcations()


class TestBifDeleteRejoin(unittest.TestCase):

  def test_exact_split_siblings_are_found_and_rejoined(self):
    g, furc = make_split_and_branch_graph(branch_offset=np.array([0.03, 0, 0]))
    pair = _find_split_sibling_pair(g, furc)
    self.assertEqual(set(pair), {1, -2})

    fixed = _apply_removal(g, furc)
    # The true split pair merged into one edge -> one fewer edge overall.
    self.assertEqual(len(fixed.edges), 2)
    self.assertTrue(all(f.N == 1 for f in fixed.furcations))
    self.assertIn(3, fixed.indices)
    merged_edge = fixed.get_edges([i for i in fixed.indices if i != 3][0])[0]
    self.assertAlmostEqual(float(merged_edge.length), 4.0, places=1)

  def test_no_exact_coincidence_leaves_edges_independent(self):
    # All three edges only ever come within ``merge_eps``-scale proximity
    # of each other -- a genuine multi-vessel junction, not a split
    # artifact. Nothing should be merged: merging here would splice two
    # unrelated vessels into one artificially elongated centerline (the
    # reported bug).
    e1 = _make_edge([-2, 0, 0], [0, 0, 0])
    e2 = _make_edge([0.02, 0, 0], [2, 0, 0])
    e3 = _make_edge([0.01, 0.01, 0], [0, -2, 0])
    g = VascularGraph.from_edges([e1.points, e2.points, e3.points])
    at_junction = {1, -2, -3}
    far_furcs = [f for f in g.furcations
                 if not (set(map(int, f.indices)) & at_junction)]
    furc = NFurcation(list(at_junction), np.zeros(3))
    g = g._edit(furcations=far_furcs + [furc])

    self.assertIsNone(_find_split_sibling_pair(g, furc))

    fixed = _apply_removal(g, furc)
    self.assertEqual(len(fixed.edges), 3)
    self.assertEqual(sorted(fixed.indices), [1, 2, 3])
    self.assertTrue(all(f.N == 1 for f in fixed.furcations))

  def test_defused_stubs_do_not_get_elongated_by_furcomplete(self):
    # VascularGraph.furcomplete() -- run only inside generate_mesh()'s
    # pipeline, never in the interactive preview -- extends any edge whose
    # trivial furcation centroid sits more than 1e-8 away from its actual
    # endpoint. A defused stub's furcation must therefore be centered on
    # that edge's own endpoint, or the resulting *mesh* silently grows
    # longer than the (correct-looking) previewed centerline.
    e1 = _make_edge([-2, 0, 0], [0, 0, 0])
    e2 = _make_edge([0.02, 0, 0], [2, 0, 0])
    e3 = _make_edge([0.01, 0.01, 0], [0, -2, 0])
    g = VascularGraph.from_edges([e1.points, e2.points, e3.points])
    at_junction = {1, -2, -3}
    far_furcs = [f for f in g.furcations
                 if not (set(map(int, f.indices)) & at_junction)]
    furc = NFurcation(list(at_junction), np.zeros(3))
    g = g._edit(furcations=far_furcs + [furc])

    fixed = _apply_removal(g, furc)
    lengths_before = {i: fixed.get_edges([i])[0].length for i in fixed.indices}
    completed = fixed.furcomplete(npoints=200)
    for i in fixed.indices:
      self.assertAlmostEqual(
        completed.get_edges([i])[0].length, lengths_before[i], places=6,
        msg=f"edge {i} was elongated by furcomplete() after being defused",
      )


if __name__ == '__main__':
  unittest.main()
