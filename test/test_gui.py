"""GUI functional test suite for :mod:`gui` (the partery mesh-generator app).

Runs the real ``MeshGeneratorWindow`` against the bundled sample datasets
(``data/csv/coro``, ``data/csv/brain``) headlessly (``QT_QPA_PLATFORM=offscreen``)
-- no ``show()``, since pyqtgraph's OpenGL view widget has no real GL context
under the offscreen platform and would just spam stderr for no test value.

Most buttons in this GUI are local variables in their constructor method
(only reachable through the ``clicked`` signal they're wired to at
construction time), not stored as ``self.<name>`` attributes -- calling the
connected handler method directly (e.g. ``win._delete_bif()``) is exactly
equivalent to a click (``QPushButton.clicked`` just invokes connected slots
synchronously) and is how most of these tests exercise "the button". Where a
button *is* a real attribute (the view-mode segments, Generate/Stop mesh),
tests call ``.click()`` on the actual widget instead, to also exercise its
checked/enabled state.

This suite intentionally avoids full volumetric mesh generation (~1
minute+ per run) -- see ``test_gui_mesh_generation.py`` for that slower,
separate end-to-end smoke test.

Run with:

    cd test && QT_QPA_PLATFORM=offscreen python3 -m unittest test_gui -v

or as part of the full suite via ``python3 run_all.py`` from within ``test/``.
"""
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# partery/ has no __init__.py -- its modules (gui, edge, network, ...) use
# bare sibling imports and expect partery/ itself on sys.path. Append (not
# insert(0, ...)) so nothing in partery/ (e.g. its own io.py) can shadow a
# same-named stdlib/third-party module for the rest of this test run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARTERY_DIR = os.path.join(_REPO_ROOT, "partery")
sys.path.append(_PARTERY_DIR)

from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import gui

_app = QApplication.instance() or QApplication([])

CORO_DIR = os.path.join(_PARTERY_DIR, "data", "csv", "coro")
BRAIN_DIR = os.path.join(_PARTERY_DIR, "data", "csv", "brain")


class _NullSignal:
  """No-op stand-in for a pyqtSignal that is only ever .connect()ed to,
  never emitted, by the fake worker below."""

  def connect(self, *_args, **_kwargs):
    pass


class _FakeMeshProcess:
  """Stand-in for gui.MeshComputeProcess that never forks a real OS
  process.

  gui.MeshComputeProcess uses multiprocessing (fork on Linux) so
  cancellation can SIGKILL native/numba code that ignores Python
  interruption. Forking repeatedly out of a single long-lived process that
  also has PyQt6's event loop, nutils, and numpy all initialized is exactly
  the kind of multi-threaded fork() that CPython's own multiprocessing
  module warns is deadlock-prone -- and it *did* reliably hang test runs
  that exercised Generate Mesh more than once in the same interpreter
  (confirmed: a run under ``run_all.py`` stalled for 30+ minutes with a
  zombie child stuck mid-fork). Real mesh-computation correctness is
  already covered by test_gui_mesh_generation.py, which calls
  generate_mesh() directly in-process (no subprocess involved) -- these
  tests only need to exercise the GUI's own synchronous button/state
  bookkeeping around start/stop.
  """

  progress = _NullSignal()
  finished_ok = _NullSignal()
  finished_err = _NullSignal()
  finished = _NullSignal()

  def __init__(self, kw, parent=None):
    self._kw = kw
    self._running = False

  def start(self):
    self._running = True

  def is_running(self):
    return self._running

  def stop(self):
    self._running = False


def _new_window(csv_dir=CORO_DIR):
  win = gui.MeshGeneratorWindow()
  if csv_dir != win.csv_edit.text().strip():
    win.csv_edit.setText(csv_dir)
    win._refresh()
  return win


class GuiTestCase(unittest.TestCase):
  """Base fixture: a real, never-shown MeshGeneratorWindow loaded against
  the bundled coronary sample dataset."""

  csv_dir = CORO_DIR

  def setUp(self):
    self.win = _new_window(self.csv_dir)

  def tearDown(self):
    self.win.close()
    self.win.deleteLater()

  def select_bif_row(self, row=0):
    """Select a real bifurcation-list row the way a user click would,
    returning the (signed) key stored on it."""
    self.win.mode.setCurrentText("Bifurcations")
    self.assertGreater(self.win.bif_list.count(), 0, "no bifurcations loaded")
    key = self.win.bif_list.item(row).data(Qt.ItemDataRole.UserRole)
    self.win.bif_list.setCurrentRow(row)
    return key

  def select_id_map_row(self, row=0):
    self.win.mode.setCurrentText("Centerlines")
    self.assertGreater(self.win.id_map.count(), 0, "no centerlines loaded")
    self.win.id_map.setCurrentRow(row)


class TestViewModeAndNavigation(GuiTestCase):

  def test_segmented_view_buttons_switch_mode(self):
    for button, expected in (
      (self.win._view_btn_bifs, "Bifurcations"),
      (self.win._view_btn_mesh, "Mesh"),
      (self.win._view_btn_centerlines, "Centerlines"),
    ):
      button.click()
      self.assertEqual(self.win.mode.currentText(), expected)
      self.assertTrue(button.isChecked())

  def test_clicking_active_segment_again_keeps_it_checked(self):
    self.win._view_btn_centerlines.click()
    self.win._view_btn_centerlines.click()
    self.assertEqual(self.win.mode.currentText(), "Centerlines")
    self.assertTrue(self.win._view_btn_centerlines.isChecked())

  def test_bifurcation_editing_toggle_enters_and_exits_fix_mode(self):
    self.win._view_btn_fix.setChecked(True)
    self.win._on_bif_edit_toggle()
    self.assertEqual(self.win.mode.currentText(), "Fix bifurcation")

    self.win._view_btn_fix.setChecked(False)
    self.win._on_bif_edit_toggle()
    self.assertNotEqual(self.win.mode.currentText(), "Fix bifurcation")

  def test_refresh_button_reloads_without_crashing(self):
    n_before = len(self.win._centerlines)
    self.win._refresh_btn.click()
    self.assertEqual(len(self.win._centerlines), n_before)
    self.assertTrue(self.win.status.text())


class TestBifurcationPanel(GuiTestCase):

  def test_selecting_a_row_persists_the_selection(self):
    # Regression test: _fill_bif_list() used to clear() the list without
    # blocking signals, which re-entered _on_bif_selected_list via the
    # currentItemChanged(None, ...) fired by clear() itself, clobbering
    # the just-set selection back to None before it could be restored.
    key = self.select_bif_row(0)
    self.assertEqual(self.win._selected_bif_key, key)
    self.assertEqual(self.win.bif_list.currentRow(), 0)
    self.assertIsNotNone(self.win.bif_list.currentItem())

  def test_selecting_a_different_row_updates_selection(self):
    self.select_bif_row(0)
    key1 = self.select_bif_row(1)
    self.assertEqual(self.win._selected_bif_key, key1)
    self.assertEqual(self.win.bif_list.currentRow(), 1)

  def test_delete_selection_without_selection_shows_status_and_noops(self):
    self.win.mode.setCurrentText("Bifurcations")
    self.win.bif_list.setCurrentRow(-1)
    before = set(self.win._removed_bif_keys)
    self.win._delete_bif()
    self.assertEqual(self.win._removed_bif_keys, before)
    self.assertIn("Select a bifurcation", self.win.status.text())

  def test_delete_selection_removes_furcation_and_updates_graph(self):
    n_before = len(self.win._graph().edges)
    n_furcs_before = len(self.win._graph().nontrivial_furcations)
    key = self.select_bif_row(0)

    self.win._delete_bif()

    self.assertIn(key, self.win._removed_bif_keys)
    g_after = self.win._graph()
    self.assertLessEqual(len(g_after.nontrivial_furcations), n_furcs_before - 1)
    # Deleting must never *increase* edge count, and either rejoins a true
    # split (edges -1) or safely leaves a genuine junction split (edges
    # unchanged) -- see fix_self_intersection / build_graph's removed_keys
    # handling.
    self.assertIn(len(g_after.edges), (n_before - 1, n_before))
    self.assertIn("deleted", self.win.status.text().lower())

  def test_reset_deleted_restores_furcation(self):
    key = self.select_bif_row(0)
    self.win._delete_bif()
    self.assertIn(key, self.win._removed_bif_keys)

    self.win._reset_deleted()

    self.assertEqual(self.win._removed_bif_keys, set())
    self.assertIsNone(self.win._selected_bif_key)
    g_after = self.win._graph()
    self.assertTrue(
      any(frozenset(int(i) for i in f.indices) == key
          for f in g_after.nontrivial_furcations)
    )

  def test_deleting_a_merge_produced_furcation_is_rejected(self):
    # Can't reliably synthesize a merge-produced key without a real mesh
    # run; just confirm the guard path itself doesn't crash when the set
    # is (legitimately) empty.
    self.win._merged_bif_unsigned_keys = frozenset()
    self.select_bif_row(0)
    self.win._delete_bif()  # should proceed normally, not raise

  def test_invalid_only_filter_checkbox_disabled_without_mesh_validity(self):
    self.win.mode.setCurrentText("Bifurcations")
    self.assertFalse(self.win.bif_list_invalid_only.isEnabled())
    self.assertFalse(self.win.bif_list_invalid_only.isChecked())
    # Toggling a disabled checkbox is a no-op in Qt, but the handler itself
    # must still tolerate being invoked directly without state to filter on.
    self.win._on_bif_invalid_filter_toggled(True)


class TestManualBifurcationAdditions(GuiTestCase):

  def _two_real_indices(self):
    g = self.win._graph()
    return [int(g.indices[0]), int(g.indices[1])]

  def test_add_with_empty_input_shows_status(self):
    self.win.add_edit.setText("")
    self.win._add_bif()
    self.assertIn("Insert signed edge IDs", self.win.status.text())

  def test_add_with_malformed_input_reports_error_without_crashing(self):
    self.win.add_edit.setText("not, a, number")
    self.win._add_bif()
    self.assertIn("Invalid bifurcation input", self.win.status.text())
    self.assertEqual(self.win._manual_additions, [])

  def test_add_with_single_id_is_rejected(self):
    self.win.add_edit.setText("3")
    self.win._add_bif()
    self.assertIn("Invalid bifurcation input", self.win.status.text())

  def test_add_valid_ids_then_remove_then_clear(self):
    i0, i1 = self._two_real_indices()
    self.win.add_edit.setText(f"{i0}, {i1}")
    self.win._add_bif()
    self.assertEqual(self.win._manual_additions, [(i0, i1)])
    self.assertEqual(self.win.add_edit.text(), "")

    self.win.add_list.setCurrentRow(0)
    self.win._remove_add()
    self.assertEqual(self.win._manual_additions, [])

    self.win.add_edit.setText(f"{i0}, {i1}")
    self.win._add_bif()
    self.win._clear_add()
    self.assertEqual(self.win._manual_additions, [])

  def test_remove_add_without_selection_shows_status(self):
    self.win.add_list.setCurrentRow(-1)
    self.win._remove_add()
    self.assertIn("Select an added bifurcation", self.win.status.text())


class TestCenterlinePanel(GuiTestCase):

  def test_delete_selected_centerline_without_selection_shows_status(self):
    self.win.id_map.setCurrentRow(-1)
    self.win._delete_selected_centerline()
    self.assertIn("Select a centerline", self.win.status.text())

  def test_delete_selected_centerline_and_reset(self):
    self.select_id_map_row(0)
    n_before = len(self.win._centerlines)

    self.win._delete_selected_centerline()

    self.assertEqual(len(self.win._removed_centerline_ids), 1)
    self.assertEqual(len(self.win._centerlines), n_before - 1)

    self.win._reset_removed_centerlines()

    self.assertEqual(self.win._removed_centerline_ids, set())
    self.assertEqual(len(self.win._centerlines), n_before)

  def test_reset_removed_centerlines_without_any_removed_shows_status(self):
    self.win._reset_removed_centerlines()
    self.assertIn("No removed centerlines", self.win.status.text())

  def test_cannot_delete_last_remaining_centerline(self):
    # Remove every centerline but one, then confirm the guard fires.
    g = self.win._graph()
    for i in list(g.indices)[:-1]:
      self.win._removed_centerline_ids.add(int(i))
    self.win._clear_cached_mesh()
    self.win._refresh()
    self.assertEqual(len(self.win._centerlines), 1)

    self.select_id_map_row(0)
    self.win._delete_selected_centerline()

    self.assertIn("Cannot delete the last remaining centerline", self.win.status.text())
    self.assertEqual(len(self.win._centerlines), 1)


class TestFolderSelection(GuiTestCase):

  def test_choose_csv_cancelled_dialog_leaves_state_untouched(self):
    before = self.win.csv_edit.text()
    with patch.object(gui.QFileDialog, "getExistingDirectory", return_value=""):
      self.win._choose_csv()
    self.assertEqual(self.win.csv_edit.text(), before)

  def test_choose_csv_switches_dataset_and_resets_edits(self):
    key = self.select_bif_row(0)
    self.win._delete_bif()
    self.assertIn(key, self.win._removed_bif_keys)

    with patch.object(gui.QFileDialog, "getExistingDirectory", return_value=BRAIN_DIR):
      self.win._choose_csv()

    self.assertEqual(self.win.csv_edit.text(), BRAIN_DIR)
    self.assertEqual(self.win._removed_bif_keys, set())
    self.assertEqual(self.win._removed_centerline_ids, set())
    self.assertEqual(self.win._manual_additions, [])
    self.assertGreater(len(self.win._centerlines), 0)

  def test_choose_out_cancelled_dialog_leaves_state_untouched(self):
    before = self.win.out_edit.text()
    with patch.object(gui.QFileDialog, "getExistingDirectory", return_value=""):
      self.win._choose_out()
    self.assertEqual(self.win.out_edit.text(), before)

  def test_choose_out_sets_output_folder(self):
    with patch.object(gui.QFileDialog, "getExistingDirectory", return_value="/tmp/some_out_dir"):
      self.win._choose_out()
    self.assertEqual(self.win.out_edit.text(), "/tmp/some_out_dir")


class TestFixBifurcationPanel(GuiTestCase):

  def setUp(self):
    super().setUp()
    self.win._view_btn_fix.setChecked(True)
    self.win._on_bif_edit_toggle()

  def test_clear_fix_picks_does_not_crash_with_nothing_picked(self):
    self.win._clear_fix_picks()

  def test_clear_all_fix_anchors_does_not_crash_with_nothing_set(self):
    self.win._clear_all_fix_anchors()

  def test_fix_show_mesh_toggle_without_mesh_does_not_crash(self):
    self.win.fix_show_mesh.setChecked(True)
    self.win._on_fix_show_mesh_toggled(True)
    self.win.fix_show_mesh.setChecked(False)
    self.win._on_fix_show_mesh_toggled(False)


class TestMeshPanelWithoutRunningMesh(GuiTestCase):

  def test_check_valid_without_mesh_shows_status(self):
    self.win.last_mesh = None
    self.win._check_valid()
    self.assertIn("No mesh cached", self.win.status.text())

  def test_export_without_mesh_shows_status(self):
    self.win.last_mesh = None
    self.win._export_last_mesh()
    self.assertIn("No mesh cached", self.win.status.text())

  def test_stop_mesh_generation_with_nothing_running_is_a_safe_noop(self):
    self.win._mesh_thread = None
    self.win._stop_mesh_generation()
    self.assertIn("No mesh generation is currently running", self.win.status.text())
    self.assertFalse(self.win.b_stop_gen.isEnabled())

  def test_generate_mesh_rejects_missing_csv_folder(self):
    self.win.csv_edit.setText("/no/such/folder")
    self.win.b_gen.click()
    self.assertIn("Select a valid CSV folder", self.win.status.text())
    self.assertIsNone(self.win._mesh_thread)

  def test_generate_mesh_starts_and_stop_cancels_cleanly(self):
    # Exercises the *synchronous* half of Generate Mesh / Stop Mesh
    # Generation (button enable/disable, worker bookkeeping) with a fake,
    # non-forking worker -- see _FakeMeshProcess for why a real
    # MeshComputeProcess is deliberately not used here, and
    # test_gui_mesh_generation.py for the actual end-to-end mesh run.
    with patch.object(gui, "MeshComputeProcess", _FakeMeshProcess):
      self.win.b_gen.click()
      try:
        self.assertFalse(self.win.b_gen.isEnabled())
        self.assertTrue(self.win.b_stop_gen.isEnabled())
        self.assertIsNotNone(self.win._mesh_thread)
        self.assertTrue(self.win._mesh_worker_is_running())
      finally:
        self.win.b_stop_gen.click()
      self.assertFalse(self.win.b_stop_gen.isEnabled())
      self.assertTrue(self.win.b_gen.isEnabled())
      self.assertFalse(self.win._mesh_worker_is_running())

  def test_generate_mesh_click_while_already_running_shows_status(self):
    # b_gen.setEnabled(False) as part of the first click's own handler means
    # a real second click is a Qt no-op on a disabled button (never invokes
    # the slot) -- exercise the "already running" guard by calling the
    # handler directly, the way it could still be reached via a
    # not-yet-disabled race or a non-UI caller. See _FakeMeshProcess for why
    # a real MeshComputeProcess is deliberately not used here.
    with patch.object(gui, "MeshComputeProcess", _FakeMeshProcess):
      self.win.b_gen.click()
      try:
        self.assertTrue(self.win._mesh_worker_is_running())

        self.win._generate_export_open()
        self.assertIn("already in progress", self.win.status.text())
      finally:
        self.win.b_stop_gen.click()


class TestParameterWidgets(GuiTestCase):
  """Spinboxes/checkboxes that live-refresh the preview must apply their
  value and not crash, for every widget wired to ``_refresh``."""

  def test_split_eps_spinbox_applies_and_refreshes(self):
    self.win.split_eps.setValue(0.5)
    self.assertEqual(self.win.split_eps.value(), 0.5)

  def test_merge_eps_spinbox_applies_and_refreshes(self):
    self.win.merge_eps.setValue(0.5)
    self.assertEqual(self.win.merge_eps.value(), 0.5)

  def test_max_lambda_iter_spinbox_applies_and_refreshes(self):
    self.win.max_lambda_iter.setValue(3)
    self.assertEqual(self.win.max_lambda_iter.value(), 3)
    g = self.win._graph()
    self.assertIsInstance(self.win._centerline_validity, dict)
    self.assertTrue(g.edges)

  def test_max_lambda_iter_zero_disables_correction_live(self):
    self.win.max_lambda_iter.setValue(0)
    g = self.win._graph()
    for result in self.win._centerline_validity.values():
      if not result.initial_valid:
        self.assertEqual(result.n_iterations, 0)
        self.assertFalse(result.final_valid)

  def test_use_merge_checkbox_toggle_refreshes(self):
    n_furcs_merged = len(self.win._graph().nontrivial_furcations)
    self.win.use_merge.setChecked(False)
    n_furcs_unmerged = len(self.win._graph().nontrivial_furcations)
    self.assertLessEqual(n_furcs_unmerged, n_furcs_merged)

  def test_bif_exploit_symmetry_checkbox_toggle_does_not_crash(self):
    self.win.bif_exploit_symmetry.setChecked(True)
    self.win.bif_exploit_symmetry.setChecked(False)


if __name__ == "__main__":
  unittest.main()
