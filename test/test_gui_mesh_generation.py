"""Slow end-to-end smoke test: actually run the volumetric mesh pipeline.

Separate from test_gui.py (which intentionally avoids full mesh generation)
because a single run here takes roughly a minute even with the coarsest
reasonable parameters. Run explicitly when you want to confirm the whole
Generate Mesh -> Check Validity -> Export chain still works end-to-end,
e.g. after touching build_graph(), generate_mesh(), or the mesh module:

    cd test && QT_QPA_PLATFORM=offscreen python3 -m unittest test_gui_mesh_generation -v

or as part of the full suite via ``python3 run_all.py`` from within ``test/``.
"""
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# partery/ has no __init__.py -- its modules (gui, edge, network, ...) use
# bare sibling imports and expect partery/ itself on sys.path. Append (not
# insert(0, ...)) so nothing in partery/ (e.g. its own io.py) can shadow a
# same-named stdlib/third-party module for the rest of this test run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARTERY_DIR = os.path.join(_REPO_ROOT, "partery")
sys.path.append(_PARTERY_DIR)

from PyQt6.QtWidgets import QApplication

import gui

_app = QApplication.instance() or QApplication([])

BRAIN_DIR = os.path.join(_PARTERY_DIR, "data", "csv", "brain")

# Coarse but valid mesh parameters -- keeps the smoke test's real hex-mesh
# solve under ~90s while still exercising the exact same code path
# 'Generate Mesh' uses.
COARSE_MESH_KW = dict(
  split_eps=0.06, merge_eps=0.07, use_merge=True,
  mesh_dx=0.02, mesh_nref=1, mesh_rfac=2.0, mesh_section_points=8,
  removed_keys=set(), manual_additions=[],
)


class TestMeshGenerationEndToEnd(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.mesh = gui.generate_mesh(BRAIN_DIR, **COARSE_MESH_KW)

  def setUp(self):
    self.win = gui.MeshGeneratorWindow()
    self.win.csv_edit.setText(BRAIN_DIR)
    self.win._refresh()
    self.win.last_mesh = self.mesh
    self.tmp_out = tempfile.mkdtemp(prefix="partery_test_export_")
    self.win.out_edit.setText(self.tmp_out)

  def tearDown(self):
    self.win.close()
    self.win.deleteLater()
    shutil.rmtree(self.tmp_out, ignore_errors=True)

  def test_generated_mesh_is_a_real_nonempty_hex_mesh(self):
    self.assertGreater(self.mesh.points.shape[0], 0)

  def test_check_valid_reports_on_the_real_mesh(self):
    self.win._check_valid()
    self.assertIn("is mesh valid", self.win.status.text())

  def test_export_writes_a_real_msh_file(self):
    self.win.filename_edit.setText("smoke_test_export")
    self.win._export_last_mesh()

    self.assertIn("Mesh exported", self.win.status.text())
    exported = os.path.join(self.tmp_out, "smoke_test_export.msh")
    self.assertTrue(os.path.isfile(exported), self.win.status.text())
    self.assertGreater(os.path.getsize(exported), 0)

  def test_export_without_output_folder_shows_status(self):
    self.win.out_edit.setText("")
    self.win._export_last_mesh()
    self.assertIn("select an output folder", self.win.status.text().lower())


if __name__ == "__main__":
  unittest.main()
