"""PyQt6 front-end for the partery vascular mesh pipeline.

Loads centerline CSVs, builds a :class:`~artery.VascularGraph`, and offers:
  - 3D centerline and bifurcation views (pyqtgraph OpenGL, no embedded VTK)
  - Bifurcation editing (delete / manual merge by signed edge IDs) with 3D picking
  - Background mesh generation (QThread) and full **boundary** wireframe preview
  - Volume mesh statistics, Gmsh ``.msh`` export, and validity check

Bifurcation picking uses the same projection matrices as pyqtgraph’s frustum;
mesh preview arrays are precomputed off the GUI thread where possible.

**Main window layout**

The central vertical layout gives **stretch** to the horizontal splitter so the 3D
view and left panel track window height. The left column wraps **Input**,
**Bifurcations** (including bifurcation editing), and **Mesh** in a
:class:`~PyQt6.QtWidgets.QScrollArea` so short windows scroll instead of clipping
widgets; the status line sits **below** the scroll area and stays visible.
"""
import os
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QRect
from PyQt6.QtGui import QColor, QFont, QVector4D, QPixmap, QIcon
from PyQt6.QtWidgets import (
  QApplication,
  QCheckBox,
  QComboBox,
  QDoubleSpinBox,
  QFileDialog,
  QFrame,
  QGridLayout,
  QGroupBox,
  QHBoxLayout,
  QLabel,
  QLineEdit,
  QListWidget,
  QListWidgetItem,
  QMainWindow,
  QPushButton,
  QRubberBand,
  QScrollArea,
  QSizeGrip,
  QSizePolicy,
  QSplashScreen,
  QSpinBox,
  QSplitter,
  QVBoxLayout,
  QWidget,
)

from artery import VascularGraph
from network import NFurcation

try:
  import pyqtgraph.opengl as gl
except Exception:
  gl = None

# --- Bifurcation 3D picking (screen pixels) ---------------------------------
PICK_TOLERANCE_PX = 40
RED_BIF_SIZE_PX = 12.0
# List- or 3D-selected bifurcation: large black centroid marker (white GL background).
SEL_BIF_BLACK_SIZE_PX = 26.0

# --- Mesh preview fallback (subsampled points if boundary wireframe unavailable)
MAX_MESH_PREVIEW_POINTS = 30000

# Reference-hex facet corners (must match ``pyhaemo.mesh.element.hex_facets``).
_HEX_FACETS = (
  (0, 1, 2, 3),
  (0, 1, 4, 5),
  (0, 2, 4, 6),
  (4, 5, 6, 7),
  (1, 3, 5, 7),
  (2, 3, 6, 7),
)


@lru_cache(maxsize=1)
def _hex_local_edge_pairs():
  """12 undirected local edges of the reference hexahedron (vertex index pairs)."""
  edges = set()
  for facet in _HEX_FACETS:
    for i in range(4):
      a, b = int(facet[i]), int(facet[(i + 1) % 4])
      if a > b:
        a, b = b, a
      edges.add((a, b))
  return np.array(sorted(edges), dtype=np.int64)


# Boundary surface quads/tris: local edge pairs from pyhaemo reference elements.
_QUAD_BOUNDARY_LOCAL_EDGES = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
_TRI_BOUNDARY_LOCAL_EDGES = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)


def _resolve_logo_path(kind: str = "logotipo"):
  """Resolve preferred logo file under repo assets."""
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  kind = str(kind).lower()
  if kind == "pictogram":
    candidates = (
      os.path.join(root, "assets", "logo", "pictogram_2x.png"),
      os.path.join(root, "assets", "pictogram_2x.png"),
      os.path.join(root, "assets", "logo", "pictogram.png"),
      os.path.join(root, "assets", "pictogram.png"),
      os.path.join(root, "assets", "logo", "logotipo.png"),
      os.path.join(root, "assets", "logotipo.png"),
      os.path.join(root, "assets", "pictogram.pdf"),
      os.path.join(root, "assets", "logotipo.pdf"),
    )
  else:
    candidates = (
      os.path.join(root, "assets", "logo", "logotipo.png"),
      os.path.join(root, "assets", "logotipo.png"),
      os.path.join(root, "assets", "logo", "pictogram_2x.png"),
      os.path.join(root, "assets", "pictogram_2x.png"),
      os.path.join(root, "assets", "logo", "pictogram.png"),
      os.path.join(root, "assets", "pictogram.png"),
      os.path.join(root, "assets", "logotipo.pdf"),
      os.path.join(root, "assets", "pictogram.pdf"),
    )
  for path in candidates:
    if os.path.isfile(path):
      return path
  return None


def _full_boundary_wireframe_positions(volume_points, boundary_elements, nverts):
  """Return ``(2 * n_edges, 3)`` positions for ``GLLinePlotItem(..., mode='lines')``.

  Deduplicates undirected edges across all boundary faces. Uses the same global
  ``mesh.points`` indexing as ``mesh.boundary.elements`` (no subsampling).
  """
  pts = np.asarray(volume_points, dtype=float)
  if pts.shape[1] < 3:
    return None
  pts = pts[:, :3]
  bel = np.asarray(boundary_elements, dtype=np.int64)
  if bel.ndim != 2 or bel.shape[0] == 0:
    return None
  if nverts == 4 and bel.shape[1] == 4:
    loc = _QUAD_BOUNDARY_LOCAL_EDGES
  elif nverts == 3 and bel.shape[1] == 3:
    loc = _TRI_BOUNDARY_LOCAL_EDGES
  else:
    return None
  vi = bel[:, loc[:, 0]].ravel()
  vj = bel[:, loc[:, 1]].ravel()
  a = np.minimum(vi, vj)
  b = np.maximum(vi, vj)
  pairs = np.column_stack([a, b])
  unique_pairs = np.unique(pairs, axis=0)
  n = unique_pairs.shape[0]
  if n == 0:
    return None
  pos = np.empty((2 * n, 3), dtype=float)
  pos[0::2] = pts[unique_pairs[:, 0]]
  pos[1::2] = pts[unique_pairs[:, 1]]
  return pos


def _boundary_preview_arrays(mesh):
  """Precompute OpenGL arrays for the **entire** exterior surface of ``mesh``.

  Returns
  -------
  wire_pos, scatter_xyz : ndarray or None
    ``wire_pos`` is line-pair vertices; ``scatter_xyz`` is one row per distinct
    boundary node. On failure (no boundary), returns ``(None, None)``.
  """
  try:
    b = mesh.boundary
  except Exception:
    return None, None
  pts = np.asarray(mesh.points, dtype=float)
  bel = np.asarray(b.elements, dtype=np.int64)
  if bel.size == 0 or pts.size == 0:
    return None, None
  nv = int(getattr(b, "nverts", bel.shape[1]))
  wpos = _full_boundary_wireframe_positions(pts, bel, nv)
  if wpos is None:
    return None, None
  idx = np.unique(bel.ravel())
  scatter = np.ascontiguousarray(pts[np.asarray(idx, dtype=np.int64), :3].copy())
  return wpos, scatter


def mesh_statistics(mesh):
  """Scalar summary of the **volume** hex mesh (not the boundary preview).

  ``mean_hex_edge_length`` is the mean over all ``12 * n_cells`` local hex edges
  (geometric interior edges appear twice, once per adjacent cell).
  """
  out = {
    "n_cells": 0,
    "n_vertices": 0,
    "mean_hex_edge_length": float("nan"),
    "kind": "unknown",
  }
  if mesh is None or not hasattr(mesh, "elements") or not hasattr(mesh, "points"):
    return out
  els = np.asarray(mesh.elements, dtype=np.int64)
  pts = np.asarray(mesh.points, dtype=float)
  if els.size == 0 or pts.size == 0:
    return out
  out["n_cells"] = int(els.shape[0])
  out["n_vertices"] = int(pts.shape[0])
  nv = int(getattr(mesh, "nverts", els.shape[1] if els.ndim == 2 else 0))
  if els.ndim != 2 or nv != 8 or els.shape[1] != 8:
    out["kind"] = f"non-hex (nverts={nv})"
    return out
  local = _hex_local_edge_pairs()
  vi = els[:, local[:, 0]]
  vj = els[:, local[:, 1]]
  p0 = pts[vi.ravel()]
  p1 = pts[vj.ravel()]
  lens = np.linalg.norm(p1 - p0, axis=1)
  out["mean_hex_edge_length"] = float(np.mean(lens))
  out["kind"] = "hexahedron"
  return out


class PickGLViewWidget(gl.GLViewWidget if gl is not None else object):
  """``GLViewWidget`` that reports left-clicks without stealing drag-to-orbit.

  Fires ``on_left_click(x, y)`` on **release** only if movement stayed within
  ``_click_drag_tol_px``, so camera orbit/pan keeps working.

  **Shift + left-drag** draws a rubber band; on release, ``on_rubber_zoom_rect``
  receives a :class:`~PyQt6.QtCore.QRect` in widget pixels (if the box is large
  enough) so the parent can zoom the camera to that screen region.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.on_left_click = None
    self.on_rubber_zoom_rect = None
    self._press_pos = None
    self._click_drag_tol_px = 6.0
    self._rubber_origin = None
    self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
    self._rubber.hide()

  def mousePressEvent(self, event):
    if gl is not None and event.button() == Qt.MouseButton.LeftButton:
      if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
        self._rubber_origin = event.position()
        p = self._rubber_origin
        self._rubber.setGeometry(int(p.x()), int(p.y()), 1, 1)
        self._rubber.show()
        self._press_pos = None
        event.accept()
        return
      self._press_pos = event.position()
    super().mousePressEvent(event)

  def mouseMoveEvent(self, event):
    if gl is not None and self._rubber_origin is not None:
      p0, p1 = self._rubber_origin, event.position()
      x = int(min(p0.x(), p1.x()))
      y = int(min(p0.y(), p1.y()))
      w = int(abs(p1.x() - p0.x()))
      h = int(abs(p1.y() - p0.y()))
      self._rubber.setGeometry(x, y, max(1, w), max(1, h))
      event.accept()
      return
    super().mouseMoveEvent(event)

  def mouseReleaseEvent(self, event):
    if gl is not None and event.button() == Qt.MouseButton.LeftButton and self._rubber_origin is not None:
      p0 = self._rubber_origin
      self._rubber_origin = None
      p1 = event.position()
      x = int(min(p0.x(), p1.x()))
      y = int(min(p0.y(), p1.y()))
      w = int(abs(p1.x() - p0.x()))
      h = int(abs(p1.y() - p0.y()))
      self._rubber.hide()
      if w >= 8 and h >= 8 and callable(self.on_rubber_zoom_rect):
        self.on_rubber_zoom_rect(QRect(x, y, w, h))
      self._press_pos = None
      event.accept()
      super().mouseReleaseEvent(event)
      return
    if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None and callable(self.on_left_click):
      rel = event.position()
      d = np.hypot(rel.x() - self._press_pos.x(), rel.y() - self._press_pos.y())
      if d <= self._click_drag_tol_px:
        self.on_left_click(rel.x(), rel.y())
    self._press_pos = None
    super().mouseReleaseEvent(event)


def is_mesh_valid(mesh) -> bool:
  """Thin wrapper around pyhaemo’s ``mesh.is_valid()`` for the GUI."""
  return bool(mesh.is_valid())


def _apply_compact_subtree_font(root: QWidget, point_size: float = 8.0):
  """Apply ``point_size`` to ``root`` and all descendant :class:`~PyQt6.QtWidgets.QWidget` fonts.

  Used for the bifurcation-editing subtree and re-applied from
  :meth:`MeshGeneratorWindow.resizeEvent` after global font scaling.
  """
  f = QFont(root.font())
  f.setPointSizeF(float(point_size))
  root.setFont(f)
  for w in root.findChildren(QWidget):
    w.setFont(f)


def build_graph(csv_folder, split_eps, merge_eps, use_merge, removed_keys, manual_additions,
                removed_centerline_ids=frozenset()):
  """Graph used for previews and mesh generation: split/merge furcs + user edits."""
  g = VascularGraph.from_csv(csv_folder, skip_indices=set(removed_centerline_ids)).normalize().refit()
  g = g.split_edges_proximity(eps=split_eps)
  if use_merge:
    g = g.merge_furcs_proximity(merge_eps).refit().recompute_centroids()
  if removed_keys:
    furcs = []
    for furc in g.furcations:
      key = frozenset(int(i) for i in furc.indices)
      if furc.N > 2 and key in removed_keys:
        for i in furc.indices:
          furcs.append(NFurcation([int(i)], furc.centroid))
      else:
        furcs.append(furc)
    g = g._edit(furcations=furcs)
  for signed_ids in manual_additions:
    if len(signed_ids) < 2:
      continue
    base = int(signed_ids[0])
    for other in map(int, signed_ids[1:]):
      try:
        _, i0 = g.get_furc(base, return_index=True)
        _, i1 = g.get_furc(other, return_index=True)
      except Exception:
        continue
      if i0 != i1:
        g = g.merge_furcs(i0, i1)
  return g


def generate_mesh(csv_folder, split_eps, merge_eps, use_merge, spline_points, mesh_dx, mesh_nref, mesh_rfac,
                  mesh_section_points, removed_keys, manual_additions,
                  boundary_layer_exponent=1.0,
                  mesh_union_eps=1e-8, bif_exploit_symmetry=False,
                  removed_centerline_ids=frozenset()):
  """Run the full artery pipeline and return the volumetric hex mesh (CPU-heavy)."""
  artery = build_graph(
    csv_folder, split_eps, merge_eps, use_merge, removed_keys, manual_additions,
    removed_centerline_ids=removed_centerline_ids,
  )
  artery = artery.orient_nontrivial()
  artery = (
    artery.align_nontrivial_normals()
    .orient_trivial()
    .rtruncate(1)
    .furcomplete(npoints=1001)
    .delete_twofurcations()
    .align_frames()
  )
  return artery.to_spline_graph(spline_points).make_mesh(
    dx=mesh_dx, nref=mesh_nref, rfac=mesh_rfac, cross_nelems=mesh_section_points,
    union_eps=mesh_union_eps, bif_exploit_symmetry=bif_exploit_symmetry,
    boundary_layer_exponent=boundary_layer_exponent)


def _extract_boundary_points_for_preview(mesh):
  """Sample boundary coordinates for **fallback** point-cloud preview only.

  Prefer :func:`_boundary_preview_arrays` for full boundary connectivity.
  Safe to call from :class:`MeshComputeThread`.
  """
  bpts = None
  for obj in (mesh, getattr(mesh, "boundary", None), getattr(getattr(mesh, "boundary", None), "interfaces", None)):
    if obj is None or not hasattr(obj, "take_boundary_points"):
      continue
    try:
      bpts = obj.take_boundary_points()
      if bpts is not None:
        break
    except Exception:
      pass
  if bpts is None:
    for obj in (getattr(mesh, "boundary", None), getattr(getattr(mesh, "boundary", None), "interfaces", None), mesh):
      if obj is None:
        continue
      pts = getattr(obj, "points", None)
      if pts is not None:
        bpts = pts
        break
  if bpts is None:
    return None
  arr = np.asarray(bpts, dtype=float)
  if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
    return None
  return np.ascontiguousarray(arr[:, :3].copy())


class MeshComputeThread(QThread):
  """Runs ``work_fn`` off the GUI thread; results are delivered via Qt signals.

  ``finished_ok`` payloads: ``(mesh, aux_points, stats, boundary_wire, boundary_scatter)``.
  """

  finished_ok = pyqtSignal(object, object, object, object, object)
  finished_err = pyqtSignal(str)

  def __init__(self, work_fn):
    super().__init__()
    self._work_fn = work_fn

  def run(self):
    try:
      mesh, bpts, stats, bwire, bscatter = self._work_fn()
      self.finished_ok.emit(mesh, bpts, stats, bwire, bscatter)
    except BaseException:
      self.finished_err.emit(traceback.format_exc())


class Scene3D(QWidget):
  """Embedded pyqtgraph OpenGL scene: centerlines, bifurcations, or mesh boundary.

  Labels are drawn as crude 7-segment polylines in world space. There is no fixed
  axis triad at the origin. Bifurcation picking uses screen projection plus a ray
  fallback aligned with ``GLViewWidget.projectionMatrix(region, region)``.

  **Shift + left-drag** on the GL view draws a rectangle; on release the camera
  dollies in and recenters so that screen region fills the view (rubber-band zoom).
  """

  # 7-segment style glyphs for in-scene numeric labels (no GL text texture).
  SEG = {"A": ((0, 2), (1, 2)), "B": ((1, 2), (1, 1)), "C": ((1, 1), (1, 0)), "D": ((0, 0), (1, 0)), "E": ((0, 1), (0, 0)), "F": ((0, 2), (0, 1)), "G": ((0, 1), (1, 1))}
  DIG = {"0": ("A", "B", "C", "D", "E", "F"), "1": ("B", "C"), "2": ("A", "B", "G", "E", "D"), "3": ("A", "B", "G", "C", "D"), "4": ("F", "G", "B", "C"), "5": ("A", "F", "G", "C", "D"), "6": ("A", "F", "G", "C", "D", "E"), "7": ("A", "B", "C"), "8": ("A", "B", "C", "D", "E", "F", "G"), "9": ("A", "B", "C", "D", "F", "G"), "-": ("G",)}

  def __init__(self, parent=None):
    super().__init__(parent)
    self.on_bif_click = None
    self.mesh_simplified = False
    self._bif_pick = []
    layout = QVBoxLayout(self)
    if gl is None:
      self.view = None
      msg = QLabel("3D viewer unavailable. Install: pip install pyqtgraph PyOpenGL")
      msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
      layout.addWidget(msg)
      return
    self.view = PickGLViewWidget()
    self.view.setBackgroundColor("white")
    self.view.setCameraPosition(distance=4.5)
    self.view.on_left_click = self._handle_left_click
    self.view.on_rubber_zoom_rect = self._apply_zoom_rect
    layout.addWidget(self.view)
    self.items = []

  def _handle_left_click(self, sx, sy):
    key = self._pick_bif(sx, sy)
    if key is not None and callable(self.on_bif_click):
      self.on_bif_click(key)

  def _viewport_region(self):
    """Match pyqtgraph paint(): same tuple for region and projection viewport."""
    return self.view.getViewport()

  def _project(self, p):
    """Map a world point to widget pixel coordinates using pyqtgraph’s frustum."""
    try:
      region = self._viewport_region()
      mv = self.view.viewMatrix()
      pm = self.view.projectionMatrix(region, region)
      v = QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0)
      clip = pm * (mv * v)
      if abs(clip.w()) < 1e-9:
        return None
      ndc_x, ndc_y = clip.x() / clip.w(), clip.y() / clip.w()
      _, _, w, h = region
      sx = (ndc_x * 0.5 + 0.5) * w
      sy = (1.0 - (ndc_y * 0.5 + 0.5)) * h
      return np.array([sx, sy], dtype=float)
    except Exception:
      return None

  def _unproject_ray(self, sx, sy):
    """World-space ray through pixel (sx, sy), consistent with pyqtgraph frustum."""
    region = self._viewport_region()
    _, _, w, h = region
    wf, hf = float(w), float(h)
    if wf < 1.0 or hf < 1.0:
      return None, None
    pm = self.view.projectionMatrix(region, region)
    vm = self.view.viewMatrix()
    m = pm * vm
    inv, ok = m.inverted()
    if not ok:
      return None, None

    x_ndc = (2.0 * float(sx) / wf) - 1.0
    y_ndc = 1.0 - (2.0 * float(sy) / hf)

    def to_world(z_ndc):
      v = inv * QVector4D(float(x_ndc), float(y_ndc), float(z_ndc), 1.0)
      ww = v.w()
      if abs(ww) < 1e-12:
        return None
      return np.array([v.x() / ww, v.y() / ww, v.z() / ww], dtype=float)

    p_near = to_world(-1.0)
    p_far = to_world(1.0)
    if p_near is None or p_far is None:
      return None, None
    d = p_far - p_near
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
      return None, None
    d /= nd
    return p_near, d

  def _apply_zoom_rect(self, rect: QRect):
    """Dolly and recenter the camera so geometry that projected into ``rect`` fills the view."""
    if self.view is None:
      return
    try:
      _, _, W, H = self._viewport_region()
      wf, hf = float(W), float(H)
      if wf < 2.0 or hf < 2.0:
        return
      rw = max(1.0, float(rect.width()))
      rh = max(1.0, float(rect.height()))
      zoom = min(wf / rw, hf / rh)
      zoom = min(max(zoom, 1.02), 80.0)

      cx = float(rect.x()) + 0.5 * float(rect.width())
      cy = float(rect.y()) + 0.5 * float(rect.height())
      ray_o, ray_d = self._unproject_ray(cx, cy)
      if ray_o is None or ray_d is None:
        return

      cam = np.asarray(self.view.cameraPosition(), dtype=float).ravel()
      C = np.array(
        [
          self.view.opts["center"].x(),
          self.view.opts["center"].y(),
          self.view.opts["center"].z(),
        ],
        dtype=float,
      )
      fwd = C - cam
      fn = float(np.linalg.norm(fwd))
      if fn < 1e-9:
        return
      fwd /= fn
      denom = float(np.dot(ray_d, fwd))
      if abs(denom) < 1e-6:
        C_new = C
      else:
        t = float(np.dot(C - ray_o, fwd) / denom)
        if t < 0.0:
          t = 0.0
        C_new = ray_o + t * ray_d

      dist = float(self.view.opts["distance"])
      new_dist = dist / zoom

      self.view.opts["center"].setX(float(C_new[0]))
      self.view.opts["center"].setY(float(C_new[1]))
      self.view.opts["center"].setZ(float(C_new[2]))
      self.view.setCameraPosition(distance=new_dist)
      self.view.update()
    except Exception:
      pass

  def _world_tol_from_px(self, p3, pick_tol_px):
    """World radius ~ pick_tol_px screen pixels at p3 (pyqtgraph pixelSize)."""
    p = np.asarray(p3, dtype=float)
    try:
      return max(1e-6, float(self.view.pixelSize(p)) * float(pick_tol_px))
    except Exception:
      dist = float(np.linalg.norm(p - np.array(self.view.cameraPosition(), dtype=float)))
      fov = float(self.view.opts.get("fov", 60.0))
      w = max(1.0, float(self._viewport_region()[2]))
      return max(1e-6, dist * 2.0 * np.tan(np.deg2rad(0.5 * fov)) / w * float(pick_tol_px))

  def _pick_bif(self, sx, sy):
    """Return bifurcation key for pixel (sx, sy), or None. Tries screen distance then ray."""
    if not self._bif_pick:
      return None
    m = np.array([sx, sy], dtype=float)
    best_key, best_d, best_tol = None, float("inf"), PICK_TOLERANCE_PX
    for key, p3, pick_tol in self._bif_pick:
      p2 = self._project(p3)
      if p2 is None:
        continue
      d = float(np.linalg.norm(m - p2))
      if d < best_d:
        best_d, best_key, best_tol = d, key, pick_tol
    if best_key is not None and best_d <= best_tol:
      return best_key
    return self._pick_bif_by_ray(sx, sy)

  def _pick_bif_by_ray(self, sx, sy):
    """Pick nearest bifurcation to the unprojected ray; tolerance scales with ``pixelSize``."""
    if self.view is None or not self._bif_pick:
      return None
    try:
      ray_o, ray_d = self._unproject_ray(sx, sy)
      if ray_o is None or ray_d is None:
        return None
      best_key, best_score = None, None
      for key, p3, pick_tol in self._bif_pick:
        p = np.asarray(p3, dtype=float)
        rel = p - ray_o
        t = float(np.dot(rel, ray_d))
        if t < 0.0:
          continue
        closest = ray_o + t * ray_d
        d_perp = float(np.linalg.norm(p - closest))
        tol_w = self._world_tol_from_px(p3, pick_tol)
        if d_perp <= tol_w:
          score = (d_perp, t)
          if best_score is None or score < best_score:
            best_score = score
            best_key = key
      return best_key
    except Exception:
      return None

  def clear(self):
    """Remove all GL drawables (no axis triad at the origin)."""
    self._bif_pick = []
    self.mesh_simplified = False
    if self.view is None:
      return
    for it in self.items:
      self.view.removeItem(it)
    self.items = []

  def _fit(self, points):
    """Frame the camera to bound ``points`` (used after loading new geometry)."""
    if self.view is None or points.size == 0:
      return 0.02
    mins, maxs = points.min(axis=0), points.max(axis=0)
    c = 0.5 * (mins + maxs)
    r = max(1e-3, 0.5 * np.max(maxs - mins))
    self.view.opts["center"].setX(float(c[0]))
    self.view.opts["center"].setY(float(c[1]))
    self.view.opts["center"].setZ(float(c[2]))
    self.view.setCameraPosition(distance=float(4.0 * r))
    return 0.016 * r

  def _line(self, pts, color, width=1.0, **gl_kw):
    item = gl.GLLinePlotItem(
      pos=np.asarray(pts, dtype=float), color=color, width=width, antialias=True, **gl_kw
    )
    self.view.addItem(item)
    self.items.append(item)

  def _digit(self, pos, text, color, scale, line_gl_kw=None):
    """Draw ``text`` as 7-segment polylines near ``pos`` (world units)."""
    line_kw = {} if line_gl_kw is None else line_gl_kw
    x = 0.0
    for ch in str(text):
      segs = self.DIG.get(ch)
      if segs is None:
        x += 1.25
        continue
      for seg in segs:
        (x0, y0), (x1, y1) = self.SEG[seg]
        p0 = np.array([x0 + x, y0, 0.0]) * scale + pos
        p1 = np.array([x1 + x, y1, 0.0]) * scale + pos
        self._line(np.vstack([p0, p1]), color, 1.0, **line_kw)
      x += 1.25

  def draw_centerlines(self, centerlines):
    """``centerlines``: iterable of ``(id, (n,3) ndarray)`` polylines."""
    self.clear()
    if self.view is None:
      return
    all_pts = []
    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      c = QColor.fromHsv((idx * 37) % 360, 220, 220)
      self._line(pts, (c.redF(), c.greenF(), c.blueF(), 1.0), 1.2)
      all_pts.append(pts)
    if not all_pts:
      return
    s = self._fit(np.concatenate(all_pts, axis=0))
    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      mid = pts[pts.shape[0] // 2]
      self._digit(mid + np.array([0.0, 0.0, 0.1 * s]), idx, (0.1, 0.1, 0.1, 1.0), 0.52 * s)

  def draw_bifurcations(self, graph, selected_key, preserve_camera=False):
    """Render graph edges, furcation nodes, and pick targets in ``self._bif_pick``."""
    self.clear()
    if self.view is None:
      return
    all_pts = []
    bif_pt_color = (1.0, 0.2, 0.2, 1.0)
    bif_sel_centroid = (0.04, 0.04, 0.04, 1.0)
    # Default pyqtgraph GL items use additive blending; stacked primaries read as yellow
    # at furcations. Opaque lines + markers keep true red / black on white.
    _bif_gl = {"glOptions": "opaque"}
    for idx, edge in zip(graph.indices, graph.edges):
      pts = edge.points[:, :3].astype(float)
      if pts.shape[0] < 2:
        continue
      c = QColor.fromHsv((idx * 37) % 360, 180, 220)
      self._line(pts, (c.redF(), c.greenF(), c.blueF(), 1.0), 1.0, **_bif_gl)
      all_pts.append(pts)
    unsel_centers, sel_centers = [], []
    for furc in graph.nontrivial_furcations:
      center = np.asarray(furc.centroid, dtype=float).ravel()
      key = frozenset(int(i) for i in furc.indices)
      sel = selected_key is not None and key == selected_key
      marker_size = SEL_BIF_BLACK_SIZE_PX if sel else RED_BIF_SIZE_PX
      # Make click area cover the visible marker radius.
      pick_tol = max(PICK_TOLERANCE_PX, 0.6 * marker_size)
      self._bif_pick.append((key, center, pick_tol))
      (sel_centers if sel else unsel_centers).append(center)
      for signed in furc.indices:
        edge = graph.get_edges([abs(signed)])[0]
        endpoint = edge.points[-1, :3] if signed > 0 else edge.points[0, :3]
        # Same red as markers; orange/green spokes looked yellow where they met the centroid.
        self._line(np.vstack([endpoint, center]), bif_pt_color, 2.4 if sel else 1.4, **_bif_gl)
    if unsel_centers:
      p = np.asarray(unsel_centers, dtype=float)
      s = gl.GLScatterPlotItem(
        pos=p, color=bif_pt_color, size=RED_BIF_SIZE_PX, glOptions="opaque"
      )
      s.setDepthValue(10)
      self.view.addItem(s); self.items.append(s); all_pts.append(p)
    if sel_centers:
      p = np.asarray(sel_centers, dtype=float)
      s = gl.GLScatterPlotItem(
        pos=p, color=bif_sel_centroid, size=SEL_BIF_BLACK_SIZE_PX, glOptions="opaque"
      )
      s.setDepthValue(11)
      self.view.addItem(s); self.items.append(s); all_pts.append(p)
    if not all_pts:
      return
    all_pts_arr = np.concatenate(all_pts, axis=0)
    if preserve_camera:
      # Keep current user camera (rotation/zoom/pan) while recomputing labels.
      mins, maxs = all_pts_arr.min(axis=0), all_pts_arr.max(axis=0)
      r = max(1e-3, 0.5 * np.max(maxs - mins))
      s = 0.016 * r
    else:
      s = self._fit(all_pts_arr)
    _bif_label_color = (1.0, 0.2, 0.2, 1.0)
    for idx, edge in zip(graph.indices, graph.edges):
      pts = edge.points[:, :3]
      if pts.shape[0] < 2:
        continue
      self._digit(
        pts[pts.shape[0] // 2] + np.array([0.0, 0.0, 0.09 * s]),
        idx,
        _bif_label_color,
        0.50 * s,
        line_gl_kw=_bif_gl,
      )
    for ib, furc in enumerate(graph.nontrivial_furcations, start=1):
      center = np.asarray(furc.centroid, dtype=float).ravel()
      key = frozenset(int(i) for i in furc.indices)
      sel = selected_key is not None and key == selected_key
      self._digit(
        center + np.array([0.0, 0.0, 0.12 * s]),
        ib,
        _bif_label_color,
        (0.60 if sel else 0.52) * s,
        line_gl_kw=_bif_gl,
      )

  def draw_mesh(self, mesh, preview_points=None, boundary_wire=None, boundary_scatter=None):
    """Draw **full** boundary wireframe + boundary vertices; fallback to point cloud.

    Cached ``boundary_wire`` / ``boundary_scatter`` from the worker avoid recomputing
    ``mesh.boundary`` on the GUI thread. Sets ``self.mesh_simplified`` if fallback
    subsampling is used.
    """
    self.clear()
    if self.view is None:
      return
    self.mesh_simplified = False
    wpos = boundary_wire
    scat = boundary_scatter
    if wpos is None and scat is None:
      wpos, scat = _boundary_preview_arrays(mesh)
    drew = False
    fit_pts = []
    # Saturated blue: strong contrast on white GL background.
    mesh_rgb = (0.02, 0.38, 0.95, 1.0)
    mesh_rgb_a = (0.02, 0.38, 0.95, 0.95)
    if wpos is not None and len(wpos) > 0:
      ln = gl.GLLinePlotItem(
        pos=wpos,
        color=mesh_rgb,
        width=1.1,
        antialias=True,
        mode="lines",
        glOptions="opaque",
      )
      self.view.addItem(ln)
      self.items.append(ln)
      fit_pts.append(wpos)
      drew = True
    if scat is not None and len(scat) > 0:
      s = gl.GLScatterPlotItem(
        pos=np.asarray(scat, dtype=float),
        color=mesh_rgb_a,
        size=3.0,
        glOptions="opaque",
      )
      self.view.addItem(s)
      self.items.append(s)
      fit_pts.append(np.asarray(scat, dtype=float))
      drew = True
    if drew:
      self._fit(np.vstack(fit_pts))
      return
    # Fallback: subsampled point cloud only.
    if preview_points is not None:
      points = np.asarray(preview_points, dtype=float)
    else:
      bpts = _extract_boundary_points_for_preview(mesh)
      if bpts is None:
        return
      points = bpts
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
      return
    points = points[:, :3]
    step = max(1, int(np.ceil(points.shape[0] / MAX_MESH_PREVIEW_POINTS)))
    self.mesh_simplified = step > 1
    p = points[::step]
    s = gl.GLScatterPlotItem(pos=p, color=mesh_rgb_a, size=2.5, glOptions="opaque")
    self.view.addItem(s)
    self.items.append(s)
    self._fit(p)


class MeshGeneratorWindow(QMainWindow):
  """Main window: CSV input, pipeline params, bifurcation list, mesh actions, 3D scene."""

  def __init__(self):
    super().__init__()
    self._init_logo()
    self.setWindowTitle("PyHaemo Mesh Generator")
    self.resize(1540, 816)
    self._base_window_width = 1540.0
    self._base_font_pt = 9.0
    self._last_font_pt = -1.0
    self.last_mesh = None
    self._mesh_preview_points = None
    self._mesh_preview_boundary_wire = None
    self._mesh_preview_boundary_scatter = None
    self._mesh_stats = None
    self._mesh_thread = None
    self.last_mesh_path = ""
    self._selected_bif_key = None
    self._removed_bif_keys = set()
    self._removed_centerline_ids = set()
    self._manual_additions = []
    self._last_view_mode = "Centerlines"

    # --- Central widget: splitter fills height; grip row fixed at bottom ------------
    root = QWidget()
    self.setCentralWidget(root)
    root_layout = QVBoxLayout(root)
    self._view_hint_label = QLabel(
      "3D view: drag to rotate · Ctrl+drag to pan · wheel zoom · Shift+drag a rectangle to zoom that region"
    )
    self._view_hint_label.setWordWrap(True)
    self._view_hint_label.setObjectName("viewHintLabel")
    hint_font = QFont(self._view_hint_label.font())
    hint_font.setPointSizeF(7.0)
    self._view_hint_label.setFont(hint_font)
    self._view_hint_label.setStyleSheet("color: #555555;")
    root_layout.addWidget(self._view_hint_label)
    splitter = QSplitter()
    root_layout.addWidget(splitter, stretch=1)

    # Status stays outside the scroll area so it remains visible when the left column scrolls.
    self.status = QLabel("Ready.")
    self.status.setWordWrap(True)
    self.status.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
    )

    # Scrollable column: input, bifurcations (incl. editing lists), mesh parameters.
    left_content = QWidget()
    left_content.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
    )
    left_layout = QVBoxLayout(left_content)
    left_layout.addWidget(self._input_group())
    left_layout.addWidget(self._bifurcation_group(), stretch=1)
    left_layout.addWidget(self._mesh_group())

    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QFrame.Shape.NoFrame)
    left_scroll.setHorizontalScrollBarPolicy(
      Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    left_scroll.setVerticalScrollBarPolicy(
      Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    left_scroll.setWidget(left_content)
    left_scroll.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
    )

    # Splitter child: scroll expands; status pinned under it.
    left_strip = QWidget()
    left_strip.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
    )
    left_strip_layout = QVBoxLayout(left_strip)
    left_strip_layout.setContentsMargins(0, 0, 0, 0)
    left_strip_layout.addWidget(left_scroll, stretch=1)
    left_strip_layout.addWidget(self.status)

    right = QWidget()
    right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    right_layout = QVBoxLayout(right)
    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.addStretch(1)
    header_logo = self._logo_label(width=130)
    if header_logo is not None:
      header_layout.addWidget(header_logo)
    header_layout.addStretch(1)
    right_layout.addWidget(header)
    self.scene = Scene3D(right)
    self.scene.on_bif_click = self._on_bif_clicked_3d
    right_layout.addWidget(self.scene, stretch=4)
    id_map_hdr = QLabel("Centerline ID map")
    id_map_hdr.setToolTip("Select a row, then use Delete Selected Centerline below.")
    right_layout.addWidget(id_map_hdr)
    self.id_map = QListWidget()
    right_layout.addWidget(self.id_map, stretch=1)
    cl_btn_row = QHBoxLayout()
    b_cl_del = QPushButton("Delete Selected Centerline")
    b_cl_del.clicked.connect(self._delete_selected_centerline)
    b_cl_reset = QPushButton("Reset Removed Centerlines")
    b_cl_reset.clicked.connect(self._reset_removed_centerlines)
    cl_btn_row.addWidget(b_cl_del)
    cl_btn_row.addWidget(b_cl_reset)
    right_layout.addLayout(cl_btn_row)

    splitter.addWidget(left_strip)
    splitter.addWidget(right)
    splitter.setSizes([600, 940])
    grip_row = QWidget()
    grip_layout = QHBoxLayout(grip_row)
    grip_layout.setContentsMargins(0, 0, 2, 2)
    grip_layout.addStretch(1)
    grip_layout.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
    root_layout.addWidget(grip_row)
    self.setMinimumSize(480, 360)
    self._update_ui_font()
    self._set_defaults()

  def resizeEvent(self, event):
    super().resizeEvent(event)
    self._update_ui_font()
    if hasattr(self, "_bif_edit_group"):
      _apply_compact_subtree_font(self._bif_edit_group, 8.0)

  def _update_ui_font(self):
    """Keep text readable with a small default and mild responsive scaling."""
    w = max(float(self.width()), 1.0)
    scale = w / self._base_window_width
    scale = min(1.06, max(0.94, scale))
    pt = self._base_font_pt * scale
    if abs(pt - self._last_font_pt) < 0.05:
      return
    f = self.font()
    f.setPointSizeF(pt)
    self.setFont(f)
    self._last_font_pt = pt
    if hasattr(self, "_view_hint_label"):
      hf = QFont(self._view_hint_label.font())
      hf.setPointSizeF(max(6.5, pt - 2.0))
      self._view_hint_label.setFont(hf)

  def _init_logo(self):
    """Set window icon from project logo if available."""
    logo_path = self._resolve_logo_path(kind="pictogram")
    if logo_path is None:
      return
    icon = QIcon(logo_path)
    if not icon.isNull():
      self.setWindowIcon(icon)

  def _resolve_logo_path(self, kind: str = "logotipo"):
    """Resolve preferred logo file under repo assets."""
    return _resolve_logo_path(kind=kind)

  def _logo_label(self, width=260):
    """Create a visible logo label (e.g. header)."""
    logo_path = self._resolve_logo_path(kind="logotipo")
    if logo_path is None:
      return None
    pix = QPixmap(logo_path)
    if pix.isNull():
      return None
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setPixmap(pix.scaledToWidth(int(width), Qt.TransformationMode.SmoothTransformation))
    return label

  def _set_status(self, msg):
    self.status.setText(msg)

  def _set_mesh_stats(self, stats):
    """Update the read-only mesh statistics label (volume counts / mean hex edge)."""
    self._mesh_stats = stats
    if not hasattr(self, "mesh_stats_label"):
      return
    if not stats:
      self.mesh_stats_label.setText("—")
      return
    avg = stats.get("mean_hex_edge_length", float("nan"))
    avg_txt = f"{avg:.6g}" if avg == avg and np.isfinite(avg) else "n/a"
    self.mesh_stats_label.setText(
      f"Cells: {stats['n_cells']:,} · Vertices: {stats['n_vertices']:,} · "
      f"Mean hex edge length: {avg_txt} · {stats.get('kind', '')}"
    )

  def _clear_cached_mesh(self):
    """Invalidate last generated mesh and all preview caches (e.g. after graph edits)."""
    self.last_mesh = None
    self._mesh_preview_points = None
    self._mesh_preview_boundary_wire = None
    self._mesh_preview_boundary_scatter = None
    self._set_mesh_stats(None)

  # --- QWidget groups (left panel) -------------------------------------------

  def _input_group(self):
    """CSV / output paths, view mode combo, and Refresh (builds ``self.csv_edit``, ``self.mode``, …)."""
    g = QGroupBox("Input - Output")
    l = QGridLayout(g)
    self.csv_edit = QLineEdit()
    b_csv = QPushButton("Browse...")
    b_csv.clicked.connect(self._choose_csv)
    self.out_edit = QLineEdit()
    b_out = QPushButton("Browse...")
    b_out.clicked.connect(self._choose_out)
    self.mode = QComboBox()
    self.mode.addItems(["Centerlines", "Bifurcations", "Mesh"])
    self.mode.currentIndexChanged.connect(self._on_view_mode_changed)
    b_refresh = QPushButton("Refresh Preview")
    b_refresh.clicked.connect(self._refresh)
    l.addWidget(QLabel("CSV folder"), 0, 0); l.addWidget(self.csv_edit, 0, 1); l.addWidget(b_csv, 0, 2)
    l.addWidget(QLabel("Output folder"), 1, 0); l.addWidget(self.out_edit, 1, 1); l.addWidget(b_out, 1, 2)
    l.addWidget(QLabel("View mode"), 2, 0); l.addWidget(self.mode, 2, 1, 1, 2)
    l.addWidget(b_refresh, 3, 0, 1, 3)
    return g

  def _bifurcation_group(self):
    """Pipeline epsilons, spline sample count, and ``self._bif_edit_group`` (lists + add/remove)."""
    g = QGroupBox("Bifurcations")
    l = QGridLayout(g)
    self.split_eps = QDoubleSpinBox(); self.split_eps.setDecimals(4); self.split_eps.setRange(0.0001, 10.0); self.split_eps.setValue(0.06); self.split_eps.valueChanged.connect(self._refresh)
    self.use_merge = QCheckBox("Use merge_furcs_proximity"); self.use_merge.setChecked(True); self.use_merge.stateChanged.connect(self._refresh)
    self.merge_eps = QDoubleSpinBox(); self.merge_eps.setDecimals(4); self.merge_eps.setRange(0.0001, 10.0); self.merge_eps.setValue(0.07); self.merge_eps.valueChanged.connect(self._refresh)
    self.spline_points = QSpinBox()
    self.spline_points.setRange(5, 500)
    self.spline_points.setValue(21)
    self.spline_points.valueChanged.connect(self._refresh)
    l.addWidget(QLabel("split_edges eps"), 0, 0); l.addWidget(self.split_eps, 0, 1)
    l.addWidget(self.use_merge, 1, 0, 1, 2)
    l.addWidget(QLabel("merge_furcs eps"), 2, 0); l.addWidget(self.merge_eps, 2, 1)
    l.addWidget(QLabel("Spline points"), 3, 0); l.addWidget(self.spline_points, 3, 1)
    self._bif_edit_group = self._edit_group()
    l.addWidget(self._bif_edit_group, 4, 0, 1, 2)
    # Let the editing block absorb extra vertical space inside this group box.
    l.setRowStretch(4, 1)
    g.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    return g

  def _edit_group(self):
    """Two-column UI: detected bifurcations vs manual edge-ID merges (compact font applied)."""
    g = QGroupBox("Bifurcation Editing")
    g.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    root = QVBoxLayout(g)
    self.bif_list = QListWidget()
    # Small floor height so the window can shrink; extra height comes from layout stretch.
    self.bif_list.setMinimumHeight(72)
    self.bif_list.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
    )
    self.bif_list.currentItemChanged.connect(self._on_bif_selected_list)
    b_del = QPushButton("Delete Selected Bifurcation"); b_del.clicked.connect(self._delete_bif)
    b_reset = QPushButton("Reset Deleted"); b_reset.clicked.connect(self._reset_deleted)
    self.add_edit = QLineEdit(); self.add_edit.setPlaceholderText("signed edge IDs, e.g. -3, 7, 12")
    b_add = QPushButton("Add Bifurcation From IDs"); b_add.clicked.connect(self._add_bif)
    self.add_list = QListWidget()
    self.add_list.setMinimumHeight(72)
    self.add_list.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
    )
    b_rm = QPushButton("Remove Selected Added"); b_rm.clicked.connect(self._remove_add)
    b_cl = QPushButton("Clear Added Bifurcations"); b_cl.clicked.connect(self._clear_add)
    r2 = QHBoxLayout(); r2.addWidget(self.add_edit); r2.addWidget(b_add)

    col_left = QVBoxLayout()
    col_left.addWidget(QLabel("Detected bifurcations (list → large black centroid in 3D; or click red marker)"))
    col_left.addWidget(self.bif_list, 1)
    col_left.addWidget(b_del)
    col_left.addWidget(b_reset)

    col_right = QVBoxLayout()
    col_right.addWidget(QLabel("Manual additions"))
    col_right.addLayout(r2)
    col_right.addWidget(self.add_list, 1)
    col_right.addWidget(b_rm)
    col_right.addWidget(b_cl)

    row = QHBoxLayout()
    row.addLayout(col_left, 1)
    row.addLayout(col_right, 1)
    # Stretch so the two-column row grows with the outer bifurcation group row.
    root.addLayout(row, 1)
    _apply_compact_subtree_font(g, 8.0)
    return g

  def _mesh_group(self):
    """Gmsh / sampling parameters and mesh action buttons (``self.mesh_dx``, …)."""
    g = QGroupBox("Mesh")
    l = QGridLayout(g)
    self.mesh_dx = QDoubleSpinBox(); self.mesh_dx.setDecimals(4); self.mesh_dx.setRange(0.0001, 1.0); self.mesh_dx.setValue(0.005)
    self.mesh_nref = QSpinBox(); self.mesh_nref.setRange(0, 10); self.mesh_nref.setValue(3)
    self.mesh_rfac = QDoubleSpinBox(); self.mesh_rfac.setDecimals(2); self.mesh_rfac.setRange(0.1, 20.0); self.mesh_rfac.setValue(2.0)
    self.mesh_section_points = QSpinBox()
    self.mesh_section_points.setRange(2, 200)
    self.mesh_section_points.setSingleStep(2)
    self.mesh_section_points.setValue(4)
    self.mesh_section_points.setToolTip(
      "Number of control points/elements per cross-section direction "
      "(must be even).")
    self.mesh_bl_exp = QDoubleSpinBox()
    self.mesh_bl_exp.setDecimals(2)
    self.mesh_bl_exp.setRange(1.0, 8.0)
    self.mesh_bl_exp.setValue(1.0)
    self.mesh_bl_exp.setSingleStep(0.05)
    self.mesh_bl_exp.setToolTip(
      "Radial grading on the unit disc: 1 = uniform; larger values push hex sampling "
      "toward the wall (boundary-layer-style clustering in parameter space).")
    self.mesh_union_eps = QDoubleSpinBox()
    self.mesh_union_eps.setDecimals(9)
    self.mesh_union_eps.setRange(1e-9, 1e-2)
    self.mesh_union_eps.setValue(1e-8)
    self.mesh_union_eps.setSingleStep(1e-9)
    self.mesh_union_eps.setToolTip(
      "KD-tree tolerance for mesh_union when gluing vessel and bifurcation volumes.")
    self.bif_exploit_symmetry = QCheckBox("Bif exploit symmetry")
    self.bif_exploit_symmetry.setToolTip("Experimental: may help symmetric bifurcation frame alignment.")
    self.b_gen = QPushButton("Generate Mesh"); self.b_gen.clicked.connect(self._generate_export_open)
    self.b_stop_gen = QPushButton("Stop Mesh Generation")
    self.b_stop_gen.setEnabled(False)
    self.b_stop_gen.clicked.connect(self._stop_mesh_generation)
    b_valid = QPushButton("Check Mesh Validity"); b_valid.clicked.connect(self._check_valid)
    b_export = QPushButton("Export .msh"); b_export.clicked.connect(self._export_last_mesh)
    r1 = QHBoxLayout(); r1.addWidget(self.b_gen); r1.addWidget(self.b_stop_gen); r1.addWidget(b_export)
    r2 = QHBoxLayout(); r2.addWidget(b_valid)
    l.addWidget(QLabel("dx"), 0, 0); l.addWidget(self.mesh_dx, 0, 1)
    l.addWidget(QLabel("nref"), 1, 0); l.addWidget(self.mesh_nref, 1, 1)
    l.addWidget(QLabel("rfac"), 2, 0); l.addWidget(self.mesh_rfac, 2, 1)
    l.addWidget(QLabel("section points"), 3, 0); l.addWidget(self.mesh_section_points, 3, 1)
    l.addWidget(QLabel("BL exp"), 4, 0); l.addWidget(self.mesh_bl_exp, 4, 1)
    l.addWidget(QLabel("union ε"), 5, 0); l.addWidget(self.mesh_union_eps, 5, 1)
    l.addWidget(self.bif_exploit_symmetry, 6, 0, 1, 2)
    l.addLayout(r1, 7, 0, 1, 2); l.addLayout(r2, 8, 0, 1, 2)
    self.mesh_stats_label = QLabel("—")
    self.mesh_stats_label.setWordWrap(True)
    l.addWidget(QLabel("Last mesh statistics"), 9, 0, 1, 2)
    l.addWidget(self.mesh_stats_label, 10, 0, 1, 2)
    return g

  # --- Slots and application logic -------------------------------------------

  def _set_defaults(self):
    root = os.path.dirname(__file__)
    csv_default = os.path.join(root, "data", "csv", "coro")
    out_default = os.path.join(root, "generated_meshes")
    os.makedirs(out_default, exist_ok=True)
    if os.path.isdir(csv_default):
      self.csv_edit.setText(csv_default)
      self._refresh()
    self.out_edit.setText(out_default)

  def _choose_csv(self):
    folder = QFileDialog.getExistingDirectory(self, "Select CSV folder")
    if not folder:
      return
    self.csv_edit.setText(folder)
    self._removed_bif_keys = set()
    self._removed_centerline_ids = set()
    self._manual_additions = []
    self._selected_bif_key = None
    self._clear_cached_mesh()
    self._refresh()

  def _choose_out(self):
    folder = QFileDialog.getExistingDirectory(self, "Select output folder")
    if folder:
      self.out_edit.setText(folder)

  def _load_centerlines(self, folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    cls, names = [], []
    display_i = 0
    for orig_i, fn in enumerate(files, start=1):
      if orig_i in self._removed_centerline_ids:
        continue
      data = np.genfromtxt(os.path.join(folder, fn), delimiter=",", skip_header=1)
      if data.size == 0:
        continue
      if data.ndim == 1:
        data = data[np.newaxis, :]
      if data.shape[1] < 3:
        continue
      display_i += 1
      cls.append((display_i, data[:, :3].astype(float)))
      names.append((orig_i, fn))
    return cls, names

  def _graph(self):
    return build_graph(
      csv_folder=self.csv_edit.text().strip(),
      split_eps=self.split_eps.value(),
      merge_eps=self.merge_eps.value(),
      use_merge=self.use_merge.isChecked(),
      removed_keys=self._removed_bif_keys,
      manual_additions=self._manual_additions,
      removed_centerline_ids=frozenset(self._removed_centerline_ids),
    )

  def _reload_preview_data(self):
    """Reload centerlines CSVs and rebuild the vascular graph + bif lists."""
    folder = self.csv_edit.text().strip()
    if not folder or not os.path.isdir(folder):
      self._set_status("Select a valid CSV folder.")
      return False
    self._centerlines, self._filenames = self._load_centerlines(folder)
    self._graph_cache = self._graph()
    self._fill_bif_list()
    self._fill_add_list()
    return True

  def _on_view_mode_changed(self, index):
    """Keep preview in sync when switching modes (no extra Refresh click for bif)."""
    if index < 0:
      return
    prev = self._last_view_mode
    mode = self.mode.itemText(index)
    if mode == "Bifurcations":
      try:
        if not self._reload_preview_data():
          self._last_view_mode = mode
          return
      except Exception:
        self._set_status(f"Preview error: {traceback.format_exc().splitlines()[-1]}")
        self._last_view_mode = mode
        return
      # Match Refresh Preview framing: refit camera to bif geometry (preserve=False).
      self._render_mode(preserve_camera=False)
      self._last_view_mode = mode
      return
    if mode == "Centerlines":
      self._render_mode(preserve_camera=(prev == "Bifurcations"))
      self._last_view_mode = mode
      return
    self._render_mode(preserve_camera=False)
    self._last_view_mode = mode

  def _refresh(self):
    folder = self.csv_edit.text().strip()
    if not folder or not os.path.isdir(folder):
      self._set_status("Select a valid CSV folder.")
      return
    try:
      if not self._reload_preview_data():
        return
      self._last_view_mode = self.mode.currentText()
      self._render_mode()
    except Exception:
      self._set_status(f"Preview error: {traceback.format_exc().splitlines()[-1]}")

  def _fill_bif_list(self):
    self.bif_list.clear()
    for ib, furc in enumerate(self._graph_cache.nontrivial_furcations, start=1):
      key = frozenset(int(v) for v in furc.indices)
      item = QListWidgetItem(f"B{ib}: indices {list(map(int, furc.indices))}")
      item.setData(Qt.ItemDataRole.UserRole, key)
      self.bif_list.addItem(item)

  def _fill_add_list(self):
    self.add_list.clear()
    for i, tpl in enumerate(self._manual_additions, start=1):
      self.add_list.addItem(f"A{i}: {list(tpl)}")

  def _select_bif_item_by_key(self, key):
    for row in range(self.bif_list.count()):
      item = self.bif_list.item(row)
      if item.data(Qt.ItemDataRole.UserRole) == key:
        self.bif_list.blockSignals(True)
        self.bif_list.setCurrentRow(row)
        self.bif_list.blockSignals(False)
        self._selected_bif_key = key
        if self.mode.currentText() != "Bifurcations":
          self.mode.blockSignals(True)
          self.mode.setCurrentText("Bifurcations")
          self.mode.blockSignals(False)
        self._last_view_mode = "Bifurcations"
        self._render_mode(preserve_camera=True)
        self._set_status(f"Selected bifurcation: {item.text()}")
        return

  def _on_bif_clicked_3d(self, key):
    self._set_status("3D click detected. Selecting nearest bifurcation...")
    self._select_bif_item_by_key(key)

  def _on_bif_selected_list(self, current, _prev):
    self._selected_bif_key = None if current is None else current.data(Qt.ItemDataRole.UserRole)
    if self.mode.currentText() != "Bifurcations":
      self.mode.blockSignals(True)
      self.mode.setCurrentText("Bifurcations")
      self.mode.blockSignals(False)
    self._last_view_mode = "Bifurcations"
    self._render_mode(preserve_camera=True)

  def _delete_bif(self):
    item = self.bif_list.currentItem()
    if item is None:
      self._set_status("Select a bifurcation first.")
      return
    self._removed_bif_keys.add(item.data(Qt.ItemDataRole.UserRole))
    self._selected_bif_key = None
    self._clear_cached_mesh()
    self._refresh()
    self._set_status("Selected bifurcation deleted.")

  def _reset_deleted(self):
    self._removed_bif_keys = set()
    self._selected_bif_key = None
    self._clear_cached_mesh()
    self._refresh()
    self._set_status("Deleted bifurcations reset.")

  def _delete_selected_centerline(self):
    item = self.id_map.currentItem()
    if item is None:
      self._set_status("Select a centerline in the ID map first.")
      return
    orig_i = item.data(Qt.ItemDataRole.UserRole)
    if orig_i is None:
      self._set_status("Select a centerline in the ID map first.")
      return
    orig_i = int(orig_i)
    if len(self._centerlines) <= 1:
      self._set_status("Cannot delete the last remaining centerline.")
      return
    self._removed_centerline_ids.add(orig_i)
    self._removed_bif_keys = set()
    self._manual_additions = []
    self._selected_bif_key = None
    self._clear_cached_mesh()
    self._refresh()
    self._set_status(
      "Centerline removed. Bifurcation edits were cleared (graph indices changed)."
    )

  def _reset_removed_centerlines(self):
    if not self._removed_centerline_ids:
      self._set_status("No removed centerlines to restore.")
      return
    self._removed_centerline_ids = set()
    self._clear_cached_mesh()
    self._refresh()
    self._set_status("Removed centerlines restored.")

  def _parse_ids(self, text):
    vals = [int(p.strip()) for p in text.replace(";", ",").split(",") if p.strip()]
    if len(vals) < 2:
      raise ValueError("Provide at least two signed edge IDs.")
    if len(set(vals)) != len(vals):
      raise ValueError("Duplicate IDs are not allowed.")
    if any(v == 0 for v in vals):
      raise ValueError("Edge IDs cannot be zero.")
    return tuple(vals)

  def _add_bif(self):
    txt = self.add_edit.text().strip()
    if not txt:
      self._set_status("Insert signed edge IDs first.")
      return
    try:
      tpl = self._parse_ids(txt)
      self._manual_additions.append(tpl)
      self.add_edit.clear()
      self._clear_cached_mesh()
      self._refresh()
      self._set_status(f"Manual bifurcation added: {list(tpl)}")
    except Exception as exc:
      self._set_status(f"Invalid bifurcation input: {exc}")

  def _remove_add(self):
    row = self.add_list.currentRow()
    if row < 0:
      self._set_status("Select an added bifurcation first.")
      return
    del self._manual_additions[row]
    self._clear_cached_mesh()
    self._refresh()
    self._set_status("Selected manual addition removed.")

  def _clear_add(self):
    self._manual_additions = []
    self._clear_cached_mesh()
    self._refresh()
    self._set_status("Manual additions cleared.")

  def _render_mode(self, preserve_camera=False):
    """Refresh the 3D scene and ID list for the current ``self.mode``."""
    self.id_map.clear()
    mode = self.mode.currentText()
    if mode == "Mesh":
      if self.last_mesh is None:
        self._set_status("No mesh loaded yet. Generate mesh first.")
        return
      self.scene.draw_mesh(
        self.last_mesh,
        self._mesh_preview_points,
        self._mesh_preview_boundary_wire,
        self._mesh_preview_boundary_scatter,
      )
      if self.scene.mesh_simplified:
        self._set_status("Mesh preview: boundary point cloud only (subsampled); statistics in the Mesh panel.")
      else:
        self._set_status("Mesh preview: full boundary surface (edges + vertices) + statistics in the Mesh panel.")
      return
    if mode == "Bifurcations":
      self.scene.draw_bifurcations(self._graph_cache, self._selected_bif_key, preserve_camera=preserve_camera)
      for disp, (orig_i, name) in enumerate(self._filenames, start=1):
        it = QListWidgetItem(f"{disp} → {name}")
        it.setData(Qt.ItemDataRole.UserRole, orig_i)
        self.id_map.addItem(it)
      self._set_status(
        f"Bifurcation view loaded: {len(self._graph_cache.nontrivial_furcations)} nodes. "
        "List selection shows a large black centroid; unselected furcs are red."
      )
      return
    self.scene.draw_centerlines(self._centerlines)
    for disp, (orig_i, name) in enumerate(self._filenames, start=1):
      it = QListWidgetItem(f"{disp} → {name}")
      it.setData(Qt.ItemDataRole.UserRole, orig_i)
      self.id_map.addItem(it)
    self._set_status(f"Centerline view loaded: {len(self._centerlines)} lines.")

  def _generate_export_open(self):
    """Start :class:`MeshComputeThread` with a snapshot of current UI parameters."""
    if self._mesh_thread is not None and self._mesh_thread.isRunning():
      self._set_status("Mesh generation already in progress.")
      return
    folder = self.csv_edit.text().strip()
    if not folder or not os.path.isdir(folder):
      self._set_status("Select a valid CSV folder.")
      return
    section_points = int(self.mesh_section_points.value())
    if section_points % 2 != 0:
      section_points += 1
      self.mesh_section_points.setValue(section_points)
      self._set_status("Section points must be even; rounded up by 1.")
    kw = dict(
      csv_folder=folder,
      split_eps=self.split_eps.value(),
      merge_eps=self.merge_eps.value(),
      use_merge=self.use_merge.isChecked(),
      spline_points=self.spline_points.value(),
      mesh_dx=self.mesh_dx.value(),
      mesh_nref=self.mesh_nref.value(),
      mesh_rfac=self.mesh_rfac.value(),
      mesh_section_points=section_points,
      boundary_layer_exponent=self.mesh_bl_exp.value(),
      mesh_union_eps=self.mesh_union_eps.value(),
      bif_exploit_symmetry=self.bif_exploit_symmetry.isChecked(),
      removed_keys=set(self._removed_bif_keys),
      manual_additions=list(self._manual_additions),
      removed_centerline_ids=frozenset(self._removed_centerline_ids),
    )

    def work():
      mesh = generate_mesh(**kw)
      bpts = _extract_boundary_points_for_preview(mesh)
      stats = mesh_statistics(mesh)
      bwire, bscatter = _boundary_preview_arrays(mesh)
      return mesh, bpts, stats, bwire, bscatter

    self.b_gen.setEnabled(False)
    self.b_stop_gen.setEnabled(True)
    self._set_status("Generating mesh in background… You can keep using the window.")
    QApplication.processEvents()
    th = MeshComputeThread(work)
    self._mesh_thread = th
    th.finished_ok.connect(self._on_mesh_thread_ok)
    th.finished_err.connect(self._on_mesh_thread_err)
    th.finished.connect(self._on_mesh_thread_finished)
    th.start()

  def _on_mesh_thread_ok(self, mesh, bpts, stats, bwire, bscatter):
    self.last_mesh = mesh
    self._mesh_preview_points = bpts
    self._mesh_preview_boundary_wire = bwire
    self._mesh_preview_boundary_scatter = bscatter
    self._set_mesh_stats(stats)
    if self.mode.currentText() != "Mesh":
      self.mode.blockSignals(True)
      self.mode.setCurrentText("Mesh")
      self.mode.blockSignals(False)
    self._render_mode()
    self._set_status("Mesh generated. Full boundary preview + statistics below.")

  def _on_mesh_thread_err(self, tb):
    self._set_status(f"Generation error: {tb.strip().splitlines()[-1]}")

  def _on_mesh_thread_finished(self):
    self.b_gen.setEnabled(True)
    self.b_stop_gen.setEnabled(False)
    th = self._mesh_thread
    self._mesh_thread = None
    if th is not None:
      th.deleteLater()

  def _stop_mesh_generation(self):
    if self._mesh_thread is None or not self._mesh_thread.isRunning():
      self._set_status("No mesh generation is currently running.")
      self.b_stop_gen.setEnabled(False)
      return
    self._set_status("Stopping mesh generation…")
    # Work function is CPU-bound and does not poll interruption flags.
    # Use terminate as an emergency stop from the GUI.
    self._mesh_thread.terminate()
    self._mesh_thread.wait(5000)

  def closeEvent(self, event):
    """Block exit until the mesh worker finishes (avoids tearing down mid-build)."""
    if self._mesh_thread is not None and self._mesh_thread.isRunning():
      self._set_status("Waiting for mesh generation to finish before closing…")
      QApplication.processEvents()
      self._mesh_thread.wait(600_000)
    event.accept()

  def _export_last_mesh(self):
    if self.last_mesh is None:
      self._set_status("No mesh cached. Generate mesh first.")
      return
    out = self.out_edit.text().strip()
    if not out:
      self._set_status("Please select an output folder.")
      return
    os.makedirs(out, exist_ok=True)
    try:
      path = self._save_msh(self.last_mesh, out)
      self.last_mesh_path = path
      self._set_status(f"Mesh exported: {path}")
    except Exception:
      self._set_status(f"Export error: {traceback.format_exc().splitlines()[-1]}")

  def _check_valid(self):
    if self.last_mesh is None:
      self._set_status("No mesh cached. Generate first.")
      return
    try:
      self._set_status(f"is mesh valid = {is_mesh_valid(self.last_mesh)}")
    except Exception:
      self._set_status(f"Mesh validity error: {traceback.format_exc().splitlines()[-1]}")

  def _save_msh(self, mesh, out_folder):
    path = os.path.join(out_folder, f"partery_mesh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.msh")
    if not hasattr(mesh, "export_gmsh"):
      raise RuntimeError(f"Generated object type {type(mesh).__name__} does not support export_gmsh().")
    mesh.export_gmsh(path)
    return path



def main():
  """Application entry: single main window, standard Qt event loop."""
  app = QApplication(sys.argv)
  app.setApplicationName("pyhaemo-mesh-generator")
  app.setApplicationDisplayName("PyHaemo Mesh Generator")
  if hasattr(app, "setDesktopFileName"):
    app.setDesktopFileName("pyhaemo-mesh-generator")
  app_logo_path = _resolve_logo_path(kind="pictogram")
  if app_logo_path is not None:
    app_icon = QIcon(app_logo_path)
    if not app_icon.isNull():
      app.setWindowIcon(app_icon)
  splash = None
  logo_path = _resolve_logo_path(kind="pictogram")
  if logo_path is not None:
    pix = QPixmap(logo_path)
    if not pix.isNull():
      pix_s = pix.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation)
      splash = QSplashScreen(pix_s)
      def _center_splash():
        scr = splash.screen() or app.primaryScreen()
        if scr is None:
          return
        ag = scr.availableGeometry()
        fg = splash.frameGeometry()
        splash.move(ag.center().x() - fg.width() // 2, ag.center().y() - fg.height() // 2)

      splash.show()
      QTimer.singleShot(0, _center_splash)
      app.processEvents()
      _center_splash()
      app.processEvents()
      time.sleep(3.0)
      app.processEvents()
  w = MeshGeneratorWindow()
  w.show()
  if splash is not None:
    splash.finish(w)
  return app.exec()


if __name__ == "__main__":
  raise SystemExit(main())
