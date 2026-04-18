"""PyQt6 front-end for the partery vascular mesh pipeline.

Loads centerline CSVs, builds a :class:`~artery.VascularGraph`, and offers:
  - 3D centerline and bifurcation views (pyqtgraph OpenGL, no embedded VTK)
  - **Fix bifurcation** view after meshing: optional mesh + **bif-view** vessel
    polylines; full branch-anchor picks are applied on **Generate mesh** (trim
    incident edges at picked points before the normal mesh pipeline).  Every
    junction with a **complete** set of picks stays in the queue across mesh
    runs so you can fix one bifurcation, generate, then fix another without
    losing earlier trims (until you clear anchors or invalidate the graph cache).
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

# --- Centerline 3D picking (screen pixels) ----------------------------------
# Max screen-space distance (click pixel → nearest polyline segment) to accept
# a centerline click. Looser than the bifurcation threshold because polylines
# are thin and the user aims for a line, not a marker.
CL_PICK_TOLERANCE_PX = 10.0

# Fallback boundary preview point cloud (matches blue wire/vert accent in ``draw_mesh``).
MESH_FALLBACK_SCATTER_RGBA = (0.02, 0.38, 0.95, 1.0)

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


def _boundary_face_tris(boundary_elements, unique_idx, nverts):
  """Triangulate boundary facets and remap indices to the compact scatter array.

  ``unique_idx`` is the sorted ``np.unique`` of ``boundary_elements`` used to
  build the scatter vertex buffer. Returns ``(n_tris, 3)`` int64 indices into
  that compact vertex buffer, or ``None`` if facet shape is unsupported.
  """
  bel = np.asarray(boundary_elements, dtype=np.int64)
  idx = np.asarray(unique_idx, dtype=np.int64)
  if bel.ndim != 2 or bel.size == 0 or idx.size == 0:
    return None
  lookup = np.empty(int(idx.max()) + 1, dtype=np.int64)
  lookup[idx] = np.arange(idx.size, dtype=np.int64)
  local = lookup[bel]
  if nverts == 4 and local.shape[1] == 4:
    # Quad local layout matches ``_QUAD_BOUNDARY_LOCAL_EDGES``:
    #   2 --- 3
    #   |     |
    #   0 --- 1
    # → triangles (0,1,3) and (0,3,2).
    t1 = np.column_stack([local[:, 0], local[:, 1], local[:, 3]])
    t2 = np.column_stack([local[:, 0], local[:, 3], local[:, 2]])
    return np.ascontiguousarray(np.concatenate([t1, t2], axis=0))
  if nverts == 3 and local.shape[1] == 3:
    return np.ascontiguousarray(local.copy())
  return None


def _boundary_preview_arrays(mesh):
  """Precompute OpenGL arrays for the **entire** exterior surface of ``mesh``.

  Returns
  -------
  wire_pos, scatter_xyz, face_tris : ndarray or None
    ``wire_pos`` is line-pair vertices for the wireframe;
    ``scatter_xyz`` is one row per distinct boundary node (vertex buffer);
    ``face_tris`` is ``(n_tris, 3)`` int indices into ``scatter_xyz`` for the
    filled boundary surface. On failure (no boundary), returns
    ``(None, None, None)``.
  """
  try:
    b = mesh.boundary
  except Exception:
    return None, None, None
  pts = np.asarray(mesh.points, dtype=float)
  bel = np.asarray(b.elements, dtype=np.int64)
  if bel.size == 0 or pts.size == 0:
    return None, None, None
  nv = int(getattr(b, "nverts", bel.shape[1]))
  wpos = _full_boundary_wireframe_positions(pts, bel, nv)
  if wpos is None:
    return None, None, None
  idx = np.unique(bel.ravel())
  scatter = np.ascontiguousarray(pts[np.asarray(idx, dtype=np.int64), :3].copy())
  tris = _boundary_face_tris(bel, idx, nv)
  return wpos, scatter, tris


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


def _trim_vascular_edge_at_anchor(edge, signed_branch: int, anchor_xyz: np.ndarray):
  """Move the bif-attached endpoint of ``edge`` to the polyline point closest to ``anchor_xyz``.

  Positive ``signed_branch`` means the bifurcation meets the edge at ``points[-1]``;
  negative means it meets at ``points[0]``.
  """
  pts = np.asarray(edge.points, dtype=float)
  n = int(pts.shape[0])
  if n < 2 or pts.shape[1] < 4:
    return edge
  p3 = pts[:, :3]
  a = np.asarray(anchor_xyz, dtype=float).ravel()[:3]
  best_seg, best_t, best_d = 0, 0.0, float("inf")
  for i in range(n - 1):
    p0, p1 = p3[i], p3[i + 1]
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-18:
      continue
    t = float(np.clip(np.dot(a - p0, d) / denom, 0.0, 1.0))
    q = p0 + t * d
    dist = float(np.linalg.norm(q - a))
    if dist < best_d:
      best_d, best_seg, best_t = dist, i, t
  seg, t = best_seg, best_t
  r0, r1 = float(pts[seg, 3]), float(pts[seg + 1, 3])
  q3 = p3[seg] + t * (p3[seg + 1] - p3[seg])
  q4 = np.array([q3[0], q3[1], q3[2], (1.0 - t) * r0 + t * r1], dtype=float)
  if signed_branch > 0:
    new_pts = np.vstack([pts[: seg + 1], q4.reshape(1, 4)])
  else:
    new_pts = np.vstack([q4.reshape(1, 4), pts[seg + 1 :]])
  try:
    out = edge._edit(points=new_pts, frame=None).remove_duplicates(thresh=1e-9)
  except Exception:
    return edge
  if out.npoints < 2:
    return edge
  return out


def apply_fix_branch_anchors_to_vgraph(vgraph, bif_key: frozenset, anchors: dict[int, np.ndarray]):
  """Trim each incident edge so bif endpoints coincide with user-picked locations."""
  if not anchors or not bif_key:
    return vgraph
  furc = None
  key_set = frozenset(int(i) for i in bif_key)
  for f in vgraph.nontrivial_furcations:
    if frozenset(int(i) for i in f.indices) == key_set:
      furc = f
      break
  if furc is None:
    return vgraph
  need = {abs(int(s)) for s in furc.indices}
  if set(int(k) for k in anchors.keys()) != need:
    return vgraph
  edge_sign = {abs(int(s)): int(s) for s in furc.indices}
  new_edges = []
  for i, e in zip(vgraph.indices, vgraph.edges):
    iabs = int(i)
    if iabs in anchors and iabs in edge_sign:
      new_edges.append(_trim_vascular_edge_at_anchor(e, edge_sign[iabs], anchors[iabs]))
    else:
      new_edges.append(e)
  return vgraph._edit(edges=new_edges).recompute_centroids()


def generate_mesh(csv_folder, split_eps, merge_eps, use_merge, mesh_dx, mesh_nref, mesh_rfac,
                  mesh_section_points, removed_keys, manual_additions,
                  boundary_layer_exponent=1.0,
                  mesh_union_eps=1e-8, bif_exploit_symmetry=False,
                  removed_centerline_ids=frozenset(),
                  merge_opt_in=(),
                  fix_anchor_bif_key=None,
                  fix_branch_anchors=None,
                  fix_anchor_specs=()):
  """Run the full artery pipeline and return the volumetric hex mesh (CPU-heavy).

  ``merge_opt_in`` is an iterable of "component specs"; each spec is itself
  an iterable of bif-keys where a bif-key is an iterable of unsigned edge
  ids.  Every listed component is collapsed into a single n-furcation at
  the :class:`VascularGraph` level — **before** ``align_nontrivial_normals``
  and ``align_frames`` run — so the merged n-furcation goes through the
  identical normal-alignment / frame-rebuild pipeline as any naturally
  occurring n-furcation.  This is what keeps
  :func:`bif.bif_from_matrices`'s CCW wing-stitching consistent for merged
  bifs: the endpoint frames it reads on the :class:`SplineGraph` side are
  indistinguishable (up to polyline-vs-spline tangent fit residual) from
  those of natural bifs.

  Centerline B-splines use ``mesh_dx`` via :meth:`artery.VascularGraph.to_spline_graph`
  (``dx=mesh_dx``): each edge gets ``max(2, int(length / mesh_dx))`` knot
  intervals, matching the longitudinal sampling of the volumetric mesh.

  ``fix_anchor_bif_key`` / ``fix_branch_anchors`` (optional): legacy single-bif
  form; same trimming as one entry of ``fix_anchor_specs``.

  ``fix_anchor_specs`` (optional): iterable of ``(bif_key_frozenset, anchors_dict)``
  for multiple junctions; each ``anchors_dict`` maps unsigned edge id → 3D point.
  Every complete spec is applied in sorted edge-id order. When **any** spec is
  applied, ``rfac_retry=False`` for the whole mesh run (see above).

  GUI **Fix bifurcation** stores one spec per bif in ``fix_anchor_specs``.

  When any fix-anchor trimming runs (``fix_anchor_specs`` or the legacy single-bif
  kwargs), the downstream :meth:`SplineGraph.make_mesh` call uses
  ``rfac_retry=False`` for that entire run.
  """
  from mesh import apply_merge_opt_in_vgraph
  artery = build_graph(
    csv_folder, split_eps, merge_eps, use_merge, removed_keys, manual_additions,
    removed_centerline_ids=removed_centerline_ids,
  )
  artery = artery.orient_nontrivial()
  # Apply opt-in merges on the VascularGraph, *before* normal / frame
  # alignment.  ``merged_key_to_members`` maps the merged n-furcation's
  # own bif-key back to the frozenset of constituent bif-keys it absorbed
  # — :func:`make_spline_mesh` uses it to back-fill validity/repair info
  # onto those member bif-keys (for diagnostics); the GUI list/3D bif view
  # show the merged n-furcation row instead of the absorbed ones.
  artery, merged_key_to_members = apply_merge_opt_in_vgraph(
    artery, merge_opt_in,
  )
  fix_anchors_applied = False
  if fix_anchor_specs:
    for _fk_raw, adict in sorted(
      fix_anchor_specs,
      key=lambda sp: tuple(sorted(abs(int(x)) for x in sp[0])),
    ):
      fk = frozenset(int(i) for i in _fk_raw)
      need = {abs(int(x)) for x in fk}
      if need != {int(k) for k in adict.keys()}:
        continue
      anchors = {
        int(k): np.asarray(v, dtype=float).reshape(3)
        for k, v in adict.items()
      }
      artery = apply_fix_branch_anchors_to_vgraph(artery, fk, anchors)
      fix_anchors_applied = True
  elif fix_branch_anchors and fix_anchor_bif_key is not None:
    fk = frozenset(int(i) for i in fix_anchor_bif_key)
    need = {abs(int(x)) for x in fk}
    if need == {int(k) for k in fix_branch_anchors.keys()}:
      anchors = {
        int(k): np.asarray(v, dtype=float).reshape(3)
        for k, v in fix_branch_anchors.items()
      }
      artery = apply_fix_branch_anchors_to_vgraph(artery, fk, anchors)
      fix_anchors_applied = True
  artery = (
    artery.align_nontrivial_normals()
    .orient_trivial()
    .rtruncate(1)
    .furcomplete(npoints=1001)
    .delete_twofurcations()
    .align_frames()
  )
  # Spline knot density along each vessel matches volumetric ``mesh_dx``:
  # ``max(2, int(edge.length / mesh_dx))`` per edge (see ``to_spline_graph``).
  return artery.to_spline_graph(dx=mesh_dx).make_mesh(
    dx=mesh_dx, nref=mesh_nref, rfac=mesh_rfac, cross_nelems=mesh_section_points,
    union_eps=mesh_union_eps, bif_exploit_symmetry=bif_exploit_symmetry,
    boundary_layer_exponent=boundary_layer_exponent,
    merged_key_to_members=merged_key_to_members,
    rfac_retry=not fix_anchors_applied,
  )


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

  ``finished_ok`` payloads:
    ``(mesh, aux_points, stats, boundary_wire, boundary_scatter,
       boundary_faces, bif_validity, merge_components)``.
  """

  finished_ok = pyqtSignal(object, object, object, object, object, object, object, object)
  finished_err = pyqtSignal(str)

  def __init__(self, work_fn):
    super().__init__()
    self._work_fn = work_fn

  def run(self):
    try:
      result = self._work_fn()
      # Tolerate both old 7-tuple and new 8-tuple layouts so an in-flight
      # worker from a previous build doesn't crash after a hot reload.
      if len(result) == 8:
        mesh, bpts, stats, bwire, bscatter, bfaces, bvalid, mcomp = result
      else:
        mesh, bpts, stats, bwire, bscatter, bfaces, bvalid = result
        mcomp = ()
      self.finished_ok.emit(
        mesh, bpts, stats, bwire, bscatter, bfaces, bvalid, mcomp,
      )
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
    self.on_cl_click = None
    self.on_cl_click_detail = None
    self.mesh_simplified = False
    self._bif_pick = []
    self._cl_pick = []
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
    # Fix-bifurcation mode: only polyline picks (no bif markers drawn).
    if callable(self.on_cl_click_detail):
      det = self._pick_centerline_detail(sx, sy)
      if det is not None:
        self.on_cl_click_detail(int(det[0]), det[1])
      return
    # Bifurcation markers are small; try them first (they beat the looser
    # centerline tolerance at close range).
    key = self._pick_bif(sx, sy)
    if key is not None and callable(self.on_bif_click):
      self.on_bif_click(key)
      return
    cl_id = self._pick_centerline(sx, sy)
    if cl_id is not None and callable(self.on_cl_click):
      self.on_cl_click(cl_id)

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

  def _pick_centerline_detail(self, sx, sy):
    """Return ``(display_cl_id, world_xyz)`` for the closest polyline segment, or None."""
    if self.view is None or not self._cl_pick:
      return None
    m = np.array([float(sx), float(sy)], dtype=float)
    best_id, best_d, best_xyz = None, float("inf"), None
    for cl_id, pts in self._cl_pick:
      if pts.shape[0] < 2:
        continue
      proj = np.full((pts.shape[0], 2), np.nan, dtype=float)
      for k in range(pts.shape[0]):
        p2 = self._project(pts[k])
        if p2 is not None:
          proj[k] = p2
      a = proj[:-1]
      b = proj[1:]
      p3a = np.asarray(pts[:-1], dtype=float)
      p3b = np.asarray(pts[1:], dtype=float)
      valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
      if not np.any(valid):
        continue
      a = a[valid]
      b = b[valid]
      p3a = p3a[valid]
      p3b = p3b[valid]
      ab = b - a
      ab_len2 = np.einsum("ij,ij->i", ab, ab)
      safe = ab_len2 > 1e-12
      t = np.zeros(a.shape[0], dtype=float)
      t[safe] = np.clip(
        np.einsum("ij,ij->i", (m - a[safe]), ab[safe]) / ab_len2[safe],
        0.0,
        1.0,
      )
      closest = a + t[:, None] * ab
      d = np.linalg.norm(m - closest, axis=1)
      i_min = int(np.argmin(d))
      dmin = float(d[i_min])
      if dmin < best_d:
        best_d = dmin
        tt = float(t[i_min])
        xyz = p3a[i_min] + tt * (p3b[i_min] - p3a[i_min])
        best_id, best_xyz = int(cl_id), np.asarray(xyz, dtype=float)
    if best_id is not None and best_d <= CL_PICK_TOLERANCE_PX:
      return best_id, best_xyz
    return None

  def _pick_centerline(self, sx, sy):
    """Return centerline display id closest (in screen pixels) to (sx, sy).

    Projects each polyline vertex to screen space and measures the minimum
    point-to-segment distance from the click to each polyline. The nearest
    polyline wins if its distance is within :data:`CL_PICK_TOLERANCE_PX`.
    """
    det = self._pick_centerline_detail(sx, sy)
    return det[0] if det is not None else None

  def clear(self):
    """Remove all GL drawables (no axis triad at the origin)."""
    self._bif_pick = []
    self._cl_pick = []
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

  def draw_centerlines(self, centerlines, selected_id=None, preserve_camera=False):
    """Render centerline polylines; optionally highlight one.

    Parameters
    ----------
    centerlines : iterable of ``(id, (n,3) ndarray)``
      Polylines with their display id (same id shown in the ID-map panel).
    selected_id : int or None
      If given, the matching polyline is drawn thicker in black so list- and
      3D-driven selections stay visually in sync.
    preserve_camera : bool
      If True, keep the current camera pose (used when re-rendering on
      selection changes so the view does not jump).
    """
    self.clear()
    if self.view is None:
      return
    centerlines = list(centerlines)
    all_pts = []
    # Base pass: one colored line per centerline + pick record (hit-testing).
    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      c = QColor.fromHsv((idx * 37) % 360, 220, 220)
      self._line(pts, (c.redF(), c.greenF(), c.blueF(), 1.0), 1.2)
      self._cl_pick.append((int(idx), np.asarray(pts, dtype=float)))
      all_pts.append(pts)
    if not all_pts:
      return
    all_pts_arr = np.concatenate(all_pts, axis=0)
    if preserve_camera:
      mins, maxs = all_pts_arr.min(axis=0), all_pts_arr.max(axis=0)
      r = max(1e-3, 0.5 * np.max(maxs - mins))
      s = 0.016 * r
    else:
      s = self._fit(all_pts_arr)
    # Overlay: thick black line for the selected centerline (drawn last so it
    # wins the depth test against its colored base line).
    if selected_id is not None:
      for idx, pts in centerlines:
        if int(idx) != int(selected_id) or pts.shape[0] < 2:
          continue
        self._line(pts, (0.04, 0.04, 0.04, 1.0), 3.0, glOptions="opaque")
        break
    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      mid = pts[pts.shape[0] // 2]
      self._digit(mid + np.array([0.0, 0.0, 0.1 * s]), idx, (0.1, 0.1, 0.1, 1.0), 0.52 * s)

  def draw_bifurcations(self, graph, selected_key, preserve_camera=False,
                        invalid_keys=frozenset()):
    """Render graph edges, furcation nodes, and pick targets in ``self._bif_pick``.

    ``invalid_keys`` is a set of unsigned-index frozensets (one per bifurcation
    whose per-bif hex mesh failed :meth:`Mesh.is_valid`). Those centroids are
    drawn with an extra large dark-red halo so the user can spot them before
    exporting.
    """
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
    unsel_centers, sel_centers, invalid_centers = [], [], []
    # Dark crimson halo for invalid bifurcations — strong enough to dominate
    # the normal red markers and the black "selected" dot.
    bif_invalid_color = (0.60, 0.00, 0.00, 1.0)
    invalid_halo_size = SEL_BIF_BLACK_SIZE_PX + 14.0
    for furc in graph.nontrivial_furcations:
      center = np.asarray(furc.centroid, dtype=float).ravel()
      key = frozenset(int(i) for i in furc.indices)
      unsigned_key = frozenset(abs(int(i)) for i in furc.indices)
      sel = selected_key is not None and key == selected_key
      invalid = unsigned_key in invalid_keys
      marker_size = SEL_BIF_BLACK_SIZE_PX if sel else RED_BIF_SIZE_PX
      pick_tol = max(PICK_TOLERANCE_PX, 0.6 * marker_size)
      self._bif_pick.append((key, center, pick_tol))
      (sel_centers if sel else unsel_centers).append(center)
      if invalid:
        invalid_centers.append(center)
      for signed in furc.indices:
        edge = graph.get_edges([abs(signed)])[0]
        endpoint = edge.points[-1, :3] if signed > 0 else edge.points[0, :3]
        # Same red as markers; orange/green spokes looked yellow where they met the centroid.
        self._line(np.vstack([endpoint, center]), bif_pt_color, 2.4 if sel else 1.4, **_bif_gl)
    # Invalid halo goes below the normal markers so the red/black dots still
    # sit visibly on top.
    if invalid_centers:
      p = np.asarray(invalid_centers, dtype=float)
      s = gl.GLScatterPlotItem(
        pos=p, color=bif_invalid_color, size=invalid_halo_size, glOptions="opaque"
      )
      s.setDepthValue(9)
      self.view.addItem(s); self.items.append(s); all_pts.append(p)
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

  def draw_mesh(self, mesh, preview_points=None, boundary_wire=None,
                boundary_scatter=None, boundary_faces=None):
    """Draw the boundary as a blue **filled surface** with its wireframe + vertices.

    Cached ``boundary_wire`` / ``boundary_scatter`` / ``boundary_faces`` from the
    worker avoid recomputing ``mesh.boundary`` on the GUI thread. Sets
    ``self.mesh_simplified`` if only the point-cloud fallback could be rendered.
    """
    self.clear()
    if self.view is None:
      return
    self.mesh_simplified = False
    wpos = boundary_wire
    scat = boundary_scatter
    tris = boundary_faces
    if wpos is None and scat is None and tris is None:
      wpos, scat, tris = _boundary_preview_arrays(mesh)
    drew = False
    fit_pts = []
    # Flat-gray faces with saturated-blue edges and vertices: the lack of
    # shading keeps every element the same tone (no face-by-face "darker blue"
    # artefacts from per-face lighting) and the blue wireframe reads clearly.
    face_rgba = (0.75, 0.75, 0.75, 1.0)
    edge_rgba = (0.02, 0.38, 0.95, 1.0)
    vert_rgba = (0.02, 0.38, 0.95, 1.0)
    # Blue filled boundary surface.
    if tris is not None and scat is not None and len(tris) > 0 and len(scat) > 0:
      verts = np.ascontiguousarray(np.asarray(scat, dtype=float))
      faces = np.ascontiguousarray(np.asarray(tris, dtype=np.int64))
      face_colors = np.tile(np.asarray(face_rgba, dtype=float), (faces.shape[0], 1))
      try:
        mesh_item = gl.GLMeshItem(
          vertexes=verts,
          faces=faces,
          faceColors=face_colors,
          smooth=False,
          drawEdges=False,
          drawFaces=True,
          shader=None,
          glOptions="opaque",
        )
        # Render faces first so lines drawn next win the z-fight on top.
        mesh_item.setDepthValue(-1)
        self.view.addItem(mesh_item)
        self.items.append(mesh_item)
        fit_pts.append(verts)
        drew = True
      except Exception:
        pass
    if wpos is not None and len(wpos) > 0:
      ln = gl.GLLinePlotItem(
        pos=wpos,
        color=edge_rgba,
        width=2.5,
        antialias=True,
        mode="lines",
        glOptions="opaque",
      )
      ln.setDepthValue(10)
      self.view.addItem(ln)
      self.items.append(ln)
      fit_pts.append(wpos)
      drew = True
    if scat is not None and len(scat) > 0:
      s = gl.GLScatterPlotItem(
        pos=np.asarray(scat, dtype=float),
        color=vert_rgba,
        size=3.0,
        glOptions="opaque",
      )
      s.setDepthValue(11)
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
    s = gl.GLScatterPlotItem(
      pos=p, color=MESH_FALLBACK_SCATTER_RGBA, size=2.5, glOptions="opaque"
    )
    self.view.addItem(s)
    self.items.append(s)
    self._fit(p)

  def draw_fix_view(
    self,
    centerlines,
    selected_id=None,
    border_points=None,
    fix_bif_ref=None,
    show_mesh=False,
    mesh=None,
    preview_points=None,
    boundary_wire=None,
    boundary_scatter=None,
    boundary_faces=None,
    preserve_camera=False,
  ):
    """Centerlines plus optional mesh boundary overlay and branch-anchor markers.

    Used by **Fix bifurcation** view: optional mesh plus **vascular-graph**
    vessel polylines (same geometry as Bifurcations mode) for branch picks.

    ``fix_bif_ref`` is ``(centroid_xyz, branch_roots)`` for the **selected**
    bifurcation: each branch root matches :func:`mesh._build_bif_mesh` —
    ``cline([interval[1]])`` on the **rfac-truncated** spline edge oriented into
    the junction (baseline graph, **no** queued fix-anchor trims).  If that
    snapshot fails, raw polyline junction endpoints are used instead.
    """
    self.clear()
    if self.view is None:
      return
    centerlines = list(centerlines)
    border_points = list(border_points or [])
    ref_centroid = None
    ref_roots_arr = None
    if fix_bif_ref is not None:
      c0, roots = fix_bif_ref
      if c0 is not None and roots:
        ref_centroid = np.asarray(c0, dtype=float).ravel()[:3]
        ref_roots_arr = np.asarray(
          [np.asarray(r, dtype=float).ravel()[:3] for r in roots],
          dtype=float,
        )
        if ref_roots_arr.ndim != 2 or ref_roots_arr.shape[0] == 0:
          ref_centroid = None
          ref_roots_arr = None
    all_pts = []
    self.mesh_simplified = False
    face_rgba = (0.75, 0.75, 0.75, 1.0)
    edge_rgba = (0.02, 0.38, 0.95, 1.0)
    vert_rgba = (0.02, 0.38, 0.95, 1.0)
    if show_mesh and mesh is not None:
      wpos = boundary_wire
      scat = boundary_scatter
      tris = boundary_faces
      if wpos is None and scat is None and tris is None:
        wpos, scat, tris = _boundary_preview_arrays(mesh)
      drew = False
      if tris is not None and scat is not None and len(tris) > 0 and len(scat) > 0:
        verts = np.ascontiguousarray(np.asarray(scat, dtype=float))
        faces = np.ascontiguousarray(np.asarray(tris, dtype=np.int64))
        face_colors = np.tile(np.asarray(face_rgba, dtype=float), (faces.shape[0], 1))
        try:
          mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=face_colors,
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            shader=None,
            glOptions="opaque",
          )
          mesh_item.setDepthValue(-1)
          self.view.addItem(mesh_item)
          self.items.append(mesh_item)
          all_pts.append(verts)
          drew = True
        except Exception:
          pass
      if wpos is not None and len(wpos) > 0:
        ln = gl.GLLinePlotItem(
          pos=wpos,
          color=edge_rgba,
          width=2.5,
          antialias=True,
          mode="lines",
          glOptions="opaque",
        )
        ln.setDepthValue(5)
        self.view.addItem(ln)
        self.items.append(ln)
        all_pts.append(wpos)
        drew = True
      if scat is not None and len(scat) > 0:
        s = gl.GLScatterPlotItem(
          pos=np.asarray(scat, dtype=float),
          color=vert_rgba,
          size=3.0,
          glOptions="opaque",
        )
        s.setDepthValue(6)
        self.view.addItem(s)
        self.items.append(s)
        all_pts.append(np.asarray(scat, dtype=float))
        drew = True
      if not drew:
        if preview_points is not None:
          points = np.asarray(preview_points, dtype=float)
        else:
          bpts = _extract_boundary_points_for_preview(mesh)
          points = bpts
        if points is not None and points.ndim == 2 and points.shape[1] >= 3:
          points = points[:, :3]
          step = max(1, int(np.ceil(points.shape[0] / MAX_MESH_PREVIEW_POINTS)))
          self.mesh_simplified = step > 1
          p = points[::step]
          s = gl.GLScatterPlotItem(
            pos=p, color=MESH_FALLBACK_SCATTER_RGBA, size=2.5, glOptions="opaque"
          )
          s.setDepthValue(4)
          self.view.addItem(s)
          self.items.append(s)
          all_pts.append(p)

    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      c = QColor.fromHsv((idx * 37) % 360, 220, 220)
      self._line(
        pts, (c.redF(), c.greenF(), c.blueF(), 1.0), 1.2, glOptions="opaque"
      )
      self._cl_pick.append((int(idx), np.asarray(pts, dtype=float)))
      all_pts.append(pts)
    if ref_centroid is not None and ref_roots_arr is not None:
      all_pts.append(ref_centroid.reshape(1, 3))
      all_pts.append(ref_roots_arr)
    if not all_pts and not border_points:
      return
    if selected_id is not None:
      for idx, pts in centerlines:
        if int(idx) != int(selected_id) or pts.shape[0] < 2:
          continue
        self._line(pts, (0.04, 0.04, 0.04, 1.0), 3.0, glOptions="opaque")
        break
    if all_pts:
      all_pts_arr = np.concatenate(all_pts, axis=0)
      if preserve_camera:
        mins, maxs = all_pts_arr.min(axis=0), all_pts_arr.max(axis=0)
        r = max(1e-3, 0.5 * np.max(maxs - mins))
        s = 0.016 * r
      else:
        s = self._fit(all_pts_arr)
    else:
      s = self._fit(np.asarray(border_points, dtype=float))
    for idx, pts in centerlines:
      if pts.shape[0] < 2:
        continue
      mid = pts[pts.shape[0] // 2]
      self._digit(
        mid + np.array([0.0, 0.0, 0.1 * s]),
        idx,
        (0.1, 0.1, 0.1, 1.0),
        0.52 * s,
      )
    if ref_centroid is not None and ref_roots_arr is not None:
      ref_rgba = (0.92, 0.48, 0.06, 1.0)
      line_kw = {"glOptions": "opaque"}
      c = ref_centroid
      for r in ref_roots_arr:
        self._line(np.vstack([r, c]), ref_rgba, 2.0, **line_kw)
      s_roots = gl.GLScatterPlotItem(
        pos=ref_roots_arr, color=ref_rgba, size=9.0, glOptions="opaque"
      )
      s_roots.setDepthValue(17)
      self.view.addItem(s_roots)
      self.items.append(s_roots)
      s_cent = gl.GLScatterPlotItem(
        pos=c.reshape(1, 3),
        color=(0.72, 0.30, 0.02, 1.0),
        size=7.0,
        glOptions="opaque",
      )
      s_cent.setDepthValue(16)
      self.view.addItem(s_cent)
      self.items.append(s_cent)
    if border_points:
      p = np.asarray(border_points, dtype=float)
      if p.size:
        s_mark = gl.GLScatterPlotItem(
          pos=p,
          color=(0.05, 0.65, 0.12, 1.0),
          size=10.0,
          glOptions="opaque",
        )
        s_mark.setDepthValue(20)
        self.view.addItem(s_mark)
        self.items.append(s_mark)


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
    self._mesh_preview_boundary_faces = None
    self._mesh_stats = None
    # {frozenset(bif indices) → bool}. Populated after every successful mesh
    # generation so the bifurcation view / list can flag invalid ones in red
    # without needing a separate re-run.
    self._bif_validity = {}
    # {frozenset(bif indices) → repair_info_dict | None}. Parallel to
    # ``_bif_validity``; carries per-branch rfac details for bifurcations
    # that the retry cascade touched (whether it succeeded or exhausted).
    self._bif_repair = {}
    # {frozenset(bif indices) → list of overlap dicts}. Each entry notes a
    # shared edge where the two adjacent bifurcations' truncations leave a
    # near-degenerate vessel stub between them → merge candidate.  The
    # list is empty for bifurcations with no close neighbours.
    self._bif_overlaps = {}
    # Connected components of the merge-candidate graph, as emitted by
    # ``make_spline_mesh``.  Each entry: {"component_id": str, "bif_keys":
    # tuple[frozenset[int], ...], "internal_edges": tuple[int, ...]}.  The
    # Bifurcations panel renders one checkbox per component at its top.
    self._merge_components: list[dict] = []
    # Set of component_ids the user has checked ⇒ will be folded into a
    # single n-furcation on the next Generate mesh click.  Persisted across
    # generations so toggling the checkbox is sticky even when the mesh is
    # regenerated (the same component usually recomputes with the same id).
    self._merge_opt_in: set[str] = set()
    # Last ``merge_opt_in`` tuple passed to :func:`generate_mesh` (immutable snapshot).
    self._last_merge_opt_in: tuple[tuple[tuple[int, ...], ...], ...] = ()
    # Post-merge :class:`~artery.VascularGraph` for bif list + Bifurcations 3D when
    # the last successful mesh used opt-in merges; else ``None`` → use ``_graph_cache``.
    self._graph_for_bif_ui = None
    # Unsigned bif keys that are user-merged n-furcations in ``_graph_for_bif_ui``.
    self._merged_bif_unsigned_keys: frozenset = frozenset()
    self._mesh_thread = None
    self.last_mesh_path = ""
    self._selected_bif_key = None
    self._selected_centerline_id = None
    self._removed_bif_keys = set()
    self._removed_centerline_ids = set()
    self._manual_additions = []
    self._last_view_mode = "Centerlines"
    # (edge_display_id, world_xyz) anchors along each branch — Fix bifurcation mode.
    self._fix_border_hits: list[tuple[int, np.ndarray]] = []
    # Completed or in-progress picks per bif key (survives switching list rows).
    self._fix_anchors_by_bif = {}
    self._pending_fix_anchor_mesh = False
    # Spline + ``mesh_rfac`` truncation snapshot for Fix-mode orange markers.
    self._fix_rfac_trunc_cache_key = None
    self._fix_rfac_trunc_data = None

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
    left_layout.addWidget(self._fix_bif_group())
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
    self.scene.on_cl_click = self._on_cl_clicked_3d
    right_layout.addWidget(self.scene, stretch=4)
    self.id_map_hdr = QLabel("Centerline ID map")
    self.id_map_hdr.setToolTip(
      "Select a row to highlight that centerline in 3D; click a centerline "
      "in 3D to select its row. Use Delete Selected Centerline below."
    )
    right_layout.addWidget(self.id_map_hdr)
    self.id_map = QListWidget()
    self.id_map.currentItemChanged.connect(self._on_id_map_item_changed)
    right_layout.addWidget(self.id_map, stretch=1)
    cl_btn_row = QHBoxLayout()
    self.b_cl_del = QPushButton("Delete Selected Centerline")
    self.b_cl_del.clicked.connect(self._delete_selected_centerline)
    self.b_cl_reset = QPushButton("Reset Removed Centerlines")
    self.b_cl_reset.clicked.connect(self._reset_removed_centerlines)
    cl_btn_row.addWidget(self.b_cl_del)
    cl_btn_row.addWidget(self.b_cl_reset)
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
    logo_path = _resolve_logo_path(kind="pictogram")
    if logo_path is None:
      return
    icon = QIcon(logo_path)
    if not icon.isNull():
      self.setWindowIcon(icon)

  def _logo_label(self, width=260):
    """Create a visible logo label (e.g. header)."""
    logo_path = _resolve_logo_path(kind="logotipo")
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
    self._mesh_preview_boundary_faces = None
    self._bif_validity = {}
    self._bif_repair = {}
    self._bif_overlaps = {}
    # Note: ``_merge_opt_in`` is *not* reset here.  The cache-invalidation
    # path is triggered by graph edits (delete/add bifurcation, reload CSV,
    # etc.) and it's surprising to lose the user's merge decisions after
    # e.g. nudging a split tolerance.  ``_merge_components`` however is
    # tied to the *current* mesh computation — drop it so stale checkboxes
    # don't linger while ``last_mesh`` is None.
    self._merge_components = []
    self._fix_border_hits = []
    self._fix_anchors_by_bif.clear()
    self._pending_fix_anchor_mesh = False
    self._last_merge_opt_in = ()
    self._graph_for_bif_ui = None
    self._merged_bif_unsigned_keys = frozenset()
    self._fix_rfac_trunc_cache_key = None
    self._fix_rfac_trunc_data = None
    if hasattr(self, "bif_list_invalid_only"):
      self.bif_list_invalid_only.blockSignals(True)
      self.bif_list_invalid_only.setChecked(False)
      self.bif_list_invalid_only.setEnabled(False)
      self.bif_list_invalid_only.blockSignals(False)
    self._set_mesh_stats(None)

  def _opt_in_components_from_ui(self):
    """Same merge tuple shape as :func:`generate_mesh` ``merge_opt_in``."""
    opt_in_components: list[tuple[tuple[int, ...], ...]] = []
    for comp in (self._merge_components or ()):
      cid = comp.get("component_id")
      if cid in self._merge_opt_in:
        keys = comp.get("bif_keys") or frozenset()
        opt_in_components.append(tuple(
          tuple(sorted(int(i) for i in k)) for k in keys
        ))
    return opt_in_components

  def _fix_rfac_trunc_overlay_cache_key(self):
    folder = self.csv_edit.text().strip()
    if not folder:
      return None
    removed_bk = tuple(sorted(
      (tuple(sorted(abs(int(i)) for i in k)) for k in self._removed_bif_keys),
      key=lambda t: t,
    ))
    manual = tuple(tuple(sorted(int(x) for x in row)) for row in self._manual_additions)
    rem_cl = tuple(sorted(self._removed_centerline_ids))
    opt_in = tuple(
      tuple(tuple(sorted(int(i) for i in k)) for k in comp)
      for comp in self._opt_in_components_from_ui()
    )
    # Intentionally **omit** queued fix-anchor specs: orange shows the baseline
    # mesh rfac interface on the graph **without** user trims so it stays fixed
    # while placing green picks (otherwise each completed bif re-trims the
    # artery and orange jumps away from the click).
    return (
      folder,
      self.split_eps.value(),
      self.merge_eps.value(),
      self.use_merge.isChecked(),
      removed_bk,
      manual,
      rem_cl,
      opt_in,
      float(self.mesh_dx.value()),
      int(self.mesh_nref.value()),
      float(self.mesh_rfac.value()),
    )

  def _sgraph_trunc_for_fix_rfac_overlay(self):
    """Build ``(SplineGraph, truncate_dict)`` for **baseline** mesh rfac (no fix anchors).

    Queued green picks are **not** applied here so amber stays aligned with the
    automatic cut while you place anchors; ``generate_mesh`` still applies
    ``fix_anchor_specs`` before the spline stage.
    """
    from mesh import apply_merge_opt_in_vgraph, _compute_truncate_dict
    ck = self._fix_rfac_trunc_overlay_cache_key()
    if ck is None:
      return None
    if ck == self._fix_rfac_trunc_cache_key and self._fix_rfac_trunc_data is not None:
      return self._fix_rfac_trunc_data
    folder = self.csv_edit.text().strip()
    try:
      artery = build_graph(
        folder,
        self.split_eps.value(),
        self.merge_eps.value(),
        self.use_merge.isChecked(),
        self._removed_bif_keys,
        self._manual_additions,
        removed_centerline_ids=frozenset(self._removed_centerline_ids),
      )
      artery = artery.orient_nontrivial()
      artery, _mk = apply_merge_opt_in_vgraph(
        artery, self._opt_in_components_from_ui(),
      )
      artery = (
        artery.align_nontrivial_normals()
        .orient_trivial()
        .rtruncate(1)
        .furcomplete(npoints=1001)
        .delete_twofurcations()
        .align_frames()
      )
      dx = float(self.mesh_dx.value())
      nref = int(self.mesh_nref.value())
      rfac = float(self.mesh_rfac.value())
      sgraph = artery.to_spline_graph(dx=dx)
      xis = [e.knotvector[0].refine(nref).knots for e in sgraph.edges]
      xidict = dict(zip(sgraph.indices, xis))
      trunc = _compute_truncate_dict(sgraph, xis, xidict, rfac)
    except Exception:
      return None
    self._fix_rfac_trunc_cache_key = ck
    self._fix_rfac_trunc_data = (sgraph, trunc)
    return self._fix_rfac_trunc_data

  def _persist_current_fix_hits(self):
    k = self._selected_bif_key
    if k is None:
      return
    if not self._fix_border_hits:
      self._fix_anchors_by_bif.pop(k, None)
    else:
      self._fix_anchors_by_bif[k] = [
        (int(e), np.asarray(p, dtype=float).copy()) for e, p in self._fix_border_hits
      ]

  def _sync_fix_hits_from_storage(self):
    k = self._selected_bif_key
    if k is None:
      self._fix_border_hits = []
      return
    stored = self._fix_anchors_by_bif.get(k)
    if not stored:
      self._fix_border_hits = []
    else:
      self._fix_border_hits = [
        (int(e), np.asarray(p, dtype=float).copy()) for e, p in stored
      ]

  def _all_fix_anchor_xyz_for_draw(self):
    out = []
    for hits in self._fix_anchors_by_bif.values():
      for _e, p in hits:
        out.append(np.asarray(p, dtype=float).ravel()[:3])
    return out

  def _complete_fix_anchor_specs(self):
    specs = []
    for bif_key, hits in self._fix_anchors_by_bif.items():
      need = {abs(int(i)) for i in bif_key}
      got = {int(p[0]) for p in hits}
      if need != got or len(hits) != len(need):
        continue
      specs.append((
        bif_key,
        {int(eid): np.asarray(xyz, dtype=float).reshape(3).copy() for eid, xyz in hits},
      ))
    return specs

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
    self.mode.addItems(["Centerlines", "Bifurcations", "Mesh", "Fix bifurcation"])
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
    l.addWidget(QLabel("split_edges eps"), 0, 0); l.addWidget(self.split_eps, 0, 1)
    l.addWidget(self.use_merge, 1, 0, 1, 2)
    l.addWidget(QLabel("merge_furcs eps"), 2, 0); l.addWidget(self.merge_eps, 2, 1)
    self._bif_edit_group = self._edit_group()
    l.addWidget(self._bif_edit_group, 3, 0, 1, 2)
    # Let the editing block absorb extra vertical space inside this group box.
    l.setRowStretch(3, 1)
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
    # Merge-candidate checkbox host: one QCheckBox per connected component
    # of the "too-close" overlap graph is injected here by
    # ``_fill_merge_checkboxes`` after each mesh generation.  Empty until a
    # mesh with detected overlaps is produced; hidden entirely when empty
    # so it doesn't take panel real estate from the bif list below.
    self._merge_panel = QGroupBox("Merge candidates")
    self._merge_panel.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
    )
    self._merge_panel_layout = QVBoxLayout(self._merge_panel)
    self._merge_panel_layout.setContentsMargins(6, 4, 6, 4)
    self._merge_panel_layout.setSpacing(2)
    self._merge_panel.setVisible(False)
    # Keep a dict of component_id → QCheckBox so toggling the stored opt-in
    # state can update widgets without a full rebuild.
    self._merge_checkboxes: dict = {}
    col_left.addWidget(self._merge_panel)
    self.bif_list_invalid_only = QCheckBox("List invalid bifurcations only")
    self.bif_list_invalid_only.setToolTip(
      "After Generate mesh, show only junctions whose bifurcation hex failed "
      "is_valid() (red [INVALID]). Uncheck to list every bifurcation again."
    )
    self.bif_list_invalid_only.setEnabled(False)
    self.bif_list_invalid_only.toggled.connect(self._on_bif_invalid_filter_toggled)
    col_left.addWidget(self.bif_list_invalid_only)
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

  def _fix_bif_group(self):
    """Widgets for **Fix bifurcation** view (mesh + bif-view vessels + picks)."""
    g = QGroupBox("Fix bifurcation")
    g.setVisible(False)
    v = QVBoxLayout(g)
    self.fix_hint = QLabel(
      "Use after a mesh run. Pick a bifurcation in the list, then click each "
      "incident vessel in 3D — same curves as Bifurcations mode (graph edges), "
      "not raw CSV polylines. Esc clears markers."
    )
    self.fix_hint.setWordWrap(True)
    self.fix_show_mesh = QCheckBox("Show mesh overlay")
    self.fix_show_mesh.setChecked(True)
    self.fix_show_mesh.setToolTip(
      "Turn off to see only bif-view vessels and your branch anchors."
    )
    self.fix_show_mesh.toggled.connect(self._on_fix_show_mesh_toggled)
    b_clear = QPushButton("Clear branch picks")
    b_clear.clicked.connect(self._clear_fix_picks)
    b_clear_all = QPushButton("Clear all fix anchors")
    b_clear_all.setToolTip(
      "Remove stored branch anchors for every bifurcation (not only the selected one)."
    )
    b_clear_all.clicked.connect(self._clear_all_fix_anchors)
    v.addWidget(self.fix_hint)
    v.addWidget(self.fix_show_mesh)
    v.addWidget(b_clear)
    v.addWidget(b_clear_all)
    self._fix_group = g
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
    self._selected_centerline_id = None
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

  def _graph_for_bif_list_and_3d(self):
    """Graph for the bif list + Bifurcations 3D (post-merge if last mesh merged)."""
    g = self._graph_for_bif_ui
    return g if g is not None else self._graph_cache

  def _fix_mode_branch_polylines(self):
    """``(edge_id, (n,3))`` polylines from the graph used for Bifurcations / bif list.

    Matches :meth:`Scene3D.draw_bifurcations` edge sampling so picks land on
    split/merged topology, not raw CSV centerlines.
    """
    g = self._graph_for_bif_list_and_3d()
    out: list[tuple[int, np.ndarray]] = []
    for idx, edge in zip(g.indices, g.edges):
      pts = np.asarray(edge.points, dtype=float)
      if pts.ndim != 2 or pts.shape[1] < 3:
        continue
      p3 = np.ascontiguousarray(pts[:, :3].copy())
      if p3.shape[0] < 2:
        continue
      out.append((int(idx), p3))
    return out

  def _fix_mode_mesh_rfac_bif_reference(self):
    """``(centroid, branch_xyz)`` on the **truncated** spline where bif mesh meets vessels.

    Matches :func:`mesh._build_bif_mesh`: after the same ``truncate_dict`` as
    :func:`mesh.make_spline_mesh` (baseline ``mesh_rfac`` / ``mesh_nref``), each
    branch uses ``get_edges([signed])`` then ``edge.cline([edge.interval[1]])`` —
    not raw ``truncate_dict`` knots evaluated on the **pre**-truncation edge
    (split/reparam and orientation differ).  Computed **without** queued fix
    anchors so amber does not move when you finish green picks.  Per-branch
    rfac repair overrides are not applied here.
    """
    pair = self._sgraph_trunc_for_fix_rfac_overlay()
    if pair is None:
      return None
    sgraph, trunc = pair
    k = self._selected_bif_key
    if k is None:
      return None
    try:
      _sgraph = sgraph.truncate_edges(trunc)
    except Exception:
      return None
    key_set = frozenset(int(i) for i in k)
    furc = None
    for f in _sgraph.nontrivial_furcations:
      if frozenset(int(i) for i in f.indices) == key_set:
        furc = f
        break
    if furc is None:
      return None
    centroid = np.asarray(furc.centroid, dtype=float).ravel()[:3]
    roots = []
    for signed in furc.indices:
      try:
        edge = _sgraph.get_edges([signed])[0]
      except Exception:
        return None
      xi_face = float(edge.interval[1])
      try:
        p = np.asarray(edge.cline([xi_face]), dtype=float).ravel()[:3]
      except Exception:
        return None
      roots.append(p.copy())
    if len(roots) != len(furc.indices):
      return None
    return (centroid, roots)

  def _fix_mode_polyline_endpoint_bif_reference(self):
    """``(centroid, branch_xyz)`` from raw graph polylines (pre-spline / full length)."""
    g = self._graph_for_bif_list_and_3d()
    k = self._selected_bif_key
    if g is None or k is None:
      return None
    key_set = frozenset(int(i) for i in k)
    for furc in g.nontrivial_furcations:
      if frozenset(int(i) for i in furc.indices) != key_set:
        continue
      centroid = np.asarray(furc.centroid, dtype=float).ravel()[:3]
      roots = []
      for signed in furc.indices:
        eid = abs(int(signed))
        try:
          edge = g.get_edges([eid])[0]
        except Exception:
          continue
        pts = np.asarray(edge.points, dtype=float)
        if pts.shape[0] < 2 or pts.shape[1] < 3:
          continue
        endpoint = pts[-1, :3] if int(signed) > 0 else pts[0, :3]
        roots.append(np.asarray(endpoint, dtype=float).ravel()[:3].copy())
      if not roots:
        return None
      return (centroid, roots)
    return None

  def _fix_mode_selected_bif_reference(self):
    """Amber reference: mesh ``rfac`` spline cuts, else polyline junction endpoints."""
    ref = self._fix_mode_mesh_rfac_bif_reference()
    if ref is not None:
      return ref
    return self._fix_mode_polyline_endpoint_bif_reference()

  def _set_right_panel_fix_mode(self, fix: bool):
    """Fix mode: graph-edge list + disable CSV centerline delete (IDs differ)."""
    if fix:
      self.id_map_hdr.setText("Graph edges (Bifurcations view vessels)")
      self.id_map_hdr.setToolTip(
        "Each row is a vascular-graph edge id — the same polylines as in "
        "Bifurcations 3D (after split/merge). Branch picks use these curves, "
        "not raw CSV centerlines."
      )
      self.b_cl_del.setEnabled(False)
      self.b_cl_reset.setEnabled(False)
    else:
      self.id_map_hdr.setText("Centerline ID map")
      self.id_map_hdr.setToolTip(
        "Select a row to highlight that centerline in 3D; click a centerline "
        "in 3D to select its row. Use Delete Selected Centerline below."
      )
      self.b_cl_del.setEnabled(True)
      self.b_cl_reset.setEnabled(True)

  def _rebuild_graph_for_bif_ui(self, merge_opt_in):
    """Rebuild post-merge :class:`~artery.VascularGraph` after a successful mesh."""
    from mesh import apply_merge_opt_in_vgraph
    self._graph_for_bif_ui = None
    self._merged_bif_unsigned_keys = frozenset()
    if not merge_opt_in:
      return
    try:
      g0 = self._graph().orient_nontrivial()
      g_merged, mk2m = apply_merge_opt_in_vgraph(g0, merge_opt_in)
      self._graph_for_bif_ui = g_merged
      self._merged_bif_unsigned_keys = frozenset(mk2m.keys())
    except Exception:
      pass

  def _reload_preview_data(self):
    """Reload centerlines CSVs and rebuild the vascular graph + bif lists."""
    folder = self.csv_edit.text().strip()
    if not folder or not os.path.isdir(folder):
      self._set_status("Select a valid CSV folder.")
      return False
    self._centerlines, self._filenames = self._load_centerlines(folder)
    self._graph_cache = self._graph()
    if self.last_mesh is not None:
      self._rebuild_graph_for_bif_ui(self._last_merge_opt_in)
    else:
      self._graph_for_bif_ui = None
      self._merged_bif_unsigned_keys = frozenset()
    self._fill_bif_list()
    self._fill_add_list()
    return True

  def _on_view_mode_changed(self, index):
    """Keep preview in sync when switching modes (no extra Refresh click for bif)."""
    if index < 0:
      return
    prev = self._last_view_mode
    mode = self.mode.itemText(index)

    if prev == "Fix bifurcation" and mode != "Fix bifurcation":
      self._persist_current_fix_hits()
      self.scene.on_cl_click_detail = None
      self.scene.on_bif_click = self._on_bif_clicked_3d
      self.scene.on_cl_click = self._on_cl_clicked_3d
      self._fix_group.setVisible(False)

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
    if mode == "Fix bifurcation":
      self.scene.on_cl_click_detail = self._on_fix_border_pick
      self.scene.on_bif_click = None
      self.scene.on_cl_click = None
      self._fix_group.setVisible(True)
      self.fix_show_mesh.setEnabled(self.last_mesh is not None)
      try:
        if not self._reload_preview_data():
          self._last_view_mode = mode
          return
      except Exception:
        self._set_status(f"Preview error: {traceback.format_exc().splitlines()[-1]}")
        self._last_view_mode = mode
        return
      self._sync_fix_hits_from_storage()
      self._update_fix_hint()
      self._render_fix_mode(
        preserve_camera=(prev in ("Mesh", "Fix bifurcation", "Centerlines")),
      )
      self._last_view_mode = mode
      return
    if mode == "Mesh":
      self._render_mode(preserve_camera=False)
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
      if self.mode.currentText() == "Fix bifurcation":
        self.fix_show_mesh.setEnabled(self.last_mesh is not None)
        self._update_fix_hint()
        self._render_fix_mode()
      else:
        self._render_mode()
    except Exception:
      self._set_status(f"Preview error: {traceback.format_exc().splitlines()[-1]}")

  def _synthetic_merge_components_from_last_opt_in(self):
    """Rebuild merge-candidate dicts when overlap detection yields an empty list.

    ``make_spline_mesh`` derives ``merge_components`` from post-mesh bif–bif
    overlap topology.  After an **opt-in** merge, those overlaps are typically
    gone, so the returned tuple is empty — but :meth:`_fill_merge_checkboxes`
    would then clear ``_merge_opt_in`` and the **next** Generate would run
    without merges (e.g. Fix-bifurcation + anchor passes).  Recreate the same
    ``component_id`` / ``bif_keys`` / ``internal_edges`` shape as
    :func:`mesh._compute_merge_components` from ``_last_merge_opt_in``, which
    was frozen for this mesh on the worker launch.
    """
    out = []
    for comp in self._last_merge_opt_in or ():
      if len(comp) < 2:
        continue
      keys = frozenset(frozenset(int(i) for i in k) for k in comp)
      sig = tuple(sorted(tuple(sorted(kk) for kk in keys)))
      component_id = "|".join(
        ",".join(str(i) for i in member) for member in sig
      )
      internal = frozenset()
      try:
        g = self._graph().orient_nontrivial()
        component_furcs = [
          f for f in g.furcations
          if frozenset(abs(int(i)) for i in f.indices) in keys
          and len(f) > 2
        ]
        if len(component_furcs) >= 2:
          counts = {}
          for furc in component_furcs:
            for i in furc.indices:
              ai = abs(int(i))
              counts[ai] = counts.get(ai, 0) + 1
          internal = frozenset(ai for ai, c in counts.items() if c >= 2)
      except Exception:
        pass
      out.append({
        "component_id": component_id,
        "bif_keys": keys,
        "internal_edges": internal,
      })
    return out

  def _fill_merge_checkboxes(self):
    """Rebuild the checkbox list at the top of the Bifurcations panel.

    One :class:`QCheckBox` is created per entry in ``self._merge_components``;
    the checkbox's ``toggled`` signal mutates ``self._merge_opt_in`` in
    place so the state is read directly when the user hits Generate mesh.

    Any component_id present in ``self._merge_opt_in`` but no longer in
    the fresh component list is dropped — the previous mesh generation's
    topology is gone, so the stale opt-in would silently do nothing.
    """
    from PyQt6.QtWidgets import QCheckBox  # local import: keeps module-top
    # import block focused on the widgets actually touched at construction
    # time; QCheckBox is only relevant here.

    # Remove any previous widgets without leaking them (blocks garbage
    # collection issues on older PyQt builds).
    while self._merge_panel_layout.count():
      item = self._merge_panel_layout.takeAt(0)
      w = item.widget() if item is not None else None
      if w is not None:
        w.setParent(None)
        w.deleteLater()
    self._merge_checkboxes.clear()

    if not self._merge_components:
      self._merge_panel.setVisible(False)
      # Purge opt-ins that no longer map to a current component; if a user
      # unchecks everything relevant (graph edit invalidated the merge),
      # the state must not leak to the next generation.
      self._merge_opt_in.clear()
      return

    # Drop any persisted opt-in ids that aren't in the newly-detected set
    # (happens e.g. when the user edits the graph then regenerates — the
    # same component might or might not re-appear with the same id).
    current_ids = {c["component_id"] for c in self._merge_components}
    self._merge_opt_in &= current_ids

    # Short-key lookup: map each unsigned bif key to its "Bk" label as
    # rendered in the bif list, so the checkbox text reads naturally.
    vkey_to_label: dict = {}
    try:
      graph = self._graph_cache
      for ib, furc in enumerate(graph.nontrivial_furcations, start=1):
        vk = frozenset(abs(int(v)) for v in furc.indices)
        vkey_to_label[vk] = f"B{ib}"
    except Exception:
      # Graph cache may be empty before the first refresh — the checkboxes
      # still work, we just fall back to raw index sets in the label.
      pass

    for comp in self._merge_components:
      cid = comp["component_id"]
      bif_keys = comp.get("bif_keys") or ()
      internal_edges = comp.get("internal_edges") or ()
      # Human-readable label: "Merge B1 + B2 + B3 into one 5-furcation (drops e3, e5)"
      members = [vkey_to_label.get(frozenset(k),
                                    "{" + ",".join(str(i) for i in sorted(k)) + "}")
                 for k in bif_keys]
      members_str = " + ".join(members)
      # External-branch count = total distinct unsigned indices across the
      # component minus the internal (shared) edges that get dropped.
      all_abs = set()
      for k in bif_keys:
        all_abs |= set(int(i) for i in k)
      n_ext = len(all_abs) - len(internal_edges)
      drops = ", ".join(f"e{i}" for i in internal_edges) or "nothing"
      cb = QCheckBox(
        f"Merge {members_str} into one {n_ext}-furcation (drops {drops})"
      )
      cb.setToolTip(
        "On the next Generate mesh click, this component will be collapsed\n"
        "into a single n-furcation:\n"
        f"  • constituent bifs: {', '.join(members)}\n"
        f"  • shared edges dropped from the graph: {drops}\n"
        f"  • merged n-furcation external branches: {n_ext}\n"
        "Leave unchecked to keep the per-branch rfac repair scan and the\n"
        "red [INVALID] flags on the individual bifurcations."
      )
      cb.setChecked(cid in self._merge_opt_in)
      cb.toggled.connect(
        # Capture cid by default arg so each closure binds its own id.
        lambda checked, _cid=cid: self._on_merge_checkbox_toggled(_cid, checked)
      )
      self._merge_panel_layout.addWidget(cb)
      self._merge_checkboxes[cid] = cb

    self._merge_panel.setVisible(True)

  def _on_merge_checkbox_toggled(self, component_id: str, checked: bool):
    """Record / forget a component's opt-in state.

    Doesn't trigger regeneration — the user explicitly clicks Generate
    mesh afterwards.  The status bar gets a one-line hint so it's obvious
    the decision has been captured and *will* take effect.
    """
    if checked:
      self._merge_opt_in.add(component_id)
    else:
      self._merge_opt_in.discard(component_id)
    n = len(self._merge_opt_in)
    if n:
      self._set_status(
        f"{n} merge candidate(s) queued — click Generate mesh to apply."
      )
    else:
      self._set_status("Merge candidates cleared.")

  def _fill_bif_list(self):
    self.bif_list.clear()
    g_src = self._graph_for_bif_list_and_3d()
    # Unsigned-key → "Bk" for rows in *this* list (merged topology when applicable).
    vkey_to_label = {
      frozenset(abs(int(v)) for v in furc.indices): f"B{ib}"
      for ib, furc in enumerate(g_src.nontrivial_furcations, start=1)
    }
    # Pre-merge labels help resolve overlap "partner" keys from the last mesh
    # when those partners no longer appear as separate rows.
    vkey_to_label_pre = {
      frozenset(abs(int(v)) for v in furc.indices): f"B{ib}"
      for ib, furc in enumerate(self._graph_cache.nontrivial_furcations, start=1)
    }
    def _partner_label(vk_partner):
      return vkey_to_label.get(vk_partner) or vkey_to_label_pre.get(vk_partner, "?")

    for ib, furc in enumerate(g_src.nontrivial_furcations, start=1):
      key = frozenset(int(v) for v in furc.indices)
      # Validity is keyed on unsigned indices (see ``make_spline_mesh``); the
      # list / 3D selection flow keeps the signed key untouched.
      vkey = frozenset(abs(int(v)) for v in furc.indices)
      validity = self._bif_validity.get(vkey) if self._bif_validity else None
      if (
        self.bif_list_invalid_only.isChecked()
        and self._bif_validity
        and validity is not False
      ):
        continue
      repair = (self._bif_repair or {}).get(vkey) if self._bif_repair else None
      overlaps = (self._bif_overlaps or {}).get(vkey) or []
      label = f"B{ib}: indices {list(map(int, furc.indices))}"
      # A repair entry may come from either path:
      #   • per-branch rfac scan  → ``repair`` has ``multipliers`` /
      #     ``branch_rfac`` / ``tried``;
      #   • opt-in merge (member rows only) → ``merged_into`` / ``merged_members``.
      #     List rows use the merged n-furcation directly, so this branch is skipped.
      is_merged_member = repair is not None and repair.get("merged_into") is not None
      if validity is False:
        label += "  [INVALID]"
      elif validity is True and is_merged_member:
        n_ext = len(repair.get("merged_into") or ())
        label += f"  [merged into {n_ext}-furcation]"
      elif validity is True and repair is not None:
        # A successful repair: show the multipliers on the label so the user
        # can see at a glance *which* branch needed extra retreat.
        mults = repair.get("multipliers")
        if mults is not None:
          label += f"  [repaired rfac×{list(mults)}]"
      if overlaps:
        # Near-degenerate shared edge(s) detected: this bifurcation is
        # too close to at least one neighbour and is a merge candidate.
        # List the partner "Bk" labels; a standalone "(term)" annotation
        # marks overlaps on a terminal edge (no partner bif).
        partner_labels = [
          _partner_label(ov.get("other"))
          if ov.get("other") is not None else "(term)"
          for ov in overlaps
        ]
        label += f"  [merge candidate: {', '.join(partner_labels)}]"
      item = QListWidgetItem(label)
      item.setData(Qt.ItemDataRole.UserRole, key)
      # Build tooltip lines in a single list so overlap info can be
      # appended uniformly regardless of validity/repair branch.
      tip_lines: list[str] = []
      if vkey in self._merged_bif_unsigned_keys:
        tip_lines.append(
          "This row is the n-furcation after merging the checked merge "
          "candidate(s); former neighbour rows are folded into this junction."
        )
      if validity is False:
        item.setForeground(QColor(200, 30, 30))
        tip_lines.append(
          "Bifurcation hex mesh self-intersects (is_valid() returned "
          "False). Exporting may still work, but this furcation is "
          "unlikely to produce a usable volume."
        )
        if repair is not None and repair.get("tried") is not None:
          n_tried = repair.get("attempts", len(repair.get("tried", [])))
          tip_lines.append(
            f"\nAutomatic per-branch rfac retry exhausted after "
            f"{n_tried} attempts. Tried multipliers (in order): "
            + ", ".join(
              str(tuple(t.get("multipliers"))) for t in repair.get("tried", [])
            )
          )
      elif validity is True:
        if repair is not None and repair.get("branch_rfac") is not None:
          # The bifurcation was rescued by a retry — make the tooltip
          # explicit about which per-branch rfac values were used so the
          # user can make it the default if the same topology recurs.
          br = repair.get("branch_rfac", {})
          mults = repair.get("multipliers")
          attempts = repair.get("attempts")
          score = repair.get("score")
          baseline_score = repair.get("baseline_score")
          tip_lines.append(
            "Bifurcation hex mesh passes is_valid() after automatic "
            "per-branch rfac scan (best-scoring attempt committed)."
          )
          tip_lines.append(f"  multipliers (by branch order): {mults}")
          tip_lines.append(
            "  absolute rfac per signed branch id: "
            + ", ".join(f"{k}→{v:g}" for k, v in br.items())
          )
          if attempts is not None:
            tip_lines.append(f"  attempts scored: {attempts}")
          if score is not None:
            # Score layout (see mesh._mesh_quality_score):
            # (valid_flag, min_scaled_J, min_det_J, mean_scaled_J, mean_det_J)
            tip_lines.append(
              f"  quality: min_scaled_J={score[1]:.3e}, "
              f"min_det_J={score[2]:.3e}, "
              f"mean_scaled_J={score[3]:.3e}"
            )
          if baseline_score is not None and baseline_score[0] == 0:
            tip_lines.append(
              f"  baseline failed "
              f"(min_scaled_J={baseline_score[1]:.3e}, "
              f"min_det_J={baseline_score[2]:.3e})"
            )
        else:
          tip_lines.append("Bifurcation hex mesh passes is_valid().")
      if overlaps:
        tip_lines.append("")  # blank line separator
        tip_lines.append(
          "⚠ Close-neighbour detection: this bifurcation shares a "
          "near-degenerate edge with another bifurcation. The vessel "
          "stub between them is too short to mesh reliably — consider "
          "merging into a single n-furcation."
        )
        for ov in overlaps:
          partner = (
            _partner_label(ov.get("other"))
            if ov.get("other") is not None else "terminal end"
          )
          tip_lines.append(
            f"  shared edge id {ov['iedge']} ↔ {partner}: "
            f"remaining span = {ov['remaining_fraction']:.1%} of full xi"
          )
      if tip_lines:
        item.setToolTip("\n".join(tip_lines))
      self.bif_list.addItem(item)

    if not self._bif_validity:
      self.bif_list_invalid_only.blockSignals(True)
      self.bif_list_invalid_only.setChecked(False)
      self.bif_list_invalid_only.blockSignals(False)
    self.bif_list_invalid_only.setEnabled(bool(self._bif_validity))

    sel_key = self._selected_bif_key
    if sel_key is not None:
      found_row = -1
      for row in range(self.bif_list.count()):
        it = self.bif_list.item(row)
        if it.data(Qt.ItemDataRole.UserRole) == sel_key:
          found_row = row
          break
      if found_row >= 0:
        self.bif_list.blockSignals(True)
        self.bif_list.setCurrentRow(found_row)
        self.bif_list.blockSignals(False)
      else:
        self.bif_list.blockSignals(True)
        self.bif_list.clearSelection()
        self.bif_list.blockSignals(False)
        if self.mode.currentText() == "Fix bifurcation":
          self._persist_current_fix_hits()
        self._selected_bif_key = None
        if self.mode.currentText() == "Fix bifurcation":
          self._sync_fix_hits_from_storage()
          self._update_fix_hint()

  def _on_bif_invalid_filter_toggled(self, _checked=False):
    self._fill_bif_list()
    mode = self.mode.currentText()
    if mode == "Bifurcations":
      self._render_mode(preserve_camera=True)
    elif mode == "Fix bifurcation":
      self._render_fix_mode(preserve_camera=True)

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
    self._set_status(
      "That bifurcation is hidden: turn off 'List invalid bifurcations only' "
      "to select it in the list or from 3D."
    )

  def _on_bif_clicked_3d(self, key):
    self._set_status("3D click detected. Selecting nearest bifurcation...")
    self._select_bif_item_by_key(key)

  def _select_centerline_by_id(self, cl_id, focus_scene=True):
    """Sync UI state to ``cl_id`` (display id): scroll the ID map and re-render.

    Used by both 3D picking and list selection; the scene is only redrawn when
    centerlines are currently visible (avoids fighting the Bifurcations/Mesh
    views).
    """
    if cl_id is None:
      return
    cl_id = int(cl_id)
    self._selected_centerline_id = cl_id
    # Move the ID-map cursor to the corresponding row without re-entering this
    # slot. Centerlines mode: rows follow 1-based file order. Fix mode: rows
    # follow ``_fix_mode_branch_polylines`` / graph edge order.
    if self.mode.currentText() == "Fix bifurcation":
      target_row = next(
        (
          r for r in range(self.id_map.count())
          if (d := self.id_map.item(r).data(Qt.ItemDataRole.UserRole + 1))
          is not None and int(d) == cl_id
        ),
        -1,
      )
    else:
      target_row = cl_id - 1
    if 0 <= target_row < self.id_map.count():
      self.id_map.blockSignals(True)
      self.id_map.setCurrentRow(target_row)
      self.id_map.blockSignals(False)
    if focus_scene and self.mode.currentText() in ("Centerlines", "Fix bifurcation"):
      if self.mode.currentText() == "Fix bifurcation":
        self._render_fix_mode(preserve_camera=True)
      else:
        self._render_mode(preserve_camera=True)

  def _on_cl_clicked_3d(self, cl_id):
    """3D viewport click on a centerline polyline → select it in the ID map."""
    self._set_status(f"3D click detected. Selecting centerline {int(cl_id)}.")
    self._select_centerline_by_id(cl_id, focus_scene=True)

  def _on_id_map_item_changed(self, current, _prev):
    """ID-map row change → highlight that centerline in 3D (Centerlines / Fix modes)."""
    if current is None:
      return
    mode = self.mode.currentText()
    if mode not in ("Centerlines", "Fix bifurcation"):
      return
    disp = current.data(Qt.ItemDataRole.UserRole + 1)
    if disp is None:
      return
    self._selected_centerline_id = int(disp)
    if mode == "Centerlines":
      self.scene.draw_centerlines(
        self._centerlines,
        selected_id=self._selected_centerline_id,
        preserve_camera=True,
      )
    else:
      self._render_fix_mode(preserve_camera=True)

  def _on_bif_selected_list(self, current, _prev):
    if self.mode.currentText() == "Fix bifurcation":
      self._persist_current_fix_hits()
    self._selected_bif_key = None if current is None else current.data(Qt.ItemDataRole.UserRole)
    if self.mode.currentText() == "Fix bifurcation":
      self._sync_fix_hits_from_storage()
      self._update_fix_hint()
      self._render_fix_mode(preserve_camera=True)
      return
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
    key = item.data(Qt.ItemDataRole.UserRole)
    vkey = frozenset(abs(int(i)) for i in key)
    if vkey in self._merged_bif_unsigned_keys:
      self._set_status(
        "Cannot delete a merge-produced n-furcation from the list — uncheck "
        "the merge candidate, Generate mesh to restore separate bifurcations, "
        "then delete if still needed."
      )
      return
    self._removed_bif_keys.add(key)
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
    self._selected_centerline_id = None
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
    self._selected_centerline_id = None
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
    if mode != "Fix bifurcation":
      self._set_right_panel_fix_mode(False)
    if mode == "Fix bifurcation":
      self._update_fix_hint()
      self._render_fix_mode(preserve_camera=preserve_camera)
      return
    if mode == "Mesh":
      if self.last_mesh is None:
        self._set_status("No mesh loaded yet. Generate mesh first.")
        return
      self.scene.draw_mesh(
        self.last_mesh,
        self._mesh_preview_points,
        self._mesh_preview_boundary_wire,
        self._mesh_preview_boundary_scatter,
        self._mesh_preview_boundary_faces,
      )
      if self.scene.mesh_simplified:
        self._set_status("Mesh preview: boundary point cloud only (subsampled); statistics in the Mesh panel.")
      else:
        self._set_status("Mesh preview: full boundary surface (edges + vertices) + statistics in the Mesh panel.")
      return
    if mode == "Bifurcations":
      invalid_keys = frozenset(
        k for k, v in (self._bif_validity or {}).items() if not v
      )
      g_bif = self._graph_for_bif_list_and_3d()
      self.scene.draw_bifurcations(
        g_bif, self._selected_bif_key,
        preserve_camera=preserve_camera, invalid_keys=invalid_keys,
      )
      for disp, (orig_i, name) in enumerate(self._filenames, start=1):
        it = QListWidgetItem(f"{disp} → {name}")
        it.setData(Qt.ItemDataRole.UserRole, orig_i)
        self.id_map.addItem(it)
      self._set_status(
        f"Bifurcation view loaded: {len(g_bif.nontrivial_furcations)} nodes. "
        "List selection shows a large black centroid; unselected furcs are red."
      )
      return
    # Keep the selected centerline id only if it still exists after edits.
    valid_ids = {int(disp) for disp, _ in self._centerlines}
    if self._selected_centerline_id not in valid_ids:
      self._selected_centerline_id = None
    self.scene.draw_centerlines(
      self._centerlines,
      selected_id=self._selected_centerline_id,
      preserve_camera=preserve_camera,
    )
    # Stash the display id on each map row (UserRole+1) so we can look them up
    # for 3D-driven selections; UserRole keeps the original file id used by
    # delete/reset-centerline flows.
    self.id_map.blockSignals(True)
    for disp, (orig_i, name) in enumerate(self._filenames, start=1):
      it = QListWidgetItem(f"{disp} → {name}")
      it.setData(Qt.ItemDataRole.UserRole, orig_i)
      it.setData(Qt.ItemDataRole.UserRole + 1, int(disp))
      self.id_map.addItem(it)
    if self._selected_centerline_id is not None:
      row = int(self._selected_centerline_id) - 1
      if 0 <= row < self.id_map.count():
        self.id_map.setCurrentRow(row)
    self.id_map.blockSignals(False)
    self._set_status(f"Centerline view loaded: {len(self._centerlines)} lines.")

  def _render_fix_mode(self, preserve_camera=False):
    """3D: vascular-graph vessels (as in Bifurcations), mesh overlay, branch picks."""
    self._set_right_panel_fix_mode(True)
    self.id_map.clear()
    folder = self.csv_edit.text().strip()
    if not folder or not os.path.isdir(folder):
      self._set_status("Select a valid CSV folder.")
      return
    branches = self._fix_mode_branch_polylines()
    valid_ids = {int(i) for i, _ in branches}
    if self._selected_centerline_id not in valid_ids:
      self._selected_centerline_id = None
    show_mesh = self.fix_show_mesh.isChecked() and self.last_mesh is not None
    self.scene.draw_fix_view(
      branches,
      selected_id=self._selected_centerline_id,
      border_points=self._all_fix_anchor_xyz_for_draw(),
      fix_bif_ref=self._fix_mode_selected_bif_reference(),
      show_mesh=show_mesh,
      mesh=self.last_mesh,
      preview_points=self._mesh_preview_points,
      boundary_wire=self._mesh_preview_boundary_wire,
      boundary_scatter=self._mesh_preview_boundary_scatter,
      boundary_faces=self._mesh_preview_boundary_faces,
      preserve_camera=preserve_camera,
    )
    self.id_map.blockSignals(True)
    for idx, pts in branches:
      arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
      it = QListWidgetItem(f"Edge {idx} · polyline L≈{arc:.5g}")
      it.setData(Qt.ItemDataRole.UserRole, int(idx))
      it.setData(Qt.ItemDataRole.UserRole + 1, int(idx))
      self.id_map.addItem(it)
    if self._selected_centerline_id is not None:
      for row in range(self.id_map.count()):
        it = self.id_map.item(row)
        d = it.data(Qt.ItemDataRole.UserRole + 1)
        if d is not None and int(d) == int(self._selected_centerline_id):
          self.id_map.setCurrentRow(row)
          break
    self.id_map.blockSignals(False)
    n_need = len(self._selected_bif_key) if self._selected_bif_key is not None else 0
    n_hit = len(self._fix_border_hits)
    self._set_status(
      f"Fix bifurcation: bif-view vessels + {'mesh' if show_mesh else 'mesh hidden'}; "
      f"branch picks {n_hit}/{max(n_need, 1)}."
    )

  def _update_fix_hint(self):
    if self._selected_bif_key is None:
      n_queued = len(self._complete_fix_anchor_specs())
      qtxt = (
        f"{n_queued} bifurcation(s) have full anchors queued for the next mesh."
        if n_queued
        else ""
      )
      self.fix_hint.setText(
        "Select a bifurcation in the Bifurcations list (left), then click each "
        "incident vessel in 3D — the same polylines as Bifurcations mode "
        "(split/merged graph), not raw CSV lines. "
        "Amber: **baseline** vessel↔bif cut (Mesh **Truncation rfac** / "
        "**Refinement**) on the spline — ignores queued green picks until "
        "**Generate mesh**. **Green** = your trim targets on the polylines. "
        "Esc clears picks for the selected bif only. Completed anchors stay queued "
        "across successful mesh runs until you use “Clear all fix anchors”. "
        "Toggle mesh overlay as needed."
        + (f" {qtxt}" if qtxt else "")
      )
      return
    abs_e = sorted({abs(int(i)) for i in self._selected_bif_key})
    n = len(self._selected_bif_key)
    n_queued = len(self._complete_fix_anchor_specs())
    qtxt = (
      f" {n_queued} bifurcation(s) have full anchors and will trim together on Generate."
      if n_queued
      else ""
    )
    self.fix_hint.setText(
      f"Target bif — unsigned edges {abs_e}. Record one click per branch "
      f"({len(self._fix_border_hits)}/{n}). "
      f"Amber = baseline **rfac** cut (ignores green until Generate); "
      f"green = closest point on the **displayed** polyline to your click. "
      f"{qtxt}"
    )

  def _clear_fix_picks(self):
    if self._selected_bif_key is not None:
      self._fix_anchors_by_bif.pop(self._selected_bif_key, None)
    self._fix_border_hits = []
    self._update_fix_hint()
    if self.mode.currentText() == "Fix bifurcation":
      self._render_fix_mode(preserve_camera=True)

  def _clear_all_fix_anchors(self):
    self._fix_anchors_by_bif.clear()
    self._fix_border_hits = []
    self._update_fix_hint()
    if self.mode.currentText() == "Fix bifurcation":
      self._render_fix_mode(preserve_camera=True)

  def _on_fix_show_mesh_toggled(self, _checked=False):
    if self.mode.currentText() != "Fix bifurcation":
      return
    self._render_fix_mode(preserve_camera=True)

  def _on_fix_border_pick(self, cl_id, xyz):
    if self._selected_bif_key is None:
      self._set_status("Fix bifurcation: select a bifurcation in the list first.")
      return
    allowed = {abs(int(i)) for i in self._selected_bif_key}
    eid = int(cl_id)
    if eid not in allowed:
      self._set_status(
        f"That vessel edge (id {eid}) is not incident to the selected bifurcation; "
        f"pick one of {sorted(allowed)}."
      )
      return
    n = len(self._selected_bif_key)
    picked_edges = {abs(int(p[0])) for p in self._fix_border_hits}
    if eid in picked_edges:
      self._set_status(f"Vessel edge {eid} already has an anchor — Clear branch picks to redo.")
      return
    if len(self._fix_border_hits) >= n:
      self._set_status(
        f"Already have {n} anchors (full). Clear branch picks to change them."
      )
      return
    self._fix_border_hits.append((eid, np.asarray(xyz, dtype=float).ravel()[:3].copy()))
    self._persist_current_fix_hits()
    self._update_fix_hint()
    self._render_fix_mode(preserve_camera=True)
    if len(self._fix_border_hits) == n:
      coords = "; ".join(
        f"e{eid}→({p[0]:.4g},{p[1]:.4g},{p[2]:.4g})"
        for eid, p in self._fix_border_hits
      )
      self._set_status(
        f"All {n} branch anchors recorded — click Generate mesh to trim vessels "
        f"at those points and rebuild. ({coords})"
      )
    else:
      self._set_status(
        f"Branch anchor {len(self._fix_border_hits)}/{n} on edge {eid}."
      )

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
    # Resolve opt-in merges: map each checked component id back to the
    # tuple-of-tuples of its constituent unsigned-index bif keys.  This is
    # the shape ``generate_mesh`` → ``make_spline_mesh`` expects.
    opt_in_components = self._opt_in_components_from_ui()
    # After a merged mesh, overlap detection often returns no merge rows; the
    # checkbox list is rebuilt from ``_last_merge_opt_in`` synthetically, but
    # ``_opt_in_components_from_ui`` can still be empty if component_id /
    # panel state glitches.  Do **not** run ``generate_mesh`` without merges
    # while the user still has merge opt-ins checked — that drops the merged
    # topology and fix-anchor bif keys no longer match ``apply_merge``.
    if (
      not opt_in_components
      and self._last_merge_opt_in
      and self._merge_opt_in
    ):
      opt_in_components = [
        tuple(tuple(int(i) for i in k) for k in comp)
        for comp in self._last_merge_opt_in
      ]
    self._persist_current_fix_hits()
    fix_anchor_specs = self._complete_fix_anchor_specs()
    self._pending_fix_anchor_mesh = bool(fix_anchor_specs)
    kw = dict(
      csv_folder=folder,
      split_eps=self.split_eps.value(),
      merge_eps=self.merge_eps.value(),
      use_merge=self.use_merge.isChecked(),
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
      merge_opt_in=opt_in_components,
      fix_anchor_specs=tuple(fix_anchor_specs),
    )
    # Never overwrite a non-empty merge snapshot with ``()`` when this click
    # failed to resolve components — that breaks the next fallback.  Clear
    # only when the user has unchecked every merge checkbox.
    if opt_in_components:
      self._last_merge_opt_in = tuple(
        tuple(tuple(int(i) for i in k) for k in comp)
        for comp in opt_in_components
      )
    elif not self._merge_opt_in:
      self._last_merge_opt_in = ()

    def work():
      mesh = generate_mesh(**kw)
      bpts = _extract_boundary_points_for_preview(mesh)
      stats = mesh_statistics(mesh)
      bwire, bscatter, bfaces = _boundary_preview_arrays(mesh)
      # Per-bifurcation validity is attached by ``make_spline_mesh`` as
      # ``mesh.bif_validity = ((key, bool, repair_or_None), ...)`` (older
      # builds emit 2-tuples without the repair slot).  Normalise to a dict
      # ``{frozenset_key: {"valid": bool, "repair": dict | None}}`` — the
      # richer value lets the bif list show *why* a bifurcation is
      # still red (no cascade entry worked) or reassure the user which
      # per-branch rfac combination fixed it.
      raw = getattr(mesh, "bif_validity", ())
      bvalid: dict = {}
      for entry in raw:
        if len(entry) == 2:
          k, v = entry
          repair = None
        else:
          k, v, repair = entry[0], entry[1], entry[2] if len(entry) > 2 else None
        bvalid[frozenset(int(i) for i in k)] = {
          "valid": bool(v),
          "repair": repair,
          "overlaps": [],
        }
      # Per-edge overlap detection from ``make_spline_mesh``: edges where
      # the two adjacent bifurcations are so close that ``rtruncate`` had
      # to clamp the vessel stub between them — merge candidates for a
      # single n-furcation.  Fold the info into the per-bif dict so every
      # offending bif shows its partner(s) in the list tooltip.
      for ov in getattr(mesh, "bif_overlaps", ()):
        bs = ov.get("bif_start")
        be = ov.get("bif_end")
        if bs is None or be is None:
          # Terminal edge — no partner bif.  Still worth recording on the
          # single attached bif so the user sees the short-stub warning.
          attached = bs or be
          if attached is None:
            continue
          bvalid.setdefault(attached, {
            "valid": True, "repair": None, "overlaps": [],
          })["overlaps"].append({
            "iedge": int(ov["iedge"]),
            "remaining_fraction": float(ov["remaining_fraction"]),
            "other": None,
          })
          continue
        info = {
          "iedge": int(ov["iedge"]),
          "remaining_fraction": float(ov["remaining_fraction"]),
        }
        bvalid.setdefault(bs, {
          "valid": True, "repair": None, "overlaps": [],
        })["overlaps"].append({**info, "other": be})
        bvalid.setdefault(be, {
          "valid": True, "repair": None, "overlaps": [],
        })["overlaps"].append({**info, "other": bs})
      # Connected components of the merge-candidate graph — one checkbox
      # per component gets rendered at the top of the Bifurcations panel.
      # Each item is a plain dict so it survives signal marshalling
      # untouched; sets/frozensets in the items are already hashable and
      # picklable.
      components_raw = getattr(mesh, "merge_components", ()) or ()
      merge_components = [
        {
          "component_id": comp["component_id"],
          "bif_keys": tuple(
            frozenset(int(i) for i in k) for k in comp["bif_keys"]
          ),
          "internal_edges": tuple(sorted(int(i) for i in comp["internal_edges"])),
        }
        for comp in components_raw
      ]
      return mesh, bpts, stats, bwire, bscatter, bfaces, bvalid, merge_components

    self.b_gen.setEnabled(False)
    self.b_stop_gen.setEnabled(True)
    if self._pending_fix_anchor_mesh:
      nfa = len(fix_anchor_specs)
      self._set_status(
        "Generating mesh (Fix bifurcation anchor(s) on "
        f"{nfa} junction(s); no rfac repair scan on this run…"
      )
    else:
      self._set_status("Generating mesh in background… You can keep using the window.")
    QApplication.processEvents()
    th = MeshComputeThread(work)
    self._mesh_thread = th
    th.finished_ok.connect(self._on_mesh_thread_ok)
    th.finished_err.connect(self._on_mesh_thread_err)
    th.finished.connect(self._on_mesh_thread_finished)
    th.start()

  def _on_mesh_thread_ok(self, mesh, bpts, stats, bwire, bscatter, bfaces,
                         bvalid, merge_components=()):
    applied_fix_anchors = self._pending_fix_anchor_mesh
    self.last_mesh = mesh
    self._mesh_preview_points = bpts
    self._mesh_preview_boundary_wire = bwire
    self._mesh_preview_boundary_scatter = bscatter
    self._mesh_preview_boundary_faces = bfaces
    # ``bvalid`` from the worker is now a dict of dicts with three slots:
    # ``{"valid": bool, "repair": info | None, "overlaps": list}``.  Split
    # into three flat dicts so all downstream code that expects plain
    # booleans keeps working unchanged; repair and overlap info are shown
    # separately in the list tooltip / status line.
    self._bif_validity = {k: bool(entry["valid"]) for k, entry in (bvalid or {}).items()}
    self._bif_repair = {k: entry.get("repair") for k, entry in (bvalid or {}).items()}
    self._bif_overlaps = {
      k: list(entry.get("overlaps") or [])
      for k, entry in (bvalid or {}).items()
    }
    if applied_fix_anchors:
      # Keep ``_fix_anchors_by_bif`` so later Generate clicks still apply **all**
      # completed junction trims (e.g. fix bif A, mesh OK, then fix bif B — A must
      # stay in the dict).  Clear only the in-memory row; resync from storage.
      self._fix_border_hits = []
      self._sync_fix_hits_from_storage()
      self._pending_fix_anchor_mesh = False
      self._update_fix_hint()
    # Connected-component list: one checkbox per entry at the top of the
    # Bifurcations panel.  ``bif_keys`` arrives as a tuple of frozensets
    # (already picklable), but normalise once here so downstream code can
    # assume a stable shape.
    raw_merge = merge_components or ()
    if not raw_merge and self._last_merge_opt_in:
      raw_merge = self._synthetic_merge_components_from_last_opt_in()
      # Overlap detection often disappears after a merge, so the worker sends
      # no ``merge_components``; the checkbox row would be empty and
      # ``_merge_opt_in`` could have been cleared.  Re-queue the same merge for
      # the next Generate unless the user explicitly unchecks it.
      for c in raw_merge:
        self._merge_opt_in.add(c["component_id"])
    self._merge_components = [
      {
        "component_id": c["component_id"],
        "bif_keys": tuple(frozenset(int(i) for i in k) for k in c["bif_keys"]),
        "internal_edges": tuple(int(i) for i in c["internal_edges"]),
      }
      for c in raw_merge
    ]
    self._rebuild_graph_for_bif_ui(self._last_merge_opt_in)
    self._set_mesh_stats(stats)
    # Rebuild the checkbox row first so the subsequent bif-list refresh
    # sees the current opt-in state and its status-bar sentence (added by
    # ``_on_merge_checkbox_toggled``) doesn't get clobbered.
    self._fill_merge_checkboxes()
    # Refresh the bifurcation list so invalid bifs turn red on the same
    # pass, even if we stay in Centerlines / Mesh view.
    self._fill_bif_list()
    if self.mode.currentText() != "Mesh":
      self.mode.blockSignals(True)
      self.mode.setCurrentText("Mesh")
      self.mode.blockSignals(False)
    self._render_mode()
    n_total = len(self._bif_validity)
    n_bad = sum(1 for v in self._bif_validity.values() if not v)
    # A "repaired" bif: originally failed validity but was rescued by the
    # per-branch rfac retry cascade.  Surface this count in the status so
    # the user immediately sees *how much* of the mesh needed repair, not
    # just the final good/bad split.
    n_repaired = sum(
      1
      for k, v in self._bif_validity.items()
      if v and (self._bif_repair or {}).get(k) is not None
      and (self._bif_repair or {}).get(k, {}).get("branch_rfac") is not None
    )
    # Merge-candidate count: number of *distinct* bifurcations that share
    # a near-degenerate edge with another (counted per bif, not per pair
    # — a bif that overlaps with two neighbours counts as one).
    n_merge_candidates = sum(
      1 for lst in (self._bif_overlaps or {}).values() if lst
    )
    if n_total == 0:
      tail = ""
    elif n_bad == 0 and n_repaired == 0:
      tail = f" All {n_total} bifurcations valid."
    elif n_bad == 0 and n_repaired > 0:
      tail = (
        f" All {n_total} bifurcations valid — "
        f"{n_repaired} rescued by per-branch rfac scan "
        "(see Bifurcations list tooltips for the multipliers)."
      )
    else:
      extra = (
        f", {n_repaired} rescued by per-branch rfac scan"
        if n_repaired else ""
      )
      tail = (
        f" {n_bad} of {n_total} bifurcations self-intersect{extra} — "
        "marked in red in the Bifurcations list / view."
      )
    if n_merge_candidates:
      tail += (
        f"  {n_merge_candidates} merge candidate(s) detected "
        "(two bifurcations too close to fit a vessel between them — "
        "see Bifurcations list tooltip)."
      )
    anchor_tail = (
      "  Fix bifurcation anchors were applied; automatic per-bif rfac repair was "
      "skipped for this run (baseline truncations only). "
      "Completed junction anchors stay queued for the next Generate until you "
      "clear them (per-bif or “Clear all fix anchors”)."
      if applied_fix_anchors else ""
    )
    self._set_status(
      "Mesh generated. Full boundary preview + statistics below." + tail + anchor_tail
    )

  def _on_mesh_thread_err(self, tb):
    self._pending_fix_anchor_mesh = False
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

  def keyPressEvent(self, event):
    if (
      event.key() == Qt.Key.Key_Escape
      and self.mode.currentText() == "Fix bifurcation"
      and self._fix_border_hits
    ):
      self._clear_fix_picks()
      event.accept()
      return
    super().keyPressEvent(event)

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
