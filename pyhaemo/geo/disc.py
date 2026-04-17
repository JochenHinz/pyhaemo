"""Five-patch unit disc: spline geometry, harmonic map, and discrete boundary-layer control nets.

The trampoline topology (eight vertices, five bilinear patches) covers the unit square
with a circular hole.  :func:`ellipse` / :func:`_disc` build Nutils multipatch splines.

Boundary-layer grading (``boundary_layer_exponent > 1``)
---------------------------------------------------------
After the harmonic map is projected onto the discontinuous B-spline basis, the
**control points** of the five patches are relocated:

* **Outer patches** (the four that touch the circular boundary): every control
  point ``p`` is slid along the straight *fiber chord* from the inner endpoint
  ``s`` to the outer endpoint ``e`` of its own tensor row/column.  The new
  position is determined by grading the point's current radius ``|p|`` with the
  uniform map  ``r → 1 − (1 − r)^exponent`` and then mapping the resulting
  radius back onto the chord as a linear fraction ``t``.  Because the map
  depends only on ``|p|``, the expansion intensity is identical at every angle.

* **Center patch**: after the outer patches are moved, the shared interface
  edges are copied from the outer neighbours, and the center interior is rebuilt
  with a Coons bilinear blend of the four boundary polylines so that
  :math:`C^0` continuity is preserved across all patch interfaces.

Relocating the control points causes the B-spline evaluation ``∑ N_ij · c_ij``
to produce sample points (mesh nodes) that cluster near the outer wall — the
boundary-layer effect in physical space.

Use :func:`plot_disc_control_net_2d` to draw the tensor-product control net in
the plane and inspect the BL distribution.
"""
from pyhaemo.util import np, _
from pyhaemo.topo import Topology
from pyhaemo.nutils.mul import multipatch, basis_spline
from pyhaemo.spl import UnivariateKnotVector, NDSpline, NDSplineArray
from pyhaemo.types import Float, Int, Numeric

from itertools import product
from functools import lru_cache
from collections import Counter

from nutils import function, solver
import treelog as log


# Patch corner indices into ``patchverts`` (see :func:`trampoline_template`).
PATCHES = (2, 0, 3, 1), \
          (2, 4, 0, 6), \
          (2, 3, 4, 5), \
          (3, 1, 5, 7), \
          (4, 5, 6, 7)

# Three independent univariate knot vectors are replicated across symmetric edges.
KNOTVECTOR_EDGES = (0, 1), (0, 6), (0, 2)


ALL_SIDES = 'left', 'right', 'top', 'bottom'

# Guard against division by zero at the disc centre in the radial BL map.
_RHO_SAFE = 1e-15


def _exterior_undirected_edges(topo: Topology):
  """Return undirected vertex pairs that belong to exactly one patch (boundary edges)."""
  cnt = Counter()
  for _ipatch, edge in topo.patch_edge_iter():
    cnt[tuple(sorted(edge))] += 1
  return {e for e, n in cnt.items() if n == 1}


def _patch_side_edges(patch_verts):
  """Map structured side name → sorted global vertex pair for one patch."""
  c0, c1, c2, c3 = patch_verts
  return {
    'left':   tuple(sorted((c0, c1))),
    'right':  tuple(sorted((c2, c3))),
    'bottom': tuple(sorted((c0, c2))),
    'top':    tuple(sorted((c1, c3))),
  }


def _graded_radius(r, exponent: Float):
  """Uniform radial BL map: ``r → 1 − (1 − min(r, 1))^exponent``.

  * ``exponent == 1``: identity.
  * ``exponent > 1``: compresses the interval ``[0, 1]`` toward ``1``,
    giving denser spacing near the outer boundary.
  The map is applied pointwise to the Euclidean radius ``r = |p|`` of each
  control point and is independent of the angular position, so all angles
  receive the same grading intensity.
  """
  return 1. - (1. - min(float(r), 1.))**float(exponent)


def _apply_discrete_bl_radial(controlpoints, side: str,
                               exponent: Float) -> None:
  """Move control points in-place for the discrete radial BL of one outer patch.

  Algorithm (for every control point ``(i, j)``):

  1. **Identify the fiber chord** – the straight segment from the inner endpoint
     ``s`` (closest to the disc centre) to the outer endpoint ``e`` (on the
     circular boundary) of the same tensor row or column, selected by ``side``.

  2. **Grade the radius** – compute the target Euclidean radius
     ``r_new = _graded_radius(|p|, exponent)``.  Because this map depends only
     on the magnitude of the point, the expansion intensity is identical for
     every angle (including the diagonal 45/135/225/315° patch interfaces).

  3. **Place back on the chord** – find the chord fraction
     ``t = (r_new − |s|) / (|e| − |s|)`` and set ``p ← s + t·(e − s)``.
     This keeps all control points on their original straight fiber so the
     control-net lines remain perfectly straight after grading.

  Parameters
  ----------
  controlpoints:
      Array of shape ``(nxi, neta, 2)`` modified **in place**.
  side:
      Name of the outer boundary side (``'top'``, ``'bottom'``, ``'left'``,
      ``'right'``).  Determines which axis is the radial direction.
  exponent:
      BL grading exponent passed to :func:`_graded_radius`.
  """
  # Keep a read-only snapshot of the original positions so that chord
  # endpoints are not affected by the in-place updates of interior points.
  ref = np.array(controlpoints, dtype=float, copy=True)
  nxi, neta = int(ref.shape[0]), int(ref.shape[1])

  for i in range(nxi):
    for j in range(neta):
      # s = inner endpoint, e = outer endpoint for this fiber.
      if side == 'top':
        s, e = ref[i, 0, :], ref[i, -1, :]
      elif side == 'bottom':
        s, e = ref[i, -1, :], ref[i, 0, :]
      elif side == 'right':
        s, e = ref[0, j, :], ref[-1, j, :]
      else:  # left
        s, e = ref[-1, j, :], ref[0, j, :]

      r_s = float(np.hypot(s[0], s[1]))
      r_e = float(np.hypot(e[0], e[1]))
      r_span = r_e - r_s
      if abs(r_span) < _RHO_SAFE:
        # Degenerate fiber (zero radial span) — skip to avoid division by zero.
        continue

      r_p = float(np.hypot(ref[i, j, 0], ref[i, j, 1]))
      r_new = _graded_radius(r_p, exponent)
      t = np.clip((r_new - r_s) / r_span, 0., 1.)
      controlpoints[i, j, :] = s + t * (e - s)


def _discrete_outer_radial_project_and_bl(controlpoints_segments, tkvs, patches,
                                          exponent: Float):
  """Apply the discrete radial BL to the four outer patches.

  Iterates over all patches.  For patches with **exactly one** exterior side
  (the four outer patches), :func:`_apply_discrete_bl_radial` is called in
  place.  The center patch (zero exterior sides) is left unchanged here and
  updated afterwards by :func:`_propagate_center_patch_bl`.

  Parameters
  ----------
  controlpoints_segments:
      List of per-patch control-point arrays (each shape ``(nxi·neta, 2)``).
  tkvs:
      List of per-patch tensor-product knot vectors.
  patches:
      The ``PATCHES`` corner-index tuples (topology connectivity).
  exponent:
      BL grading exponent forwarded to :func:`_apply_discrete_bl_radial`.

  Returns
  -------
  list
      Updated per-patch control-point arrays, same structure as input.
  """
  exterior = _exterior_undirected_edges(Topology(patches))
  out = []
  for p, cps in enumerate(controlpoints_segments):
    kv = tkvs[p]
    nxi, neta = int(kv.dim[0]), int(kv.dim[1])
    prow = patches[p]
    cps = np.asarray(cps, dtype=float).reshape(nxi, neta, 2, order='C')
    outer_sides = [side for side, ek in _patch_side_edges(prow).items()
                   if ek in exterior]

    if len(outer_sides) == 1:
      _apply_discrete_bl_radial(cps, outer_sides[0], exponent)

    out.append(cps.reshape(-1, 2, order='C'))
  return out


def _get_side_line(cps, side: str):
  """Return a copy of the boundary polyline on ``side`` of ``cps`` (shape ``(n, 2)``)."""
  if side == 'left':
    return np.array(cps[0, :, :], copy=True)
  if side == 'right':
    return np.array(cps[-1, :, :], copy=True)
  if side == 'bottom':
    return np.array(cps[:, 0, :], copy=True)
  return np.array(cps[:, -1, :], copy=True)  # top


def _set_side_line(cps, side: str, line):
  """Overwrite the boundary polyline on ``side`` of ``cps`` in place."""
  if side == 'left':
    cps[0, :, :] = line
  elif side == 'right':
    cps[-1, :, :] = line
  elif side == 'bottom':
    cps[:, 0, :] = line
  else:  # top
    cps[:, -1, :] = line


def _orient_line_like(line, ref):
  """Return ``line`` (or its reversal) oriented consistently with ``ref``.

  Patch edge traversal directions may differ between neighbours; flipping
  makes the shared interface polyline flow in the same direction before it
  is written onto the center patch boundary.
  """
  d0 = np.linalg.norm(line[0] - ref[0]) + np.linalg.norm(line[-1] - ref[-1])
  d1 = np.linalg.norm(line[-1] - ref[0]) + np.linalg.norm(line[0] - ref[-1])
  return line if d0 <= d1 else line[::-1]


def _propagate_center_patch_bl(controlpoints_segments, tkvs, patches):
  """Update the center patch so it is :math:`C^0`-consistent with the moved outer patches.

  The outer patches have already been graded by :func:`_apply_discrete_bl_radial`,
  which moves their control points radially.  The center patch shares four
  interface edges with those outer patches; simply leaving its original harmonic
  control points would break :math:`C^0` continuity.  This function:

  1. **Copies interface lines** – for every side of the center patch, overwrites
     the boundary polyline with the corresponding (oriented) line taken from the
     adjacent outer patch.

  2. **Rebuilds the interior with Coons blending** – fills the center interior
     as a bilinear Coons patch from the four updated boundary polylines.
     The classical formula is ``blend_LR + blend_BT − corner_bilinear``, which
     is exact at the boundaries and smooth in-between.

  3. **Re-applies the interface lines** – overwrites the boundary polylines
     once more after the Coons fill to cancel any floating-point drift.

  Parameters
  ----------
  controlpoints_segments:
      List of per-patch control-point arrays after the outer BL step.
  tkvs, patches:
      Same knot-vector list and topology connectivity as the rest of the pipeline.

  Returns
  -------
  list
      Updated list with the center patch replaced.
  """
  exterior = _exterior_undirected_edges(Topology(patches))
  side_edges = [_patch_side_edges(prow) for prow in patches]
  outer_sides = [{s for s, e in se.items() if e in exterior} for se in side_edges]

  center_ids = [i for i, os in enumerate(outer_sides) if len(os) == 0]
  if len(center_ids) != 1:
    return controlpoints_segments

  ic = center_ids[0]
  nxi, neta = map(int, tkvs[ic].dim)
  center = np.asarray(controlpoints_segments[ic], dtype=float).reshape(nxi, neta, 2, order='C')

  # Step 1: copy the moved boundary polylines from each outer neighbour.
  for cside, cedge in side_edges[ic].items():
    for j in range(len(patches)):
      if j == ic:
        continue
      if cedge not in side_edges[j].values() or len(outer_sides[j]) != 1:
        continue
      nside = next(s for s, e in side_edges[j].items() if e == cedge)
      nj, mj = map(int, tkvs[j].dim)
      neigh = np.asarray(controlpoints_segments[j], dtype=float).reshape(nj, mj, 2, order='C')
      nline = _get_side_line(neigh, nside)
      cline = _get_side_line(center, cside)
      _set_side_line(center, cside, _orient_line_like(nline, cline))
      break

  # Step 2: bilinear Coons patch from the four boundary polylines.
  # p(u,v) = C_lr(u,v) + C_bt(u,v) - C_corners(u,v)   (Gordon-Hall blending)
  left   = _get_side_line(center, 'left')    # shape (neta, 2), indexed by v
  right  = _get_side_line(center, 'right')   # shape (neta, 2), indexed by v
  bottom = _get_side_line(center, 'bottom')  # shape (nxi, 2),  indexed by u
  top    = _get_side_line(center, 'top')     # shape (nxi, 2),  indexed by u
  p00, p01, p10, p11 = bottom[0], top[0], bottom[-1], top[-1]

  for i in range(nxi):
    u = i / max(nxi - 1, 1)
    for k in range(neta):
      v = k / max(neta - 1, 1)
      blend_lr = (1 - u) * left[k]   + u       * right[k]
      blend_bt = (1 - v) * bottom[i] + v       * top[i]
      corner   = (1 - u) * (1 - v)   * p00     + \
                 (1 - u) * v         * p01     + \
                 u       * (1 - v)   * p10     + \
                 u       * v         * p11
      center[i, k] = blend_lr + blend_bt - corner

  # Step 3: re-enforce boundary lines to remove any Coons drift at edges.
  for cside, cedge in side_edges[ic].items():
    for j in range(len(patches)):
      if j == ic:
        continue
      if cedge not in side_edges[j].values() or len(outer_sides[j]) != 1:
        continue
      nside = next(s for s, e in side_edges[j].items() if e == cedge)
      nj, mj = map(int, tkvs[j].dim)
      neigh = np.asarray(controlpoints_segments[j], dtype=float).reshape(nj, mj, 2, order='C')
      nline = _get_side_line(neigh, nside)
      cline = _get_side_line(center, cside)
      _set_side_line(center, cside, _orient_line_like(nline, cline))
      break

  ret = list(controlpoints_segments)
  ret[ic] = center.reshape(-1, 2, order='C')
  return ret


def _radial_boundary_layer_unit_disc(u, exponent: Float):
  """Continuous radial BL map on a 2D vector field: ``ρ → 1 − (1 − ρ)^exponent``.

  Applied as a Nutils function expression so it can be used directly in the
  variational projection.  Used only when ``boundary_layer_exponent == 1``
  (identity) or when the caller opts into the continuous rather than discrete
  BL path.  The unit circle and the origin are fixed points of the map.
  """
  if exponent == 1:
    return u
  rho = function.sqrt((u * u).sum())
  rho_c = function.min(rho, 1.)
  t = 1. - rho_c
  scaled_r = 1. - t**exponent
  return u * (scaled_r / function.max(rho, _RHO_SAFE))


def trampoline_template(inner_height: Float = .5,
                        inner_width: Float = .5):
  r"""
    1 - - - - - 7
    | \   D   / |
    |  3 - - 5  |
    |A |  C  | E|
    |  |     |  |
    |  2 - - 4  |
    | /   B   \ |
    0 - - - - - 6
  """
  assert 0 < inner_height < 1
  assert 0 < inner_width < 1

  h, w = (1 - inner_height) / 2, (1 - inner_width) / 2

  # the vertices of the multipatch topology numbered from 0 to 7
  patchverts = ( (0, 0), (0, 1),
                 (w, h), (w, 1 - h),
                 (1 - w, h), (1 - w, 1 - h),
                 (1, 0), (1, 1) )

  # a patch is characterised by its patch vertices
  # patch = (0, 1, 2, 3) means it is oriented like this:
  """
      1 ----- 3
      |       |
      |       |
      |       |
      0-------2
  """

  # so the left side of the patch is given by edge 0 --- 1
  # right by 2 --- 3
  # bottom by 0 --- 2
  # top by 1 --- 3

  # patches and patchverts are tuple. Immutable objects.

  return Topology(PATCHES), patchverts


@lru_cache(maxsize=32)
def _disc(kv0, kv1, kv2, reparam: bool = True,
          boundary_layer_exponent: Float = 1.) -> NDSplineArray:
  """Build and cache the five-patch unit-disc spline array.

  Pipeline
  --------
  1. Assemble the trampoline knot-vector map (three independent KVs propagated
     to all symmetric edges via the topology opposites).
  2. Solve for the harmonic (Dirichlet-energy-minimising) map via
     :func:`make_unit_disc`, optionally with trace-penalty reparameterisation
     to smooth the patch interface angles.
  3. Project the harmonic map onto the **discontinuous** B-spline basis
     ``basisd`` (patch-continuous=False) to obtain per-patch control points.
  4. If ``boundary_layer_exponent != 1``, apply the discrete radial BL:
     move outer-patch control points along their fiber chords
     (:func:`_discrete_outer_radial_project_and_bl`) and rebuild the center
     patch with Coons blending (:func:`_propagate_center_patch_bl`).
     A fallback restores the harmonic values for any non-finite DOFs.
  5. Pack the control-point arrays into :class:`NDSpline` objects (with a
     trailing zero for the third spatial dimension) and return them as an
     :class:`NDSplineArray`.

  The result is cached by ``(kv0, kv1, kv2, reparam, exponent)`` so that
  repeated calls with the same arguments are free.
  """

  topo, patchverts = trampoline_template()
  opposites = topo.opposites

  # Propagate the three independent KVs to all symmetric edges.
  kvs = {}
  for edge, kv in zip(KNOTVECTOR_EDGES, (kv0, kv1, kv2)):
    for _edge in (edge, *opposites[edge]):
      kvs[_edge] = kv
      kvs[_edge[::-1]] = kv.flip()

  degree = max(kv.degree for kv in kvs.values())

  domain, geom, localgeom = multipatch(patches=topo.topo,
                                       knotvectors=kvs,
                                       patchverts=patchverts)

  basis  = basis_spline(domain, kvs)
  basisd = basis_spline(domain, kvs, patchcontinuous=False)

  tkvs     = [kvs[patch[::2]] @ kvs[patch[:2]] for patch in PATCHES]
  segments = np.array([kv.ndofs for kv in tkvs]).cumsum()[:-1]

  # When discrete BL is active, skip the continuous radial map inside
  # make_unit_disc; the grading will be applied on the projected DOFs instead.
  disc_bl    = boundary_layer_exponent != 1.
  controlmap = make_unit_disc(domain, basis, geom, localgeom, topo.topo,
                              reparam=reparam,
                              apply_finalize_bl=not disc_bl,
                              boundary_layer_exponent=boundary_layer_exponent)

  # Project onto the discontinuous basis to get per-patch control points.
  controlpoints = domain.project(controlmap, onto=basisd.vector(2),
                                             geometry=geom,
                                             degree=2 * degree).reshape(-1, 2)

  if disc_bl:
    harmonic_cp = np.array(controlpoints, copy=True)
    segs = list(np.array_split(controlpoints, segments, axis=0))
    segs = _discrete_outer_radial_project_and_bl(
        segs, tkvs, PATCHES, boundary_layer_exponent)
    segs = _propagate_center_patch_bl(segs, tkvs, PATCHES)
    controlpoints = np.concatenate(segs, axis=0)
    # Safety net: revert any DOFs that became NaN/Inf during grading.
    bad = ~np.isfinite(controlpoints).all(axis=1)
    if bad.any():
      log.warning('Radial BL produced non-finite control points; reverting '
                  'those DOFs to the harmonic projection.')
      controlpoints[bad] = harmonic_cp[bad]

  # Append a zero z-coordinate so all patches live in the xy-plane (z=0).
  spls = [NDSpline(kv, np.concatenate([cps, np.zeros(len(cps))[:, _]], axis=1))
          for kv, cps in zip(tkvs, np.array_split(controlpoints, segments, axis=0))]

  return NDSplineArray(spls)


def ellipse(a: Numeric, b: Numeric, kv0: Int | UnivariateKnotVector = 5,
                                    kv1: Int | UnivariateKnotVector | None = None,
                                    kv2: Int | UnivariateKnotVector | None = None,
                                    reparam: bool = True,
                                    boundary_layer_exponent: Float = 1.) -> NDSplineArray:
  """
  Create a five patch ellipse spline with major axes of length `a` and `b`.

  Parameters
  ----------
  a : Numeric
    First axis length.
  b : Numeric
    Second axis length.
  kv0 : Int | UnivariateKnotVector
    The knotvector assigned to edge (0, 1). If of type `Int`, it gets
    coerced into a `UnivariateKnotVector` with `degree=3` and `kv0` internal
    elements.
  kv1 : Int | UnivariateKnotVector | None
    The knotvector assigned to edge (0, 6). `Int` inputs get coerced and
    if `None`, defaults to `kv0`.
  kv2 : Int | UnivariateKnotVector | None
    The knotvector assigned to edge (0, 2). `Int` inputs get coerced and
    if `None`, defaults to `kv0`.
  reparam : bool
    Whether to suppress the disc's interface angles.
  boundary_layer_exponent : :class:`float`
    Radial grading on the unit disc before scaling by ``a`` and ``b``: ``1`` is
    uniform; larger values push the control net toward the circular boundary.
  """

  if isinstance(kv0, Int):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, kv0 + 1), degree=3)

  if isinstance(kv1, Int):
    kv1 = UnivariateKnotVector(np.linspace(0, 1, kv1 + 1), degree=3)
  elif kv1 is None:
    kv1 = kv0

  if isinstance(kv2, Int):
    kv2 = UnivariateKnotVector(np.linspace(0, 1, kv2 + 1), degree=3)
  elif kv2 is None:
    kv2 = kv0

  return _disc(kv0, kv1, kv2, reparam=reparam,
               boundary_layer_exponent=boundary_layer_exponent) * np.array([a, b, 0])[_]


def plot_disc_control_net_2d(disc: NDSplineArray, *, ax=None,
                             patch_colors: tuple | None = None,
                             unit_circle: bool = True,
                             linewidth: Float = 0.85,
                             line_kw: dict | None = None,
                             circle_kw: dict | None = None):
  """
  Plot the 2D control-net grid for each patch of a five-patch disc (e.g. :func:`ellipse`).

  Requires ``matplotlib``. Pass ``ax`` or a new figure is created via ``plt.subplots()``.
  """
  import matplotlib.pyplot as plt

  if disc.ndim < 2:
    raise ValueError('disc must be an NDSplineArray with shape (n_patches, …), e.g. (5, 3).')
  npatch = int(disc.shape[0])
  if patch_colors is None:
    patch_colors = tuple(f'C{i % 10}' for i in range(npatch))
  elif len(patch_colors) < npatch:
    raise ValueError('patch_colors must have at least one entry per patch.')

  if ax is None:
    _, ax = plt.subplots(figsize=(6, 6))

  line_kw = {'linewidth': linewidth, 'alpha': 0.9, **(line_kw or {})}
  circle_kw = {'linestyle': '--', 'linewidth': 0.9, 'color': '0.45', **(circle_kw or {})}

  for p in range(npatch):
    block = disc[p]
    spl = np.asarray(block.arr, dtype=object).ravel()
    if spl.size != 1:
      raise ValueError(f'expected one vector spline per patch index, got {spl.size} for patch {p}.')
    xy = np.asarray(spl[0].tcontrolpoints[..., :2], dtype=float)
    nu, nv = xy.shape[:2]
    c = patch_colors[p]
    for j in range(nv):
      ax.plot(xy[:, j, 0], xy[:, j, 1], color=c, **line_kw)
    for i in range(nu):
      ax.plot(xy[i, :, 0], xy[i, :, 1], color=c, **line_kw)

  if unit_circle:
    t = np.linspace(0., 2. * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), **circle_kw)

  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid(True, alpha=0.25)
  return ax


def get_all_vertices(domain, geom, patches):
  """Collect unique patch-corner positions in ``patches`` (global vertex indices → R^3)."""

  verts = {}

  for i, patch in enumerate(patches):
    for ii, j in enumerate(patch):
      if j in verts: continue
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = domain._topos[i].boundary[side0] \
                                 .boundary[side1] \
                                 .sample('vertex', 0) \
                                 .eval(geom).ravel()

  return np.stack([verts[key] for key in sorted(verts.keys())], axis=0)


def multipatch_trace_penalty_stab(domain,
                                  basis,
                                  geom,
                                  localgeom,
                                  patches,
                                  stabmat=None,
                                  mu=800):

  """
  Nutils reparameterization of the harmonic disc map for more regular patch interfaces.

  Solves a linear problem derived from a trace-weighted stiffness with optional
  per-point stabilization ``stabmat(x)``. Large ``mu`` sharpens the vertex mollifiers
  in ``blending`` (see implementation).
  """

  assert mu >= 0

  if stabmat is None:
    stabmat = lambda x: function.eye(2)

  Jmu = function.normalized(geom.grad(localgeom), axis=0)
  D = function.matmat(Jmu, Jmu.T)

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  stabmat = stabmat(x)

  allverts = get_all_vertices(domain, geom, patches)
  blending = sum( np.exp(-mu * ((vert - geom)**2).sum()) for vert in allverts )

  D = (1 - blending) * D + blending * stabmat
  Dx = function.matmat(x.grad(geom), D)

  integrand = (basis.vector(2).grad(geom) * Dx[None]).sum([1, 2])
  res = domain.integral( integrand * function.J(geom), ischeme='gauss6')

  cons = domain.boundary.project(geom, onto=basis.vector(2),
                                       geometry=geom,
                                       ischeme='gauss6')

  controlmap = basis.vector(2) \
                    .dot(solver.solve_linear('target', res, constrain=cons))

  return controlmap


def make_unit_disc(domain, basis, geom, localgeom, patches, reparam=False,
                   boundary_layer_exponent: Float = 1.,
                   apply_finalize_bl: bool = True,
                   **kwargs):

  """
  Create a multipatch covering of the unit disc from the topology.

  Parameters
  ----------
  boundary_layer_exponent : :class:`float`
    After the harmonic map, apply radial grading in unit-disc coordinates
    (ρ → 1 - (1-ρ)^exponent). ``1`` leaves the map unchanged; values greater
    than one cluster the control net toward the circular boundary.
    For ``exponent != 1``, grading is applied on discrete control points after
    projection (with exterior points aligned to reference radials first).
  apply_finalize_bl : :class:`bool`
    If ``False``, return the harmonic map in unit-disc coordinates without the
    continuous radial boundary-layer map (used when applying that map on the
    projected control net instead).
  """

  if reparam is False and kwargs:
    log.warning('reparam is False so the keyword arguments will be ignored.')

  center = np.array([.5, .5])

  func = function.normalized(geom - center) / np.sqrt(2) + center
  cons = domain.boundary.project(func, geometry=geom,
                                       onto=basis.vector(2),
                                       ischeme='gauss6')

  for (i, patch), side in product(enumerate(domain._topos), ALL_SIDES):
    cons |= patch.boundary[side].project(geom, geometry=geom,
                                               onto=basis.vector(2),
                                               ischeme='gauss10')

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  J = x.grad(geom)
  G = function.matmat(J.T, J)

  costfunc = domain.integral(np.trace(G) * function.J(geom), degree=10)
  controlmap = basis.vector(2) \
                    .dot(solver.optimize('target', costfunc, constrain=cons))

  def finalize(ctrl):
    u = (ctrl - center) * np.sqrt(2)
    if not apply_finalize_bl:
      return u
    return _radial_boundary_layer_unit_disc(u, boundary_layer_exponent)

  if not reparam:
    return finalize(controlmap)

  return finalize(multipatch_trace_penalty_stab(domain,
                                                basis,
                                                controlmap,
                                                localgeom,
                                                patches,
                                                **kwargs))


if __name__ == '__main__':
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser(description='Plot five-patch disc control nets (see plot_disc_control_net_2d).')
  parser.add_argument('--save', metavar='PATH', help='write figure to PATH (e.g. .png) instead of opening a window')
  ns = parser.parse_args()

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  plot_disc_control_net_2d(ellipse(1., 1., 5, boundary_layer_exponent=3.), ax=axes[0])
  axes[0].set_title('boundary_layer_exponent = 3')
  plot_disc_control_net_2d(ellipse(1., 1., 5, boundary_layer_exponent=10), ax=axes[1])
  axes[1].set_title('boundary_layer_exponent = 10')
  fig.suptitle('Five-patch disc: control nets (dashed: unit circle)')
  fig.tight_layout()
  if ns.save:
    fig.savefig(ns.save, dpi=150)
    print(ns.save)
  else:
    plt.show()
