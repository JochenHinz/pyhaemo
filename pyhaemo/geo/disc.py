"""Five-patch unit disc: spline geometry, harmonic map, and discrete boundary-layer control nets.

The trampoline topology (eight vertices, five bilinear patches) covers the unit square
with a circular hole.  :func:`ellipse` / :func:`_disc` build Nutils multipatch splines.

Boundary-layer expansion (``boundary_layer_exponent > 1``)
------------------------------------------------------------
After the harmonic map is projected onto the discontinuous B-spline basis, the
**control points** of the five patches are relocated.

**Terms (discrete step on each outer-patch tensor fiber)**

* **Curved fiber** — The polyline of control points along one tensor index from
  inner endpoint ``i`` to outer endpoint ``e`` *before* the step.  Its vertices
  usually do **not** lie on a single straight line.

* **Reference chord** — The straight segment **``i``–``e``** in the plane, using
  the same ``i`` and ``e`` as the curved fiber's endpoints.  It is read from a
  **frozen snapshot** ``ref`` of the control net *before* that fiber is
  overwritten.  The reference chord is **only** the line used for **orthogonal
  projection**; it is **not** the curved polyline itself.

* **u** — A scalar in ``[0, 1]``: the **affine coordinate along the reference
  chord** of the orthogonal projection of a control point ``p`` onto the
  infinite line through ``i`` and ``e``.  By construction ``u = 0`` at ``i`` and
  ``u = 1`` at ``e``; values are clipped to ``[0, 1]`` if the projection falls
  outside the segment.

* **Straight fiber** — *After* the step, every control point on that fiber lies
  on the segment **``i_new``–``e``** (inner end moved radially to ``i_new``,
  outer end ``e`` fixed).

* **Outer patches** (the four that touch the circular boundary): discrete
  **boundary-layer expansion** (see :func:`_apply_discrete_bl_expansion`) uses the
  reference chord to obtain ``u`` for each ``p``, radially expands ``i`` to
  ``i_new``, applies the same one-dimensional expansion law to ``u`` to get
  ``u_new``, and sets ``p_new = i_new + u_new·(e − i_new)`` so the fiber becomes
  straight.  The harmonic projection places outer control points slightly
  outside the unit circle (cubic B-spline overshoot along the outer edge); that
  overshoot is larger at the middle of each side than at the diagonal corners.
  Treating the outer target as ``r = 1`` while ``|e| > 1`` would thicken the
  last span at the cardinals; keeping ``e`` fixed (``u = 1`` unchanged) keeps
  ``r_e = |e|`` and the circle approximation along the outer edge.  :math:`C^0`
  continuity on diagonal interfaces holds because both patches share the same
  ``i``, ``e``, and the same expansion.

* **Center patch**: after the outer patches are moved, the shared interface
  edges are copied from the outer neighbours, and the center interior is rebuilt
  with a Coons bilinear blend of the four boundary polylines so that
  :math:`C^0` continuity is preserved across all patch interfaces.

Relocating the control points causes the B-spline evaluation ``∑ N_ij · c_ij``
to produce sample points (mesh nodes) that cluster near the outer wall — the
boundary-layer effect in physical space.

Use :func:`plot_disc_control_net_2d` to draw the tensor-product control net in
the plane and inspect the boundary-layer expansion.
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

# Guard against division by zero at the disc centre in the radial expansion map.
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


def _scalar_expansion(r, exponent: Float):
  """One-dimensional **scalar expansion** on ``[0, 1]``, **identity for ``r > 1``**.

  * ``r ∈ [0, 1]``:   ``r ↦ 1 − (1 − r)^exponent``.
  * ``r > 1``:        **``r ↦ r``** (no clamp to ``1``).

  The behaviour that mattered in practice is **keeping ``r > 1`` unchanged**:
  cubic B-spline outer-ring control points often sit slightly **outside** the
  unit circle (``|p| > 1`` or overshoot in ``u``).  An older ``min(r, 1)``
  formulation folded those onto ``1``; combined with placement along non-radial
  chords that **changed angles** and produced **cusps** at diagonal patch
  interfaces.  The identity branch preserves overshoot — those scalars are left
  exactly where they are — which is what fixes that issue.

  Used for inner-end radial expansion (on ``|i|`` when ``i`` is the inner
  endpoint) and for expanding the chord parameter ``u`` along the straight fiber.

  * ``exponent == 1``: identity everywhere.
  * ``exponent > 1``: on ``[0, 1]``, pushes values toward ``1`` (stronger
    boundary-layer concentration near the outer end); above ``1``, still
    identity.
  """
  rf = float(r)
  if rf >= 1.:
    return rf
  return 1. - (1. - rf)**float(exponent)


def _fiber_endpoints(ref, side: str, fiber_idx: int):
  """Return ``(i, e)`` — inner and outer endpoints of one tensor fiber.

  Together ``i`` and ``e`` span the **reference chord** (see module docstring)
  used by :func:`_apply_discrete_bl_expansion`.  ``ref`` is an ``(nxi, neta, 2)``
  snapshot; ``side`` selects which patch boundary is the outer one.
  """
  if side == 'top':
    return ref[fiber_idx, 0, :], ref[fiber_idx, -1, :]
  if side == 'bottom':
    return ref[fiber_idx, -1, :], ref[fiber_idx, 0, :]
  if side == 'right':
    return ref[0, fiber_idx, :], ref[-1, fiber_idx, :]
  # left
  return ref[-1, fiber_idx, :], ref[0, fiber_idx, :]


def _fiber_iter(nxi: int, neta: int, side: str, fiber_idx: int):
  """Yield ``(ii, jj)`` tensor indices of every control point on fiber ``fiber_idx``."""
  if side in ('top', 'bottom'):
    for jj in range(neta):
      yield fiber_idx, jj
  else:  # left / right
    for ii in range(nxi):
      yield ii, fiber_idx


def _apply_discrete_bl_expansion(controlpoints, side: str,
                               exponent: Float) -> None:
  """Expand boundary-layer control spacing on one outer patch (in place).

  For each tensor **fiber** (fixed index along ``side``), inner and outer ring
  endpoints ``i`` and ``e`` are read from a frozen copy ``ref`` of the control
  net.  Every control point on that fiber is then rewritten so the fiber becomes
  a straight segment from an expanded inner end ``i_new`` to the **unchanged**
  outer end ``e``.

  Terminology
  -----------
  **Curved fiber**
      Polyline of control points from ``i`` to ``e`` *before* this call.  Its
      vertices need not be collinear with ``i``–``e``.

  **Reference chord**
      The straight line through ``i`` and ``e`` (vector ``e − i`` from ``ref``).
      Only this line is used to define **u**; it is not the curved polyline.

  **u**
      Affine coordinate in ``[0, 1]`` along the reference chord: orthogonal
      projection of each ``p`` onto the line through ``i`` and ``e`` (``0`` at
      ``i``, ``1`` at ``e``), clipped to the segment.

  **Straight fiber**
      After the call, all control points on the fiber lie on ``i_new``–``e``.

  Algorithm (per fiber)
  ----------------------
  1. ``u = clip(((p − i) · (e − i)) / |e − i|², 0, 1)`` for each ``p`` (uses ``i``,
     ``e``, ``d = e − i`` from ``ref`` only).
  2. ``i_new = i · (_scalar_expansion(|i|, exponent) / |i|)`` if ``|i| > 0``; else
     ``i_new = i``.
  3. ``u_new = _scalar_expansion(u, exponent)`` then
     ``p_new = i_new + u_new · (e − i_new)``.

  The map ``u ↦ u_new`` fixes ``0`` and ``1``, so **``e`` is never moved** and
  ``r_e = |e|`` stays as from the harmonic map (including cubic overshoot
  ``> 1`` at cardinals).  Expansion along ``u`` pulls interior points toward
  ``e`` without retargeting to ``r = 1``, which keeps the last control-net span
  thin everywhere.  :func:`_scalar_expansion` leaves scalars ``> 1`` unchanged,
  which is consistent with preserving overshoot.  Inner ``i`` moves outward so
  the center patch grows.  Neighbouring outer patches share the same ``(i, e)``
  on diagonal interfaces, so :math:`C^0` continuity is preserved.

  ``exponent`` is the same as ``boundary_layer_exponent`` in :func:`ellipse` /
  :func:`_disc`.

  Parameters
  ----------
  controlpoints:
      ``(nxi, neta, 2)`` array, modified **in place**.
  side:
      Which patch side is the **outer** boundary: ``'top'``, ``'bottom'``,
      ``'left'``, or ``'right'`` (selects the fiber direction in the tensor grid).
  exponent:
      ``1`` = identity on each fiber; larger values move interior control points
      closer to ``e`` along the straight fiber.
  """
  # Frozen copy: inner i, outer e, and chord direction d must stay fixed while
  # we overwrite controlpoints[ii, jj] for this fiber.
  ref = np.array(controlpoints, dtype=float, copy=True)
  nxi, neta = int(ref.shape[0]), int(ref.shape[1])

  # One fiber per row (top/bottom) or per column (left/right).
  n_fibers = nxi if side in ('top', 'bottom') else neta
  exp_in = float(exponent)

  for fib in range(n_fibers):
    # Inner endpoint i and outer endpoint e (2-vectors from ref only).
    i, e = _fiber_endpoints(ref, side, fib)
    r_i = float(np.hypot(i[0], i[1]))

    # Expand inner point i radially from the origin → i_new (grows center patch).
    if r_i > _RHO_SAFE:
      r_i_new = _scalar_expansion(r_i, exp_in)
      i_new = np.asarray(i, dtype=float) * (r_i_new / r_i)
    else:
      i_new = np.asarray(i, dtype=float)

    # Reference chord i → e (defines u); outer e is not moved.
    d = np.asarray(e, dtype=float) - np.asarray(i, dtype=float)
    d2 = float(d[0] * d[0] + d[1] * d[1])
    if d2 < _RHO_SAFE * _RHO_SAFE:
      continue

    # Straight fiber: fixed e, expanded inner i_new.
    d_new = np.asarray(e, dtype=float) - i_new
    for ii, jj in _fiber_iter(nxi, neta, side, fib):
      p = ref[ii, jj, :]
      # u = affine coordinate of p on the reference chord (clip to segment).
      u = float((p[0] - i[0]) * d[0] + (p[1] - i[1]) * d[1]) / d2
      u = min(1., max(0., u))
      u_new = _scalar_expansion(u, exp_in)
      controlpoints[ii, jj, :] = i_new + u_new * d_new


def _discrete_outer_bl_expansion(controlpoints_segments, tkvs, patches,
                                          exponent: Float):
  """Apply discrete boundary-layer **expansion** to the four outer patches.

  Iterates over all patches.  For patches with **exactly one** exterior side
  (the four outer patches), :func:`_apply_discrete_bl_expansion` is called in
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
      ``boundary_layer_exponent`` forwarded to :func:`_apply_discrete_bl_expansion`.

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
      _apply_discrete_bl_expansion(cps, outer_sides[0], exponent)

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

  The outer patches have already undergone boundary-layer **expansion** via
  :func:`_apply_discrete_bl_expansion` (projection on each reference chord, radial
  expansion of ``i``, then relocation on ``i_new``–``e``).  The center patch
  shares four interface edges with those outer patches; simply leaving its
  original harmonic control points would break :math:`C^0` continuity.  This
  function:

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
      List of per-patch control-point arrays after the outer expansion step.
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
  4. If ``boundary_layer_exponent != 1``, apply **discrete** boundary-layer
     expansion on the four outer patches (:func:`_discrete_outer_bl_expansion`)
     and rebuild the center patch with Coons blending
     (:func:`_propagate_center_patch_bl`).  A fallback restores the harmonic values
     for any non-finite DOFs.
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

  disc_bl    = boundary_layer_exponent != 1.
  controlmap = make_unit_disc(domain, basis, geom, localgeom, topo.topo,
                              reparam=reparam)

  # Project onto the discontinuous basis to get per-patch control points.
  controlpoints = domain.project(controlmap, onto=basisd.vector(2),
                                             geometry=geom,
                                             degree=2 * degree).reshape(-1, 2)

  if disc_bl:
    harmonic_cp = np.array(controlpoints, copy=True)
    segs = list(np.array_split(controlpoints, segments, axis=0))
    segs = _discrete_outer_bl_expansion(
        segs, tkvs, PATCHES, boundary_layer_exponent)
    segs = _propagate_center_patch_bl(segs, tkvs, PATCHES)
    controlpoints = np.concatenate(segs, axis=0)
    # Safety net: revert any DOFs that became NaN/Inf during expansion.
    bad = ~np.isfinite(controlpoints).all(axis=1)
    if bad.any():
      log.warning('Boundary-layer expansion produced non-finite control points; '
                  'reverting those DOFs to the harmonic projection.')
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
    Boundary-layer **expansion** strength on the unit disc before scaling by ``a``
    and ``b``.  ``1`` is uniform; larger values concentrate outer-patch control
    points toward the circular boundary (discrete expansion on the projected
    control net; see module docstring).
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
                   **kwargs):

  """
  Create a multipatch covering of the unit disc from the topology.

  Returns the harmonic (Dirichlet-energy-minimising) map in unit-disc
  coordinates.  Optional boundary-layer expansion is **not** done here: it is
  applied on the projected control net in :func:`_disc` via
  :func:`_apply_discrete_bl_expansion`.
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

  # Convert from unit-square patch coordinates (domain of the harmonic map,
  # centred at (0.5, 0.5)) back to unit-disc coordinates centred at the origin.
  def to_unit_disc(ctrl):
    return (ctrl - center) * np.sqrt(2)

  if not reparam:
    return to_unit_disc(controlmap)

  return to_unit_disc(multipatch_trace_penalty_stab(domain,
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
