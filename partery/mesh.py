import warnings

from pyhaemo.util import np, _
from pyhaemo.spl import NDSplineArray, NDSpline
from pyhaemo.mesh import mesh_union

from bif import bif_from_matrices
from sgraph import SplineGraph
from edge import SplineEdge
from artery import VascularGraph
from fit import fit_constrained
from preprocessing import rtruncate_per_branch
from network import NFurcation
from geo import estimate_normal


# Multiplier ladders used to build the per-bifurcation repair cascade.  They
# multiply the user-provided ``rfac`` (default 2) per branch of the failing
# furcation, matched positionally to ``furc.indices``.  Rationale:
#   * Uniform boosts first — the most common failure mode is "pull all
#     branches back a bit further".
#   * Single-branch boosts next — when one branch is thinner/closer than
#     the others, the self-intersection usually sits on that branch only.
#   * Two-branch boosts last — catch the common asymmetric-trifurcation
#     case where one daughter is much larger than the other two.
# The cascade is built dynamically by ``_build_retry_cascade(k)`` so it
# scales to any branch count k ≥ 3 (trifurcations, quadfurcations, …) and
# every tuple has length k, so indexing ``mults[i]`` for each branch is
# guaranteed safe.  Unlike the previous break-on-first-valid loop, the
# cascade is now fully scanned per failing bif and the best valid attempt
# (scored by :func:`_mesh_quality_score`) is committed.
_RFAC_UNIFORM_STEPS: tuple[float, ...] = (1.25, 1.5)
_RFAC_SINGLE_BRANCH_STEPS: tuple[float, ...] = (1.5, 2.0)


def _build_retry_cascade(k: int) -> tuple[tuple[float, ...], ...]:
  """Build a per-branch rfac retry ladder for a k-furcation.

  The cascade is deliberately small because the scoring in
  :func:`_combined_retry_score` prefers smaller multipliers and the outer
  retry loop short-circuits as soon as a gentler attempt is valid — big
  boosts only get built when everything smaller has failed anyway.  Size
  scales as ``len(uniform) + k*len(single)``:

    k=3 : 2 + 3*2 =  8 attempts
    k=4 : 2 + 4*2 = 10 attempts
    k=5 : 2 + 5*2 = 12 attempts

  Pair-branch boosts (previously 2×C(k,2) entries) are dropped entirely:
  they stretch two vessels at once and the user explicitly prefers
  asymmetric single-branch rescues in that stretching regime, so their
  scoring contribution is dominated by the single-branch variants.

  Entries are emitted in ascending (max(mults), sum(mults)) order so the
  retry loop can break the moment ``max(mults)`` exceeds the current best
  valid attempt — i.e. at-most one "tie group" of attempts gets fully
  evaluated past the first success.
  """
  assert k >= 3, f'Nontrivial furcation must have at least 3 branches (got {k})'
  cascade: list[tuple[float, ...]] = []
  # Gentle uniform 1.25× first: smallest possible max over the whole
  # cascade, so a valid attempt here triggers an immediate early exit.
  cascade.append((1.25,) * k)
  # All max=1.5 variants as one tie group — single-branch (sum = k + 0.5)
  # before uniform (sum = 1.5 * k) because the asymmetric choice spreads
  # less stretching and therefore ranks strictly higher on the
  # ``-sum(mults)`` tiebreak.
  for i in range(k):
    tup = [1.0] * k
    tup[i] = 1.5
    cascade.append(tuple(tup))
  cascade.append((1.5,) * k)
  # Final escalation: single-branch 2.0× rescues (max=2.0).  No uniform
  # 2.0× or pair boosts — those produce visibly stretched meshes and the
  # scoring would penalise them anyway, so evaluating them is pure waste.
  for i in range(k):
    tup = [1.0] * k
    tup[i] = 2.0
    cascade.append(tuple(tup))
  return tuple(cascade)


def _build_retry_cascade_fallback(k: int) -> tuple[tuple[float, ...], ...]:
  """Stronger per-branch multipliers after :func:`_build_retry_cascade` finds
  no valid bif mesh.

  Only used when the furcation still fails strict quality after the standard
  ladder (and not for merged-only quality scans where baseline was valid).
  Adds uniform 2.0× (omitted from the standard ladder), then asymmetric /
  uniform 2.5× and 3.0× escalations.  Entries are ordered by increasing
  ``(max(mults), sum(mults))``; the repair loop scans the whole fallback
  without ``max(mults)`` early-exit so the best valid attempt wins on
  :func:`_combined_retry_score` among these coarser truncations.
  """
  assert k >= 3, f'Nontrivial furcation must have at least 3 branches (got {k})'
  c: list[tuple[float, ...]] = []
  c.append((2.0,) * k)
  for i in range(k):
    tup = [1.0] * k
    tup[i] = 2.5
    c.append(tuple(tup))
  c.append((2.5,) * k)
  for i in range(k):
    tup = [1.0] * k
    tup[i] = 3.0
    c.append(tuple(tup))
  c.append((3.0,) * k)
  return tuple(c)


# Minimum acceptable scaled Jacobian.  An element's scaled Jacobian is
# ``det(J) / ∏ ||J[:, i]||`` (range [-1, 1] for volumetric meshes, [0, 1]
# for surface meshes); values close to 1 are well-shaped, values near 0
# are near-degenerate, and negative values (volumetric only) indicate
# inversion.  ``is_valid()`` only checks ``det(J) > 0`` at first-order
# nodal points, which passes "butterfly" bifurcation meshes whose wings
# cross but whose element corners happen to stay positively oriented.
# Sampling the scaled Jacobian at a denser quadrature (``_QUALITY_ORDER``)
# catches those: at the interior quadrature points of a crossed element
# the mapping degenerates, so ``min_sJ`` drops below threshold even when
# ``is_valid()`` returns True.
_SCALED_JAC_THRESH: float = 0.02
# Quadrature order used for the stricter sampling.  Order 3 gives ~27
# interior points per 3D element / ~9 per 2D element — dense enough to
# catch interior inversion without blowing up the cost of the retry
# cascade.
_QUALITY_ORDER: int = 3


def _scaled_jacobian(JK: "np.ndarray") -> "np.ndarray":
  """Compute per-element, per-point scaled Jacobian from a stack of
  Jacobian matrices.

  ``JK`` is the tensor returned by :meth:`Mesh.JK`, with shape
  ``(nelems, npoints, 3, ndims)`` (``ndims=3`` for volumetric meshes,
  ``ndims=2`` for boundary surfaces embedded in 3D).  Returns an array
  of shape ``(nelems, npoints)`` whose values lie in ``[-1, 1]`` (3D)
  or ``[0, 1]`` (2D, cross-product magnitude is always non-negative).
  """
  col_norms = np.linalg.norm(JK, axis=-2)  # (nelems, npoints, ndims)
  denom = np.prod(col_norms, axis=-1)
  # Guard against degenerate elements with a zero-length edge — they
  # won't produce a meaningful scaled-J anyway, so flag them as -inf
  # so they dominate ``min``.
  safe = denom > 1e-14
  if JK.shape[-1] == 3:
    numer = np.linalg.det(JK)
  else:
    # Surface mesh: cross-product magnitude is the "area element".
    numer = np.linalg.norm(np.cross(JK[..., 0], JK[..., 1]), axis=-1)
  sJ = np.where(safe, numer / np.where(safe, denom, 1.0), -np.inf)
  return sJ


def _mesh_quality_score(fmesh) -> tuple:
  """Composite quality score for a bifurcation mesh.

  Returns a sortable tuple
  ``(valid_flag, min_scaled_jac, min_det_jac, mean_scaled_jac,
  mean_det_jac)`` — larger is better under Python's natural tuple
  comparison.  Scoring logic:

  * ``valid_flag`` is ``1`` iff the mesh passes both ``is_valid()``
    (``det(J) > 0`` at order-1 nodal points, the weak/nutils check) *and*
    the stricter scaled-Jacobian gate ``min_scaled_jac >= _SCALED_JAC_THRESH``
    sampled at order ``_QUALITY_ORDER`` quadrature.  The stricter gate
    catches the "butterfly" failure mode where individual element
    corners are positively oriented (so ``is_valid()`` returns True) but
    the mapping degenerates in the interior of each crossed element —
    the exact pathology we see on some merged n-furcations when the
    merged normal is slightly off from the constituents' convention.
  * Primary ordering criterion among valid meshes is ``min_scaled_jac``:
    a merge whose worst element has ``sJ = 0.4`` is far more robust than
    one with ``sJ = 0.05`` (both clear the threshold but the second is
    one numerical perturbation away from inversion).
  * Secondary / tertiary are ``min_det_jac`` and the means, as
    tiebreakers for near-identical attempts.

  Invalid meshes score ``(0, -inf, -inf, -inf, -inf)`` so any valid
  attempt beats any invalid one regardless of magnitude.  Exceptions
  during scoring (e.g. a JK evaluation that blows up on a degenerate
  element layout) are treated as invalid so the scan can continue.
  """
  sentinel = (0, float('-inf'), float('-inf'),
              float('-inf'), float('-inf'))
  try:
    weak_valid = bool(fmesh.is_valid())
  except Exception:
    return sentinel
  try:
    # Stricter sampling: order-3 quadrature catches interior inversion
    # that ``is_valid()`` (order-1 nodes) misses on butterfly meshes.
    pts_hi = fmesh._local_ordinances(_QUALITY_ORDER)
    JK_hi = fmesh.JK(pts_hi)
    sJ = _scaled_jacobian(JK_hi)
    # ``det(J)`` is only defined for square Jacobians (volumetric).
    # For boundary surfaces (3x2 JK) we fall back to reusing the
    # scaled Jacobian numerator as a robust non-square stand-in.
    if JK_hi.shape[-1] == 3:
      dets = np.linalg.det(JK_hi)
    else:
      dets = np.linalg.norm(np.cross(JK_hi[..., 0], JK_hi[..., 1]),
                             axis=-1)
    min_sJ = float(np.min(sJ))
    mean_sJ = float(np.mean(sJ))
    min_det = float(np.min(dets))
    mean_det = float(np.mean(dets))
  except Exception:
    return sentinel
  strict_valid = weak_valid and (min_sJ >= _SCALED_JAC_THRESH) \
                 and (min_det > 0)
  if not strict_valid:
    return (0, min_sJ, min_det, mean_sJ, mean_det)
  return (1, min_sJ, min_det, mean_sJ, mean_det)


def _combined_retry_score(quality_score: tuple, mults: tuple) -> tuple:
  """Combine raw mesh quality with a penalty for large rfac multipliers.

  Larger = better under Python tuple comparison.  Layout:

    (valid_flag, -max(mults), -sum(mults),
     min_scaled_jac, min_det_jac, mean_scaled_jac, mean_det_jac)

  So among *valid* attempts:

  1. The one with the smallest **worst-branch** multiplier wins first —
     a single branch at rfac×3 stretches the vessel mesh disproportionately
     and is visually noticeable, so we veto it before looking at quality.
  2. Ties on ``max`` are broken by the **total** multiplier sum, which
     penalises "every branch boosted uniformly" over "one branch boosted":
     spreading the pain across all branches makes the bifurcation mesh
     end further from the centroid in every direction, which is usually
     worse than asymmetrically retreating just the problematic branch.
  3. Only when *both* rfac-magnitude criteria tie do we fall back to
     raw quality — ``min_scaled_jac`` first (the most sensitive indicator
     of near-degeneracy), then ``min_det_jac``, then the means.  This
     keeps the scan from ever picking a borderline-valid mesh just
     because an earlier cascade entry happened to have one branch at
     rfac×1.5 and this one has it at rfac×1.25 — if the mesh is on the
     edge of inversion the user sees it as invalid anyway.

  ``mults`` is the raw multipliers tuple straight from the cascade; the
  baseline rfac is the same for every entry so there's no point scaling
  it in before the max/sum.
  """
  valid = quality_score[0]
  if valid != 1:
    # Invalid attempts never get promoted regardless of their mults —
    # keep every slot at its sentinel so the comparison is unambiguous.
    return (0, float('-inf'), float('-inf'),
            float('-inf'), float('-inf'),
            float('-inf'), float('-inf'))
  max_m = float(max(mults)) if mults else 1.0
  sum_m = float(sum(mults)) if mults else 1.0
  return (1, -max_m, -sum_m,
          quality_score[1], quality_score[2],
          quality_score[3], quality_score[4])


def _build_vessel_mesh(sedge, cross, dx, nref, nsec0, nsec1, union_eps):
  """Build the vessel geometry and boundary mesh for a single truncated
  spline edge.  Returns ``(vessel_nda, redge, vessel_mesh)``.

  Extracted as a helper so the per-bif rfac retry loop can rebuild a single
  vessel in isolation when its truncation changes.
  """
  kv = sedge.knotvector
  assert -1 not in kv[0].continuity[1:-1]
  cline, radius = sedge.rcenterline[:3], sedge.rcenterline[3]
  frame = sedge.frame
  evl = kv[0].refine(nref).knots
  rR = fit_constrained(kv, evl, radius(evl)[:, _, _] * frame(evl))
  redge = sedge._edit(frame=rR)
  vessel = (cross[:, _] @ rR[_]).sum(-1) + (cross.unity @ cline)
  npoints = max(2, int(sedge.length / dx))
  vessel_mesh = mesh_union(
    *vessel.quick_sample((nsec0, nsec1, npoints)).ravel(),
    boundary=True, eps=union_eps,
  )
  return vessel, redge, vessel_mesh


def _coorent_frame_tangent_inward(frame: np.ndarray,
                                  centroid: np.ndarray,
                                  centerpoint: np.ndarray,
                                  eps: float = 1e-9) -> np.ndarray:
  """Force the frame's tangent (column 2) to point *into* the junction.

  :func:`bif.bif_from_matrices` assumes consecutive branches in
  ``furc.indices`` order are paired with Hermite spines whose end tangents
  use ``mat[:, 2]`` as the inward vessel direction.  :func:`geo.sort_ccw`
  only permutes branches by endpoint *positions* in the bif plane — it
  never checks that every spline's parametric tangent at the bif end
  actually points toward ``centroid``.  One outward-pointing tangent
  flips the binormal (``B = T × N``) and swaps which disc quadrants the
  wing logic connects, which looks like "sections in the wrong order" on
  n-furcations (n > 3) even when the cyclic *geometry* is CCW.

  If ``tangent · normalize(centroid - centerpoint) < 0``, negate columns
  1 and 2 together so the frame stays right-handed and ``det`` sign is
  unchanged.
  """
  F = np.asarray(frame, dtype=float).reshape(3, 3).copy()
  toward = (np.asarray(centroid, dtype=float).ravel()
            - np.asarray(centerpoint, dtype=float).ravel())
  tn = float(np.linalg.norm(toward))
  if tn < eps:
    return F
  toward = toward / tn
  tcol = F[:, 2]
  tnorm = float(np.linalg.norm(tcol))
  if tnorm < eps:
    return F
  tcol = tcol / tnorm
  if float(tcol @ toward) < 0.0:
    F[:, 1] *= -1.0
    F[:, 2] *= -1.0
  return F


def _build_bif_mesh(furc, sgraph_r, cross, nsec0, nsec1, mat,
                    bif_exploit_symmetry, union_eps, dx: float):
  """Build the bifurcation geometry and boundary mesh for one furcation.

  Returns ``(furcation_nda, fmesh, score)`` where ``score`` is the composite
  quality score from :func:`_mesh_quality_score` — a sortable tuple where
  larger-is-better, and ``score[0] == 1`` iff the mesh is valid.  The retry
  loop uses ``score`` both for the validity gate (scan-all still needs to
  know which attempts cleared ``is_valid()``) and for picking the highest
  quality attempt among the valid ones.

  Merged n-furcations (user opt-in from the GUI) are collapsed upstream on
  the :class:`VascularGraph` — see :func:`apply_merge_opt_in_vgraph` — so
  by the time we get here :func:`preprocessing.align_frames` has already
  aligned every edge's frame to the merged n-furcation's normal using the
  exact same no-twist algorithm applied to naturally-occurring furcations.
  The endpoint frame read here is therefore indistinguishable (up to the
  polyline-vs-spline tangent fit residual) from a natural case: there is
  no separate code path for merged bifs downstream of ``to_spline_graph``.

  Immediately before :func:`bif.bif_from_matrices`, each endpoint frame is
  passed through :func:`_coorent_frame_tangent_inward` so column 2 points
  toward ``centroid`` (into the junction).  CCW branch order from
  :meth:`NFurcation.orient` does not imply this, and one flipped tangent
  permutes which disc quadrants the wing patches connect.

  The third ``quick_sample`` axis (along-wing / junction extent) uses the
  same ``dx`` rule as :func:`_build_vessel_mesh`: ``max(5, int(L / dx))``
  with ``L`` the largest distance from ``centroid`` to a branch endpoint,
  capped so pathological graphs do not allocate huge meshes.
  """
  normal = furc.normal
  centroid = furc.centroid
  myedges = sgraph_r.get_edges(furc.indices)  # head directed into furcation
  centerpoints = [edge.cline([edge.interval[1]]).ravel() for edge in myedges]
  radii = [edge.radius([edge.interval[1]])[0] for edge in myedges]
  bB = bT = float(np.mean(radii))
  frames = [
    _coorent_frame_tangent_inward(
      (edge.frame([edge.interval[1]]).reshape(3, 3) @ mat),
      centroid,
      cp,
    )
    for edge, cp in zip(myedges, centerpoints)
  ]
  furcation = bif_from_matrices(
    frames, centerpoints, normal, centroid, bB, bT, cross,
    exploit_symmetry=bif_exploit_symmetry,
  )
  c0 = np.asarray(centroid, dtype=float).ravel()
  L_junc = 0.0
  for cp in centerpoints:
    L_junc = max(
      L_junc,
      float(np.linalg.norm(np.asarray(cp, dtype=float).ravel() - c0)),
    )
  if L_junc < 1e-12:
    L_junc = float(bB)
  dxp = float(dx)
  if dxp <= 0:
    dxp = 1e-3
  # Match vessel ``npoints = max(2, int(length / dx))`` spirit; floor 5 for
  # the bif patch stack (was a hard-coded 7).
  n_bif_axial = max(5, int(L_junc / dxp))
  n_bif_axial = min(n_bif_axial, 512)
  fmesh = mesh_union(
    *furcation.quick_sample((nsec0, nsec1, n_bif_axial)).ravel(),
    boundary=True, eps=union_eps,
  )
  score = _mesh_quality_score(fmesh)
  return furcation, fmesh, score


def _detect_bif_overlaps(sgraph, truncate_dict, xidict, min_remaining=0.1):
  """Detect shared edges where the two adjacent bifurcations' truncations
  leave almost nothing of the vessel — i.e. the two bifs are so close that
  ``rtruncate`` either has to clamp them to a degenerate mid-point window
  (``[npoints_2 - 1, npoints_2 + 1]``) or leaves ``< min_remaining`` of the
  original xi span between them.

  Returns a list of dicts, one per offending edge:

    ``{"iedge", "bif_start", "bif_end", "remaining_fraction",
       "xi_start", "xi_end", "span_xi"}``

  ``bif_start`` / ``bif_end`` are the ``frozenset`` keys used by the GUI's
  bifurcation list (unsigned index sets); ``bif_start`` corresponds to the
  bifurcation at the edge's start (negative-signed branch), ``bif_end`` to
  the one at the end (positive-signed branch).  Either may be ``None`` if
  the edge is terminal (only one bif attached).

  Convention matches :func:`preprocessing.rtruncate`:
    * ``truncate_dict[a][0]`` is set from the bif holding ``-a`` in its
      indices (edge's "start" side).
    * ``truncate_dict[a][1]`` from the bif holding ``+a`` (edge's "end").
  """
  edge_bifs: dict[int, dict[str, frozenset]] = {}
  for furc in sgraph.nontrivial_furcations:
    key = frozenset(abs(int(j)) for j in furc.indices)
    for i in furc.indices:
      entry = edge_bifs.setdefault(abs(int(i)), {})
      entry["start" if int(i) < 0 else "end"] = key

  overlaps: list[dict] = []
  for iedge, slots in truncate_dict.items():
    start_xi, end_xi = slots[0], slots[1]
    if start_xi is None or end_xi is None:
      continue
    # Use the pre-refinement knot array for the xi span reference: this is
    # the edge's full parametric domain that any truncation eats into.
    myxi = xidict[iedge]
    xi_min, xi_max = float(myxi[0]), float(myxi[-1])
    full_span = xi_max - xi_min
    if full_span <= 0:
      continue
    remaining = float(end_xi) - float(start_xi)
    fraction = remaining / full_span
    if fraction < min_remaining:
      bifs = edge_bifs.get(iedge, {})
      overlaps.append({
        "iedge": int(iedge),
        "bif_start": bifs.get("start"),
        "bif_end": bifs.get("end"),
        "remaining_fraction": float(fraction),
        "xi_start": float(start_xi),
        "xi_end": float(end_xi),
        "span_xi": float(full_span),
      })
  return overlaps


def _compute_merge_components(overlap_info):
  """Group overlap candidates into connected components.

  An overlap is a 1-edge in the "too-close" graph whose nodes are the
  two adjacent bifurcations (``bif_start`` / ``bif_end`` from
  :func:`_detect_bif_overlaps`); a connected component in that graph is
  exactly the set of bifurcations the user might want to collapse into a
  single n-furcation.

  Returns a list of dicts, one per component with ≥ 2 bifurcations::

    {
      "bif_keys"      : frozenset[frozenset[int]],  # constituent bifs' keys
      "internal_edges": frozenset[int],             # shared (absorbed) edges
      "component_id" : str,                         # stable id for GUI state
    }

  The ``component_id`` is deterministic (sorted members), so the GUI can
  persist the user's opt-in selection across regenerations without
  caring about iteration order.
  """
  parent: dict = {}
  rank: dict = {}

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry:
      return
    # Union by rank keeps the find chain short; not critical here (graphs
    # are tiny) but cheap and avoids any pathological case.
    if rank.get(rx, 0) < rank.get(ry, 0):
      rx, ry = ry, rx
    parent[ry] = rx
    if rank.get(rx, 0) == rank.get(ry, 0):
      rank[rx] = rank.get(rx, 0) + 1

  edges_by_pair: dict = {}
  for ov in overlap_info:
    bs, be = ov.get("bif_start"), ov.get("bif_end")
    if bs is None or be is None:
      # Terminal-edge overlap: only one bif attached, so it can't be part
      # of a merge component.  Skip.
      continue
    parent.setdefault(bs, bs)
    parent.setdefault(be, be)
    union(bs, be)
    pair = frozenset((bs, be))
    edges_by_pair.setdefault(pair, set()).add(int(ov["iedge"]))

  by_root: dict = {}
  for node in parent:
    by_root.setdefault(find(node), set()).add(node)

  components: list[dict] = []
  for nodes in by_root.values():
    if len(nodes) < 2:
      continue
    internal: set[int] = set()
    for pair, eids in edges_by_pair.items():
      if pair.issubset(nodes):
        internal |= eids
    frozen_keys = frozenset(nodes)
    # Build a deterministic id: sorted tuple of sorted-tuple constituent
    # keys, joined.  Tuple-of-tuples so the id is hashable and stable
    # regardless of the internal ordering Python uses for frozenset
    # iteration.
    sig = tuple(sorted(
      tuple(sorted(k)) for k in frozen_keys
    ))
    component_id = "|".join(
      ",".join(str(i) for i in member) for member in sig
    )
    components.append({
      "bif_keys": frozen_keys,
      "internal_edges": frozenset(internal),
      "component_id": component_id,
    })
  return components


def _head_position(edge):
  """Return the head endpoint of an edge as a ``(3,)`` ``np.ndarray``.

  Graph-agnostic wrapper so :func:`_merge_bif_component` can build the
  CCW-ordering ``endpoint_dict`` for either a :class:`VascularGraph`
  (:class:`edge.VascularEdge`, ``.points[-1][:3]``) or a
  :class:`SplineGraph` (:class:`edge.SplineEdge`,
  ``.cline([interval[1]])``) without caring which one it got.
  """
  if hasattr(edge, "cline"):
    return np.asarray(edge.cline([edge.interval[1]])).ravel()
  return np.asarray(edge.points[-1][:3])


def _merge_bif_component(graph, bif_keys):
  """Collapse a connected component of bifurcations into one n-furcation.

  ``bif_keys`` is an iterable of ``frozenset(abs(int(i)) for i in
  furc.indices)`` keys (same shape as the keys used by the bifurcation
  list in the GUI).  The matching :class:`NFurcation` objects in
  ``graph.furcations`` are identified by that unsigned-indices key, and
  the internal edges — those that appear in two or more of them — are
  dropped from the graph.  The merged n-furcation inherits:

  * ``indices``  = concatenation of every constituent bif's signed
                   indices *excluding* the internal ones, re-ordered
                   CCW around the merged centroid via
                   :meth:`NFurcation.orient` so
                   :func:`bif_from_matrices`'s ``np.roll`` wing-stitching
                   walks around adjacent branches.
  * ``centroid`` = branch-count-weighted mean of the constituents'
                   centroids.  Weighting by ``N`` (branches) rather than
                   by radius puts the new centroid closer to the
                   "bigger" junction.
  * ``normal``   = PCA plane normal of the external branch endpoints
                   (same construction as :meth:`NFurcation.set_normal`),
                   evaluated at ``merged_centroid`` so the plane matches
                   :meth:`NFurcation.orient`'s sort.  **Do not** average
                   constituent ``furc.normal`` here: this runs *before*
                   :func:`preprocessing.align_nontrivial_normals`, so those
                   normals can still lie in opposite half-spaces; their
                   vector sum is a bad axis for :func:`geo.sort_ccw` and
                   permutes branches into an order where
                   :func:`bif.bif_from_matrices` connects the wrong disc
                   corners.

  This function is graph-agnostic (works on any :class:`EdgeGraph`
  subclass) and **topology-only** — it does not touch edge frames.
  The intended entry point is :func:`apply_merge_opt_in_vgraph`, which
  applies the merge on the :class:`VascularGraph` just before the
  :func:`preprocessing.align_nontrivial_normals` /
  :func:`preprocessing.align_frames` passes, so the merged n-furcation
  goes through the *identical* frame-alignment code path as any
  naturally occurring n-furcation.  There is then no separate frame
  rebuild on the :class:`SplineGraph` side — the previous approach
  (end-to-end ``compute_notwistframe_from_spline`` refit with projected
  boundary conditions) is gone, because it slightly differed from
  :func:`align_frames`'s polyline-tangent refit and that tiny
  discrepancy was enough to swap wing connectivity in
  :func:`bif.bif_from_matrices` on some merged components.
  """
  target_keys = set(bif_keys)
  component_furcs = [
    f for f in graph.furcations
    if frozenset(abs(int(i)) for i in f.indices) in target_keys
    and len(f) > 2
  ]
  if len(component_furcs) < 2:
    return graph

  # Internal edges: abs-index appears in ≥ 2 constituent bifs.
  counts: dict = {}
  for furc in component_furcs:
    for i in furc.indices:
      ai = abs(int(i))
      counts[ai] = counts.get(ai, 0) + 1
  internal_edges = {ai for ai, c in counts.items() if c >= 2}

  # External branches: every signed index whose abs is not internal.
  external: list[int] = []
  seen_abs: set[int] = set()
  for furc in component_furcs:
    for i in furc.indices:
      ai = abs(int(i))
      if ai in internal_edges or ai in seen_abs:
        continue
      external.append(int(i))
      seen_abs.add(ai)

  weights = np.array([float(len(f)) for f in component_furcs], dtype=float)
  total_w = float(weights.sum())
  centroids = np.stack(
    [np.asarray(f.centroid, dtype=float) for f in component_furcs]
  )
  merged_centroid = (centroids * weights[:, None]).sum(axis=0) / total_w

  # Endpoint positions for CCW reordering must match
  # :meth:`VascularGraph.orient_nontrivial`: that pass permutes *natural*
  # n-furcations using ``self.rtruncate(rfac).endpoints`` (GUI default
  # ``rfac=1``), then returns ``self._edit(furcations=…)`` with **full**
  # polylines still in ``self.edges``.  If we sort a merged n-furcation
  # from the raw ``get_edges`` heads instead, the cyclic order can differ
  # from every other n-furcation on the same graph →
  # :func:`bif.bif_from_matrices` connects the wrong quarter-disc corners
  # and strict validity (scaled Jacobian) fails even when the junction
  # “looks” oriented.  Non-:class:`VascularGraph` callers keep the generic
  # ``_head_position`` path.
  if isinstance(graph, VascularGraph):
    trunc = graph.rtruncate(1)
    endpoint_dict = {
      int(idx): np.asarray(trunc.endpoints[idx], dtype=float).ravel()
      for idx in external
    }
  else:
    ext_edges = graph.get_edges(external)
    endpoint_dict = {
      idx: _head_position(edge)
      for idx, edge in zip(external, ext_edges)
    }

  # Plane normal for CCW ordering: geometry only (endpoint PCA at the
  # merged centroid), *not* a sum of constituent ``furc.normal`` — those
  # are not globally aligned until ``align_nontrivial_normals`` runs later.
  endpoint_stack = np.stack(
    [endpoint_dict[i] for i in external], dtype=float
  )
  merged_normal = estimate_normal(endpoint_stack, centroid=merged_centroid)

  # Drop constituents, add the merged n-furc, and insert trivial
  # placeholders for the internal edges so ``delete_edges`` can remove
  # them (same pattern as :meth:`EdgeGraph.merge_furcs`).
  keep = [f for f in graph.furcations if f not in component_furcs]
  merged = NFurcation(external, merged_centroid, merged_normal)
  merged = merged.orient(endpoint_dict)
  new_furcs = list(keep) + [merged]
  for i in internal_edges:
    new_furcs.append(NFurcation([-int(i)], merged_centroid))
    new_furcs.append(NFurcation([int(i)], merged_centroid))

  return graph._edit(furcations=new_furcs) \
              .delete_edges(list(internal_edges))


def apply_merge_opt_in_vgraph(vgraph, merge_opt_in):
  """Apply user opt-in merges on a :class:`VascularGraph`.

  ``merge_opt_in`` is the same spec shape the GUI hands to
  :func:`make_spline_mesh`: an iterable of components, where each
  component is itself an iterable of bif-keys (a bif-key is an iterable
  of unsigned edge ids).  Components with fewer than two constituent
  bifurcations are silently skipped.

  Runs **before** :func:`preprocessing.align_nontrivial_normals` /
  :func:`preprocessing.align_frames` so every merged n-furcation goes
  through the identical normal-alignment / frame-rebuild pass as a
  natural n-furcation.  That removes the last observable difference
  between merged and natural paths: :func:`bif.bif_from_matrices` sees
  the same endpoint frames either way, so its CCW wing-stitching is
  correct in both cases.

  Returns ``(merged_vgraph, merged_key_to_members)``.
  ``merged_key_to_members`` maps the merged n-furcation's own bif-key
  (frozenset of external abs-indices) back to the frozenset of
  constituent bif-keys it absorbed, so the GUI can propagate
  validity/repair info from the merged n-furcation back to the rows it
  still displays (the pre-merge bifurcations).
  """
  merged_key_to_members: dict[frozenset, frozenset] = {}
  for comp in merge_opt_in or ():
    keys = frozenset(frozenset(int(i) for i in k) for k in comp)
    if len(keys) < 2:
      continue
    before = {id(f) for f in vgraph.furcations}
    vgraph = _merge_bif_component(vgraph, keys)
    new_furc = next(
      (f for f in vgraph.furcations
       if id(f) not in before and len(f) > 2),
      None,
    )
    if new_furc is not None:
      merged_key = frozenset(abs(int(i)) for i in new_furc.indices)
      merged_key_to_members[merged_key] = keys
  return vgraph, merged_key_to_members


def _compute_truncate_dict(sgraph, xis, xidict, rfac_default, branch_overrides=None):
  """Compute the xi-space ``truncate_dict`` consumed by
  ``SplineGraph.truncate_edges``, optionally with per-branch rfac overrides.

  When ``branch_overrides`` is empty this matches the original behaviour of
  calling ``VascularGraph._rtruncate(rfac=rfac_default)``.  With overrides,
  only the furcations touching an overridden branch see asymmetric rfac; all
  other pairs still use ``rfac_default``, so bifurcations that were already
  valid keep their baseline truncation bit-for-bit.
  """
  vgraph = VascularGraph(
    sgraph.indices,
    [e.sample(_xi) for e, _xi in zip(sgraph.edges, xis)],
    sgraph.furcations,
  )
  if branch_overrides:
    raw = rtruncate_per_branch(
      vgraph, rfac_default=rfac_default, branch_overrides=branch_overrides,
    )
  else:
    raw = vgraph._rtruncate(rfac=rfac_default)

  truncate_dict = {}
  for iedge, indices in raw.items():
    myxi = xidict[iedge]
    truncate_dict[iedge] = [
      myxi[idx] if idx is not None else None for idx in indices
    ]
  return truncate_dict


def make_spline_mesh(sgraph: SplineGraph,
                     cross: NDSplineArray,
                     dx: float = 0.02,
                     nref: int = 3,
                     rfac: int = 2,
                     show_interfaces: bool = False,
                     **kwargs):
  """Create a mesh from a SplineGraph.

  Parameters
  ----------
  dx : float
    Target spacing along vessel centerlines (``_build_vessel_mesh``) and
    along the bifurcation patch ``quick_sample`` third axis, taken as
    ``max(5, int(L / dx))`` samples with ``L`` the max distance from the
    junction centroid to an inlet (capped at 512).  Cross-sectional
    resolution in the bif still follows ``cross`` (``nsec0``, ``nsec1``).

  union_eps : float, optional
    Tolerance for :func:`pyhaemo.mesh.bool.mesh_union` when gluing volumes.
  bif_exploit_symmetry : bool, optional
    Passed to :func:`bif.bif_from_matrices`.
  rfac_retry : bool, optional
    If True (default), failing bifurcations are retried with a per-branch
    rfac cascade before being reported as invalid.  If the standard ladder
    still finds no valid mesh, a second **fallback** ladder (uniform /
    asymmetric 2.0×–3.0× multipliers) runs for that bif only.  Set to False
    to skip repairs and return the raw validity flags from the first pass.

  Side-channel attributes on the returned mesh (in addition to
  ``bif_validity``) when GUI merges were applied:

  * ``merged_nfurc_check`` — tuple of dicts, one per merged n-furcation:
    ``merged_key``, ``member_keys``, ``valid`` (strict quality recheck),
    ``quality_score`` (tuple from :func:`_mesh_quality_score`),
    ``valid_pipeline`` (flag from the main bif pass), ``repair``.
  * ``all_merged_nfurc_valid`` — ``True`` if there were no merges or every
    merged n-furcation passed the strict recheck.

  If any merged n-furcation fails the recheck, a :class:`UserWarning` is
  emitted so batch runs surface the issue without opening the GUI.

  Merged n-furcations (keys in ``merged_key_to_members``) also participate
  in the per-branch ``rfac`` cascade when the baseline mesh is **already**
  valid: the scan prepends multiplier ``(1,\\dots,1)`` so the baseline
  competes with gentler boosts, and early-exit on ``max(mults)`` is
  disabled for that scan so a higher-quality valid mesh can win without
  stretching every branch.
  """
  union_eps = float(kwargs.pop("union_eps", 1e-8))
  bif_exploit_symmetry = bool(kwargs.pop("bif_exploit_symmetry", False))
  rfac_retry = bool(kwargs.pop("rfac_retry", True))
  # ``merged_key_to_members`` maps the merged n-furcation's bif-key to the
  # frozenset of constituent bif-keys it absorbed.  The caller is expected
  # to have already applied the merges at the :class:`VascularGraph` level
  # (see :func:`apply_merge_opt_in_vgraph` in this module) so the
  # ``sgraph`` we receive already contains the merged n-furcations; this
  # mapping is used only to back-fill validity/repair info onto the
  # pre-merge bif-keys the GUI still shows in its Bifurcations list.  An
  # empty mapping means "no merges" and the mesh build behaves identically
  # to a graph that never had overlap candidates.
  merged_key_to_members: dict = dict(
    kwargs.pop("merged_key_to_members", None) or {}
  )
  kwargs.pop("boundary_layer_exponent", None)

  xis = [edge.knotvector[0].refine(nref).knots for edge in sgraph.edges]
  xidict = dict(zip(sgraph.indices, xis))

  # ------------------------------------------------------------------ #
  # Baseline pass: uniform rfac for every pair, build every vessel and
  # every bifurcation, record per-bif validity.
  # ------------------------------------------------------------------ #
  truncate_dict = _compute_truncate_dict(sgraph, xis, xidict, rfac)
  # Detection-only: find edges where the two adjacent bifurcations' required
  # truncations overlap or leave a near-degenerate vessel stub between them.
  # These are candidates for "merge into a single n-furcation" — flagged in
  # the GUI's bif list so the user can decide topology manually.  We detect
  # against the *baseline* truncate_dict because that's the topology-level
  # diagnostic; the repair cascade only shifts rfac, it can't fix a true
  # geometric overlap.
  overlap_info = _detect_bif_overlaps(sgraph, truncate_dict, xidict)
  # Connected-component grouping of overlaps: each component is a set of
  # bifurcations the user *could* merge into one n-furcation from the GUI.
  merge_components = _compute_merge_components(overlap_info)
  _sgraph = sgraph.truncate_edges(truncate_dict)

  kv0 = cross.knotvector[0]
  kv1 = cross.knotvector[1]
  if not hasattr(kv0, "knotvalues") and len(kv0):
    kv0 = kv0[0]
  if not hasattr(kv1, "knotvalues") and len(kv1):
    kv1 = kv1[0]
  # Tie sampling density to the actual cross-section discretization
  # (e.g. repeated_knot_disc(nelems): points ~= nelems + 1).
  nsec0 = max(5, int(len(kv0.knotvalues)))
  nsec1 = max(5, int(len(kv1.knotvalues)))

  # Store vessels / meshes indexed by edge id so the retry loop can update
  # just the three edges touching a failing bifurcation without disturbing
  # the rest of the graph.
  vessel_by_id: dict[int, "NDSplineArray"] = {}
  redge_by_id = {}
  vmesh_by_id = {}
  for sedge, iedge in zip(_sgraph.edges, _sgraph.indices):
    vessel, redge, vmesh = _build_vessel_mesh(
      sedge, cross, dx, nref, nsec0, nsec1, union_eps,
    )
    vessel_by_id[iedge] = vessel
    redge_by_id[iedge] = redge
    vmesh_by_id[iedge] = vmesh

  sgraph_r = _sgraph._edit(edges=[redge_by_id[i] for i in _sgraph.indices])
  mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
  mat = np.eye(3)

  # Per-bifurcation validity: ``(key, valid, repair)`` entries keyed by
  # ``frozenset(abs(int(i)) for i in furc.indices)`` so the GUI can match a
  # failure back to the row in ``bif_list`` (which uses the same key).
  # ``repair`` is either ``None`` (no retry needed or all retries failed)
  # or a dict describing the per-branch rfac multipliers that succeeded.
  furc_by_key: dict[frozenset, "NDSplineArray"] = {}
  fmesh_by_key: dict[frozenset, object] = {}
  bif_validity: list[tuple[frozenset, bool, dict | None]] = []
  failing_furcs: list = []

  baseline_score_by_key: dict[frozenset, tuple] = {}
  for furc in _sgraph.nontrivial_furcations:
    key = frozenset(abs(int(i)) for i in furc.indices)
    furcation, fmesh, score = _build_bif_mesh(
      furc, sgraph_r, cross, nsec0, nsec1, mat,
      bif_exploit_symmetry, union_eps, dx,
    )
    furc_by_key[key] = furcation
    fmesh_by_key[key] = fmesh
    baseline_score_by_key[key] = score
    valid = score[0] == 1
    bif_validity.append((key, valid, None))
    if not valid:
      failing_furcs.append(furc)

  # Keys of user-merged n-furcations (post-``apply_merge_opt_in_vgraph``).
  merged_key_set: frozenset = frozenset(
    merged_key_to_members.keys(),
  ) if merged_key_to_members else frozenset()
  # Rfac cascade runs for every *invalid* bif plus every *merged* n-furcation
  # that was valid at baseline so we still scan multipliers for a better
  # strict-quality score (same machinery as failing bifs, with a longer
  # cascade when baseline was already valid).
  furcs_for_rfac: list = list(failing_furcs)
  if merged_key_set:
    keys_failing = {
      frozenset(abs(int(i)) for i in f.indices) for f in failing_furcs
    }
    for furc_m in _sgraph.nontrivial_furcations:
      km = frozenset(abs(int(i)) for i in furc_m.indices)
      if km in merged_key_set and km not in keys_failing:
        furcs_for_rfac.append(furc_m)

  # ------------------------------------------------------------------ #
  # Repair / optimization pass: per-branch rfac cascade.  Invalid bifs
  # must be rescued; merged n-furcations that are already valid still run
  # the scan to pick the best valid attempt (quality vs multiplier penalty).
  # ------------------------------------------------------------------ #
  if rfac_retry and furcs_for_rfac:
    # Index map from key -> position in bif_validity, so we can patch the
    # validity entry once a repair succeeds.
    key_to_pos = {k: p for p, (k, _v, _r) in enumerate(bif_validity)}

    # Accumulator of successful per-branch overrides across all failing bifs
    # processed so far.  Each time a bif's retry succeeds we merge that
    # bif's per-branch rfacs into this dict.  Subsequent retries *must*
    # include these committed values in every ``_compute_truncate_dict``
    # call — otherwise the next retry re-runs ``rtruncate`` from scratch
    # with only its own overrides, silently reverting the already-committed
    # rfacs on shared edges.  That would push the shared edge's truncation
    # on the far side back to the baseline ``rfac``, so the vessel mesh
    # we rebuild in this pass would extend past the endpoint where the
    # previously-committed bifurcation mesh already begins → overlap.
    branch_overrides_committed: dict[int, float] = {}

    for furc in furcs_for_rfac:
      key = frozenset(abs(int(i)) for i in furc.indices)
      signed = tuple(int(i) for i in furc.indices)
      k_n = len(signed)
      is_merged = key in merged_key_set
      baseline_ok = baseline_score_by_key[key][0] == 1
      # Merged + already valid: include explicit 1× multipliers first so the
      # baseline trunc competes; do not early-exit on max(mult) so later
      # entries can beat on Jacobian quality without forcing a coarser max.
      merged_quality_scan = is_merged and baseline_ok
      cascade = (
        ((1.0,) * k_n,) + tuple(_build_retry_cascade(k_n))
        if merged_quality_scan
        else _build_retry_cascade(k_n)
      )
      tried: list[dict] = []

      # Scan-all-then-commit-best: walk the whole cascade and keep a
      # running best-scoring valid attempt.  The score is no longer raw
      # mesh quality — it's a composite from :func:`_combined_retry_score`
      # whose primary term is ``-max(mults)``: among valid attempts, the
      # one with the smallest worst-branch rfac wins, because high rfac
      # stretches the adjacent vessel mesh visibly and that's worse for
      # the user than a slightly lower ``min(det J)``.  Raw Jacobian
      # quality is kept as a tertiary tiebreaker (see the docstring for
      # the full lexicographic order).
      best_score = (0, float('-inf'), float('-inf'),
                    float('-inf'), float('-inf'),
                    float('-inf'), float('-inf'))
      best_commit: dict | None = None  # payload to commit if this stays best
      # ``(max(mults), sum(mults))`` of the best valid attempt seen so
      # far.  The cascade is emitted in ascending order of this pair, so
      # once a cascade entry's pair exceeds it the entry can no longer
      # beat ``best`` on the primary two score slots — skip the (much
      # more expensive) mesh rebuild and move on.
      best_rfac_key: tuple[float, float] | None = None

      phases: list[tuple[str, tuple[tuple[float, ...], ...]]] = [
        ("standard", cascade),
      ]
      if not merged_quality_scan:
        phases.append(("fallback", _build_retry_cascade_fallback(k_n)))

      for phase_name, phase_cascade in phases:
        if phase_name == "fallback" and best_commit is not None:
          break
        # Fallback: always scan the whole phase (no max-mult early exit)
        # so the least damaging *valid* coarse truncation can win.
        no_early_exit = (phase_name == "fallback") or merged_quality_scan

        for mults in phase_cascade:
          mults_key = (float(max(mults)), float(sum(mults)))
          # Early-exit: if we already have a valid best and this entry is
          # strictly worse on ``(max, sum)``, skip it.  Since the cascade
          # is sorted ascending on that key, once we see a worse entry the
          # rest of the cascade is all worse — ``break`` out entirely.
          # Skip for merged quality scans and for the fallback phase.
          if (not no_early_exit) and best_rfac_key is not None \
             and mults_key > best_rfac_key:
            break
          overrides_this = {
            signed[b]: float(mults[b]) * float(rfac) for b in range(len(signed))
          }
          # Merge the in-flight trial with every override committed by a
          # previous bif's successful retry.  Later writes win for any
          # overlap, but for two different bifurcations the signed branch
          # ids never collide (each bif's indices are distinct signed ints,
          # and a shared edge appears with opposite signs in the two bifs).
          overrides_all = dict(branch_overrides_committed)
          overrides_all.update(overrides_this)
          try:
            trunc_retry = _compute_truncate_dict(
              sgraph, xis, xidict, rfac, branch_overrides=overrides_all,
            )
          except ValueError:
            # rtruncate_per_branch gave up on this combination (edges too
            # short for the requested per-branch rfac).  Move on.
            tried.append({
              "phase": phase_name,
              "multipliers": mults,
              "status": "truncate_failed",
            })
            continue

          # Build the retry sgraph carrying *all* committed overrides plus
          # the current trial.  Every edge in ``affected_ids`` is re-baked
          # against this sgraph so the vessel mesh endpoints match the new
          # bif mesh endpoints by construction.
          _sgraph_retry = sgraph.truncate_edges(trunc_retry)
          try:
            new_vessels = {}
            new_redges = {}
            new_vmeshes = {}
            affected_ids = sorted({abs(int(i)) for i in furc.indices})
            for iedge in affected_ids:
              pos = _sgraph_retry.indices.index(iedge)
              sedge_new = _sgraph_retry.edges[pos]
              v_new, r_new, m_new = _build_vessel_mesh(
                sedge_new, cross, dx, nref, nsec0, nsec1, union_eps,
              )
              new_vessels[iedge] = v_new
              new_redges[iedge] = r_new
              new_vmeshes[iedge] = m_new

            # Rebuild sgraph_r for this retry: override the affected edges
            # with their new redges; keep the rest (which already carry the
            # per-edge radius×frame fit from the baseline pass or an earlier
            # committed repair).
            redges_retry = [
              new_redges[i] if i in new_redges else redge_by_id[i]
              for i in _sgraph_retry.indices
            ]
            sgraph_r_retry = _sgraph_retry._edit(edges=redges_retry)

            # Find the corresponding NFurcation object in the retry sgraph
            # (identity may differ; match on its unsigned index set).
            furc_retry = next(
              (f for f in _sgraph_retry.nontrivial_furcations
               if frozenset(abs(int(i)) for i in f.indices) == key),
              None,
            )
            if furc_retry is None:
              tried.append({
                "phase": phase_name,
                "multipliers": mults,
                "status": "furc_missing",
              })
              continue

            furcation_new, fmesh_new, score_new = _build_bif_mesh(
              furc_retry, sgraph_r_retry, cross, nsec0, nsec1, mat,
              bif_exploit_symmetry, union_eps, dx,
            )
          except Exception as exc:
            tried.append({
              "phase": phase_name,
              "multipliers": mults,
              "status": f"build_error: {type(exc).__name__}",
            })
            continue

          valid_new = score_new[0] == 1
          # Combined score embeds the cascade-multiplier penalty so
          # smaller-rfac attempts outrank larger ones at equal-or-comparable
          # quality.  ``score_new`` (raw quality) is retained in ``tried``
          # for the GUI tooltip so the user can still inspect the Jacobian
          # numbers if they care about the mesh quality alone.
          combined_new = _combined_retry_score(score_new, mults)
          tried.append({
            "phase": phase_name,
            "multipliers": mults,
            "status": "valid" if valid_new else "invalid",
            "score": score_new,
            "combined": combined_new,
          })

          # Scan-all + keep-best: only consider valid attempts, and only
          # replace the running best when the combined score is strictly
          # better.  We store every piece we'd need to commit later so the
          # commit step after the cascade loop is a single dict lookup.
          if valid_new and combined_new > best_score:
            best_score = combined_new
            best_rfac_key = mults_key
            best_commit = {
              "mults": tuple(mults),
              "overrides_this": dict(overrides_this),
              "new_vessels": new_vessels,
              "new_redges": new_redges,
              "new_vmeshes": new_vmeshes,
              "furcation_new": furcation_new,
              "fmesh_new": fmesh_new,
              "score_new": score_new,
              "combined_new": combined_new,
              "trunc_retry": trunc_retry,
              "_sgraph_retry": _sgraph_retry,
              "_from_fallback": phase_name == "fallback",
            }

      # Cascade exhausted: commit the highest-scoring valid attempt (if
      # any) and fold its overrides into the accumulator so shared edges
      # stay consistent across subsequent bif repairs.
      repaired = best_commit is not None
      if repaired:
        affected_ids = sorted({abs(int(i)) for i in furc.indices})
        for iedge in affected_ids:
          vessel_by_id[iedge] = best_commit["new_vessels"][iedge]
          redge_by_id[iedge] = best_commit["new_redges"][iedge]
          vmesh_by_id[iedge] = best_commit["new_vmeshes"][iedge]
        furc_by_key[key] = best_commit["furcation_new"]
        fmesh_by_key[key] = best_commit["fmesh_new"]
        branch_overrides_committed.update(best_commit["overrides_this"])
        repair_info = {
          "branch_rfac": {
            signed[k]: float(best_commit["mults"][k]) * float(rfac)
            for k in range(len(signed))
          },
          "multipliers": best_commit["mults"],
          "attempts": len(tried),
          # ``score`` is raw quality (``min_det_J`` / ``mean_det_J``);
          # ``combined_score`` is the lexicographic tuple actually used
          # to pick this attempt — exposed so the GUI can explain why a
          # lower-quality-but-smaller-rfac attempt was preferred.
          "score": best_commit["score_new"],
          "combined_score": best_commit["combined_new"],
          "baseline_score": baseline_score_by_key.get(key),
          "fallback_escalation": bool(best_commit.get("_from_fallback")),
        }
        bif_validity[key_to_pos[key]] = (key, True, repair_info)
        truncate_dict = best_commit["trunc_retry"]
        _sgraph = best_commit["_sgraph_retry"]
        sgraph_r = _sgraph._edit(
          edges=[redge_by_id[i] for i in _sgraph.indices],
        )

      if not repaired:
        if merged_quality_scan:
          # Baseline was already strictly valid; optional scan found nothing
          # better to commit — keep validity True and retain baseline meshes.
          bif_validity[key_to_pos[key]] = (
            key, True,
            {
              "branch_rfac": None,
              "attempts": len(tried),
              "tried": tried,
              "merged_rfac_scan": "no_valid_improvement",
              "baseline_score": baseline_score_by_key.get(key),
            },
          )
        else:
          # Keep the original mesh and validity=False, but record what we tried
          # so the GUI can show a useful tooltip.
          bif_validity[key_to_pos[key]] = (
            key, False,
            {"branch_rfac": None, "attempts": len(tried), "tried": tried},
          )

  # ------------------------------------------------------------------ #
  # Assemble the final volumetric mesh from the (possibly partially
  # repaired) per-edge and per-bif meshes.
  # ------------------------------------------------------------------ #
  vessel_meshes = [vmesh_by_id[i] for i in _sgraph.indices]
  furcmeshes = [fmesh_by_key[
    frozenset(abs(int(i)) for i in furc.indices)
  ] for furc in _sgraph.nontrivial_furcations]

  vessels_union = mesh_union(*vessel_meshes, boundary=True, eps=union_eps)
  furcs_union = mesh_union(*furcmeshes, boundary=True, eps=union_eps)
  mesh = mesh_union(vessels_union, furcs_union, boundary=True, eps=union_eps)

  # --- Explicit merged n-furcation validity check ------------------- #
  # Re-score the *final* per-bif boundary mesh for each merged junction so
  # callers (and warnings) do not have to infer validity only from the
  # tuple ``bif_validity``.  ``valid`` is ``score[0] == 1`` from
  # :func:`_mesh_quality_score` (``is_valid`` ∧ scaled-Jacobian gate).
  merged_nfurc_check: list[dict] = []
  if merged_key_to_members:
    pipeline_valid = {k: bool(v) for k, v, _r in bif_validity}
    for merged_key, member_keys in merged_key_to_members.items():
      fmesh_m = fmesh_by_key.get(merged_key)
      if fmesh_m is None:
        score_m = (0, float('-inf'), float('-inf'),
                   float('-inf'), float('-inf'))
        strict = False
      else:
        score_m = _mesh_quality_score(fmesh_m)
        strict = score_m[0] == 1
      pipe_v = pipeline_valid.get(merged_key, False)
      merged_nfurc_check.append({
        "merged_key": merged_key,
        "member_keys": member_keys,
        "valid": strict,
        "quality_score": score_m,
        "valid_pipeline": pipe_v,
        "repair": next((r for k, _v, r in bif_validity if k == merged_key),
                       None),
      })
      if pipe_v != strict:
        warnings.warn(
          f"Merged n-furcation {sorted(merged_key)}: bif_validity says "
          f"valid={pipe_v} but quality recheck says valid={strict} — "
          f"inspect quality_score {score_m}.",
          RuntimeWarning,
          stacklevel=2,
        )
    if any(not e["valid"] for e in merged_nfurc_check):
      bad = [sorted(e["merged_key"]) for e in merged_nfurc_check
             if not e["valid"]]
      warnings.warn(
        "Merged n-furcation boundary mesh failed strict validity "
        f"(scaled Jacobian / det J): {bad}",
        UserWarning,
        stacklevel=2,
      )

  # Propagate validity/repair info from merged n-furcations back to their
  # constituent bifurcation keys so tooltips / diagnostics can still cite
  # absorbed junctions; the GUI bif list shows the merged n-furcation row.
  if merged_key_to_members:
    merged_entries = {k: (v, r) for k, v, r in bif_validity
                      if k in merged_key_to_members}
    for merged_key, (valid, repair) in merged_entries.items():
      member_keys = merged_key_to_members[merged_key]
      # Build a synthetic repair record tagged so the GUI can show a
      # distinct "merged into n-furcation" label rather than the usual
      # per-branch rfac repair annotation.
      merged_info = {
        "merged_into": merged_key,
        "merged_members": tuple(sorted(
          tuple(sorted(k)) for k in member_keys
        )),
      }
      if repair is not None:
        merged_info["repair"] = repair
      for mk in member_keys:
        bif_validity.append((mk, valid, merged_info))

  # Side-channel: per-bifurcation validity (including repair info) and
  # per-edge overlap detection.  Attached as plain attributes so downstream
  # callers that don't know about them keep working unchanged.
  try:
    mesh.bif_validity = tuple(bif_validity)
  except Exception:
    pass
  try:
    mesh.bif_overlaps = tuple(overlap_info)
  except Exception:
    pass
  try:
    # List of dicts with frozen keys: ``bif_keys`` (frozenset of frozensets),
    # ``internal_edges`` (frozenset of unsigned ints), ``component_id`` (str).
    mesh.merge_components = tuple(merge_components)
  except Exception:
    pass
  try:
    mesh.merged_nfurc_check = tuple(merged_nfurc_check)
    mesh.all_merged_nfurc_valid = (
      (not merged_nfurc_check)
      or all(e["valid"] for e in merged_nfurc_check)
    )
  except Exception:
    pass

  if show_interfaces:
    mesh.boundary.interfaces.plot()
  return mesh
