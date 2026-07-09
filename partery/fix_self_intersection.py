from dataclasses import dataclass

from pyhaemo.util import np
from pyhaemo.log import logger as log


@dataclass
class ValidationResult:
  """Outcome of validating (and possibly auto-smoothing) a single centerline.

  ``edge`` is the (possibly replaced) :class:`~edge.VascularEdge`: unchanged
  when the centerline was already valid or when correction was disabled
  (``max_iter=0``), otherwise resampled from the spline that resolved the
  max-radius violation (or, failing that, the spline from the last attempt).
  """
  edge: object
  initial_valid: bool
  final_valid: bool
  lam1_final: float = 0.0
  n_iterations: int = 0
  violation_frac_initial: float = 0.0
  violation_frac_final: float = 0.0


def curvature_mask(cline, radius, n=200):
  """
  Flags parameter samples where the local vessel radius exceeds the local
  minimum bend radius (``1 / curvature``) of the fitted centerline spline --
  i.e. where the tube swept by the centerline and its radius self-intersects
  (the "maximum-radius condition").

  Parameters
  ----------
  cline
    A 3-vector-valued spline (e.g. ``SplineEdge.cline``) supporting
    ``cline(xi, dx=k)`` for the k-th derivative.
  radius
    A scalar-valued spline (e.g. ``SplineEdge.radius``) giving the local
    vessel radius as a function of the same parameter.
  n
    Number of evaluation points along the curve.

  Returns
  -------
  numpy.ndarray
    Boolean mask, ``True`` where the max-radius condition is violated.
  """
  xi = np.linspace(0, 1, n)
  velocities = cline(xi, dx=1)
  accelerations = cline(xi, dx=2)

  with np.errstate(divide='ignore', invalid='ignore'):
    curvatures = np.linalg.norm(np.cross(velocities, accelerations), axis=1) \
               / np.linalg.norm(velocities, axis=1) ** 3
    minimum_bend_radius = 1 / curvatures

  radius_eval = radius(xi)

  return radius_eval > minimum_bend_radius


def _violation_fraction(mask):
  return float(np.count_nonzero(mask)) / len(mask) if len(mask) else 0.0


def refit_until_valid(edge, n, lam0=1e-5, max_iter=8, eval_n=200):
  """Validate a centerline and, if needed, iteratively refit it to satisfy
  the max-radius condition.

  The centerline+radius is first fit into a spline with no curvature
  regularization (``lam1=0``, but still endpoint/tangent-constrained via
  :meth:`~edge.VascularEdge.to_spline_edge` / :func:`~fit.fit_constrained`).
  If that violates the max-radius condition, ``lam1`` is increased over
  ``max_iter`` refits (spanning ``~1e-8`` to ``1e4``) until the violation
  clears or the iteration budget is exhausted. ``max_iter=0`` disables
  correction: violations are reported but the edge is left untouched.

  The escalation refits use ``constrain_endpoints=True,
  constrain_tangents=False`` (only the initial ``lam1=0`` fidelity check
  constrains tangents too). Endpoint *position* must stay fixed to preserve
  vessel/junction topology, but pinning the endpoint *tangent* to the
  original (already-too-tight) secant direction imposes a curvature floor
  that regularization cannot get under -- empirically, increasing ``lam1``
  with tangents constrained does not resolve genuine curvature violations at
  all. Downstream, :func:`generate_mesh` already reconciles per-edge frame
  differences at junctions via ``align_nontrivial_normals``/``align_frames``,
  so a slightly rotated endpoint tangent on a corrected edge is not a new
  class of problem for that pipeline.

  Returns a :class:`ValidationResult`.
  """
  spline_edge = edge.to_spline_edge(n, lam0=lam0, lam1=0)
  mask = curvature_mask(spline_edge.cline, spline_edge.radius, n=eval_n)
  violation_frac_initial = _violation_fraction(mask)
  initial_valid = not mask.any()

  if initial_valid or max_iter <= 0:
    return ValidationResult(
      edge=edge,
      initial_valid=initial_valid,
      final_valid=initial_valid,
      lam1_final=0.0,
      n_iterations=0,
      violation_frac_initial=violation_frac_initial,
      violation_frac_final=violation_frac_initial,
    )

  final_valid = False
  lam1_final = 0.0
  violation_frac_final = violation_frac_initial

  for k in range(1, max_iter + 1):
    lam1_final = 10 ** (-8 + 12 * k / max_iter)
    spline_edge = edge.to_spline_edge(n, lam0=lam0, lam1=lam1_final,
                                      constrain_tangents=False)
    mask = curvature_mask(spline_edge.cline, spline_edge.radius, n=eval_n)
    violation_frac_final = _violation_fraction(mask)
    if not mask.any():
      final_valid = True
      break

  new_points = spline_edge.rcenterline(np.linspace(0, 1, edge.npoints))
  new_edge = edge._edit(points=new_points, frame=None)

  return ValidationResult(
    edge=new_edge,
    initial_valid=False,
    final_valid=final_valid,
    lam1_final=lam1_final,
    n_iterations=max_iter if not final_valid else k,
    violation_frac_initial=violation_frac_initial,
    violation_frac_final=violation_frac_final,
  )


def validate_and_smooth_graph(graph, mesh_dx=None, spline_nmax=500, lam0=1e-5,
                              max_iter=8, report=None):
  """Validate every edge of a :class:`~network.EdgeGraph` against the
  max-radius condition, auto-smoothing violations in place.

  ``mesh_dx``, when given, sets the per-edge spline resolution the same way
  :meth:`~artery.VascularGraph.to_spline_graph` does (``max(2, length/dx)``,
  capped at ``spline_nmax``) so validation checks the geometry at the
  resolution it will actually be meshed at; otherwise a fixed resolution is
  used. ``report``, if given a dict, is populated with
  ``{centerline_id: ValidationResult}`` for diagnostics / GUI status.

  Returns a new graph with corrected edges (edges needing no correction, or
  for which correction failed or was disabled, are unchanged).
  """
  new_edges = []
  for i, edge in zip(graph.indices, graph.edges):
    if mesh_dx is not None:
      n = min(max(2, int(edge.length / mesh_dx)), spline_nmax)
    else:
      n = 21

    result = refit_until_valid(edge, n, lam0=lam0, max_iter=max_iter)
    new_edges.append(result.edge)

    if report is not None:
      report[i] = result

    if result.initial_valid:
      log.info(f"centerline {i}: valid (original)")
    elif result.final_valid:
      log.info(
        f"centerline {i}: valid (smoothed, lambda={result.lam1_final:.1e}) "
        f"after {result.n_iterations} refit iteration(s) "
        f"(violation {result.violation_frac_initial:.1%} -> 0%)"
      )
    else:
      log.warning(
        f"centerline {i}: invalid (automatic correction failed) -- "
        f"max-radius condition still violated on "
        f"{result.violation_frac_final:.1%} of the centerline after "
        f"{result.n_iterations} refit iteration(s) up to "
        f"lambda={result.lam1_final:.1e}"
      )

  return graph._edit(edges=new_edges)
