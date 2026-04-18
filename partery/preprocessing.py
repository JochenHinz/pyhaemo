from pyhaemo.util import np, normalize, _
from pyhaemo.log import logger as log

from edge import VascularEdge
from notwist import compute_notwistframe

from itertools import combinations, repeat
from functools import lru_cache

from numpy.typing import NDArray
from numba import njit

from scipy.spatial import cKDTree


@lru_cache(maxsize=128)
def _kdtree(edge: VascularEdge, endpoints: bool = True) -> NDArray:
  """
  Create a kdtree from the centerline.
  """
  points = edge.points[:, :3]
  if not endpoints:
    assert len(points) > 2
    points = points[1:-1]
  return cKDTree(points)


def split_edges_proximity(graph, eps=1e-5, endpoint_fac=1.0):
  """
  Split edges that are close to the endpoint of another edge.
  If the minimum distance in the interior is bigger than the endpoint distance
  with the same edge (by a factor 1 / endpoint_fac) the edge is not split.
  """
  assert len(graph.nontrivial_furcations) == 0, 'The graph has nontrivial furcations'
  trees = [ _kdtree(edge, endpoints=False) for edge in graph.edges ]
  endpoints = np.concatenate([ edge.points[:, :3][[0, -1]] for edge in graph.edges ])
  endpoint_tree = cKDTree(endpoints)
  endpoint_dict = graph.endpoints

  # transform from ordering [0, 1, 2, 3, 4, 5, ...] to [-1, 1, -2, 2, -3, 3, ...]
  transform = lambda ind: -(ind // 2 + 1) if ind % 2 == 0 else (ind // 2 + 1)

  # b => (i, a, val),
  # b = endpoint, i = target edge, a = local parameteric value, val = distance
  mindices = {}
  for i, tree in enumerate(trees, 1):
    # get all distances to all endpoints
    mymatch = tree.sparse_distance_matrix(endpoint_tree, eps)

    distances = np.array(list(mymatch.values()))
    myindices = list(mymatch.keys())

    # match the endpoint edge to the corresponding local point index and minvalue
    # b => (a, val)
    for (a, b), val in map(lambda ind, val: ((ind[0], transform(ind[1])), val),
                           myindices,
                           distances):
      if b in (i, -i): continue  # skip the endpoints of the current edge
      if val < mindices.setdefault(b, (i, a, np.inf))[-1]:
        mindices[b] = (i, a, val)

  split_indices = {}

  # b = endpoint, i = centerline, a = local parameteric value, val = distance
  for b, (i, a, val) in mindices.items():
    if val < np.linalg.norm(endpoint_dict[i] - endpoint_dict[b]) / endpoint_fac and \
       val < np.linalg.norm(endpoint_dict[-i] - endpoint_dict[b]) / endpoint_fac:
      split_indices.setdefault(i, set()).add(a)

  edges = []
  for i, edge in enumerate(graph.edges, 1):
    if i not in split_indices:
      myedges = edge,
    else:
      positions = np.array(sorted(split_indices[i]), dtype=int) + 1
      myedges = edge.split(positions)
    edges.extend(myedges)

  return graph.from_edges(edges)


def merge_furcs_proximity(graph, eps=1e-5):
  allfurcs = graph.furcations
  centroids = [ furc.centroid for furc in allfurcs ]

  tree = cKDTree(centroids)
  pairs = list(tree.query_pairs(eps))

  # we filter the pairs that create an empty bifurcation on one or both
  # sides if the connecting edge(s) get deleted
  delpairs = set()
  for i, (i0, i1) in enumerate(pairs):
    bif0, bif1 = allfurcs[i0], allfurcs[i1]
    intersection = set(map(abs, bif0)) & set(map(abs, bif1))
    if any( len([ i for i in bif if abs(i) not in intersection]) == 0 for bif in (bif0, bif1)):
      delpairs.add(i)

  pairs = [pairs[i] for i in range(len(pairs)) if i not in delpairs]

  if not pairs:
    return graph

  minpair, *pairs = pairs
  mindist = np.linalg.norm(tree.data[minpair[0]] -
                           tree.data[minpair[1]])
  for (a, b) in pairs:
    if (y := np.linalg.norm(tree.data[a] - tree.data[b])) < mindist:
      minpair = (a, b)
      mindist = y

  ret = graph.merge_furcs(minpair[0], minpair[1])
  return merge_furcs_proximity(ret, eps=eps)


@njit(cache=True)
def _find_distance_core(points0, points1, r0, r1, centroid, rfac0, rfac1):
  """Numba-compiled core of :func:`find_distance` with two scalar rfacs.

  Split out so the JIT has a single concrete signature (numba rejects unions
  like ``float | tuple`` in nopython mode).  The Python-level
  :func:`find_distance` wrapper handles scalar/tuple input dispatching.

  Walks the two branch polylines toward the furcation until each endpoint is
  more than ``rfac_k · r_k`` from the centroid and the two endpoints are at
  least ``max(rfac_0 · r_0, rfac_1 · r_1)`` apart.
  """
  n0, n1 = len(points0), len(points1)
  i, j, success = 0, 0, True

  while i < n0 and j < n1:
    p0, p1 = points0[i], points1[j]
    # Per-branch scaled radii: each branch must retreat to at least
    # rfac_k · r_k from the centroid; for the pair-separation test we
    # need a shared threshold so take the larger of the two.
    sep0 = rfac0 * r0[i]
    sep1 = rfac1 * r1[j]
    sep_max = max(sep0, sep1)

    dist = np.linalg.norm(p0 - p1)
    if dist < sep_max:
      i += 1
      j += 1
      continue

    distance_center = ((np.stack((p0, p1), axis=0) - centroid)**2).sum(-1)**.5

    if distance_center[0] > sep0 and distance_center[1] > sep1:
      break

    if distance_center[0] < sep0:
      i = i + 1

    if distance_center[1] < sep1:
      j = j + 1

  else:
    i, j, success = 0, 0, False

  return i, j, success


def find_distance(points0, points1, r0, r1, centroid, rfac=2):
  """Walk the two branch polylines toward the furcation until both endpoints
  clear their per-branch pullback threshold.  Thin Python wrapper around
  :func:`_find_distance_core`; accepts ``rfac`` as either a scalar (applied
  to both branches) or a ``(rfac0, rfac1)`` pair.  Asymmetric values make
  each branch pull back by its own multiplier of its own local radius —
  used by the per-branch retry loop in :func:`rtruncate_per_branch` when
  one branch of a failing bifurcation needs to retreat further than the
  others.
  """
  if isinstance(rfac, (tuple, list, np.ndarray)):
    rfac0, rfac1 = float(rfac[0]), float(rfac[1])
  else:
    rfac0 = rfac1 = float(rfac)
  return _find_distance_core(points0, points1, r0, r1, centroid, rfac0, rfac1)


def rtruncate(graph, rfac=2):
  """
  Docstring forthcoming.
  """
  truncate_dict = {}

  for furc in graph.nontrivial_furcations:
    centroid = furc.centroid

    myparams = {}
    for pair in combinations(furc.indices, 2):
      myfac = rfac
      success = False
      while not success:
        if myfac < rfac / 10:
          raise ValueError(f"Could not find indices for furcation {furc} with centroid {centroid}.")
        e0, e1 = graph.get_edges([-i for i in pair])
        *indices, success = find_distance(e0.cpoints, e1.cpoints,
                                          e0.radii, e1.radii,
                                          centroid,
                                          myfac)
        if success is False:
          log.warning(f"No indices found for edge pair {pair} and centroid {centroid}")
          myfac /= 2
          log.warning(f"Trying again with rfac = {myfac}")
        else:
          for i, lindex in zip(pair, indices):
            myparams.setdefault(i, []).append(lindex if i < 0 else -lindex - 1)
          break

    for i, params in myparams.items():
      truncate_dict.setdefault(abs(i), [None, None])[0 if i < 0 else 1] = max(params)

  for key, vals in truncate_dict.items():
    if any(val is None for val in vals):
      continue
    edge, = graph.get_edges(key)
    _vals = [ val % edge.npoints for val in vals ]
    if _vals[0] >= _vals[1]:
      npoints_2 = edge.npoints // 2
      log.warning(f'Found invalid truncation indices {vals} for edge with index {key}.')
      truncate_dict[key] = [npoints_2 - 1, npoints_2 + 1]

  return truncate_dict


def rtruncate_per_branch(graph, rfac_default=2, branch_overrides=None):
  """Like :func:`rtruncate`, but with **per-branch** ``rfac`` overrides.

  ``branch_overrides`` is an optional mapping ``{signed_branch_id: rfac}``.
  Signed ids match ``furc.indices`` (positive/negative = branch direction into
  the furcation), so the caller can pull back one branch of a failing
  bifurcation further than the other two without touching their truncation.

  Pairs where at least one side has an override use the asymmetric
  ``find_distance(..., rfac=(r0, r1))`` path; pairs without overrides behave
  exactly like :func:`rtruncate` with the single ``rfac_default`` multiplier.

  Returns the same ``{iedge: [start_index_or_None, end_index_or_None]}``
  format so the downstream pipeline (``VascularGraph.truncate_edges`` /
  ``SplineGraph.truncate_edges``) is unchanged.
  """
  overrides = dict(branch_overrides or {})
  truncate_dict = {}

  for furc in graph.nontrivial_furcations:
    centroid = furc.centroid

    myparams = {}
    for pair in combinations(furc.indices, 2):
      # Per-branch multipliers: default to rfac_default on each side.
      r0 = float(overrides.get(pair[0], rfac_default))
      r1 = float(overrides.get(pair[1], rfac_default))
      # We still keep the global shrink-on-failure safety net from rtruncate:
      # if find_distance fails, halve both sides uniformly until it converges
      # or we drop below 10% of the user-requested value.
      r0_try, r1_try = r0, r1
      success = False
      while not success:
        if r0_try < r0 / 10 or r1_try < r1 / 10:
          raise ValueError(
            f'Could not find indices for furcation {furc} with centroid '
            f'{centroid} (per-branch rfac={r0}, {r1}).'
          )
        e0, e1 = graph.get_edges([-i for i in pair])
        *indices, success = find_distance(
          e0.cpoints, e1.cpoints, e0.radii, e1.radii, centroid,
          rfac=(r0_try, r1_try),
        )
        if success is False:
          log.warning(
            f'No indices found for edge pair {pair} (rfac={r0_try}, {r1_try})'
          )
          r0_try *= 0.5
          r1_try *= 0.5
          log.warning(f'Trying again with rfac=({r0_try}, {r1_try})')
        else:
          for k, lindex in zip(pair, indices):
            myparams.setdefault(k, []).append(lindex if k < 0 else -lindex - 1)
          break

    for k, params in myparams.items():
      truncate_dict.setdefault(abs(k), [None, None])[0 if k < 0 else 1] = max(params)

  for key, vals in truncate_dict.items():
    if any(val is None for val in vals):
      continue
    edge, = graph.get_edges(key)
    _vals = [val % edge.npoints for val in vals]
    if _vals[0] >= _vals[1]:
      npoints_2 = edge.npoints // 2
      log.warning(
        f'Found invalid per-branch truncation indices {vals} for edge with '
        f'index {key}; falling back to midpoint window.'
      )
      truncate_dict[key] = [npoints_2 - 1, npoints_2 + 1]

  return truncate_dict


def align_nontrivial_normals(graph):
  """
  Align normals of nontrivial furcations to the furcation centroid.
  """
  nontriv_indices = [i for i, furc in enumerate(graph.furcations) if len(furc) > 2]
  nfurcs = set(graph.furcations[i] for i in nontriv_indices)
  if not nfurcs:
    return graph

  signs = dict(zip(nfurcs, repeat(1)))  # map each furcation to a sign +1 or -1
  queue = []

  # while there are still orientable furcations
  while nfurcs:

    if not queue:
      queue = [nfurcs.pop()]

    furc, *queue = queue  # take first from the queue

    nfurcs.discard(furc)

    normal = signs[furc] * furc.normal
    for i in furc.indices:
      otherfurc = graph.get_furc(-i)
      if otherfurc not in nfurcs or len(otherfurc) <= 2:  # skip already oriented or trivial furcations
        continue

      # get the oriented other normal
      othernormal = otherfurc.normal * signs[otherfurc]
      edge, = graph.get_edges(-i)  # directed away from furc

      nfinal = edge.frame.project_and_transport(normal)
      if nfinal @ (edge.frame[-1:].project_tangent_plane(othernormal)).ravel() < 0:
        signs[otherfurc] *= -1
      queue.append(otherfurc)

  furcations = list(graph.furcations)
  for i in nontriv_indices:
    furc = graph.furcations[i]
    if signs[furc] == -1:
      log.warning(f"Reversing the orientation of the {i}-th furcation with indices{furc.indices.indices}")
      furcations[i] = -furc

  return graph._edit(furcations=furcations)


def align_frames(graph):
  """
  Align the edges' frames with the bifurcations' normals.
  """
  normal_dict = {}

  for furc in graph.furcations:
    mynormal = furc.normal
    for i in furc.indices:
      e, = graph.get_edges(i)
      normal_dict[i] = normalize( e.frame[-1:]
                                   .project_tangent_plane(mynormal).ravel() )
  edges = []
  for i, e in zip(graph.indices, graph.edges):
    frame = e.frame
    tangents = frame.frame[..., 2]
    newframe = compute_notwistframe(tangents, n0=normal_dict[-i],
                                              nfinal=normal_dict[i])
    edges.append( e._edit(frame=frame._edit(frame=newframe)) )

  return graph._edit(edges=edges)
