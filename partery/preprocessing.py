from pyhaemo.util import np

from edge import VascularEdge

from functools import lru_cache

from numpy.typing import NDArray

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
