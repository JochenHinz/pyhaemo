from pyhaemo.util import np, _, _round_array
from pyhaemo.types import Numeric, Int
from pyhaemo.spl import NDSpline

from functools import lru_cache
from itertools import combinations, chain
from typing import Dict, Sequence

from numpy.typing import NDArray
import networkx as nx

from scipy import optimize
from scipy.spatial import cKDTree


def fill_diagonal(arr: NDArray, val: float) -> NDArray:
  arr = np.asarray(arr)
  assert arr.ndim >= 2
  assert (n := arr.shape[0]) == arr.shape[1]

  indices = np.arange(n)
  arr[indices, indices] = val

  return arr


@lru_cache(maxsize=128)
def _kdtree(cline: NDSpline, nref: Int = 2, endpoints: bool = True) -> NDArray:
  """
  Create a kdtree from the centerline.
  """
  evalpoints = cline.knotvector[0].refine(nref).knots
  if not endpoints:
    assert len(evalpoints) > 2
    evalpoints = evalpoints[1:-1]
  return cKDTree(cline(evalpoints).reshape(-1, 3))


def split_edges_proximity(artery, eps=1e-5, nref=4, endpoint_fac=1.0):
  """
  Split edges that are close to the endpoint of another edge.
  If the minimum distance in the interior is bigger than the endpoint distance
  with the same edge (by a factor 1 / endpoint_fac) the edge is not split.
  """
  assert len(artery.nontrivial_bifs) == 0, 'The artery has nontrivial bifurcations'
  trees = [ _kdtree(cline, nref=nref, endpoints=False) for cline in artery.centerlines ]
  endpoints = np.concatenate([ spl(spl.knots[0][[0, -1]]) for spl in artery.centerlines ])
  endpoint_tree = cKDTree(endpoints)
  endpoint_dict = artery.endpoints()

  transform = lambda ind: -(ind // 2 + 1) if ind % 2 == 0 else (ind // 2 + 1)

  split_indices = {}
  for i, tree in enumerate(trees, 1):
    # get all distances to all endpoints
    mymatch = tree.sparse_distance_matrix(endpoint_tree, eps)

    distances = np.array(list(mymatch.values()))
    myindices = list(mymatch.keys())

    # match the endpoint edge to the corresponding local point index and minvalue
    # b => (a, val)
    mindices = {}
    for (a, b), val in map(lambda ind, val: ((ind[0], transform(ind[1])), val),
                           myindices,
                           distances):
      if b in (i, -i): continue  # skip the endpoints of the current edge
      if val < mindices.setdefault(b, (a, np.inf))[1]:
        mindices[b] = (a, val)

    mysplit = []
    for b, (a, val) in mindices.items():
      if val < np.linalg.norm(endpoint_dict[i] - endpoint_dict[b]) / endpoint_fac and \
         val < np.linalg.norm(endpoint_dict[-i] - endpoint_dict[b]) / endpoint_fac:
        mysplit.append(a)

    split_indices[i] = sorted(set(mysplit))

  centerlines, radii = [], []
  for i, (cline, r) in enumerate(zip(artery.centerlines, artery.radii), 1):
    if i not in split_indices:
      clines = cline,
      rs = r,
    else:
      xi = cline.knotvector[0].refine(nref).knots[1:-1]
      clines = cline.split(0, xvals=xi[split_indices[i]])
      rs = r.split(0, xvals=xi[split_indices[i]])
    centerlines.extend(clines)
    radii.extend(rs)

  return artery._edit(centerlines=centerlines,
                      radii=radii,
                      bifurcations=())


transform = lambda ind: -(ind // 2 + 1) if ind % 2 == 0 else (ind // 2 + 1)


def merge_pairs_to_chains(pairs):
  G = nx.Graph()
  G.add_edges_from(pairs)
  chains = []
  for component in nx.connected_components(G):
    subgraph = G.subgraph(component)
    # Try to find an endpoint (node with degree 1)
    endpoints = [node for node in subgraph.nodes if subgraph.degree[node] == 1]
    if not endpoints:  # cycle
      path = list(nx.dfs_preorder_nodes(subgraph))
    else:
      path = list(nx.shortest_path(subgraph, source=endpoints[0]).values())[0]
    chains.append(tuple(path))

  return chains


def bifmerge_(artery, eps=1e-5):
  endpoints = artery.endpoints()
  centroids = []
  for bif in artery.bifurcations:
    # get the coordinates of the bifurcation
    coords = np.array([ endpoints[i] for i in bif ])
    # compute the centroid
    centroid = np.mean(coords, axis=0)
    centroids.append(centroid)

  tree = cKDTree(centroids)
  pairs = tree.query_pairs(eps)

  chains = merge_pairs_to_chains(pairs)

  new_bifs = []
  for mypairs in chains:
    new_bif = tuple(chain(*(artery.bifurcations[i] for i in mypairs)))
    if len(new_bif) == 2 and new_bif[0] == -new_bif[1]:
      continue
    new_bifs.append(new_bif)

  print(new_bifs)

  return artery._edit(bifurcations=new_bifs)


def bifmerge(artery, eps=1e-5):
  endpoints = artery.endpoints()
  centroids = []
  for bif in artery.bifurcations:
    # get the coordinates of the bifurcation
    coords = np.array([ endpoints[i] for i in bif ])
    # compute the centroid
    centroid = np.mean(coords, axis=0)
    centroids.append(centroid)

  tree = cKDTree(centroids)
  pairs = tree.query_pairs(eps)

  if not pairs:
    return artery

  minpair, *pairs = pairs
  mindist = np.linalg.norm(tree.data[minpair[0]] -
                           tree.data[minpair[1]])
  for (a, b) in pairs:
    if np.linalg.norm(tree.data[a] - tree.data[b]) < mindist:
      minpair = (a, b)
      mindist = np.linalg.norm(tree.data[a] - tree.data[b])

  try:
    return bifmerge(artery.merge_bifs(minpair[0], minpair[1]), eps=eps)
  except Exception:
    import ipdb
    ipdb.set_trace()
    return bifmerge(artery.merge_bifs(minpair[0], minpair[1]), eps=eps)


def _compute_endpoint_distances(endpoint_tree: cKDTree, eps: float) -> NDArray:
  pass


def _create_proximity_tree(artery, eps=1e-5, nref=4, endpoint_fac=2.0):
  assert endpoint_fac > 1.0
  endpoints = np.concatenate([ spl(spl.knots[0][[0, -1]]) for spl in artery.centerlines ])
  endpoint_tree = cKDTree(endpoints)

  # create a kdtree for the centerline endpoints
  endpoint_pairs = endpoint_tree.query_pairs(endpoint_fac*eps)

  # compute corresponding scaled endpoint distances
  endpoint_distances = np.array([ np.linalg.norm(endpoints[a] - endpoints[b]) / endpoint_fac
                                  for a, b in endpoint_pairs ])

  # transformation function 0 => -1, 1 => 1, 2 => -2, 3 => 2, 4 => -3, 5 => 3
  # this one goes from the endpoint kDTree indexing to the original edge
  # indexing
  transform = lambda ind: -(ind // 2 + 1) if ind % 2 == 0 else (ind // 2 + 1)

  # transform indexing
  endpoint_pairs_transformed = [ tuple(sorted(map(transform, pair))) for pair in endpoint_pairs ]

  # get the existing endpoint pairs
  existing_pairs = set(artery.endpoint_pairs)

  # make a copy that disregards the sign of the indices
  abspairs = set( artery.abs_endpoint_pairs )

  # bring into the right format (+- i0, +- i1) and remove pairs that
  # share the same centerline
  keepindices = [ i for i, (a, b) in enumerate(endpoint_pairs_transformed)
                  if (a, b) not in existing_pairs
                  and a != -b
                  and len(set(map(abs, artery.get_bif(a))) & set(map(abs, artery.get_bif(b)))) == 0]

  endpoint_pairs = [ endpoint_pairs_transformed[i] for i in keepindices ]
  endpoint_distances = endpoint_distances[keepindices]

  trees = [ _kdtree(cline, nref=nref,
                           endpoints=False) for cline in artery.centerlines ]

  # create a list containing the minimum distances of the interior centerlines
  # to the endpoints and the corresponding indices in the form
  # (iedge0, ipoint, iedge1) where iedge0 is always > 0
  mindistances, mindices = [], []
  for i, (tree, cline) in enumerate(zip(trees, artery.centerlines), 1):

    # get all distances to all endpoints
    mydistances = tree.sparse_distance_matrix(endpoint_tree, eps)

    # (ipoint, +-iedge)
    indices = [ (a, transform(b)) for a, b in mydistances.keys() ]

    # keep only the ones that are not already in a shared bif
    # or originate from the same edge
    keepindices = [ j for j in range(len(indices))
                          if (tuple(sorted((i, abs(indices[j][1])))) not in abspairs
                              and abs(indices[j][1]) != i) ]

    if not keepindices:
      continue

    # all distances between the edge and an endpoint the edge is not already
    # matched to
    distances = np.array(list(mydistances.values()))[keepindices]

    # corresponding (ipoint, iedge) pair
    mindex = indices[keepindices[np.argmin(distances)]]

    # append minimum distance to mindistances
    mindistances.append(distances.min())

    # append the corresponding (iedge0, ipoint, iedge1) tuple to mindices
    mindices.append( (i, *mindex) )

  if not mindistances and not endpoint_distances:  # no matches found
    return artery

  # minimum distance found in the interior of an edge
  if min(mindistances, default=np.inf) < min(endpoint_distances, default=np.inf):
    iedge0, ipoint, iedge1 = mindices[np.argmin(mindistances)]

    cline = artery.centerlines[iedge0 - 1]
    radius = artery.radii[iedge0 - 1]

    xi = cline.knotvector[0].refine(nref).knots[1:-1]

    c0, c1 = cline.split(0, xvals=xi[ipoint])
    r0, r1 = radius.split(0, xvals=xi[ipoint])

    # get the bif and the bif index of the edge that we matched to
    bifindex, bif = artery.get_bif(iedge1, return_index=True)

    centerlines = artery.centerlines[:iedge0-1] + (c0,) + artery.centerlines[iedge0:] + (c1,)
    radii = artery.radii[:iedge0-1] + (r0,) + artery.radii[iedge0:] + (r1,)

    newindex = artery.nedges + 1
    index_map = {iedge0: newindex}

    new_bifs = tuple( tuple(index_map.get(k, k) for k in bif) for bif in
                      artery.bifurcations[:bifindex] + artery.bifurcations[bifindex+1:] )

    new_bifs = new_bifs + ((*bif, iedge0, -newindex),)

    artery = artery._edit(centerlines=centerlines,
                          radii=radii,
                          bifurcations=new_bifs)

  # minimum distance found at the endpoints
  else:
    # get minimum pair
    i0, i1 = endpoint_pairs[np.argmin(endpoint_distances)]
    # get the bifurcations of the edges
    bif0, bif1 = map(artery.get_bif, (i0, i1))

    # merge the bifs
    new_bifs = tuple(bif for bif in artery.nontrivial_bifs if bif not in (bif0, bif1)) + ((*bif0, *bif1),)
    artery = artery._edit(bifurcations=new_bifs)

  artery.plot_pyvista()
  return artery.create_proximity_tree(eps=eps, nref=nref)


def _create_proximity_tree_(artery, eps=1e-5, nref=4):

  # (ncenterlines, 2, 3)
  endpoints = np.array([ spl(spl.knots[0][[0, -1]]) for spl in artery.centerlines ])

  # (ncenterlines, 1, 2, 1, 3) and (1, ncenterlines, 1, 2, 3) -> (ncenterlines, ncenterlines, 2, 2)
  endpoint_diffs = fill_diagonal(
                    ((endpoints[:, _, :, _] - endpoints[_, :, _])**2).sum(-1),
                    np.inf
                )

  # TODO: This triple for loop needs to be simplified

  # itertate over all nontrivial bifurcations and make sure that the endpoint
  # distance is set to infinity on the bifurcation endpoint pairs.
  for bif in artery.nontrivial_bifs:
    for edges in combinations(bif, 2):
      for e0, e1 in (edges, edges[::-1]):
        endpoint_diffs[abs(e0)-1,
                       abs(e1)-1] = np.inf
  ####

  argmin = np.unravel_index( np.argmin(endpoint_diffs), endpoint_diffs.shape )
  iedge0, iedge1, endpoint0, endpoint1 = argmin

  iedge0 = (iedge0 + 1) * ((-1) ** (endpoint0 + 1))
  iedge1 = (iedge1 + 1) * ((-1) ** (endpoint1 + 1))

  bif0, bif1 = map(artery.get_bif, (iedge0, iedge1))

  # we found a match at an endpoint, we do not split an edge but just
  # create a new bifurcation
  if endpoint_diffs[argmin] < eps and all(-i not in bif1 for i in bif0):
    new_bifs = tuple(bif for bif in artery.nontrivial_bifs if bif not in (bif0, bif1)) + ((*bif0, *bif1),)
    return artery._edit( bifurcations=new_bifs ).create_proximity_tree(eps=eps, nref=nref)

  # TODO: This block is too verbose

  # match in the interior of the edges
  for i, (centerline, radius) in enumerate(zip(artery.centerlines, artery.radii)):
    xi = _round_array(centerline.knotvector[0].refine(nref).knots)
    x = centerline(xi)  # (len(xi), 3)

    # (len(xi), ncenterlines, 2)
    dist = np.sqrt(((endpoints[_] - x[:, _, _, :])**2).sum(-1))

    dist[:, i] = np.inf

    # get the bif that contains the current index
    for index in artery.get_bif(i+1) + artery.get_bif(-i-1):
      if abs(index) == (i+1): continue
      k = 0 if index < 0 else 1
      dist[:, abs(index) - 1, k] = np.inf

    j, iedge, direction = np.unravel_index(np.argmin(dist), dist.shape)
    mydist = dist[j, iedge, direction]

    if mydist > eps:
      continue

    bifindex, bif = artery.get_bif( (iedge+1) * (-1) ** (direction + 1), return_index=True)
    assert i+1 not in tuple(map(int, bif))

    c0, c1 = centerline.split(0, xvals=xi[j])
    r0, r1 = radius.split(0, xvals=xi[j])

    if not np.linalg.norm(c0([xi[j]]) - endpoints[iedge, direction]) < eps:
      raise ValueError("The centerline is not close to the bifurcation")

    centerlines = artery.centerlines[:i] + (c0,) + artery.centerlines[i+1:] + (c1,)
    radii = artery.radii[:i] + (r0,) + artery.radii[i+1:] + (r1,)

    newindex = artery.nedges + 1
    index_map = {i+1: newindex}

    new_bifs = tuple( tuple(index_map.get(k, k) for k in bif) for bif in
                      artery.bifurcations[:bifindex] + artery.bifurcations[bifindex+1:] )

    new_bifs = new_bifs + ((*bif, i+1, -newindex),)

    ret = artery._edit(centerlines=centerlines,
                       radii=radii,
                       bifurcations=new_bifs)

    return ret.create_proximity_tree(eps=eps, nref=nref)

  else:
    return artery


def biftruncate(artery, bif: Sequence[Int], distances: Dict[Int, Numeric] | Numeric, eps=0.03):
  """
  Truncate the artery at the bifurcation with the given index.
  """

  if np.isscalar(distances):
    distances = {i: distances for i in bif}

  bif, = (_bif for _bif in artery.nontrivial_bifs if set(_bif) == set(bif))
  assert set(distances.keys()) == set(bif)

  splines = [ artery.centerlines[abs(i) - 1] for i in bif ]
  bounds = tuple( (0, spl.knots[0][-1] - spl.knots[0][0]) for spl in splines )
  distances = np.array([ distances[i] for i in bif ])
  intervals = [ spl.knots[0][-1] - spl.knots[0][0] for spl in splines ]

  splfuncs = []
  for i, spl in zip(bif, splines):
    if i > 0:
      def func(x, spl=spl):
        root = spl.knots[0][-1]
        return spl(np.atleast_1d(root - x)).reshape(-1, 3)
    else:
      def func(x, spl=spl):
        root = spl.knots[0][0]
        return spl(np.atleast_1d(root + x)).reshape(-1, 3)

    splfuncs.append(func)

  def costfunc(displacements: NDArray):
    evals = np.array([spl(x).ravel() for spl, x in zip(splfuncs, displacements)])
    centroid = np.mean(evals, axis=0)

    mydistances = np.linalg.norm(evals - centroid, axis=1)

    ret = ((mydistances - distances)**2).sum()
    return ret

  centroid = np.mean([splfunc(0).ravel() for splfunc in splfuncs], axis=0)

  initial_guess = []
  for func, mydist, (a, upper_bound) in zip(splfuncs, distances, bounds):
    xi = np.linspace(0, upper_bound, 101)
    initial_guess.append( xi[((func(xi) - centroid) ** 2).sum(1).argmin()] )

  sol = optimize.minimize(costfunc, initial_guess, bounds=bounds)

  print(sol.success)
  x = sol.x

  kwargs = {}

  for i, spl, displ, intv in zip(bif, splines, x, intervals):
    if displ < eps * intv or displ > (1 - eps) * intv:
      continue
    if i > 0:
      kwargs[str(i)] = None, spl.knots[0][-1] - displ
    else:
      kwargs[str(-i)] = spl.knots[0][0] + displ, None

  return artery.truncate_edges(**kwargs)
