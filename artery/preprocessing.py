from pyhaemo.util import np, normalize
from pyhaemo.types import Numeric, Int
from pyhaemo.spl import NDSpline

from functools import lru_cache
from itertools import chain, combinations
from typing import Dict, Sequence

from numpy.typing import NDArray
import networkx as nx

from scipy import optimize
from scipy.spatial import cKDTree


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
      split_indices.setdefault(i, []).append(a)

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


def bifmerge(artery, eps=1e-5):
  endpoints = artery.endpoints()
  allbifs = artery.bifurcations

  centroids = []
  for bif in artery.bifurcations:
    # get the coordinates of the bifurcation
    coords = np.array([ endpoints[i] for i in bif ])
    # compute the centroid
    centroid = np.mean(coords, axis=0)
    centroids.append(centroid)

  tree = cKDTree(centroids)
  pairs = list(tree.query_pairs(eps))

  # we filter the pairs that create an empty bifurcation on one or both
  # sides if the connecting edge(s) get deleted
  delpairs = set()
  for i, (i0, i1) in enumerate(pairs):
    bif0, bif1 = allbifs[i0], allbifs[i1]
    intersection = set(map(abs, bif0)) & set(map(abs, bif1))
    if any( len([ i for i in bif if abs(i) not in intersection]) == 0 for bif in (bif0, bif1)):
      delpairs.add(i)

  pairs = [pairs[i] for i in range(len(pairs)) if i not in delpairs]

  if not pairs:
    return artery

  minpair, *pairs = pairs
  mindist = np.linalg.norm(tree.data[minpair[0]] -
                           tree.data[minpair[1]])
  for (a, b) in pairs:
    if (y := np.linalg.norm(tree.data[a] - tree.data[b])) < mindist:
      minpair = (a, b)
      mindist = y

  ret = artery.merge_bifs(minpair[0], minpair[1])
  return bifmerge(ret, eps=eps)


def biftruncate_opt_dist(artery, bif: Sequence[Int], distances: Dict[Int, Numeric] | Numeric, eps=0.03):
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


def biftruncate_opt_r(artery, bif: Sequence[Int], rfac: Numeric = 1.3, eps=0.03):
  """
  Truncate the artery at the bifurcation with the given index.
  """

  bif, = (_bif for _bif in artery.nontrivial_bifs if set(_bif) == set(bif))

  clines = [ artery.centerlines[abs(i) - 1] for i in bif ]
  radii = [artery.radii[abs(i) - 1] for i in bif]

  bounds = tuple( (0, spl.knots[0][-1] - spl.knots[0][0]) for spl in clines )
  intervals = [ spl.knots[0][-1] - spl.knots[0][0] for spl in clines ]

  splfuncs, radfuncs = [], []
  for i, spl, r in zip(bif, clines, radii):
    if i > 0:
      def sfunc(x, spl=spl, dx=0):
        root = spl.knots[0][-1]
        return spl(np.atleast_1d(root - x), dx=(dx,)).reshape(-1, 3)

      def rfunc(x, r=r):
        root = r.knots[0][-1]
        return r(np.atleast_1d(root - x)).ravel()
    else:
      def sfunc(x, spl=spl, dx=0):
        root = spl.knots[0][0]
        return spl(np.atleast_1d(root + x), dx=(dx,)).reshape(-1, 3)

      def rfunc(x, r=r):
        root = r.knots[0][0]
        return r(np.atleast_1d(root + x)).ravel()

    splfuncs.append(sfunc)
    radfuncs.append(rfunc)

  constraints = []
  for indices in combinations(range(len(clines)), 2):
    for i0, i1 in (indices, indices[::-1]):

      def func(x, c0=splfuncs[i0], c1=splfuncs[i1],
                  r0=radfuncs[i0], r1=radfuncs[i1],
                  i0=i0, i1=i1):
        _x, _y = x[i0], x[i1]
        # n0 = normalize(c0([_x], dx=1).ravel())
        p0, p1 = c0([_x]).ravel(), c1([_y]).ravel()
        _r0, _r1 = r0([_x]).ravel(), r1([_y]).ravel()

        return np.linalg.norm(p0 - p1) - 2 * rfac * max(_r0, _r1)

      constraints.append({'type': 'ineq', 'fun': func})

  def costfunc(displacements: NDArray):
    return displacements.sum()

  initial_guess = [bound[-1] for bound in bounds]

  sol = optimize.minimize(costfunc, initial_guess, bounds=bounds,
                                                   constraints=constraints)

  print(sol.success)
  x = sol.x

  print(x)

  kwargs = {}

  for i, spl, displ, intv in zip(bif, clines, x, intervals):
    if displ < eps * intv or displ > (1 - eps) * intv:
      continue
    if i > 0:
      kwargs[str(i)] = None, spl.knots[0][-1] - displ
    else:
      kwargs[str(-i)] = spl.knots[0][0] + displ, None

  return artery.truncate_edges(**kwargs)
