from pyhaemo.util import np, _, _round_array
from pyhaemo.types import Numeric, Int
from pyhaemo.spl import NDSpline

from functools import lru_cache
from itertools import combinations
from typing import Dict, Sequence

from numpy.typing import NDArray

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
def _kdtree(cline: NDSpline, nref: Int = 3, endpoints: bool = True) -> NDArray:
  """
  Create a kdtree from the centerline.
  """
  evalpoints = cline.knotvector[0].refine(nref).knots
  if not endpoints:
    assert len(evalpoints) > 2
    evalpoints = evalpoints[1:-1]
  return cKDTree(cline(evalpoints).reshape(-1, 3))


def _create_proximity_tree(artery, eps=1e-5, nref=4, endpoint_fac=2.0):
  endpoints = np.array([ spl(spl.knots[0][[0, -1]]) for spl in artery.centerlines ])
  endpoint_tree = _kdtree(endpoints)

  # create a kdtree for the centerline endpoints
  endpoint_pairs = endpoint_tree.query_pairs(eps=endpoint_fac*eps)
  transform = lambda ind: -(ind // 2) if ind % 2 == 0 else (ind // 2) + 1

  # bring into the right format (+- i0, +- i1) and remove pairs that
  # share the same centerline
  endpoint_pairs = [ tuple(map(transform, (a, b))) for a, b in map(sorted, endpoint_pairs)
                                                 if not (a % 2 == 0 and b == a + 1) ]

  # get the existing endpoint pairs
  existing_pairs = set(artery.endpoint_pairs)

  # remove pairs that are already in the same bifurcation
  endpoint_pairs = [ pair for pair in endpoint_pairs if pair not in existing_pairs ]

  # compute scaled endpoint distances (excluding equal centerline endpoints)
  endpoint_distances = [ np.linalg.norm(np.diff(endpoints[[a, b]])) / endpoint_fac
                                                   for a, b in endpoint_pairs ]

  trees = [ _kdtree(cline, nref=nref,
                           endpoints=False) for cline in artery.centerlines ]

  mindistances = []
  for i, tree in enumerate(trees):
    mydistances = tree.sparse_distance_matrix(endpoint_tree, eps=eps)


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
