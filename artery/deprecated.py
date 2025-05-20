from pyhaemo.util import np, Int
from pyhaemo.spl import NDSpline

from functools import lru_cache

from scipy.spatial import cKDTree
from numpy.typing import NDArray


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



