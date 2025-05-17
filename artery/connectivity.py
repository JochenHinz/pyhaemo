from pyhaemo.util import np, _, _round_array

from itertools import combinations

from numpy.typing import NDArray


def fill_diagonal(arr: NDArray, val: float) -> NDArray:
  arr = np.asarray(arr)
  assert arr.ndim >= 2
  assert (n := arr.shape[0]) == arr.shape[1]

  indices = np.arange(n)
  arr[indices, indices] = val

  return arr


def _create_proximity_tree(artery, eps=1e-5, nref=4):

  # (ncenterlines, 2, 3)
  endpoints = np.array([ spl(spl.knots[0][[0, -1]]) for spl in artery.centerlines ])

  # (ncenterlines, 1, 2, 1, 3) and (1, ncenterlines, 1, 2, 3) -> (ncenterlines, ncenterlines, 2, 2)
  endpoint_diffs = fill_diagonal(
                    ((endpoints[:, _, :, _] - endpoints[_, :, _])**2).sum(-1),
                    np.inf
                )

  # TODO: This triple for loop needs to be simplified
  for bif in artery.nontrivial_bifs:
    for edges in combinations(bif, 2):
      for e0, e1 in (edges, edges[::-1]):
        endpoint_diffs[abs(e0)-1,
                       abs(e1)-1,
                       0 if e0 < 0 else 1,
                       0 if e1 < 0 else 1] = np.inf
  ####

  argmin = np.unravel_index( np.argmin(endpoint_diffs), endpoint_diffs.shape )

  # we found a match at an endpoint, we do not split an edge but just
  # create a new bifurcation
  if endpoint_diffs[argmin] < eps:
    iedge0, iedge1, endpoint0, endpoint1 = argmin

    iedge0 = (iedge0 + 1) * ((-1) ** (endpoint0 + 1))
    iedge1 = (iedge1 + 1) * ((-1) ** (endpoint1 + 1))

    bif0, bif1 = map(artery.get_bif, (iedge0, iedge1))

    new_bifs = tuple( bif for bif in artery.nontrivial_bifs if bif not in (bif0, bif1) ) + ((*bif0, *bif1),)
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

    new_bifs = tuple( tuple(index_map.get(k, k) for k in bif)
                      for bif in artery.bifurcations[:bifindex] + artery.bifurcations[bifindex+1:] )

    new_bifs = new_bifs + ((*bif, i+1, -newindex),)

    ret = artery._edit(centerlines=centerlines,
                       radii=radii,
                       bifurcations=new_bifs)
    return ret.create_proximity_tree(eps=eps, nref=nref)

  else:
    return artery
