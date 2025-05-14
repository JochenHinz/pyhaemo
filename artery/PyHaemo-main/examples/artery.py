from splico.util import np, _
from splico.types import Immutable, Int
from splico.spl import NDSplineArray, NDSpline, UnivariateKnotVector
from splico.geo.interp import linear_interpolation

from itertools import chain, combinations
from typing import Tuple, Sequence
from data_handler import csv_converter
from numpy.typing import NDArray
import os

def fill_diagonal(arr: NDArray, val: float) -> NDArray:
  arr = np.asarray(arr)
  assert arr.ndim >= 2
  assert (n := arr.shape[0]) == arr.shape[1]

  indices = np.arange(n)
  arr[indices, indices] = val

  return arr


class Artery(Immutable):

  @classmethod
  def from_csv(cls, foldername: str, *args, **kwargs):
      """
      Create an Artery instance from a directory containing CSV files.

      Args:
            filename (str): Path to the directory containing CSV files.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

      Returns:
          Artery: An instance of the Artery class.

      Note:
          The `filename` parameter should be the path to the directory that
          contains the CSV files, not the path to a specific CSV file.
      """
      centerlines, radii = [], []

      # generator expression
      filenames = (os.path.join(foldername, f) for f
                   in os.listdir(foldername)
                   if f.endswith('.csv'))

      # Loop over each file
      for cspline, rspline in map(csv_converter, filenames):
          centerlines.append(cspline)
          radii.append(rspline)

      return cls(centerlines, radii, *args, **kwargs)


  def __init__(self, centerlines: Sequence[NDSpline],
                     radii: Sequence[NDSpline],
                     bifurcations: Tuple[Tuple[Tuple[Int]]] = ()):

    self.centerlines = tuple(map(NDSpline, centerlines))
    self.radii = tuple(map(NDSpline, radii))

    assert len(self.centerlines) == len(self.radii)
    assert all( c.knotvector == r.knotvector for c, r in zip(self.centerlines, self.radii) )

    # TODO: This block is too verbose

    nedges = len(self.centerlines)
    all_endpoints = set( chain(*( (((i,), (-i,))) for i in range(1, nedges+1) )) )
    seen = set()

    _bifurcations = []

    for bif in map(tuple, bifurcations):
      mybif = []
      for val in map(int, bif):
        assert val not in seen and 0 < abs(val) <= nedges
        all_endpoints -= {(val,)}
        seen.add(val)
        mybif.append(val)
      _bifurcations.append(tuple(mybif))
    ####

    self.bifurcations = tuple(sorted( tuple(_bifurcations) + tuple(all_endpoints) ))

    assert len( tuple(chain(*self.bifurcations)) ) == len(self.centerlines) * 2
    assert tuple(sorted(set(map(abs, chain(*self.bifurcations))))) == tuple(range(1, nedges+1))

  @property
  def nedges(self):
    return len(self.centerlines)

  def get_bif(self, index: Int, return_index=False):
    """
    Get the bifurcation that conains the index.
    """
    ret, = [ (i, bif) for i, bif in enumerate(self.bifurcations) if index in bif ]
    if not return_index:
      return ret[1]
    return ret

  @property
  def nontrivial_bifs(self):
    return tuple( bif for bif in self.bifurcations if len(bif) > 1 )

  def create_proximity_tree(self, eps=1e-5, npoints=101):

    # (ncenterlines, 2, 3)
    endpoints = np.array([ spl(spl.knots[0][[0, -1]]) for spl in self.centerlines ])

    # (ncenterlines, 1, 2, 1, 3) and (1, ncenterlines, 1, 2, 3) -> (ncenterlines, ncenterlines, 2, 2)
    endpoint_diffs = fill_diagonal(
                      ((endpoints[:, _, :, _] - endpoints[_, :, _])**2).sum(-1),
                      np.inf
                  )

  
    # TODO: This triple for loop needs to be simplified
    for bif in self.nontrivial_bifs:
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

      bif0, bif1 = map(self.get_bif, (iedge0, iedge1))

      new_bifs = tuple( bif for bif in self.nontrivial_bifs if bif not in (bif0, bif1) ) + ((*bif0, *bif1),)

      return self._edit( bifurcations=new_bifs ).create_proximity_tree(eps=eps, npoints=npoints)

    # TODO: This block is too verbose

    # match in the interior of the edges
    for i, (centerline, radius) in enumerate(zip(self.centerlines, self.radii)):
      xi = np.linspace(*centerline.knots[0][[0, -1]], npoints + 2)[1: -1]
      x = centerline(xi)  # (len(xi), 3)

      # (len(xi), ncenterlines, 2)
      dist = ((endpoints[_] - x[:, _, _, :])**2).sum(-1)

      # get the bif that contains the current index
      dist[:, i] = np.inf

      for index in self.get_bif(i+1):
        if abs(index) == (i+1): continue
        k = 0 if index < 0 else 1
        dist[:, abs(index) - 1, k] = np.inf

      j, iedge, direction = np.unravel_index(np.argmin(dist), dist.shape)

      if dist[j, iedge, direction] > eps:
        continue

      bifindex, bif = self.get_bif( (iedge+1) * (-1) ** (direction + 1), return_index=True)
      assert i+1 not in tuple(map(int, bif))

      c0, c1 = centerline.split(0, xvals=xi[j])
      r0, r1 = radius.split(0, xvals=xi[j])

      centerlines = self.centerlines[:i] + (c0,) + self.centerlines[i+1:] + (c1,)
      radii = self.radii[:i] + (r0,) + self.radii[i+1:] + (r1,)

      newindex = self.nedges + 1

      index_map = {i+1: newindex}

      new_bifs = tuple( tuple(index_map.get(k, k) for k in bif)
                        for bif in self.bifurcations[:bifindex] + self.bifurcations[bifindex+1:] )

      new_bifs = new_bifs + ((*bif, i+1, -newindex),)

      return self._edit(centerlines=centerlines,
                        radii=radii,
                        bifurcations=new_bifs).create_proximity_tree(eps=eps, npoints=npoints)

    else:
      return self

  def plot(self, npoints=101):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for centerline in self.centerlines:
      xi = np.linspace(*centerline.knots[0][[0, -1]], npoints)
      plt.plot(*centerline(xi).T[:2])
    plt.show()


if __name__ == '__main__':
  kv = UnivariateKnotVector(np.linspace(0, 1, 5)).to_tensor()
  r = NDSpline.one(kv)

  p0 = NDSpline([], [[0, 0, 0]])
  p1 = NDSpline([], [[1, 1, 0]])

  p2 = NDSpline([], [[1, -1, 0]])
  p3 = NDSpline([], [[.5, .5, 0]])

  p4 = NDSpline([], [[0, -1, 0]])

  c0 = linear_interpolation(p0, p1, kvz=kv[0])
  c1 = linear_interpolation(p2, p3, kvz=kv[0])
  c2 = linear_interpolation(p4, p3, kvz=kv[0])

  artery = Artery([c0, c1, c2], [r, r, r])

  test = artery.create_proximity_tree()

  ########## real data ########
  foldername = "/home/fabio/PhD/PyHaemo/examples/test_PyHaemo/data/centerlines_csv"
  
  test1 = artery.from_csv(foldername).create_proximity_tree(eps=0.35)
  