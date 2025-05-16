from pyhaemo.util import np, _, _round_array, clparam, normalize
from pyhaemo.types import Immutable, Int
from pyhaemo.spl import NDSpline, UnivariateKnotVector, NDSplineArray
from pyhaemo.spl.aux import sparse_kron
from pyhaemo.geo.interp import linear_interpolation
# from pyhaemo.geo.notwist import compute_notwistframe_from_spline
from notwist import compute_notwistframe
from pyhaemo.geo import ellipse
from pyhaemo.mesh.bool import mesh_union
from pyhaemo.log import logger as log

from datahandler import csv_converter, swc_converter
from aux import plot_bifurcation
from plot import plot_frames, plot_artery, pyvista_mesh
import pyvista as pv

from itertools import chain, combinations, count
from typing import Tuple, Sequence
import os

from numpy.typing import NDArray
from scipy.spatial import cKDTree


def fill_diagonal(arr: NDArray, val: float) -> NDArray:
  arr = np.asarray(arr)
  assert arr.ndim >= 2
  assert (n := arr.shape[0]) == arr.shape[1]

  indices = np.arange(n)
  arr[indices, indices] = val

  return arr


def centerline_length(centerline: NDSpline) -> float:
  """
  Compute the length of a centerline spline using the Gauss points.
  """
  xi, = centerline.knotvector.gauss_abscissae
  assert len(xi) > 1
  x = centerline(xi)
  return ((np.diff(x, axis=0)**2).sum(1) ** .5).sum()


def make_vessel(centerline: NDSpline,
                radius: NDSpline,
                cross_section: NDSplineArray, **kwargs):

  assert -1 not in centerline.knotvector[0].continuity[1:-1]
  xi, = centerline.knotvector.greville
  tangents = normalize(centerline(xi, dx=1))

  # compute the notwistframe rotation matrices
  Rs = compute_notwistframe(tangents, **kwargs)
  rRs = centerline.knotvector.fit([xi], radius(xi)[:, _, _] * Rs)

  return (cross_section[:, _] @ rRs[_]).sum(-1) + (cross_section.unity @ centerline)


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

  @classmethod
  def from_swc(cls, filename: str, *args, **kwargs):
    centerlines, radii, _ = swc_converter(filename)
    return cls(centerlines, radii)

  def __init__(self, centerlines: Sequence[NDSpline],
                     radii: Sequence[NDSpline],
                     bifurcations: Tuple[Tuple[Tuple[Int]]] = ()):

    self.centerlines = tuple(map(NDSpline, centerlines))
    self.radii = tuple(map(NDSpline, radii))

    assert len(self.centerlines) == len(self.radii)
    assert all( c.knotvector == r.knotvector for c, r in zip(self.centerlines, self.radii) )

    # TODO: This block is too verbose

    nedges = len(self.centerlines)
    all_endpoints = set(range(1, self.nedges+1)) | set(-i for i in range(1, self.nedges+1))
    seen = set()

    _bifurcations = []

    for bif in map(lambda x: tuple(map(int, x)), bifurcations):
      for val in bif:
        assert val not in seen and 0 < abs(val) <= nedges
        all_endpoints -= {val}
        seen.add(val)

      _bifurcations.append(bif)

    self.bifurcations = tuple(sorted(_bifurcations + [(i,) for i in all_endpoints]))
    ####

    assert len( tuple(chain(*self.bifurcations)) ) == len(self.centerlines) * 2
    assert tuple(sorted(set(map(abs, chain(*self.bifurcations))))) == tuple(range(1, nedges+1))

  @property
  def nedges(self):
    return len(self.centerlines)

  def endpoints(self, dx=0):
    ret = {}
    for i, cline in enumerate(self.centerlines, 1):
      ret[i] = cline(cline.knots[0][-1:], dx=dx).ravel()
      ret[-i] = cline(cline.knots[0][:1], dx=dx).ravel()

    return ret

  def get_bif(self, index: Int, return_index=False):
    """
    Get the bifurcation that conains the index.
    """
    ret, = [ (i, bif) for i, bif in enumerate(self.bifurcations) if index in bif ]
    if not return_index:
      return ret[1]
    return ret

  def reparameterize(self):
    """
    Reparameterize such that every curve goes from 0 to 1.
    """
    centerlines = [centerline.reparameterize(...) for centerline in self.centerlines]
    radii = [r.reparameterize(...) for r in self.radii]
    return self._edit(centerlines=centerlines, radii=radii)

  def bifcenters(self, eps=1e-9):
    endpoints = self.endpoints()
    centers = []
    for bif in self.bifurcations:
      mypoints = []
      for index in bif:
        mypoint = endpoints[index]
        if not any( np.linalg.norm(point - mypoint) < eps for point in mypoints ):
          mypoints.append(mypoint)
      centers.append( np.mean(mypoints, axis=0) )
    return np.stack(centers, axis=0)

  def chordlength_parameterize(self, npoints=1001):
    """
    Reparameterize the centerlines to approximate arc length.
    """
    xi = np.linspace(0, 1, npoints)
    centerlines, radii = [], []
    for cl, r in zip(self.centerlines, self.radii):
      y = cl(xi)
      x = clparam(y)
      cl_new = cl.knotvector.fit([x], y)
      r_new = r.knotvector.fit([x], r(xi))

      centerlines.append(cl_new)
      radii.append(r_new)

    return self._edit(centerlines=centerlines, radii=radii)

  @property
  def nontrivial_bifs(self):
    return tuple( bif for bif in self.bifurcations if len(bif) > 1 )

  def create_proximity_tree(self, eps=1e-5, nref=4):

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
      return self._edit( bifurcations=new_bifs ).create_proximity_tree(eps=eps, nref=nref)

    # TODO: This block is too verbose

    # match in the interior of the edges
    for i, (centerline, radius) in enumerate(zip(self.centerlines, self.radii)):
      xi = _round_array(centerline.knotvector[0].refine(nref).knots)
      x = centerline(xi)  # (len(xi), 3)

      # (len(xi), ncenterlines, 2)
      dist = np.sqrt(((endpoints[_] - x[:, _, _, :])**2).sum(-1))

      dist[:, i] = np.inf

      # get the bif that contains the current index
      for index in self.get_bif(i+1) + self.get_bif(-i-1):
        if abs(index) == (i+1): continue
        k = 0 if index < 0 else 1
        dist[:, abs(index) - 1, k] = np.inf

      j, iedge, direction = np.unravel_index(np.argmin(dist), dist.shape)
      mydist = dist[j, iedge, direction]

      if mydist > eps:
        continue

      bifindex, bif = self.get_bif( (iedge+1) * (-1) ** (direction + 1), return_index=True)
      assert i+1 not in tuple(map(int, bif))

      c0, c1 = centerline.split(0, xvals=xi[j])
      r0, r1 = radius.split(0, xvals=xi[j])

      if not np.linalg.norm(c0([xi[j]]) - endpoints[iedge, direction]) < eps:
        raise ValueError("The centerline is not close to the bifurcation")

      centerlines = self.centerlines[:i] + (c0,) + self.centerlines[i+1:] + (c1,)
      radii = self.radii[:i] + (r0,) + self.radii[i+1:] + (r1,)

      newindex = self.nedges + 1
      index_map = {i+1: newindex}

      new_bifs = tuple( tuple(index_map.get(k, k) for k in bif)
                        for bif in self.bifurcations[:bifindex] + self.bifurcations[bifindex+1:] )

      new_bifs = new_bifs + ((*bif, i+1, -newindex),)

      ret = self._edit(centerlines=centerlines,
                       radii=radii,
                       bifurcations=new_bifs)
      return ret.create_proximity_tree(eps=eps, nref=nref)

    else:
      return self

  def create_bifurcations(self):
    from aux import Bifurcation
    endpoints = self.endpoints()
    tangents = self.endpoints(dx=1)

    bifs = []
    for bif in self.nontrivial_bifs:
      biff = Bifurcation([endpoints[i] for i in bif],
                         [tangents[i] * (1 if i > 0 else -1) for i in bif],
                         indices=bif)
      bifs.append(biff)

    return tuple(bifs)

  def plot_pyvista(self, npoints=1001, **kwargs):
    return plot_artery(self, npoints=npoints, **kwargs)

  def plot_mesh(self, npoints=5, dx=0.05, plotter=None, show=True, color='tan'):
    from aux import compute_bif_input

    bifs = compute_bif_input(self)
    map_edge_endpoint_normals = {i: [None, None] for i in range(1, self.nedges + 1)}
    frames = {}

    for bif in bifs:
      for myframe, index in zip(bif.compute_frames(), bif.indices):
        frames[index] = myframe
        if index < 0:
          map_edge_endpoint_normals[abs(index)][0] = myframe[:, 0]
        else:
          map_edge_endpoint_normals[index][1] = myframe[:, 0]

    cross = ellipse(1, 1, npoints)

    if plotter is None:
      plotter = pv.Plotter()

    centers = self.bifcenters()
    plotter.add_points(centers, color='red')
    plotter.add_point_labels(centers, [str(i) for i in range(len(self.bifurcations))])

    for i, (centerline, radius) in enumerate(zip(self.centerlines, self.radii)):
      nsteps = max(int(centerline_length(centerline) / dx), 2)
      assert nsteps >= 2
      myvessel = mesh_union(*make_vessel(centerline,
                                         radius,
                                         cross,
                                         n0=map_edge_endpoint_normals[i+1][0],
                                         nfinal=map_edge_endpoint_normals[i+1][1]
                                         ).quick_sample((npoints, npoints, nsteps)), boundary=True)
      plotter.add_mesh(pyvista_mesh(myvessel), color=color,
                                               line_width=1,
                                               show_edges=True)

    if show:
      plotter.show()

    return plotter

  def truncate_edges(self, **kwargs):
    """
    Truncate the edges of the centerlines and radii at the specified
    locations. The locations are specified as a dictionary where the keys
    are the edge indices (1-based) and the values are tuples of the form
    (xi0, xi1). If either xi0 or xi1 is None, the corresponding end of the
    centerline will not be truncated.
    """
    if not kwargs:
      return self

    kwargs = {int(key): val for key, val in kwargs.items()}
    assert set(kwargs.keys()).issubset(set(range(1, self.nedges + 1)))

    centerlines, radii = list(self.centerlines), list(self.radii)

    for key, (xi0, xi1) in kwargs.items():
      if xi0 is None and xi1 is None: continue
      if xi0 is None: vals, index = [xi1], 0
      elif xi1 is None: vals, index = [xi0], 1
      else: vals, index = [xi0, xi1], 1
      centerlines[key-1] = self.centerlines[key-1].split(0, xvals=vals)[index]
      radii[key-1] = self.radii[key-1].split(0, xvals=vals)[index]

    return self._edit(centerlines=centerlines, radii=radii)

  def biftruncate(self, rfac=2.5, eps=1e-9):
    """
    Truncate the centerlines and radii at the bifurcations such that their
    distance to the bifurcation is at least rfac times the radius of the
    centerline closest to the bifurcation.

    Assumes that the centerlines are arc-length parameterized.
    """
    kwargs = dict(zip(range(1, self.nedges + 1), ([None, None] for _ in range(self.nedges))))

    endpoints = self.endpoints()

    for bif in self.nontrivial_bifs:
      points = []

      for index in bif:
        mypoint = endpoints[index]

        # don't count equal points twice
        if not any( np.allclose(my, mypoint) for my in points ):
          points.append(mypoint)

      centerpoint = np.mean(points, axis=0)
      for index in bif:
        k = 0 if index < 0 else 1
        cline, r = self.centerlines[abs(index)-1], self.radii[abs(index)-1]
        mypoint, (myradius,) = endpoints[index], r([cline.knots[0][-k]]).ravel()
        distance_center = np.linalg.norm(mypoint - centerpoint)
        radius_eff = rfac * myradius - distance_center

        if radius_eff < 0:
          continue

        mylength = centerline_length(cline)
        intv = cline.knots[0][-1] - cline.knots[0][0]
        kwargs[abs(index)][k] = cline.knots[0][-k] + \
                                radius_eff / mylength * intv * (-1 if index > 0 else 1)

    kwargs = {str(key): val for key, val in kwargs.items()
                            if not all(_val is None for _val in val)}
    for key, (a, b) in kwargs.items():
      if any( val is None for val in (a, b) ): continue
      if b < a:
        log.warning(f"Invalid truncation values for edge {key}: {a} > {b}."
                     "Truncating to ensure that the vessel is not empty.")
        mylength = centerline_length(self.centerlines[int(key)-1])
        kwargs[key] = ((b + a) / 2 - 0.2 * mylength, (b + a) / 2 + 0.2 * mylength)

    return self.truncate_edges(**kwargs)

  def take(self, indices):
    indices = tuple(map(abs, indices))
    assert set(indices).issubset(set(range(1, self.nedges + 1)))

    map_indices = dict(zip(indices, count(1)))
    map_indices = {**map_indices, **{-k: -v for k, v in map_indices.items()}}

    bifs = []
    for bif in self.nontrivial_bifs:
      mybif = tuple(map_indices[k] for k in bif if abs(k) in indices)
      if mybif:
        bifs.append(mybif)

    centerlines = [self.centerlines[index-1] for index in indices]
    radii = [self.radii[index-1] for index in indices]
    return self._edit(centerlines=centerlines,
                      radii=radii,
                      bifurcations=bifs)

  def orient(self):
    from geom import sort_ccw
    endpoints = self.endpoints()
    bifs = []
    for i, bif in enumerate(self.nontrivial_bifs):
      mypoints = np.stack([endpoints[index] for index in bif])
      if len(cKDTree(mypoints).query_pairs(0.001)):
        raise ValueError("Duplicate points in bifurcation")
      shuffle = sort_ccw([endpoints[index] for index in bif])
      bifs.append( tuple(bif[i] for i in shuffle) )
    return self._edit(bifurcations=bifs)


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

  artery = Artery([c0, c1, c2], [r, r, r]).create_proximity_tree(eps=0.01)

  foldername = "data/centerlines_csv"

  test1 = artery.from_csv(foldername)
  # test1 = artery._edit(centerlines=test1.centerlines,
  #                      radii=test1.radii,
  #                      bifurcations=())

  test1 = test1.create_proximity_tree(eps=0.05).reparameterize()
  test2 = test1.biftruncate(rfac=1, eps=1e-9)
  # test2 = test1.truncate_edges(**{str(i): [0.1, 0.9] for i in range(1, test1.nedges + 1)})

  plotter = test1.plot_mesh(show=False)
  test2.plot_mesh(plotter=plotter, color='red')

  import ipdb
  ipdb.set_trace()

  test2.plot_mesh()

  # test1.plot_pyvista(npoints=1001)
  plot_frames(test1)
