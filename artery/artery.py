from pyhaemo.util import np, clparam
from pyhaemo.types import Immutable, Int
from pyhaemo.spl import NDSpline, UnivariateKnotVector
from pyhaemo.geo.interp import linear_interpolation
from pyhaemo.log import logger as log

from datahandler import csv_converter, swc_converter
from aux import centerline_length
from plot import plot_frames, plot_artery
from connectivity import _create_proximity_tree

from itertools import chain, count
from functools import cached_property
from typing import Tuple, Sequence
import os

from scipy.spatial import cKDTree


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
    assert all( all(-i not in bif for i in bif) for bif in self.bifurcations )

  @cached_property
  def all_indices(self):
    return tuple(range(-self.nedges, 0)) + tuple(range(1, self.nedges + 1))

  def normalize(self):
    lengths = [ centerline_length(cline) for cline in self.centerlines ]
    mean = np.mean(lengths)
    return self._edit(centerlines=[cline / mean for cline in self.centerlines],
                      radii=[r / mean for r in self.radii])

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

  def create_proximity_tree(self, *args, **kwargs):
    return _create_proximity_tree(self, *args, **kwargs)

  def plot_pyvista(self, npoints=1001, **kwargs):
    return plot_artery(self, npoints=npoints, **kwargs)

  def plot_mesh(self, *args, **kwargs):
    from plot import plot_mesh
    return plot_mesh(self, *args, **kwargs)

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
      # if bif == (-14, -15, 22, -24):
      #   import ipdb
      #   ipdb.set_trace()
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
        cline = self.centerlines[int(key)-1]
        intv = cline.knots[0][-1] - cline.knots[0][0]
        mylength = centerline_length(cline)
        kwargs[key] = ((b + a) / 2 - 0.2 * mylength * intv,
                       (b + a) / 2 + 0.2 * mylength * intv)

    return self.truncate_edges(**kwargs)

  def bifcut(self, eps=1e-9, fac=.05):
    assert 0 < fac < 1
    endpoints = self.endpoints()
    kwargs = {}
    for bif in self.nontrivial_bifs:
      mypoints = np.stack([endpoints[index] for index in bif])
      for i, point in zip(bif, mypoints):

        kwargs.setdefault(abs(i), [None, None])

        if np.sort( np.linalg.norm(mypoints - point, axis=1) )[1] < eps:
          myknots = self.centerlines[abs(i)-1].knots[0]
          if i < 0:
            kwargs[abs(i)][0] = myknots[0] + fac * (myknots[-1] - myknots[0])
          else:
            kwargs[abs(i)][1] = myknots[-1] - fac * (myknots[-1] - myknots[0])
    return self.truncate_edges(**{str(k): v for k, v in kwargs.items()})

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
      if len(cKDTree(mypoints).query_pairs(0.0001)):
        raise ValueError("Duplicate points in bifurcation")
      shuffle = sort_ccw([endpoints[index] for index in bif])
      bifs.append( tuple(bif[i] for i in shuffle) )

    return self._edit(bifurcations=bifs)

  def delete_edge(self, iedge):
    assert 1 <= iedge <= self.nedges
    centerlines = self.centerlines[:iedge] + self.centerlines[iedge+1:]
    radii = self.radii[:iedge] + self.radii[iedge+1:]

    bifs = []
    for bif in self.nontrivial_bifs:
      bifs.append(tuple((index if index < iedge else index - 1) for index in bif if index != iedge))

    return self._edit(centerlines=centerlines,
                      radii=radii,
                      bifurcations=bifs)


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
