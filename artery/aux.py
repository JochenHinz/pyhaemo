from pyhaemo.util import np, _, normalize, frozen
from pyhaemo.types import Immutable
from pyhaemo.geo.notwist import compute_notwistframe
from pyhaemo.spl import NDSpline
from pyhaemo.log import logger as log

from itertools import combinations

from typing import Sequence

from numpy.typing import NDArray


def distance(arr0: np.ndarray, arr1: np.ndarray) -> np.ndarray:
  shp0, shp1 = arr0.shape, arr1.shape
  assert shp0[-1] == shp1[-1]
  # (2, 3, 3) => (1, 1, 1, 2, 3, 3)
  arr0 = arr0[(_,) * len(arr1.shape[:-1])]
  arr1 = arr1[(slice(None),) * len(arr0.shape[:-1]) + (_,) * len(arr1.shape[:-1])]

  return np.sqrt(((arr0 - arr1)**2).sum(-1))


def project_onto_plane(points, normal):
  """
  Project 3D points onto a plane defined by a normal vector.

  Args:
      points (np.ndarray): Array of 3D points with shape (n, 3).
      normal (np.ndarray): Normal vector of the plane.

  Returns:
      np.ndarray: Projected 2D points.
  """
  normal = normalize(normal)
  projection_matrix = np.eye(3) - np.outer(normal, normal)
  return points @ projection_matrix.T


def estimate_normal(points: NDArray | Sequence[NDArray]) -> NDArray:
  """
  Estimate the normal vector of a set of points in 3D space.
  The function computes the centroid of the points, estimates the normal
  vector of the plane defined by the points, and returns it.
  """
  points = np.asarray(points)
  points = points[np.lexsort(points.T)]  # use lexsort ordering to make independent of the order

  center = points.mean(axis=0)

  # Estimate normal using PCA
  u, s, vh = np.linalg.svd(points - center)
  normal = vh[-1]  # smallest principal component = normal to plane

  return normalize(normal)


# todo: add a test when the vector are on xy plane, the normal has to be (0,0,1)
def compute_orientation(tangents):
  """
  Compute the orientation for a general number of tangents.

  Args:
      tangents (list of np.ndarray): List of tangent vectors for each branch.

  Returns:
      dict: A dictionary containing the computed normal and combined tangent.
  """
  # Normalize the tangents
  tangents = normalize(tangents)

  if len(tangents) == 1:
    # TODO: think about a linear operation that takes the vector (0, 0, 1)
    #       to (1, 0, 0)
    (a, b, c), = tangents
    return normalize(np.array([1, 1, -(b + a) / c])[_])

  if len(tangents) < 2:
      if np.allclose(np.abs(np.dot(*tangents)), 1):
        log.warning("The two tangents are parallel. Returning a default normal.")
      return compute_orientation(tangents[:1])

  return normalize( np.cross(tangents[0], tangents[1:]).sum(0) )


class Bifurcation(Immutable):

  def __init__(self, points, tangents, indices, normal=None):
    self.points = frozen(points, dtype=float)
    self.tangents = frozen(tangents, dtype=float)
    self.indices = tuple(map(int, indices))

    assert self.points.shape == self.tangents.shape and \
           self.points.shape[1:] == (3,)
    assert self.points.shape[0] == len(self.indices) >= 2

    if normal is None:
      normal = compute_orientation(self.points - self.points.sum(0) / len(self.points))

    self.normal = frozen(normalize(normal), dtype=float)

    assert self.normal.shape == (3,)

  def __len__(self):
    return len(self.points)

  def orient(self, return_indices=False):
    ppoints = []
    for point in self.points:
      ppoints.append( project_onto_plane(point, self.normal) )

    ppoints = np.array(ppoints).reshape(len(self), -1)
    vectors = normalize(ppoints - np.mean(ppoints, axis=0))

    angles = [0]
    for i in range(1, len(vectors)):
      cross = normalize(np.cross(vectors[0], vectors[i]))
      dots = np.sum(vectors[0] * vectors[i])
      scalar_prod = (cross * self.normal).sum()
      angles.append((+1 if scalar_prod > 0 else -1) * np.arccos(dots))

    indices = np.argsort(angles)

    ret = self._edit( points=self.points[indices],
                      tangents=self.tangents[indices],
                      indices=tuple(self.indices[i] for i in indices),
                      normal=self.normal )
    if return_indices:
      return ret, tuple(self.indices[i] for i in indices)

    return ret

  def compute_frames(self):
    t0 = self.tangents[0]
    normal_proj = normalize(project_onto_plane(self.normal, t0))
    binormal = normalize(np.cross(t0, self.normal))

    Fr1 = np.stack((normal_proj, binormal, t0), axis=1)

    frames = [Fr1]

    for tangent in self.tangents[1:]:
      tangents_ = np.stack((t0, -tangent), axis=0)
      frame = compute_notwistframe(tangents_,
                                   n0=normal_proj)

      frames.append(frame[-1] @ np.diag([1, -1, -1]))

    return tuple(frames)


def compute_bif_input(artery):
  """
  Computes the frames, points, tangents, and bifurcation order for a given bifurcation.
  Parameters:
  -----------
  lines : list of callable
      A list of functions representing the parametric curves of the lines.
      Each function takes a single parameter (e.g., `eval`) and returns the
      corresponding point on the curve.
  radii : list of callable
      A list of functions representing the radii of the lines. Each function
      takes a single parameter (e.g., `[k]`) and returns the radius at that
      point.
  bif : iterable
      An iterable of integers representing the bifurcation configuration.
      Positive integers indicate forward direction, and negative integers
      indicate reverse direction.
  Returns:
  --------
  frames : list
      A list of computed frames for the bifurcation.
  points_clockwise : list
      A list of points ordered in a clockwise manner around the bifurcation.
  tangents_clockwise : list
      A list of tangent vectors ordered in a clockwise manner around the bifurcation.
  bif_clockwise : list
      A list of bifurcation indices ordered in a clockwise manner.
  Notes:
  ------
  - The function assumes a shape of (, , , ) and processes the bifurcation
    configuration accordingly.
  - The `find_clock_wise_order` function is used to determine the clockwise
    order of points, tangents, and bifurcation indices.
  - The `compute_frames` function is used to compute the frames based on the
    normal vector, points, and tangents.
  """
  centerlines = artery.centerlines
  bifurcations = artery.nontrivial_bifs

  bifs = []

  for bifindices in bifurcations:
    mytangents, mypoints = [], []

    for val in bifindices:
      k = 0 if val < 0 else 1
      myline = centerlines[abs(val) - 1]
      startpoint = myline.knots[0][-k]

      # by convention we have the tangent point into the bifurcation
      mytangents.append(normalize(myline([startpoint], dx=1).ravel() * (1 if val > 0 else -1)))
      mypoints.append(myline([startpoint]).ravel())

    bif = Bifurcation(mypoints, mytangents, bifindices).orient()
    bifs.append(bif)

  return tuple(bifs)


def centerline_length(centerline: NDSpline) -> float:
  """
  Compute the length of a centerline spline using the Gauss points.
  """
  xi, = centerline.knotvector.gauss_abscissae
  assert len(xi) > 1
  x = centerline(xi)
  return np.linalg.norm(x[1:] - x[:-1], axis=1).sum()


def sort_ccw(points: NDArray | Sequence[NDArray]) -> NDArray:
  """
  Sort points in counter-clockwise order around their centroid.
  The function computes the centroid of the points, estimates the normal
  vector of the plane defined by the points, and then sorts the points
  based on their angles relative to the centroid in the plane.
  """
  points = np.asarray(points, dtype=float)
  normal = estimate_normal(points)

  # Create orthonormal basis (u, v) in the plane
  u = np.cross([1, 0, 0], normal)
  if np.linalg.norm(u) < 1e-5:
    u = np.cross([0, 1, 0], normal)
  u = normalize(u)
  v = np.cross(normal, u)

  # Project points into 2D plane and compute angles
  rel = points - np.mean(points, axis=0)
  x = rel @ u
  y = rel @ v
  angles = np.arctan2(y, x)

  return np.argsort(angles)


def compute_frames(tangents: NDArray | Sequence[NDArray], normal) -> NDArray:
  t0 = tangents[0]
  normal_proj = normalize(project_onto_plane(normal, t0))
  binormal = normalize(np.cross(t0, normal))

  frames = [np.stack((normal_proj, binormal, t0), axis=1)]

  for tangent in tangents[1:]:
    tangents_ = np.stack((t0, -tangent), axis=0)
    frame = compute_notwistframe(tangents_,
                                 n0=normal_proj)

    frames.append(frame[-1] @ np.diag([1, -1, -1]))

  return np.stack(frames)


def take_lower_triangular(arr: NDArray) -> NDArray:
  indices = np.tril_indices_from(arr, k=-1)
  return arr[indices]


def biftruncate_(artery, ibif, rfac=2, npoints=201, nref=2):
  """
  Truncate the centerlines in a way that all the distances of the vessel
  to one another are at least rfac times the radii.

  Also ensure that the centerlines are at a distance rfac times the radius
  from the bifurcation center.
  """
  bif = artery.bifurcations[ibif]
  assert len(bif) > 1

  centerlines = [artery.centerlines[abs(i) - 1] for i in bif]
  radii = [artery.radii[abs(i) - 1] for i in bif]
  evalpoints = [np.linspace(*cl.knots[0][[0, -1]], npoints+2)[1:-1]
                            [slice(None) if i < 0 else slice(None, None, -1)]
                            for i, cl in zip(bif, centerlines)]

  # (npoints, ncenterlines, 3)
  ceval = np.stack([cl(evalf) for cl, evalf in zip(centerlines, evalpoints)], axis=1)
  r0s = np.array([r([r.knots[0][0 if i < 0 else -1]]).ravel()
                  for i, r, evalf in zip(bif, radii, evalpoints)], dtype=float)
  maxr = r0s.max()
  mean = np.mean(ceval[0], axis=0)

  i = 0
  while i < npoints:

    points = ceval[i]
    d = ((points[:, _] - points[_])**2).sum(-1) ** .5

    if not (take_lower_triangular(d) > 2 * maxr).all() or \
       not (((points - mean)**2).sum(-1)**.5 > 2 * maxr).all():
      i += 1
      continue
    else:
      break

  else:
    raise ValueError("No points satisfy the distance condition")

  truncation_points = [points[i] for points in evalpoints]

  kwargs = {}
  for i, mypoint, myspline in zip(bif, truncation_points, centerlines):
    myknots = myspline.knotvector[0].refine(nref).knots[1: -1]
    xi = myknots[np.abs( mypoint - myknots ).argmin()]
    if i < 0:
      kwargs[str(abs(i))] = (xi, None)
    else:
      kwargs[str(abs(i))] = (None, xi)

  return artery.truncate_edges(**kwargs)


def _biftruncate(artery, ibif, rfac=2, npoints=201, nref=2):
  """
  Truncate the centerlines in a way that all the distances of the vessel
  to one another are at least rfac times the radii.

  Also ensure that the centerlines are at a distance rfac times the radius
  from the bifurcation center.
  """
  bif = artery.bifurcations[ibif]
  assert len(bif) > 1

  centerlines = [artery.centerlines[abs(i) - 1] for i in bif]
  radii = [artery.radii[abs(i) - 1] for i in bif]
  evalpoints = [np.linspace(*cl.knots[0][[0, -1]], npoints+2)[1:-1]
                            [slice(None) if i < 0 else slice(None, None, -1)]
                            for i, cl in zip(bif, centerlines)]

  # (npoints, ncenterlines, 3)
  ceval = np.stack([cl(evalf) for cl, evalf in zip(centerlines, evalpoints)], axis=1)

  # (ncenterlines,)
  r0s = np.array([r([r.knots[0][0 if i < 0 else -1]]).ravel()
                  for i, r, evalf in zip(bif, radii, evalpoints)], dtype=float)
  maxr = r0s.max()
  mean = np.mean(ceval[0], axis=0)

  i = 0
  while i < npoints:

    points = ceval[i]

    # (ncenterlines, ncenterlines)
    d = ((points[:, _] - points[_])**2).sum(-1) ** .5

    mutual_distances = take_lower_triangular(d)
    distance_center = ((points - mean)**2).sum(-1)**.5

    if not (mutual_distances > rfac * maxr).all() or \
       not (distance_center > rfac * maxr).all():
      i += 1
    else:
      break

  else:
    raise ValueError("No points satisfy the distance condition")

  truncation_points = [points[i] for points in evalpoints]

  kwargs = {}
  for i, mypoint, myspline in zip(bif, truncation_points, centerlines):
    # myknots = myspline.knotvector[0].refine(nref).knots[1: -1]
    # xi = myknots[np.abs( mypoint - myknots ).argmin()]
    xi = mypoint
    if i < 0:
      kwargs[str(abs(i))] = (xi, None)
    else:
      kwargs[str(abs(i))] = (None, xi)

  return artery.truncate_edges(**kwargs)


def find_distance(artery, subset, centroid, rfac=2, npoints=201):
  """
  Given a subset of length 2 of a bifurcation.

  It finds the points of the centerlines that have a certain distance.
  """
  assert len(subset) == 2

  clines = [artery.centerlines[abs(val) - 1] for val in subset]
  radii = [artery.radii[abs(val) - 1] for val in subset]

  evalpoints = [np.linspace(*cl.knots[0][[0, -1]], npoints+2)[1:-1]
                            [slice(None) if i < 0 else slice(None, None, -1)]
                            for i, cl in zip(subset, clines)]

  # (npoints, 2, 3)
  ceval = np.stack([cline(evalf) for cline, evalf in zip(clines, evalpoints)], axis=1)
  reval = np.stack([r(evalf) for r, evalf in zip(radii, evalpoints)], axis=1)

  i, j = 0, 0
  while i < npoints and j < npoints:
    p0, p1 = ceval[i][0], ceval[j][1]
    maxr = max(reval[i][0], reval[j][1])

    dist = np.linalg.norm(p0 - p1)
    if dist < rfac * maxr:
      i += 1
      j += 1
      continue

    distance_center = ((np.stack([p0, p1], axis=0) - centroid)**2).sum(-1)**.5

    if (distance_center > rfac * maxr).all():
      break

    if distance_center[0] < rfac * maxr:
      i = i + 1

    if distance_center[1] < rfac * maxr:
      j = j + 1

  else:
    raise ValueError("No points satisfy the distance condition")

  return {subset[0]: evalpoints[0][i],
          subset[1]: evalpoints[1][j]}


def biftruncate(artery, rfac=2, npoints=201):

  endpoints = artery.endpoints()
  kwargs = dict(zip(range(1, artery.nedges + 1), ([None, None] for _ in range(artery.nedges))))

  for bif in artery.nontrivial_bifs:

    centroid = np.mean( [endpoints[i] for i in bif], axis=0 )

    param_list = []
    for myset in combinations(bif, 2):
      param = find_distance(artery, myset, centroid, rfac=rfac, npoints=npoints)
      param_list.append(param)

    for key in bif:
      values = [d[key] for d in param_list if key in d]
      if key > 0:
        kwargs[abs(key)][1] = min(values)
      else:
        kwargs[abs(key)][0] = max(values)

  _kwargs = {str(key): val for key, val in kwargs.items()}
  kwargs = {}
  for k, v in _kwargs.items():
    if any(_v is None for _v in v) or v[0] < v[1]:
      kwargs[k] = v
    else:
      mean = np.mean(v)
      kwargs[k] = (mean - 0.1, mean + 0.1)

  try:
    return artery.truncate_edges(**kwargs)
  except Exception:
    import ipdb
    ipdb.set_trace()
    return artery.truncate_edges(**kwargs)
