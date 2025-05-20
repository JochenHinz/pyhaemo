from pyhaemo.util import np, _, normalize, frozen
from pyhaemo.types import Immutable
from pyhaemo.geo.notwist import compute_notwistframe
from pyhaemo.spl import NDSpline
from pyhaemo.log import logger as log


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
