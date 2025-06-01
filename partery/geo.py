from pyhaemo.util import np, normalize

from typing import Optional

from numpy.typing import ArrayLike, NDArray


ZERO_THRESH = 1e-8


def estimate_normal(points: ArrayLike, centroid=None) -> NDArray:
  """
  Estimate the normal vector of a set of points in 3D space.
  The function computes the centroid of the points, estimates the normal
  vector of the plane defined by the points, and returns it.
  """
  points = np.asarray(points)

  # use lexsort ordering to make independent of the order
  points = points[np.lexsort(points.T)]

  if centroid is None:
    centroid = points.mean(axis=0)

  # Estimate normal using PCA
  u, s, vh = np.linalg.svd(points - centroid)
  normal = vh[-1]  # smallest principal component = normal to plane

  return normalize(normal)


def sort_ccw(points: ArrayLike, centroid: Optional[ArrayLike] = None,
                                normal: Optional[ArrayLike] = None) -> NDArray:
  """
  Sort points in counter-clockwise order around their centroid.
  The function computes the centroid of the points, estimates the normal
  vector of the plane defined by the points, and then sorts the points
  based on their angles relative to the centroid in the plane.
  """
  points = np.asarray(points, dtype=float)

  if centroid is None:
    centroid = np.mean(points, axis=0)

  if normal is None:
    normal = estimate_normal(points, centroid=centroid)

  centroid, normal = map(np.asarray, (centroid, normal))

  # Create orthonormal basis (u, v) in the plane
  u = np.cross([1, 0, 0], normal)
  if np.linalg.norm(u) < 1e-5:
    u = np.cross([0, 1, 0], normal)
  u = normalize(u)
  v = np.cross(normal, u)

  # Project points into 2D plane and compute angles
  rel = points - centroid
  x = rel @ u
  y = rel @ v
  angles = np.arctan2(y, x)

  return np.argsort(angles)


def project_onto_plane(vectors: ArrayLike, normal: ArrayLike) -> NDArray:
  """
  Project 3D points onto a plane defined by a normal vector.

  Args:
      points (np.ndarray): Array of 3D points with shape (n, 3).
      normal (np.ndarray): Normal vector of the plane.

  Returns:
      np.ndarray: Projected 2D points.
  """
  normal = normal / np.linalg.norm(normal)
  projection_matrix = np.eye(3) - np.outer(normal, normal)
  return vectors @ projection_matrix.T
