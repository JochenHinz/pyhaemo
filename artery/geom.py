from pyhaemo.util import np, normalize
from pyhaemo.geo.notwist import compute_notwistframe

from typing import Sequence

from numpy.typing import NDArray


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
  center = points.mean(axis=0)

  # Estimate normal using PCA
  u, s, vh = np.linalg.svd(points - center)
  normal = vh[2]  # smallest principal component = normal to plane

  return normalize(normal)


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
