import numpy as np
from pyhaemo.util import np, _, clparam, normalize, frozen
from pyhaemo.types import Immutable
from pyhaemo.geo.notwist import compute_notwistframe

from typing import Optional

import matplotlib.pyplot as plt


def distance(arr0: np.ndarray, arr1: np.ndarray) -> np.ndarray:
  shp0, shp1 = arr0.shape, arr1.shape
  assert shp0[-1] == shp1[-1]
  # (2, 3, 3) => (1, 1, 1, 2, 3, 3)
  arr0 = arr0[(_,) * len(arr1.shape[:-1])]
  arr1 = arr1[(slice(None),) * len(arr0.shape[:-1]) + (_,) * len(arr1.shape[:-1])]

  return np.sqrt(((arr0 - arr1)**2).sum(-1))


class Plane(Immutable):
  def __init__(self, normal: np.ndarray, point: Optional[np.ndarray] = None):
    self.normal = frozen(normalize(normal), dtype=float)
    if point is None:
      point = [0, 0, 0]
    self.point = frozen(point, dtype=float)
    assert self.point.shape == self.normal.shape == (3,)

  def project(self, points: np.ndarray) -> np.ndarray:
    """
    Projects points onto the plane defined by the normal vector and point.
    Works both for a single point and a batch of points.
    """
    points = np.asarray(points)
    ndims = points.ndim - 1

    return points - ((points - self.point) @ self.normal)[(slice(None),) * ndims + (_,)] * self.normal


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

  def orient(self):
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

    return self._edit( points=self.points[indices],
                       tangents=self.tangents[indices],
                       indices=tuple(self.indices[i] for i in indices),
                       normal=self.normal )

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


def find_clock_wise_order(points, tangents, bif):
    """
    Determines the clockwise order of a set of points and tangents around a bifurcation point.
    Parameters:
        points (list or np.ndarray): A list or array of 3D points to be ordered.
        tangents (list or np.ndarray): A list or array of tangent vectors corresponding to the points.
        bif (tuple): A tuple containing bifurcation-related data.
    Returns:
        tuple: A tuple containing:
            - normal (np.ndarray): The normal vector of the plane used for projection.
            - points_clockwise (np.ndarray): The points ordered in clockwise direction.
            - tangents_clockwise (np.ndarray): The tangents ordered in clockwise direction.
            - bif_clockwise (tuple): The bifurcation data reordered to match the clockwise order.
    """

    # XXX: this method will fail when two of the tangents are (close to) parallel
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)

    arrows = normalize(points - points.sum(0) / len(points))

    normal = compute_orientation(arrows)

    # Project points onto the plane
    projected_points = []
    for point in points:
      p = project_onto_plane(point, normal)
      projected_points.append(p)

    projected_points = np.array(projected_points).reshape(len(tangents), -1)
    vectors = normalize(projected_points - np.mean(projected_points, axis=0))

    angles = [0]
    for i in range(1, len(vectors)):
      cross = normalize(np.cross(vectors[0], vectors[i]))
      dots = np.sum(vectors[0] * vectors[i])
      scalar_prod = (cross * normal).sum()
      angles.append((+1 if scalar_prod > 0 else -1) * np.arccos(dots))

    angles = np.array(angles)
    index = np.argsort(angles)

    angles = angles[index]
    points_ccw = np.array(points)[index]
    tangents_ccw = np.array(tangents)[index]
    bif_ccw = tuple( bif[i] for i in index )

    return normal, points_ccw, tangents_ccw, bif_ccw


def compute_frames(normal_bif, tangents, points):
    """
    Computes a series of frames for a bifurcation geometry based on the provided
    normal vector, tangents, and points.
    Parameters:
    -----------
    normal_bif : numpy.ndarray
        The normal vector at the bifurcation point.
    tangents : list of numpy.ndarray
        A list of tangent vectors corresponding to the bifurcation branches.
    points : list of numpy.ndarray
        A list of points defining the bifurcation geometry.
    Returns:
    --------
    frames : list of numpy.ndarray
        A list of 3x3 frame matrices, where each frame is defined by three
        orthonormal vectors (normal, binormal, tangent) for each branch of the
        bifurcation.
    """
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)

    assert points.shape == tangents.shape and points.shape[1:] == (3,)

    normal_proj = normalize(project_onto_plane(normal_bif, tangents[0]))
    binormal = normalize(np.cross(tangents[0], normal_proj))

    Fr1 = np.stack((normal_proj, binormal, tangents[0]), axis=1)

    frames = [Fr1]

    for tangent in tangents[1:]:
      tangents_ = np.stack((tangents[0], -tangent), axis=0)
      frame = compute_notwistframe(tangents_,
                                   n0=normal_proj)

      frames.append(frame[-1] @ np.diag([1, -1, -1]))

    # frames = compute_notwistframe(tangents,
    #                               n0=normal_proj)

    return frames


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


# todo: add a test when the vector are on xy plane, the normal has to be (0,0,1)
def compute_orientation(tangents):
    """
    Compute the orientation for a general number of tangents.

    Args:
        tangents (list of np.ndarray): List of tangent vectors for each branch.

    Returns:
        dict: A dictionary containing the computed normal and combined tangent.
    """
    if len(tangents) < 2:
        raise ValueError("At least two tangents are required to compute orientation.")

    # Normalize the tangents
    tangents = [normalize(tangent) for tangent in tangents]

    # Compute the normal as a mix of the tangents
    normal = np.zeros_like(tangents[0])

    for i in range(len(tangents) - 1):
        cross_product = np.cross(tangents[0], tangents[i + 1])
        normal += cross_product

    return normalize(normal)


def plot_bifurcation(artery):
    """
    Plot the bifurcation geometry for a given artery and bifurcation configuration.

    Parameters:
    ----------
    artery : object
        An object representing the artery geometry. It should provide the `compute_bif_input` function.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the bifurcation. If None, a new figure and axes will be created.

    Returns:
    -------
    ax : matplotlib.axes.Axes
        The axes on which the bifurcation is plotted.
    """

    # Compute the frames, points, tangents, and bifurcation configuration
    all_frames, all_points, _, _, _ = compute_bif_input(artery)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Plot the bifurcation geometry
    for myframes, mypoints in zip(all_frames, all_points):
      for frame, point in zip(myframes, mypoints):

        # Plot the point itself
        ax.scatter(*point, color='red', s=50)

        # Plot the frame at each point using the frame's columns as vectors
        for j, vec in enumerate(frame.T):
            ax.quiver(*point, *vec,
                      color=['blue', 'green', 'orange'][j],  # Different colors for each axis
                      length=1.0, normalize=True)

    plt.show()
