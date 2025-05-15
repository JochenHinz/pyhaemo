import numpy as np
from pyhaemo.util import np, _, clparam, normalize
from pyhaemo.geo.notwist import compute_notwistframe


def distance(arr0: np.ndarray, arr1: np.ndarray) -> np.ndarray:
  shp0, shp1 = arr0.shape, arr1.shape
  assert shp0[-1] == shp1[-1]
  # (2, 3, 3) => (1, 1, 1, 2, 3, 3)
  arr0 = arr0[(_,) * len(arr1.shape[:-1])]
  arr1 = arr1[(slice(None),) * len(arr0.shape[:-1]) + (_,) * len(arr1.shape[:-1])]

  return np.sqrt(((arr0 - arr1)**2).sum(-1))


def compute_frame(T, normal_first_slice, starting_frame=None):

    T = normalize(T)

    zero_thresh = 1e-8
    assert zero_thresh >= 0

    normals = np.zeros(T.shape, dtype=np.float64)
    binormals = np.zeros(T.shape, dtype=np.float64)

    t0 = T[0]

    if starting_frame is not None and normal_first_slice is None:
        normals[0] = normalize(starting_frame[:, 1])
        binormals[0] = normalize(starting_frame[:, 0])
    if starting_frame is None and normal_first_slice is not None:
        normals[0] = normalize(normal_first_slice)
        binormals[0] = normalize((np.cross(t0, normals[0])))

    i = 1
    for told, tnew in zip(T, T[1:]):
      omega = np.cross(told, tnew)
      norm = (omega**2).sum()**.5
      if norm <= zero_thresh:
        binormal = binormals[i-1]
        normal = normals[i-1]
      else:
        omega = omega / norm
        phi = np.arccos(np.dot(told, tnew))
        binormal = (binormals[i-1] * np.cos(phi) + \
                  np.cross(omega, binormals[i-1]) * np.sin(phi) + \
                  omega * np.dot(omega, binormals[i-1]) * (1 - np.cos(phi)))
        normal = (normals[i-1] * np.cos(phi) + \
                np.cross(omega, normals[i-1]) * np.sin(phi) + \
                omega * np.dot(omega, normals[i-1]) * (1 - np.cos(phi)))

      binormals[i] = normalize(binormal)
      normals[i] = normalize(normal)
      i += 1

    Rs = np.stack([ np.stack([n, bn, t], axis=1) for n, bn, t
                      in zip(binormals, normals, T) ], axis=0)

    assert np.allclose(np.linalg.det(Rs), 1)

    return Rs


def compute_bif_input(artery, rfac=5):
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
    radii = artery.radii
    bifurcations = artery.nontrivial_bifs

    tangents_ccw, points_ccw, frames_ccw, bif_ccw = [], [], [], []

    for bif in bifurcations:
      mytangents, mypoints = [], []

      for val in bif:
        k = 0 if val < 0 else 1

        myradius = radii[abs(val) - 1]
        myline = centerlines[abs(val) - 1]

        xi = np.linspace(*myradius.knotvector.knots[0][[0, -1]], 1000)
        startpoint = xi[-k]  # startpoint of the centerline

        myradius = myradius([startpoint])
        evalpoints = myline(xi)

        d = ((evalpoints - evalpoints[-k][_])**2).sum(1) ** .5

        myparam = xi[ np.argmin(np.abs(d - rfac * myradius)) ]
        mytangents.append( normalize(myline([myparam], dx=1).ravel() * (1 if val > 0 else -1)) )
        mypoints.append( myline([myparam]) )

      normal, points_ordered, tangents_ordered, bif_ordered = \
          find_clock_wise_order(mypoints, mytangents, bif)

      points_ccw.append(points_ordered)
      tangents_ccw.append(tangents_ordered)
      bif_ccw.append(bif_ordered)
      frames_ccw.append(compute_frames(normal, tangents_ordered, points_ordered))

    return tuple(map(tuple, (frames_ccw, points_ccw, tangents_ccw, bif_ccw)))


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
    normal_proj = normalize(project_onto_plane(normal_bif, tangents[0]))
    binormal = normalize(np.cross(tangents[0], normal_proj))

    Fr1 = np.stack((normal_proj, binormal, tangents[0]), axis=1)

    frames = [Fr1]

    for i in range(len(points)-1):
      tangents_ = np.stack((tangents[0], tangents[i+1]), axis=0)
      frame = compute_notwistframe(tangents_,
                                   n0=normal_proj)

      frame = compute_frame(tangents_, normal_proj, frame)


      frames.append(frame[-1])

    return frames


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

    normal = compute_orientation(tangents)

    # Project points onto the plane
    projected_points = []
    for i in range(len(points)):
      p = project_onto_plane(points[i], normal)
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

    # Normalize the final normal vector
    normal = normalize(normal)

    return normal
