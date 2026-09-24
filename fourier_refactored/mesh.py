import numpy as np

from pyhaemo.geo import compute_notwistframe_from_spline, ellipse
from pyhaemo.mesh import rectilinear, mesh_union
from pyhaemo.mesh.mesh import LineMesh
from pyhaemo.spl import UnivariateKnotVector
from pyhaemo.util import np, _, clparam


def min_edge_length(mesh):
    """
    h_min: the shortest edge length across every element of `mesh`, in the
    same units as its points (mesh.vessel() works in meters). Useful for a
    CFL-based timestep estimate. Same edge decomposition as
    pyhaemo.mesh.qual.aspect_ratio, but returns the global minimum distance
    instead of a per-element ratio.
    """

    assert mesh.is_valid(), "mesh is not valid"

    elements = mesh.elements[:, mesh._submesh_indices]
    if mesh._submesh_type is not LineMesh:
        elements = elements[:, :, mesh.submesh._submesh_indices]

    elem_point = mesh.points[elements]
    distances = np.linalg.norm(elem_point[..., 1, :] - elem_point[..., 0, :], axis=-1)

    return float(distances.min())


# How to use it: #export.export_gmsh(mesh, filename_mesh, surface_tags= selecters)
def create_selecters(X, vessel):
    
    xin = X([0]).ravel()
    xout = X([1]).ravel()    
    points = np.squeeze(vessel([.5], [.5], [0]))
    tin = np.cross(points[0] - points[1], points[0] - points[2])
    points = np.squeeze(vessel([.5], [.5], [1]))
    tout = np.cross(points[0] - points[1], points[0] - points[2])
    
    selecters = {}
    def fin(points, xin=xin, tin=tin):
        return (np.abs(((points - xin[_, _]) * tin[_, _]).sum(-1)) < 1e-12).all(-1)
    def fout(points, xout=xout, tout=tout):
        return (np.abs(((points - xout[_, _]) * tout[_, _]).sum(-1)) < 1e-12).all(-1)
    
    selecters['in'] = lambda points: np.where(fin(points))[0]
    selecters['out'] = lambda points: np.where(fout(points))[0]
    selecters['surface'] = lambda points: np.where((~fin(points)) & (~fout(points)))[0]
    
    return selecters


def vessel(nelems_centerline,
           nelems_cross_section,
           nelems_z,
           radii,
           centerline_points,
           normal=None,
           export=False,
           filename='vessel_mesh.msh',
           boundary_layer_exponent=1.):

  """
  Make a 5-patch spline of a vessel and convert to mesh for visualization.

  Parameters
  ----------
  nelems_centerline : :class:`int`
      The number of spline elements to use for the knotvector that fits
      the centerline.
  nelems_cross_section : :class:`int`
      The number of elements of the five cross section patches in each
      direction.
  radii : :class:`np.ndarray`
      Radius information at the centerline points.
      Must be of shape centerline_points.shape[:1] (and stritly positive).
  centerline_points : :class:`np.ndarray`
      The centerline point information. Must be of shape (npoints, 3).
  boundary_layer_exponent : :class:`float`
      Forwarded to :func:`ellipse`. Values other than 1 concentrate the
      cross-section grid points toward the circular boundary.

  Version has been adapted to make use of the NDSplineArray class.
  """

  # make a knotvector with the specified number of elements
  kv = UnivariateKnotVector(np.linspace(0, 1, nelems_centerline + 1)).to_tensor()

  assert centerline_points.shape[1:] == (3,)
  assert radii.shape == centerline_points.shape[:1]
  assert (radii > 0).all()

  # compute the chord length parameter values of the centerline points
  xi = clparam(centerline_points)

  # fit a univariate spline to the data
  X = kv.fit([xi], centerline_points)

  # compute the notwistframe rotation matrices
  framekargs = {'n0': normal}
  Rs = compute_notwistframe_from_spline(X, xi, framekwargs=framekargs)

  # make the cross section of radius one
  disc = ellipse(1, 1, nelems_cross_section,
                  boundary_layer_exponent=boundary_layer_exponent)

  # fit a spline to radius and rotational frame information
  rRs = kv.fit([xi], radii[:, _, _] * Rs)

  # create the vessel spline using a tensor product
  # disc.shape == (5, 3) 5 patches 3 coordinates
  # rRs.shape == (3, 3) 3x3 matrix
  # disc[:, _] * rRs[_] = (5, 1, 3) * (1, 3, 3)
  # taking .sum(-1) => (5, 3), representing a matrix multiplication
  vessel = (disc[:, _] @ rRs[_]).sum(-1) + (disc.unity @ X)

  # create dense mesh
  eval_mesh = rectilinear([nelems_cross_section,
                           nelems_cross_section,
                                        nelems_z])

  # create a mesh sampled from each patch and take its boundary union
  # to provide one hexmesh for the entire geometry.
  mesh = mesh_union(*vessel.sample_mesh(eval_mesh), boundary=True)

  selecters = create_selecters(X, vessel)

  if export:
    mesh.export_gmsh(filename,surface_tags=selecters)

  print(f'The mesh is valid ?: {mesh.is_valid()}.')

  return mesh
