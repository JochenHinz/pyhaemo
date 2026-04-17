from pyhaemo.util import np, _
from pyhaemo.spl import NDSplineArray, NDSpline
from pyhaemo.mesh import mesh_union

from bif import bif_from_matrices
from sgraph import SplineGraph
from edge import SplineEdge
from artery import VascularGraph
from fit import fit_constrained


def make_spline_mesh(sgraph: SplineGraph,
                     cross: NDSplineArray,
                     dx: float = 0.02,
                     nref: int = 3,
                     rfac: int = 2,
                     show_interfaces: bool = False,
                     **kwargs):
  """Create a mesh from a SplineGraph.

  Parameters
  ----------.
  union_eps : float, optional
    Tolerance for :func:`pyhaemo.mesh.bool.mesh_union` when gluing volumes.
  bif_exploit_symmetry : bool, optional
    Passed to :func:`bif.bif_from_matrices`.
  """
  union_eps = float(kwargs.pop("union_eps", 1e-8))
  bif_exploit_symmetry = bool(kwargs.pop("bif_exploit_symmetry", False))
  kwargs.pop("boundary_layer_exponent", None)

  xis = [ edge.knotvector[0].refine(nref).knots for edge in sgraph.edges ]
  xidict = dict(zip(sgraph.indices, xis))
  _truncate_dict = VascularGraph(sgraph.indices,
                                 [e.sample(_xi) for e, _xi in zip(sgraph.edges, xis)],
                                 sgraph.furcations)._rtruncate(rfac=rfac)
  truncate_dict = {}
  for iedge, indices in _truncate_dict.items():
    myxi = xidict[iedge]
    truncate_dict[iedge] = [ myxi[index] if index is not None else None
                             for index in indices ]

  _sgraph = sgraph.truncate_edges(truncate_dict)
  kv0 = cross.knotvector[0]
  kv1 = cross.knotvector[1]
  if not hasattr(kv0, "knotvalues") and len(kv0):
    kv0 = kv0[0]
  if not hasattr(kv1, "knotvalues") and len(kv1):
    kv1 = kv1[0]
  # Tie sampling density to the actual cross-section discretization
  # (e.g. repeated_knot_disc(nelems): points ~= nelems + 1).
  nsec0 = max(5, int(len(kv0.knotvalues)))
  nsec1 = max(5, int(len(kv1.knotvalues)))

  vessels = []
  redges = []  # SplineEdges with radius and frame fused
  vessel_meshes = []
  for i, sedge in enumerate(_sgraph.edges, 1):
    kv = sedge.knotvector
    assert -1 not in kv[0].continuity[1:-1]
    cline, radius = sedge.rcenterline[:3], sedge.rcenterline[3]
    frame = sedge.frame
    evl = kv[0].refine(nref).knots
    rR = fit_constrained(kv, evl, radius(evl)[:, _, _] * frame(evl))
    redges.append(sedge._edit(frame=rR))
    vessels.append( (cross[:, _] @ rR[_]).sum(-1) + (cross.unity @ cline) )
    npoints = max(2, int(sedge.length / dx))
    vessel_meshes.append(
      mesh_union(*vessels[-1].quick_sample((nsec0, nsec1, npoints)).ravel(),
                 boundary=True, eps=union_eps))

  sgraph_r = _sgraph._edit(edges=redges)
  mat = np.array([ [0, 1, 0], [-1, 0, 0], [0, 0, 1] ])
  mat = np.eye(3)

  furcations, furcmeshes = [], []
  for furc in _sgraph.nontrivial_furcations:
    normal = furc.normal
    centroid = furc.centroid
    myedges = sgraph_r.get_edges(furc.indices)  # head directed into furcation
    centerpoints = [edge.cline([edge.interval[1]]).ravel() for edge in myedges]
    radii = [edge.radius([edge.interval[1]])[0] for edge in myedges]
    bB = bT = np.mean(radii)
    frames = [edge.frame([edge.interval[1]]).reshape(3, 3) @ mat for edge in myedges]
    furcations.append(
      bif_from_matrices(frames,
                        centerpoints,
                        normal,
                        centroid,
                        bB, bT, cross,
                        exploit_symmetry=bif_exploit_symmetry))
    furcmeshes.append(
      mesh_union(*furcations[-1].quick_sample((nsec0, nsec1, 7)).ravel(),
                 boundary=True, eps=union_eps))

  vessels_union = mesh_union(*vessel_meshes, boundary=True, eps=union_eps)
  furcs_union = mesh_union(*furcmeshes, boundary=True, eps=union_eps)
  mesh = mesh_union(vessels_union, furcs_union, boundary=True, eps=union_eps)

  if show_interfaces:
    mesh.boundary.interfaces.plot()
  return mesh
