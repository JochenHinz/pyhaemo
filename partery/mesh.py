from pyhaemo.util import np, _
from pyhaemo.spl import NDSplineArray
from pyhaemo.mesh import mesh_union

from bif import bif_from_matrices
from sgraph import SplineGraph
from artery import VascularGraph
from fit import fit_constrained


def make_spline_mesh(sgraph: SplineGraph,
                     cross: NDSplineArray,
                     dx: float = 0.02,
                     nref: int = 3,
                     rfac: int = 2,
                     npoints_cross=5,
                     **kwargs):
  """Create a mesh from a SplineGraph."""
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
    vessel_meshes.append(mesh_union(*vessels[-1].quick_sample((npoints_cross, npoints_cross, npoints)).ravel(), boundary=True))

  sgraph_r = _sgraph._edit(edges=redges)

  furcations, furcmeshes = [], []
  for furc in _sgraph.nontrivial_furcations:
    normal = furc.normal
    centroid = furc.centroid
    myedges = sgraph_r.get_edges(furc.indices)  # head directed into furcation
    centerpoints = [edge.cline([edge.interval[1]]).ravel() for edge in myedges]
    radii = [edge.radius([edge.interval[1]])[0] for edge in myedges]
    bB = bT = np.mean(radii)
    frames = [edge.frame([edge.interval[1]]).reshape(3, 3) for edge in myedges]
    furcations.append( bif_from_matrices(frames,
                                         centerpoints,
                                         normal,
                                         centroid,
                                         bB, bT, cross) )
    furcmeshes.append(
      mesh_union(*furcations[-1].quick_sample((npoints_cross, npoints_cross, 7)).ravel(), boundary=True))

  mesh = mesh_union(*vessel_meshes, boundary=True)

  # Heuristically, 1e-5 is better when symmetries are exploited.
  # Else round-off can cause the interfaces to not be detected.
  mesh = mesh_union(mesh, *furcmeshes, boundary=True, eps=1e-5)

  mesh.boundary.interfaces.plot()
