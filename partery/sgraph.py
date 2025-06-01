from pyhaemo.types import Int, Float
from pyhaemo.geo import ellipse

from typing import Sequence

from network import EdgeGraph, NFurcation
from artery import VascularGraph
from edge import SplineEdge


class SplineNFurcation(NFurcation):

  def __tojax__(self):
    raise NotImplementedError


def as_SplineNFurcation(furcation: NFurcation | SplineNFurcation) -> SplineNFurcation:
  if isinstance(furcation, SplineNFurcation):
    return furcation
  return SplineNFurcation(**furcation._lib)


class SplineGraph(EdgeGraph):

  def __init__(self, indices: Sequence[Int],
                     edges: Sequence[SplineEdge],
                     furcations: Sequence[NFurcation | SplineNFurcation]):
    super().__init__(indices, tuple(map(SplineEdge, edges)),
                              tuple(map(as_SplineNFurcation, furcations)))
    if any( len(furc) == 2 for furc in furcations ):
      raise ValueError("SplineGraph cannot have furcations with two edges")

  def plot(self, npoints=1001, *args, **kwargs):
    return VascularGraph(self.indices,
                         [e.sample(npoints) for e in self.edges],
                         self.furcations).plot(*args,
                                               show_frames=False,
                                               **kwargs)

  def make_mesh(self, dx: Float = 0.005, nref=3, rfac=2, **kwargs):
    from mesh import make_spline_mesh
    cross = ellipse(1, 1, 4)
    return make_spline_mesh(self, cross, dx=dx, nref=nref, rfac=rfac, **kwargs)
