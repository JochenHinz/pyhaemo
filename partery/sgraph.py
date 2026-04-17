import warnings

from pyhaemo.types import Int, Float
from pyhaemo.geo.bif import repeated_knot_disc

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

  def make_mesh(self, dx: Float = 0.005, nref=3, rfac=2, cross_nelems: Int = 4,
                boundary_layer_exponent: Float = 1., **kwargs):
    from mesh import make_spline_mesh

    try:
      cross = repeated_knot_disc(cross_nelems,
                                 boundary_layer_exponent=boundary_layer_exponent)
    except TypeError as err:
      msg = str(err).lower()
      if 'unexpected keyword' in msg and 'boundary_layer' in msg:
        if boundary_layer_exponent != 1.:
          warnings.warn(
            'Installed pyhaemo does not support boundary_layer_exponent on '
            'repeated_knot_disc; reinstall pyhaemo from the source tree. '
            'Using the default disc.',
            stacklevel=2)
        cross = repeated_knot_disc(cross_nelems)
      else:
        raise
    return make_spline_mesh(self, cross, dx=dx, nref=nref, rfac=rfac, **kwargs)
