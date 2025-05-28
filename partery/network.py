from . import aux
from .index import transform_index, as_edge, \
                   CannotSplitEdgeAtThisPositionError, CannotMergeMultiEdgesError, \
                   Edge, as_IndexSequence, IndexSequence, MultiEdge, arclength_verts
from .template import FaceTemplate, face_template
from .fit import univariate_fit, make_knotvectors
from .aux import reverse_univariate_gridobject, log, EdgeNotFoundError

from .edge import OrientedEdge, IntegerTuple

from pyhaemo.util import np, frozen, frozen_cached_property
from pyhaemo.types import Immutable, Int, Float, lock

from abc import abstractmethod
from functools import cached_property
from typing import Tuple, Self, Sequence
from collections.abc import Hashable


from scipy import interpolate
from numpy.typing import NDArray


UNDORECURRENCELENGTH = 5


class Graph(Immutable):

  def __init__(self, vertex_indices: Sequence[Int],
                     vertices: Sequence[Hashable],
                     edge_indices: Sequence[IntegerTuple | Tuple],
                     edges: Sequence[OrientedEdge]):

    self.vertex_indices = tuple(map(int, vertex_indices))
    assert all( i >= 0 for i in self.vertex_indices ), \
      "All indices must be positive."
    assert len(self.vertex_indices) == len(set(self.vertex_indices)), \
      "Vertex indices must be unique."

    self.vertices = tuple(vertices)
    assert len(self.vertex_indices) == len(self.vertices)
    assert all(isinstance(v, Hashable) for v in self.vertices), \
      "All vertices must be hashable."

    self.edge_indices = tuple(map(IntegerTuple, edge_indices))
    assert all(len(ie) == len(set(ie)) == 2 and all(i > 0 for i in ie)
                                       for ie in self.edge_indices), \
      "All edge indices must be pairs of integers."
    self.edges = tuple(edges)
    assert all(isinstance(e, OrientedEdge) for e in self.edges), \
      "All edges must be OrientedEdge instances."

    set_indices = set.union(*map(set, self.edge_indices))

    # TODO: comprehensive error message
    assert set_indices == set(self.vertex_indices)

  @cached_property
  @lock
  def edgedict(self):
    return dict(zip(self.edge_indices, self.edges))

  def get_edges(self, indices):
    edgedict = self._edgedict
    return tuple(edgedict[index] if index in edgedict else -edgedict[-index]
                 for index in map(IntegerTuple, indices))

  def get_iedges(self, indices):
    return tuple(self.edges[i] if i >= 0 else -self.edges[-i] for i in indices)


class EdgeGraph(Immutable):
  """
  Only defined by edges, not by vertices.
  """
  ...
