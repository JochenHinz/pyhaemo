from pyhaemo.util import np, frozen, normalize
from pyhaemo.types import Immutable, Int
from pyhaemo.log import logger as log

from geo import sort_ccw, estimate_normal
from edge import Truncatable, Orientable, IntegerTuple

from itertools import count
from functools import cached_property
from typing import Self, Sequence, Optional, Dict, Tuple

from numpy.typing import ArrayLike


UNDORECURRENCELENGTH = 5


class NFurcation(Orientable):

  def __init__(self, indices: Sequence[Int] | IntegerTuple,
                     centroid: ArrayLike,
                     normal: Optional[ArrayLike] = None):
    self.indices = IntegerTuple(indices)
    setindices = set(self.indices)
    assert len(self.indices) == len(setindices)
    assert all(-i not in setindices for i in setindices), \
      "Furcation indices must not contain both positive and negative values."

    self.centroid = frozen(centroid, dtype=float)

    if normal is None:
      normal = np.array([0, 0, 1], dtype=float)
    self.normal = frozen(normalize(normal), dtype=float)

    assert self.centroid.shape == self.normal.shape == (3,)

  @property
  def N(self) -> Int:
    return len(self.indices)

  def __len__(self) -> Int:
    return self.N

  def __bool__(self) -> bool:
    return bool(self.indices)

  def renumber(self, mapping: dict[Int, Int]) -> Self:
    return self._edit(indices=self.indices.renumber(mapping))

  def _neg(self) -> Self:
    return self._edit(indices=-self.indices,
                      normal=-self.normal)

  def __iter__(self):
    yield from self.indices

  def set_normal(self, endpoint_dict: Dict[Int, ArrayLike]) -> Self:
    if len(self) in (0, 1):
      return self._edit(normal=[0, 0, 1])
    normal = estimate_normal([endpoint_dict[i] for i in self.indices])
    return self._edit(normal=normal)

  def orient(self, endpoint_dict: Dict[Int, ArrayLike]) -> Self:
    if len(self) in (0, 1):
      return self
    order = sort_ccw([endpoint_dict[i] for i in self.indices],
                      centroid=self.centroid,
                      normal=self.normal)
    return self._edit(indices=[self.indices[i] for i in order])


class EdgeGraph(Immutable):
  """
  Only defined by edges, not by vertices.
  """

  @staticmethod
  def _check_furcations(graph):
    """
    Check if the furcations of the vascular graph are valid.
    """
    assert len(graph.furcations) == len(set(map(frozenset, graph.furcations))), \
      "Furcations must be unique."

    seen = set()
    for furc in map(set, graph.furcations):
      for i in furc:
        assert i not in seen and -i not in furc
        seen.add(i)

    assert seen == set(graph.indices) | {-i for i in graph.indices}, \
      "Furcations must only contain indices from the graph."

  def __init__(self, indices: Sequence[Int],
                     edges: Sequence[Truncatable],
                     furcations: Sequence[NFurcation]):
    self.indices = tuple(map(int, indices))
    assert all( i > 0 for i in self.indices )
    assert len(self.indices) == len(set(self.indices)), "Indices must be unique."

    self.edges = tuple(edges)
    assert all( isinstance(e, Orientable) for e in self.edges ), \
      "Edges must be Orientable instances."
    assert len(self.edges) == len(self.indices), "Edges must match indices in length."

    self.furcations = tuple(furcations)
    assert all(isinstance(furc, NFurcation) for furc in self.furcations), \
      "Furcations must be instances of NFurcation."
    self._check_furcations(self)

  def get_edges(self, iedges: Int | Sequence[Int]) -> Sequence[Orientable]:
    if isinstance(iedges, Int):
      iedges = iedges,
    edgedict = dict(zip(self.indices, self.edges))
    edges = [edgedict[abs(i)] for i in iedges]
    return tuple( edge if i > 0 else -edge for i, edge in zip(iedges, edges) )

  def get_furc(self, index: Int, return_index=False):
    """
    Get the bifurcation that conains the index.
    """
    if not hasattr(self, '__furcmap'):
      furcmap = {}
      for i, furc in enumerate(self.furcations):
        for findex in furc:
          furcmap[findex] = i
      self.__furcmap = furcmap

    findex = self.__furcmap[index]
    ret = self.furcations[findex]

    if return_index:
      return ret, findex
    return ret

  def truncate_edges(self, truncate_dict: Dict[Int, Sequence[Int | None]]) -> Self:
    assert set(truncate_dict.keys()).issubset(set(self.indices)), \
      "Indices to truncate must be a subset of the positive graph indices."

    if not truncate_dict:
      return self

    edges = []
    for i, edge in zip(self.indices, self.edges):
      if i not in truncate_dict:
        edges.append(edge)
        continue

      edges.append( edge.truncate(*truncate_dict[i]) )

    return self._edit(edges=edges)

  def take(self, iedges: Int | Sequence[Int]) -> Self:
    if isinstance(iedges, Int):
      iedges = iedges,

    iedges = set(map(abs, iedges))
    assert iedges and set(iedges).issubset(set(self.indices)), \
      "Indices must be a subset of the graph indices."

    indices = [i for i in self.indices if i in iedges]
    edges = [ edge for i, edge in zip(self.indices, self.edges) if i in iedges ]
    furcs = [ furc._edit(indices=[i for i in furc if abs(i) in iedges])
              for furc in self.furcations ]

    return self._edit(indices=indices,
                      edges=edges,
                      furcations=[furc for furc in furcs if furc])

  def delete_edges(self, iedges: Int | Sequence[Int]):

    if isinstance(iedges, Int):
      iedges = iedges,

    iedges = set(map(abs, iedges))

    if not iedges:
      return self

    return self.take(set(self.indices) - iedges)

  def merge_furcs(self, i0: Int, i1: Int, centroid=None, normal=None) -> Self:
    f0, f1 = self.furcations[i0], self.furcations[i1]
    if len(f0) == 1 and len(f1) == 1:
      assert f0.indices[0] != -f1.indices[0]

    if centroid is None:
      centroid = (f0.centroid + f1.centroid) / 2

    intersection = set(map(abs, f0)) & set(map(abs, f1))

    if intersection:
      log.warning(f"Bifmerge deleted the following edges: {sorted(intersection)}")

    # keep all furcs except `f0` and `f1`
    new_furcs = [f for f in self.furcations if f not in (f0, f1)]

    # add the merged furcation
    new_furcs.append(NFurcation([i for i in f0 if abs(i) not in intersection] +
                                [i for i in f1 if abs(i) not in intersection],
                                  centroid, normal))

    # add trivial furcations for the intersection (they get removed later)
    for i in intersection:
      new_furcs.append(NFurcation([-i], centroid))
      new_furcs.append(NFurcation([i], centroid))

    return self._edit(furcations=new_furcs).delete_edges(list(intersection))

  @cached_property
  def nontrivial_furcations(self) -> Tuple[NFurcation]:
    return tuple(furc for furc in self.furcations if furc.N > 2)

  def renumber(self, mapping: Optional[dict[Int, Int]] = None) -> Self:
    """
    Remumber the edges using a mapping.
    If no mapping is provided, it will create a mapping that assigns
    consecutive integers starting from 1 to the indices.
    """
    mapping = dict(mapping or dict(zip(self.indices, count(1))))
    return self._edit(indices=[mapping[i] for i in self.indices],
                      furcations=[furc.renumber(mapping) for furc in self.furcations])
