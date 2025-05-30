from pyhaemo.util import np, _, frozen, normalize
from pyhaemo.types import Immutable, Float, Int, lock
from pyhaemo.log import logger as log

from edge import IntegerTuple, VascularEdge, Orientable

from typing import Self, Sequence, Optional, Tuple, Dict, Callable
from itertools import count, repeat
from functools import cached_property, wraps
import os

from numpy.typing import ArrayLike


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


def no_empty_edges(func: Callable) -> Callable:

  @wraps(func)
  def wrapper(self, *args, **kwargs):
    if self.has_empty_edges():
      raise ValueError("Cannot perform operation on a vascular graph with empty edges.")
    return func(self, *args, **kwargs)

  return wrapper


class VascularGraph(Immutable):

  @classmethod
  def from_edges(cls, data: Sequence[ArrayLike | VascularEdge]) -> Self:
    edges = [ VascularEdge(pointset) for pointset in data ]
    indices = list(range(1, len(edges) + 1))

    furcations = tuple( (i,) for i in indices ) + \
                 tuple( (-i,) for i in indices )
    centroids = tuple( edge.points[-1][:3] for edge in edges ) + \
                tuple( edge.points[0][:3] for edge in edges )

    return cls(indices, edges, list(map(NFurcation, furcations, centroids)))

  @classmethod
  def from_centerlines_radii(cls, centerlines: Sequence[ArrayLike], radii: Sequence[ArrayLike]):
    return cls.from_edges([ np.concatenate([cline, rad[:, _]], axis=1)
                            for cline, rad in zip(centerlines, radii, strict=True) ])

  @classmethod
  def from_csv(cls, foldername: str):
    """
    Create a vascular graph from CSV files in the specified folder.
    """

    filenames = (os.path.join(foldername, f) for f
                 in os.listdir(foldername)
                 if f.endswith('.csv'))

    edges = []
    for data in map(lambda fn: np.genfromtxt(fn, delimiter=',', skip_header=1), filenames):
      edges.append(data)

    assert edges

    return cls.from_edges(edges)

  @classmethod
  def from_swc(cls, *args, **kwargs):
    """
    Create a vascular graph from SWC files.
    This method is a placeholder and should be implemented.
    """
    raise NotImplementedError("SWC file parsing is not implemented yet.")

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
                     edges: Sequence[VascularEdge],
                     furcations: Sequence[Sequence[Int] | IntegerTuple]):
    self.indices = tuple(map(int, indices))
    assert all( i > 0 for i in self.indices )
    assert len(self.indices) == len(set(self.indices)), "Indices must be unique."

    self.edges = tuple(map(VascularEdge, edges))
    assert len(self.edges) == len(self.indices), "Edges must match indices in length."

    self.furcations = tuple(furcations)
    assert all(isinstance(furc, NFurcation) for furc in self.furcations), \
      "Furcations must be instances of NFurcation."
    self._check_furcations(self)

  def has_empty_edges(self):
    return any(edge.is_empty() for edge in self.edges)

  def get_edges(self, iedges: Int | Sequence[Int]) -> Sequence[VascularEdge]:
    if isinstance(iedges, Int):
      iedges = iedges,
    edges = [self.edge[abs(i) - 1] for i in iedges]
    return tuple( np.sign(i) * edge for i, edge in zip(iedges, edges) )

  @cached_property
  @lock
  @no_empty_edges
  def endpoints(self):
    ret = { i: edge.points[-1][:3] for i, edge in zip(self.indices, self.edges) }
    ret.update( {-i: edge.points[0][:3] for i, edge in zip(self.indices, self.edges) } )
    return ret

  @cached_property
  @lock
  @no_empty_edges
  def endopint_tangents(self):
    ret = {i: (e.points[-1, :3] - e.points[-2, :3]) / (e.verts[-1] - e.verts[-2])
               for i, e in zip(self.indices, self.edges)}
    ret.update( {-i: (e.points[1, :3] - e.points[0, :3]) / (e.verts[1] - e.verts[0])
                    for i, e in zip(self.indices, self.edges) } )
    return ret

  @cached_property
  @lock
  @no_empty_edges
  def endpoint_radii(self):
    ret = { i: e.points[-1, -1] for i, e in zip(self.indices, self.edges) }
    ret.update( {-i: e.points[0, -1] for i, e in zip(self.indices, self.edges) } )
    return ret

  @cached_property
  def nontrivial_furcations(self) -> Tuple[NFurcation]:
    return tuple(furc for furc in self.furcations if furc.N > 1)

  def renumber(self, mapping: Optional[dict[Int, Int]] = None) -> Self:
    """
    Remumber the edges using a mapping.
    If no mapping is provided, it will create a mapping that assigns
    consecutive integers starting from 1 to the indices.
    """
    if mapping is None:
      mapping = dict(zip(self.indices, count(1)))

    mapping = dict(mapping)

    return self._edit(indices=[mapping[i] for i in self.indices],
                      furcations=[furc.renumber(mapping) for furc in self.furcations])

  def normalize(self):
    lengths = [ edge.length for edge in self.edges ]
    mean = np.mean(lengths)
    edges = [ edge._edit(points=edge.points / mean) for edge in self.edges ]
    furcs = [ f._edit(centroid=f.centroid / mean) for f in self.furcations ]

    return self._edit(edges=edges, furcations=furcs)

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

  def delete_small_edges(self, eps: Float):
    assert eps > 0

    delindices = []
    for i, edge in enumerate(self.edges, 1):
      if edge.length < eps:
        delindices.append(i)

    return self.delete_edges(delindices)

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

  def merge_furcs_proximity(self, *args, **kwargs):
    from preprocessing import merge_furcs_proximity
    return merge_furcs_proximity(self, *args, **kwargs)

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

  def recompute_centroids(self, eps=1e-6) -> Self:
    endpoints = self.endpoints

    centroids = []
    for furc in self.furcations:
      mypoints = []
      for mypoint in map(endpoints.__getitem__, furc):
        if any(np.linalg.norm(point - mypoint) < eps for point in mypoints):
          continue
        mypoints.append(mypoint)

      centroids.append( np.mean(mypoints, axis=0) )

    furcs = [furc._edit(centroid=centroid)
             for furc, centroid in zip(self.furcations, centroids)]

    return self._edit(furcations=furcs)

  def truncate_furcdist(self, dist_dict: Dict[Int, Sequence[Int | None]]) -> Self:
    pass

  def plot(self, *args, **kwargs):
    from plot import plot_vgraph
    return plot_vgraph(self, *args, **kwargs)

  def refit(self, xi: Int | ArrayLike = 1001) -> Self:
    return self._edit(edges=[edge.refit(xi) for edge in self.edges])

  def furcomplete(self, npoints=1001, eps=1e-8):
    endpoints = self.endpoints
    radii = self.endpoint_radii
    edgedict = dict(zip(self.indices, self.edges))

    for furc in self.furcations:
      center = furc.centroid

      for i in furc:
        mypoint = endpoints[i]
        if np.linalg.norm(mypoint - center) < eps:
          continue

        edge = edgedict[abs(i)]
        rad = radii[i]

        # TODO: Find better solution for distinguishing cases
        if i < 0:
          edge = edge._edit(points=np.concatenate([[[*center, rad]],
                                                   edge.points], axis=0))
        else:
          edge = edge._edit(points=np.concatenate([edge.points,
                                                   [[*center, rad]]], axis=0))
        edgedict[abs(i)] = edge

    xi = np.linspace(0, 1, npoints)
    for i, edge in edgedict.items():
      edgedict[i] = edgedict[i].refit(xi)

    return self._edit(edges=[edgedict[i] for i in self.indices])

  def split_edges_proximity(self, *args, **kwargs):
    from preprocessing import split_edges_proximity
    return split_edges_proximity(self, *args, **kwargs)


if __name__ == '__main__':
  artery = VascularGraph.from_csv('data/csv/coro') \
                        .normalize() \
                        .refit() \
                        .split_edges_proximity(eps=0.04) \
                        .merge_furcs_proximity(0.07) \
                        .refit() \
                        .recompute_centroids()

  trunc = dict(zip(artery.indices, repeat((10, -10))))
  artery = artery.truncate_edges(trunc).furcomplete()

  artery.plot(show_normals=False)
