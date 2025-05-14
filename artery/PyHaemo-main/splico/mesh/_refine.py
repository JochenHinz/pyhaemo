"""
Routines dedicated to refining various mesh types.
@author: Jochen Hinz
"""


from ..util import np
from .._jit import product, arange_product, ravel_multi_index, mul_reduce, float2str, multiply

from functools import lru_cache
from itertools import count

from numba import njit


@njit(cache=True)
def _format_indices_weights(indices, weights):
  """
  Given a container of vertex indices and
  an equally-sized container of weights with respect to these indices,
  remove indices with weight 0 and argsort both remaining containers
  according to the remaining indices.
  >>> _format_indices_weights([5, 4, 1, 7], [.5, .5, 0, 0])
      (4, 5), (.5, .5)
  """
  mask = weights != 0
  indices, weights = indices[mask], weights[mask]
  argsort = np.argsort(indices)
  return '((' + ', '.join([str(i) for i in indices[argsort]]) + \
          '), ' '(' + ', '.join([float2str(w) for w in weights[argsort]]) + '))'


@njit(cache=True)
def _formatter(point: np.ndarray):
  ret = np.empty((2 ** len(point),), dtype=point.dtype)
  for i, weights in enumerate(product([np.array((1 - x, x)) for x in point])):
    ret[i] = multiply.reduce(weights, axis=0)
  return ret


@njit(cache=True)
def _refine_structured(elements: np.ndarray, points: np.ndarray, ndims: int):
  """
  Refine any structured mesh type (LineMesh, QuadMesh, HexMesh).
  """
  # XXX: this routine works but is a bit difficult to read and therefore
  #      difficult to maintain. Find better solution.

  # map the range(len(self.points)) to existing points
  map_index_coord = {}
  for i in range(len(points)):
    map_index_coord[i] = points[i]

  index = len(map_index_coord)
  map_format_index = {}

  # map ((i), (1.0)) which represents 1 * self.points[i] to index i
  one = np.array([1.0], dtype=np.float64)
  for i in range(len(points)):
    map_format_index[_format_indices_weights(np.array([i], dtype=np.int64), one)] = i

  xi = np.linspace(0, 1, 3)

  # make a subdivision of [0, 1] ** n with 3 points in each direction
  # _X = np.stack(list(map(np.ravel, np.meshgrid(*[xi] * ndims))), axis=1)
  _X = product([xi for i in range(ndims)])

  # format X into the form (i0, i1, ...), (w0, w1, ...)
  X = [_formatter(_x) for _x in _X]

  # map tensor index to global index using `loc_indices`
  loc_indices = np.arange(3 ** ndims, dtype=np.int64)

  ravel_dims = 3 * np.ones((ndims,), dtype=np.int64)

  elems = np.empty((int(2 ** ndims * elements.shape[0]), int(elements.shape[1])), dtype=np.int64)
  k = 0
  for ielem, indices in enumerate(elements):

    for j, weights in enumerate(X):
      # get index that corresponds to the formatted indices, weights
      # if not yet available, create new index via defaultdict constructor
      _hash = _format_indices_weights(indices, weights)

      if _hash not in map_format_index:
        map_format_index[_hash] = index
        map_index_coord[index] = (points[indices] * weights[:, None]).sum(0)
        index += 1

    # loop over all subdivision (hyber)cubes and look up their vertices' global indices.
    for my_multi_index in arange_product([2 for _ in range(ndims)]):
      # XXX: simplify following line
      my_elem_weights = [X[loc_indices[ravel_multi_index(_index, ravel_dims)]]
                         for _index in [my_multi_index + myindex0
                                        for myindex0 in arange_product([2 for _ in range(ndims)])]]
      for j, myweights in enumerate(my_elem_weights):
        elems[k, j] = map_format_index[_format_indices_weights(indices, myweights)]
      k += 1

  newpoints = np.empty((len(map_index_coord), 3), dtype=np.float64)
  for i in range(len(map_index_coord)):
    newpoints[i] = map_index_coord[i]

  return elems, newpoints


@lru_cache(maxsize=8)
def refine_structured(mesh):
  return mesh.__class__(*_refine_structured(mesh.elements, mesh.points, mesh.ndims))


def abs_tuple(tpl):
  """
  [5, 6] -> (5, 6)
  [6, 5] -> (5, 6)
  (6, 5) -> (5, 6)
  """
  a, b = tpl
  if a > b: return b, a
  return tuple(tpl)


@lru_cache(maxsize=8)
def _refine_Triangulation(mesh):
  r"""
  Uniformly refine the entire mesh once.
        i1                             i1
        / \                            / \
       /   \                          /   \
      /     \         becomes      i01 ---- i12
     /       \                      / \   /  \
    /         \                    /   \ /    \
  i0 --------- i2                i0 -- i02 --- i2

  Returns
  -------
  The refined mesh of class `Triangulation`
  """

  # XXX: JIT compile

  points = mesh.points
  maxindex = len(points)
  # [0, 1], [0, 2], [1, 2]
  slices = list(mesh._submesh_indices)
  all_edges = list(set(map(abs_tuple, np.concatenate(mesh.elements[:, slices]))))
  newpoints = points[np.array(all_edges)].sum(1) / 2
  map_edge_number = dict(zip(all_edges, count(maxindex)))

  triangles = []
  for tri in mesh.elements:
    i01, i02, i12 = [map_edge_number[edge] for edge in map(abs_tuple, tri[slices])]
    i0, i1, i2 = tri
    triangles.extend([
        [i0, i01, i02],
        [i01, i1, i12],
        [i02, i01, i12],
        [i02, i12, i2]
    ])

  elements = np.array(triangles, dtype=int)
  points = np.concatenate([points, newpoints])

  return mesh.__class__(elements, points)
