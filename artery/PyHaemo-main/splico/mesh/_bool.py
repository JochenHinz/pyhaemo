"""
Module containing various JIT-compiled helper routines for use in `bool.py`

@author: Jochen Hinz
"""


from ..util import np

from typing import Any

from numba import njit, prange


# TODO: Make use of Numba parallelization


""" JIT-compiled routines for normal mesh union / difference """

# In Numba, tuples of differing length are differing types


@njit(cache=True)
def array_to_tuple1(arr):
  return (arr[0],)


@njit(cache=True)
def array_to_tuple2(arr):
  return (arr[0], arr[1])


@njit(cache=True)
def array_to_tuple3(arr):
  return (arr[0], arr[1], arr[2])


@njit(cache=True)
def array_to_tuple4(arr):
  return (arr[0], arr[1], arr[2], arr[3])


@njit(cache=True)
def array_to_tuple5(arr):
  return (arr[0], arr[1], arr[2], arr[3], arr[4])


def array_to_tupleN(nelems: int):
  """
  Dynamic code generator allowing for numba tuple conversion with an arbitrary
  number of entries.
  """
  # XXX: add unit test
  assert nelems >= 6
  funcstr = """def func(arr): return ({})""".\
            format(', '.join(map('arr[{}]'.format, range(nelems))) + ',')

  local_namespace: dict[Any, Any] = {}
  # Execute the code in the given namespace
  exec(funcstr, globals(), local_namespace)

  # Extract the function from the namespace
  func = local_namespace['func']

  return njit(func)


def default_converter(nelems):
  if 1 <= nelems <= 5:
     return {i: globals().get(f'array_to_tuple{i}') for i in range(1, 6)}[nelems]

  # dynamically create converter
  return array_to_tupleN(nelems)


def make_numba_indexmap(points):
  assert points.ndim == 2
  nargs = points.shape[1]
  if nargs == 3:
    return _make_numba_indexmap3(points)

  converter = default_converter(points.shape[1])
  return _make_numba_indexmap(points, converter)


@njit(cache=True)
def _make_numba_indexmap(points, point_converter):
  """
  Create a hashmap that maps each point in `points` to a unique running index.
  Assumes the points in `points` to already be unique.
  """
  map_coord_index = {}
  i = 0
  for point in points:
    map_coord_index[point_converter(point)] = i
    i += 1

  return map_coord_index


# XXX: Defining this function as
#      ``partial(_make_numba_indexmap, point_converter=array_to_tuple3)``
#      prevents function caching. Find way to write cached version of this function
#      without code copy.
#      This may be possible by dynamically creating the function string when
#      loading the script.
@njit(cache=True)
def _make_numba_indexmap3(points):
  """ ``partial(_make_numba_indexmap, point_converter=array_to_tuple3)`` """
  map_coord_index = {}
  i = 0
  for point in points:
    map_coord_index[array_to_tuple3(point)] = i
    i += 1

  return map_coord_index


def renumber_elements_from_indexmap(elements, points, map_coord_index):
  """
  Given an element array `elements` and corresponding points ``points``
  create a new element array wherein each entry ``newelements[i, j]`` is given
  by ``map_coord_index[points[elements[i, j]]]``.
  The indexmap must take ``points.shape[1]``-tuples of coordinates and return
  a new index.
  If an entry is not contained, undefined behavior results.
  """
  nargs = len(list(map_coord_index.keys())[0])
  assert points.shape[1:] == (nargs,)

  # more efficient non-dynamical routine
  if nargs == 3:
    return _renumber_elements_from_indexmap3(elements, points, map_coord_index)

  converter = default_converter(len(list(map_coord_index.keys())[0]))

  return _renumber_elements_from_indexmap(elements,
                                          points,
                                          map_coord_index,
                                          converter)


@njit(cache=True, parallel=True)
def _renumber_elements_from_indexmap(elements, points,
                                     map_coord_index, point_converter):
  newelems = np.empty(elements.shape, dtype=np.int64)
  for i in prange(len(elements)):
    myvertices = elements[i]
    for j, point in enumerate(points[myvertices]):
      newelems[i, j] = map_coord_index[point_converter(point)]
  return newelems


# XXX: Defining this function as
#      ``partial(_renumber_elements_from_indexmap, point_converter=array_to_tuple3)``
#      prevents function caching. Find way to write cached version of this function
#      without code copy.
@njit(cache=True, parallel=True)
def _renumber_elements_from_indexmap3(elements, points, map_coord_index):
  """
  ``partial(_renumber_elements_from_indexmap, point_converter=array_to_tuple3)``
  """
  newelems = np.empty(elements.shape, dtype=np.int64)
  for i in prange(len(elements)):
    myvertices = elements[i]
    for j, point in enumerate(points[myvertices]):
      newelems[i, j] = map_coord_index[array_to_tuple3(point)]
  return newelems


""" Routines for boundary mesh union """


@njit(cache=True)
def _make_matching(V, col_index, row_index):
  """
  Given a sparse distance matrix in CSR format, create a correspondence
  between points in pairs. If two points have been matched their indices
  will be removed from the set of indices that are still eligible for
  matching to avoid duplicate matching.
  Whether two points are close enough to be eligible for matching is
  the ``eps`` parameter in ``compute_distance_matrix``.

  Parameters
  ----------
  V : :class:`np.ndarray`
      The array containing the distances between the two meshes' points.
  colIndex : :class:`np.ndarray`
      The column index of the distance matrix in CSR format.
  row_index : :class:`np.ndarray`
      The row index of the distance matrix in CSR format.

  Returns
  -------
  matching : :class:`np.ndarray` of integers
      An `(nmatchings, 2)` - shaped integer array containing the matches
      as rows.
  """
  # TODO: the efficiency of this routine would benefit greatly from
  #       parallelization.

  # empty input => immediately return empty array
  if len(V) == 0:
    return np.empty((0, 2), dtype=np.int64)

  # make set of available right matches
  matched = set()

  match0, match1 = [], []

  # for all points in point0
  for i, (row_start, row_end) in enumerate(zip(row_index, row_index[1:])):

    # smallest match and distance initialized to dummy values
    smallest_match = -1
    smallest_distance = np.inf

    for index, distance in zip(col_index[row_start: row_end],
                               V[row_start: row_end]):

      # index already matched ? raise error
      if index in matched:
        raise ValueError("Index already matched, reduce the distance threshold.")

      # if distance is smaller than the smallest distance, update
      # smallest distance and smallest match index
      if distance < smallest_distance:
        smallest_distance = distance
        smallest_match = index

    # no match found
    if smallest_match == -1:
      continue

    match0.append(i)
    match1.append(smallest_match)

    matched.add(index)

  if len(match0) == 0:
    return np.empty((0, 2), dtype=np.int64)

  ret = np.empty((len(match0), 2), dtype=np.int64)

  for i in range(len(match0)):
    ret[i, 0] = match0[i]
    ret[i, 1] = match1[i]

  return ret


@njit(cache=True)
def _remap_elements(all_elements, all_points, all_matches):
  """
  Given an array of elements, an array of corresponding points and an array
  of matched point pairs, create a new element array wherein the two distinct
  indices of pairs are replaced by a single index for both while only the
  corresponding point in ``all_points`` that corresponds to the lower of
  the two indices is kept.

  Parameters
  ----------
  all_elements : :class:`np.ndarray[int, 2]`
      The element array containing all element indices before re-indexing.
  all_points : :class:`np.ndarray[float, 2]`
      The point array. Is assumed to be exhaustive, i.e., there are exact
      as many points as there are unique element indices.
  all_matches : :class:`np.ndarray[int, 2]`
      An (nmatches, 2) integer array containing matched point indices which
      are then coupled.

  Returns
  -------
  elements : :class:`np.ndarray[int, 2]`
      The renumbered element array.
  points : :class:`np.ndarray[float, 2]`
      The corresponding point array that no longer contains two distinct
      points for matched pairs.
  """

  # ascending index array
  merge = np.arange(len(all_points))
  while True:
    _merge = merge.copy()

    # iterate over all pairs and always assign the lower index of the
    # two to the corresponding position in `merge`
    for pair in all_matches:
      # normally this would be a one-liner but numba doesn't support the necessary
      # syntax.
      a, b = merge[pair[0]], merge[pair[1]]
      _min = min((a, b))
      merge[pair[0]] = merge[pair[1]] = _min

    # repeat until the merge array no longer changes
    if (_merge == merge).all():
      break

  # get all unique indices that are left
  active_indices = np.unique(merge)

  new_elements = np.empty(all_elements.shape, dtype=np.int64)
  for i, elem in enumerate(all_elements):
    for j, index in enumerate(elem):

      # the new index is the position of the mapped old index in `active_indices`
      new_elements[i, j] = np.searchsorted(active_indices, merge[index])

  # only keep active points
  all_points = all_points[active_indices]

  return new_elements, all_points
