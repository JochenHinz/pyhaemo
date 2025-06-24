import numpy as np
from typing import Tuple, Sequence

import jax.numpy as jnp


def _normshape(shape: int | Sequence[int] | None, ndims: int) -> Tuple[int, ...]:
  """
    Normalize axes.
  """
  if shape is None: return tuple(range(ndims))
  shape = tuple(shape if isinstance(shape, Sequence) else [shape])
  if len(shape) == 0: return ()
  shape = tuple( axis if axis >= 0 else axis + ndims for axis in shape )
  assert all( -ndims < axis < ndims for axis in shape )
  return tuple(sorted(shape))


def _sum_shape(shape: Sequence[int], axes: int | Sequence[int] | None) -> Tuple[int, ...]:
  axes = _normshape(axes, len(shape))
  return tuple( dim for i, dim in enumerate(shape) if i not in axes )


def _moveaxis_shape(shape, source, destination):
  assert len(source) == len(destination)
  ndims = len(shape)
  source = _normshape(source, ndims)
  destination = _normshape(destination, ndims)
  assert len(source) == len(destination) == 1, NotImplementedError
  if source == destination: return shape
  s, = source
  d, = destination
  if s < d:
    return shape[:s] + shape[s+1: d] + (shape[s],) + shape[d:]
  return shape[:d] + (shape[s],) + shape[d:s] + shape[s+1:]


def _reshape_shape(fromshape, toshape):
  assert toshape.count(-1) in (0, 1)
  if -1 in toshape:
    toshape = list(toshape)
    index = toshape.index(-1)
    assert index in (0, len(toshape) - 1)
    if index == 0:
      ignore, *_shape = toshape
    else:
      *_shape, ignore = toshape
    nfrom, nto = np.prod(fromshape), np.prod(_shape)
    assert nfrom % nto == 0
    axis = nfrom // nto
    toshape[index] = axis
  assert np.prod(fromshape) == np.prod(toshape)
  return toshape


def _swapaxes_shape(shape, *axes):
  axes = _normshape(axes, len(shape))
  n, m = axes
  shape = list(shape)
  shape[n], shape[m] = shape[m], shape[n]
  return tuple(shape)


def _mul_add_shape(shape0, shape1):
  assert len(shape0) == len(shape1)
  shape = []
  for i, j in zip(shape0, shape1):
    if i != j:
      assert 1 in (i, j)
      shape.append(max([i, j]))
    else:
      shape.append(i)
  return tuple(shape)


def _take_shape_and_indices(shape: Sequence[int], indices: int | Sequence[int] | np.ndarray | jnp.ndarray, axis: int) -> Tuple[Tuple[int, ...], jnp.ndarray]:
  if isinstance(indices, int):
    indices = indices,
  indices = jnp.asarray(indices, dtype=int)
  n = shape[axis]
  indices = indices + (indices < 0).astype(int) * n
  assert np.min(indices) >= 0 and np.max(indices) < shape[axis]
  shape = tuple(shape)
  if indices.ndim == 1:
    return shape[:axis] + (len(indices),) + shape[axis+1:], indices
  return shape[:axis] + shape[axis+1:], indices


def _kron_shape(shape0, shape1):
  if len(shape0) != len(shape1):
    raise NotImplementedError
  return tuple( i * j for i, j in zip(shape0, shape1) )


def _insertaxis_shape(shape: Sequence[int], axes: int | Sequence[int]) -> Tuple[int, ...]:
  if not isinstance(axes, Sequence):
    axes = axes,
  return tuple(np.insert(np.asarray(shape, dtype=int), axes, 1))


def _insert_axes(oldshape: Sequence[int] | jnp.ndarray | np.ndarray, new_one_axes: Sequence[int] | jnp.ndarray | np.ndarray) -> Tuple[int, ...]:
  """ Return the positions at which artificial axes have been inserted
      in order to acquire the resulting new axes w.r.t. the newshape.
      ret = _insert_axes(oldshape, new_one_axes)
      np.insert(oldshape, ret, 1)[new_one_axes] = [1, 1, ...].
      Used in arr.InsertAxis."""
  # XXX: make more readable
  if len(oldshape) == 0: return tuple(new_one_axes)
  oldshape = np.asarray(oldshape)
  new_one_axes = list(new_one_axes)
  if len(new_one_axes) == 0: return tuple()
  oldshape[oldshape == 1] = -1
  shape = []
  offset = 0
  for i in range(len(oldshape) + len(new_one_axes)):
    if i in new_one_axes:
      shape.append(1)
    else:
      shape.append(oldshape[offset])
      offset += 1
  ones_before, ones_after = [], []
  for i in sorted(set(np.arange(len(shape))) - set(new_one_axes)):
    ones_before.append( shape[:i].count(1) )
    ones_after.append( shape[i:].count(1) )
  ones_before = [0] + ones_before
  before = np.asarray(ones_before)[1:] - np.asarray(ones_before)[:-1]
  if ones_after[-1] != 0:  # add last entry
    before = np.concatenate([before, ones_after[-1:]])
  return tuple(np.concatenate([ [i] * j for i, j in zip(np.arange(len(before)), before) ]).astype(int))


def _concatenate_insertaxes(shape, axes0, axes1):
  # XXX: add detailed description
  shape = np.asarray(shape)
  shape[shape == 1] = -1  # ensure that shape ain't contain any ones
  newshape = np.insert(np.insert(shape, axes0, 1), axes1, 1)
  newshape = np.concatenate([newshape, [-1]])  # add artificial last axis
  non_one_indices = [i for i, j in enumerate(newshape) if j != 1]  # these are all axes that were not 1 at the beginning
  ones_before = np.asarray([list(newshape[:i]).count(1) for i in non_one_indices])
  diff = np.concatenate([ones_before[0:1], ones_before[1:] - ones_before[:-1]]).astype(int)
  return np.concatenate([ [i] * j for i, j in zip(np.arange(len(diff)), diff) ]).astype(int)
