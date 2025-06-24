#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a class-based library of jax functions parameterised
by a number of input arguments.
Each function is supposedly differentiable.

@author: Jochen Hinz
"""


from pyhaemo.types import Immutable, Int
import shp

from typing import Sequence, Tuple, Union, Hashable, Self, List

from functools import cached_property, reduce
from abc import abstractmethod

import jax.numpy as jnp
import numpy as np

_ = None


Number = Union[Int, float]


class JaxFunction(Immutable):

  @cached_property
  def dependencies(self):
    """ Only instances of class ``Argument`` will return a set containing their
        name. All other instances return the dependencies of their arguments.
        I.e., for instances other than ``Argument`` this returns a set containing
        the names of the instances of ``Argument`` in their dependency-tree. """
    dependencies = [ arg.dependencies for arg in self.args
                                              if isinstance(arg, JaxFunction) ]
    return frozenset.union(*(dependencies or [frozenset()]))

  @property
  def dependency_names(self):
    return frozenset(map(lambda x: x.name, self.dependencies))

  def __init__(self, args: Sequence[Hashable]):
    self.args = tuple(args)

  @cached_property
  @abstractmethod
  def func(self):
    ...

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)


EVALARGS = JaxFunction(args=())


NumberOrArrayType = Union[Number, np.ndarray, jnp.ndarray]
ArraySliceType = Union[Int, slice, None]


class Array(JaxFunction):

  def __init__(self, args: Sequence[Hashable], shape: Sequence[Int]):
    self.shape = tuple(map(int, shape))
    assert all(i >= 1 for i in self.shape)
    super().__init__(args)

  def __getitem__(self, item: ArraySliceType | Tuple[ArraySliceType, ...]):
    if not isinstance(item, tuple):
      item = item,
    assert isinstance(item, tuple), NotImplementedError
    shape: Tuple[Int, ...] = ()
    _ = slice(None)
    assert all( it == _ or it is Ellipsis or it is jnp.newaxis for it in item)
    assert item.count(Ellipsis) in (0, 1)
    if Ellipsis in item:
      i = item.index(Ellipsis)
      nslices = item[:i].count(_) + item[i+1:].count(_)
      assert nslices <= self.ndim
      item = item[:i] + (_,) * (self.ndim - nslices) + item[i+1:]
    slicecount = 0
    for it in item:
      if it == _:
        shape = shape + (self.shape[slicecount],)
        slicecount += 1
      else:  # newaxis
        shape = shape + (-1,)
    shape = shape + self.shape[slicecount:]
    new_one_axes = tuple(i for i, j in enumerate(shape) if j == -1)
    # find the insert axes that are needed to have new one axes at the positions in `new_one_axes`
    insert_axes = shp._insert_axes(self.shape, new_one_axes)
    if len(insert_axes):
      return InsertAxis(self, insert_axes)
    return self

  def __iter__(self):
    assert self.shape
    for i in range(self.shape[0]):
      yield Take(self, i, 0)

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def __add__(self, other: NumberOrArrayType | Self):
    return add([self, asarray(other)])

  __radd__ = __add__

  def __mul__(self, other: NumberOrArrayType | Self):
    return multiply([self, asarray(other)])

  __rmul__ = __mul__

  def __truediv__(self, other: NumberOrArrayType | Self):
    return multiply([self, Power(other, -1)])

  __rtruediv__ = __truediv__

  def __neg__(self):
    return (-1) * self

  def __sub__(self, other: NumberOrArrayType | Self):
    return self + (-1) * other

  def __rsub__(self, other):
    return (-1) * self + other

  def __pow__(self, power: Number):
    return Power(self, power)

  def sum(self, axes=None):
    return Contract(self, axes)

  def reshape(self, shape):
    return Reshape(self, shape)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
      The purpose of this function is to prevent numpy from
      taking precedence over own __rmul__ operation.
      >>> weights = np.arange(5)
      >>> weights * myclass
        np.array([Array, Array, Array, Array, Array], dtype=object)
      The above can no longer happen.
      Instead, we get
      >>> weights = np.arange(5)
      >>> weights * myclass
        <class Multiply>
    """
    # Handle multiplication specifically
    if ufunc == np.multiply:
      for input in inputs:
        if isinstance(input, np.ndarray):
          return self.__mul__(input)
    # Handle addition specifically
    elif ufunc == np.add:
      for input in inputs:
        if isinstance(input, np.ndarray):
          return self.__add__(input)
    # XXX: __pow__
    # For other ufuncs or methods, return NotImplemented
    return NotImplemented

  def __repr__(self):
    return f"{self.__class__.__name__}({tuple(sorted(self.dependency_names))}, shape={self.shape})"


ArrayType = Union[Number, np.ndarray, jnp.ndarray, Array]


def asarray(func: ArrayType) -> Array:
  if isinstance(func, Array):
    return func
  return Constant(jnp.asarray(func))


def concatenate(args: Sequence[ArrayType]):
  assert args
  args = list(map(asarray, args))
  if len(args) == 1:
    return args[0]
  return Concatenate(args)


def stack(args: Sequence[ArrayType], axis: Int = 0):
  assert args
  args = list(map(asarray, args))
  if len(args) == 1:
    return args[0]
  return Stack(args)


class Argument(Array):

  @cached_property
  def dependencies(self):
    return frozenset({self})

  def __init__(self, name: str, shape: Sequence[Int]):
    self.name = str(name)
    assert self.name == self.name.strip(), "The argument name contains trailing whitespace"
    super().__init__([EVALARGS], shape)

  @cached_property
  def func(self):
    def _func(safe=True, **kwargs):
      x = jnp.asarray(kwargs[self.name])
      if safe:
        assert x.shape == self.shape
      return x
    return _func


class Zeros(Array):

  def __init__(self, shape: Sequence[Int]):
    super().__init__(args=[EVALARGS], shape=shape)

  @cached_property
  def _zeros(self) -> jnp.ndarray:
    return jnp.zeros(self.shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return self._zeros
    return _func


class Constant(Array):

  def __init__(self, value: np.ndarray | jnp.ndarray | Sequence[Number] | Number,
                     dtype=None):
    value = jnp.asarray(value)
    if dtype is None:
      dtype = value.dtype
    self.value = value.astype(dtype)
    self.dtype = dtype
    super().__init__([EVALARGS], self.value.shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return self.value
    return _func


def asconstant(arr: Constant | Number | np.ndarray | jnp.ndarray, dtype=None) -> Constant:
  if isinstance(arr, Constant):
    if dtype is None:
      return arr
    return arr if arr.dtype == dtype else Constant(arr.value.astype(dtype))
  return Constant(arr, dtype=dtype)


class Pointwise(Array):

  def __init__(self, argument: ArrayType):
    self.argument = asarray(argument)
    super().__init__([self.argument], self.argument.shape)

  @property
  def func(self):
    raise NotImplementedError("Each derived class of `Pointwise` needs to implement its evaluation.")


class Sin(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.sin(self.argument(**kwargs))
    return _func


class Cos(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.cos(self.argument(**kwargs))
    return _func


class Exp(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.exp(self.argument(**kwargs))
    return _func


class Log(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.log(self.argument(**kwargs))
    return _func


class Arccos(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.arccos(self.argument(**kwargs))
    return _func


class Arcsin(Pointwise):
  @property
  def func(self):
    def _func(**kwargs):
      return jnp.arcsin(self.argument(**kwargs))
    return _func


class FunctionTuple(Array):

  def __init__(self, functions: Sequence[ArrayType]):
    self.functions = tuple(map(asarray, functions))
    shape = np.broadcast_shapes(*map(np.shape, self.functions))
    super().__init__(self.functions, shape)


class Power(FunctionTuple):

  def __init__(self, base: ArrayType, power: ArrayType):
    self.base = asarray(base)
    self.power = asarray(power)
    super().__init__((base, power))

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.power(self.base(**kwargs), self.power(**kwargs))
    return _func


class Add(FunctionTuple):
  @cached_property
  def func(self):
    def _func(**kwargs):
      args = [func(**kwargs) for func in self.functions]
      return reduce(lambda x, y: x + y, args)
    return _func


def add(args: Sequence[ArrayType]) -> Add | Zeros | Array:
  if not args: return Zeros(())
  if len(args) == 1:
    return asarray(args[0])
  return Add(args)


class Multiply(FunctionTuple):
  @cached_property
  def func(self):
    def _func(**kwargs):
      args = [func(**kwargs) for func in self.functions]
      return reduce(lambda x, y: x * y, args)
    return _func


def multiply(args: Sequence[ArrayType]) -> Multiply | Constant | Array:
  if not args: return Constant(1.0)
  if len(args) == 1:
    return asarray(args[0])
  return Multiply(args)


class Contract(Array):

  def __init__(self, argument: ArrayType, axes: Int | None | Sequence[Int]):
    self.argument = asarray(argument)
    self.axes = shp._normshape(axes, self.argument.ndim)
    shape = shp._sum_shape(self.argument.shape, self.axes)
    super().__init__([self.argument], shape=shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.sum(self.argument(**kwargs), axis=self.axes)
    return _func


class Concatenate(Array):

  def __init__(self, arguments: Sequence[ArrayType], axis: Int = 0):
    self.arguments = tuple(map(asarray, arguments))
    assert self.arguments
    assert len(set(map(len, (arg.shape for arg in self.arguments)))) == 1

    f, *ignore = self.arguments
    assert (ndim := f.ndim) > 0

    assert -ndim < (axis := int(axis)) < ndim
    self.axis = axis % ndimints(surf(**kwargs), color='blue', point_size=5, render_points_as_spheres=True)

    shp = f.shape[:self.axis] + f.shape[self.axis+1:]
    assert all( (arg.shape[:axis] + arg.shape[axis+1:]) == shp for arg in self.arguments )

    shape = f.shape[:axis] + (sum(arg.shape[axis] for arg in self.arguments),) + f.shape[axis+1:]
    super().__init__(self.arguments, shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.concatenate([arg(**kwargs) for arg in self.arguments], axis=self.axis)
    return _func


class Reshape(Array):

  def __init__(self, argument: ArrayType, shape: Sequence[Int]):
    self.argument = asarray(argument)
    nelems = np.prod(self.argument.shape, dtype=int)
    shape = tuple(map(int, shape))
    if -1 in shape:
      assert shape.count(-1) == 1
      index = shape.index(-1)
      shape = list(shape)
      shape[index] = nelems // np.prod(shape[:index] + shape[index+1:], dtype=int)
    assert np.prod(shape, dtype=int) == nelems
    super().__init__([self.argument], shape=tuple(shape))

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.reshape(self.argument(**kwargs), self.shape)
    return _func


class Ravel(Array):

  def __init__(self, argument: ArrayType):
    self.argument = asarray(argument)
    shape = (np.prod(self.argument.shape, dtype=int),)
    super().__init__([self.argument], shape=shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.ravel(self.argument(**kwargs))
    return _func


class InsertAxis(Array):

  def __init__(self, argument: ArrayType, axes: Int | Sequence[Int]):
    self.argument = asarray(argument)
    self.axes = tuple(axes if isinstance(axes, tuple) else [axes])
    shape = shp._insertaxis_shape(self.argument.shape, self.axes)
    super().__init__([self.argument], shape=shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.reshape(self.argument(**kwargs), self.shape)
    return _func


class Stack(Array):

  def __init__(self, arguments: Sequence[ArrayType], axis: Int = 0):
    self.arguments = tuple(map(asarray, arguments))
    assert self.arguments
    assert len(set(map(len, (arg.shape for arg in self.arguments)))) == 1
    assert len(set(map(np.shape, self.arguments))) == 1, 'All input arrays need to have the same shape'

    f, *tail = self.arguments
    result_ndim = f.ndim + 1

    assert -result_ndim <= (axis := int(axis)) > -result_ndim
    self.axis = axis % result_ndim

    shape = f.shape[:self.axis] + (len(self.arguments),) + f.shape[self.axis:]
    super().__init__(self.arguments, shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.stack([arg(**kwargs) for arg in self.arguments], axis=self.axis)
    return _func


class Take(Array):

  def __init__(self, argument: ArrayType,
                     indices: Int | NumberOrArrayType | slice | Constant,
                     axis: Int):
    self.argument = asarray(argument)
    assert axis > -self.argument.ndim
    self.axis = int(axis) % self.argument.ndim
    if isinstance(indices, slice):
      indices = jnp.arange(self.argument.shape[self.axis], dtype=int)[indices]
    if isinstance(indices, Constant):
      assert indices.dtype == int
      indices = indices.value
    indices = jnp.asarray(indices, dtype=int)
    assert 0 <= axis < self.argument.ndim
    # indices are sorted upon return
    shape, indices = shp._take_shape_and_indices(self.argument.shape, indices, self.axis)
    self.indices = asconstant(indices)
    super().__init__(args=[self.argument], shape=shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      return jnp.take(self.argument(**kwargs), self.indices.value, self.axis)
    return _func


class Cross(Array):

  def __init__(self, t: ArrayType, v: ArrayType):
    self.t = asarray(t)
    self.v = asarray(v)
    assert self.t.shape == self.v.shape
    shape = self.t.shape
    super().__init__([self.t, self.v], shape=shape)

  @cached_property
  def func(self):
    def _func(**kwargs):
      t = self.t(**kwargs)
      v = self.v(**kwargs)
      return jnp.cross(t, v)
    return _func


def array_split(arr: ArrayType, indices_or_sections: Int | Sequence[Int], axis: Int = 0) -> List[Take]:
  arr = asarray(arr)
  indices = jnp.split(jnp.arange(arr.shape[axis]), indices_or_sections)
  return [ Take(arr, myindices, axis) for myindices in indices ]


def polynomial(argument: Argument, weights: np.ndarray | jnp.ndarray) -> Array:
  power = (argument ** jnp.arange(weights.shape[1]))[_]
  return (weights * power).sum(1)


def hermite_curve(x: Array, p0: Array, t0: Array, p1: Array, t1: Array):
  x, p0, t0, p1, t1 = map(asarray, (x, p0, t0, p1, t1))
  shape = p0.shape
  assert shape
  assert all(item.shape == shape for item in (p0, t0, p1, t1))
  x = x[:, _]
  h00 = 2 * x**3 - 3 * x**2 + 1
  h10 = x**3 - 2 * x**2 + x
  h01 = -2 * x**3 + 3 * x**2
  h11 = x**3 - x**2
  return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


""" Tests """


def test_polynomial():
  argument = Argument('arg', (1,))
  weights = np.arange(20).reshape(4, 5)

  pol = polynomial(argument, weights)

  assert (pol(arg=[5]) == (weights * (5 ** np.arange(5))[_]).sum(1)).all()


def test_hermite():

  x = Argument('arg', (1001,))
  p0, t0, p1, t1 = [[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]

  curve = hermite_curve(x, p0, t0, p1, t1)
  from matplotlib import pyplot as plt

  plt.plot(*curve(arg=np.linspace(0, 1, 1001)).T)
  plt.show()


if __name__ == '__main__':
  import ipdb
  ipdb.set_trace()
  test_polynomial()
  test_hermite()
