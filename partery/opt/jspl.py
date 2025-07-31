from pyhaemo.util import np, _
from pyhaemo.spl import NDSpline, UnivariateKnotVector

import arr
from cox import univariate_spline_basis

from functools import cached_property
from abc import abstractmethod

import jax
from jax import numpy as jnp

from numpy.typing import ArrayLike


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


sl = slice(None),


class Differentiable(arr.Array):

  @abstractmethod
  def _derivative(self):
    ...

  def derivative(self, n=1):
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._derivative().derivative(n=n-1)


class Polynomial(Differentiable):

  def __init__(self, argument: arr.Argument, weights: arr.ArrayType):
    """
     Polynomial of shape argument.shape + weights.shape[1:] while the order is weights.shape[0]
    """
    # assert isinstance(argument, arr.Argument)
    self.weights = arr.asarray(weights)
    self.order = self.weights.shape[0]
    assert self.weights.ndim >= 1
    self.argument = argument
    # assert argument.ndim >= 1, NotImplementedError

    # has shape self.argument.shape + (self.order,) + (1,) * len(self.weights.shape[1:])
    x = (self.argument[..., _] ** jnp.arange(self.order)). \
        reshape(self.argument.shape + (self.order,) + (1,) * (self.weights.ndim-1))

    # has shape self.argument.shape + (self.order,) + (1,) * len(self.weights.shape[1:])
    weights = self.weights.reshape((1,) * self.argument.ndim + self.weights.shape)

    # sum the axis that corresponds to self.weights[0]
    self._func = (x * weights).sum(self.argument.ndim)

    super().__init__([argument, weights],
                      shape=self.argument.shape + self.weights.shape[1:])

  def _derivative(self):
    indices = jnp.arange(1, self.order).astype(float)
    weights = arr.Take(self.weights, np.arange(1, self.order), 0) * indices.reshape(-1, *(1,) * (self.weights.ndim - 1))
    return Polynomial(self.argument, weights)

  @cached_property
  def func(self):
    return self._func


class PieceWisePolynomial(Differentiable):
  """
  Piecewise polynomial function, defined by a list of Polynomial objects.
  """

  @classmethod
  def from_UnivariateKnotVector(cls, argument: arr.Array,
                                     weights: arr.Array,
                                     kv: UnivariateKnotVector):

    # coeffs:  (n, degree+1)
    funcindices, elemindices, coeffs = univariate_spline_basis(kv)

    # weigts: (n, *spl.shape)
    if isinstance(weights, np.ndarray):
      weights = weights[funcindices]
    elif isinstance(weights, arr.Constant):
      weights = weights.value[funcindices]
    else:
      weights = arr.Take(weights, funcindices, 0)

    # (n, degree + 1, *spl.shape)
    _coeffs = coeffs[(...,) + (_,) * (len(weights.shape) - 1)] * weights[:, _]

    breaks, degree = kv.knots, kv.degree

    _coeffs = _coeffs.reshape((-1, degree + 1, *_coeffs.shape[1:])).sum(1)

    return cls(argument, _coeffs, breaks)

  @classmethod
  def from_NDSpline(cls, argument: arr.Array, spl: NDSpline):
    assert spl.nvars == 1, NotImplementedError
    return cls.from_UnivariateKnotVector(argument, spl.controlpoints, spl.knotvector[0])

  def __init__(self, argument: arr.Array, weights: arr.Array, breaks: arr.Array):
    self.argument = arr.asarray(argument)
    self.weights = arr.Constant(weights)
    self.breaks = arr.asarray(breaks)

    assert len(self.weights.shape) >= 2

    self.order = self.weights.shape[1]
    shape = self.weights.shape[2:]

    assert self.breaks.shape == (self.weights.shape[0] + 1,)
    super().__init__([self.argument, self.weights, self.breaks], shape=shape)

  def _derivative(self):
    indices = jnp.arange(1, self.order).astype(float)
    weights = self.weights.value[:, 1:] * indices[(_,) + sl + (_,) * len(self.shape)]
    return self._edit(weights=weights)

  @cached_property
  def func(self):
    def _func(**kwargs):
      vals = self.breaks(**kwargs)
      x = jnp.clip(self.argument(**kwargs), vals[0], vals[-1])
      idx = jnp.searchsorted(vals, x, side='right') - 1

      powers = jnp.power(x, jnp.arange(self.order, dtype=float))
      coeffs = self.weights(**kwargs)[idx]  # shape: (order, ...)
      result = jnp.tensordot(powers, coeffs, axes=([0], [0]))  # broadcasted dot product

      return result
    return _func


if __name__ == '__main__':
  arg = arr.Argument('x', shape=())
  pols = [Polynomial(arg, np.random.randn(3, 3)) for i in range(5)]
  breaks = jnp.linspace(0, 1, len(pols) + 1)

  x = jnp.array(.5)
  ppol = PieceWisePolynomial(arg,
                             jnp.stack([pol.weights.value for pol in pols]),
                             breaks)
  test = ppol ** 2
  func = test.func

  func(x=x)

  @jax.jit
  def wrapper(x):
    return func(x=x)

  grad = jax.jacobian(wrapper)

  import ipdb
  ipdb.set_trace()

  kv = UnivariateKnotVector(np.linspace(0, 1, 11))
  spl = NDSpline([kv], np.random.randn(kv.dim, 3))

  test = PieceWisePolynomial.from_NDSpline(arg, spl)
