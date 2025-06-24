from pyhaemo.util import np, _
from pyhaemo.spl import NDSpline, UnivariateKnotVector

import arr

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
  def from_NDSpline(cls, argument: arr.Array, spl: NDSpline):
    from cox import univariate_spline_basis

    # coeffs:  (n, degree+1)
    funcindices, elemindices, coeffs = univariate_spline_basis(spl.knotvector[0])

    # weigts: (n, *spl.shape)
    weights = spl.controlpoints[funcindices]

    # (n, degree + 1, *spl.shape)
    _coeffs = coeffs[(...,) + (_,) * spl.ndim] * weights[:, _]

    (breaks,), (degree,) = spl.knots, spl.degree

    _coeffs = _coeffs.reshape(-1, degree + 1, *_coeffs.shape[1:]).sum(1)

    return cls(argument,
               _coeffs,
               breaks)

  def __init__(self, argument: arr.Array,
                     weights: ArrayLike | arr.Constant,
                     breaks: ArrayLike | arr.Constant):
    self.argument = arr.asarray(argument)
    self.weights = arr.asarray(weights)
    self.breaks = arr.asarray(breaks)
    assert len(self.weights.shape) >= 2
    self.order = self.weights.shape[1]
    shape = self.weights.shape[2:]
    self.interval = jnp.array([self.breaks.value[0], self.breaks.value[-1] - 1e-10])
    assert self.breaks.shape == (self.weights.shape[0] + 1,)
    super().__init__([self.argument], shape=shape)

  def _derivative(self):
    indices = jnp.arange(1, self.order).astype(float)
    weights = self.weights.value[:, 1:] * indices[(_,) + sl + (_,) * len(self.shape)]
    return self._edit(weights=weights)

  @cached_property
  def func(self):
    def _func(**kwargs):
      x = jnp.clip(self.argument(**kwargs), self.interval[0], self.interval[1])
      vals = self.breaks(**kwargs)
      idx = jnp.searchsorted(vals, x, side='right') - 1

      powers = jnp.power(x, jnp.arange(self.order, dtype=float))
      coeffs = self.weights.value[idx]  # shape: (order, ...)
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
