import numpy as np
from scipy import optimize
from scipy.integrate import quad
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from functools import wraps, lru_cache

_ = np.newaxis


def frozen(arr, dtype=float):
  arr = np.asarray(arr, dtype=dtype)
  arr.flags.writeable = False
  return arr


def default_interval(fn):
  @wraps(fn)
  def wrapper(self, interval=None, **kwargs):
    if interval is None:
      interval = [0, 1]
    return fn(self, interval, **kwargs)
  return wrapper


@lru_cache
def _cached_derivative(pol):
  # print(PolynomialCurve(pol.weights[1:] * np.arange(1, pol.order)[:, _]).weights)
  return PolynomialCurve(pol.weights[1:] * np.arange(1, pol.order)[:, _])


class PolynomialCurve:

  def __init__(self, weights):
    self.weights = frozen(weights)
    assert len(self.weights.shape) == 2, NotImplementedError
    assert self.weights.shape[1:] == (3,), NotImplementedError
    self.order = len(self.weights)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.weights.tobytes(), self.weights.shape))
    return self._hash

  def __eq__(self, other):
    if self.__class__ != other.__class__: return False
    if self.weights.shape != other.weights.shape: return False
    return self.__hash__() == other.__hash__()
  
  def _derivative(self):
    return _cached_derivative(self)

  def derivative(self, n=1):
    if n == 0:
      return self
    return self._derivative().derivative(n=int(n-1))

# da chiedere bene a jochen cosa fa questo comando.
  def __call__(self, xi):
    if np.isscalar(xi):
      xi = np.array([xi])
    xi = xi[_]
    return (self.weights[:, _] * (xi ** (np.arange(self.order)[:, _]))[..., _]).sum(0)

  @default_interval
  def length(self, interval):
    a, b = interval
    assert a < b
    der = self.derivative()
    return quad(lambda s: np.sqrt((der(s)**2).sum()), a, b)[0]

  @default_interval
  def smoothness(self, interval):
    a, b = interval
    assert a < b
    der = self.derivative(2)
    return quad(lambda s: (der(s) ** 2).sum(), a, b)[0]

  @default_interval
  def find_approximate_arclength_breakpoints(self, interval, abscissae: np.ndarray, npoints: int):
      abscissae = np.asarray(abscissae)
      assert np.logical_and(abscissae >= 0, abscissae <= 1).all()
      assert npoints > 1
      xi = np.linspace(*interval, npoints)  # [xi0, xi1, ...]
      points = self(xi)  # [p0; p1; p2; ...]
      dx = np.array([0, *np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()])  # [0, L0, L0 + L1, ...]  Li = distance between pi+1 and pi
      dx = dx / dx[-1]  # total length of the discrete curve normalised
      argmins = np.argmin(np.abs(dx[:, _] - abscissae[_]), axis=0)  # find the positions in dx such that |dx[argmins[i]] - L * abscissae[i]| is minimised
      return np.asarray(argmins, dtype=int)  # return the xi that correspond to the entries in 'argmins'

  @default_interval
  def plot(self, interval, npoints=1001, ax=None, show=True):
    if ax is None:
      fig = plt.figure()
      ax = plt.axes(projection='3d')
    else:
      fig = ax.figure

    xi = np.linspace(0, 1, npoints)
    ax.plot3D(*self(xi).T)

    if show:
      plt.show()

    return fig, ax


def cubic_hermite_curve(p0, t0, p1, t1):
  weights = [p0, t0, -3 * p0 - 2 * t0 + 3 * p1 - t1, 2 * p0 + t0 - 2 * p1 + t1]
  return PolynomialCurve(weights)


def optimize_smoothness(p0, t0, p1, t1, p2, t2, v0=None, method='smooth', ax=None, show=False):
  assert method in ('smooth', 'length')
  p0, t0, p1, t1, p2, t2 = map(np.asarray, (p0, t0, p1, t1, p2, t2))

  if ax is None:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
  else:
    fig = ax.figure
  
  if v0 is None:
      def make_curves(c):
        v0, t0p, t1p, t2p = c.reshape(-1, 3)
        curve0 = cubic_hermite_curve(p0, t0, v0, t0p)
        curve1 = cubic_hermite_curve(p1, t1, v0, t1p)
        curve2 = cubic_hermite_curve(p2, t2, v0, t2p)
        return curve0, curve1, curve2  

      x0 = np.asarray([(p0 + p1 + p2) / 3, t0, t1, t2]).ravel()
  else:
      v0 = np.asarray(v0)
      def make_curves(c):
        t0p, t1p, t2p = c.reshape(-1, 3)
        curve0 = cubic_hermite_curve(p0, t0, v0, t0p)
        curve1 = cubic_hermite_curve(p1, t1, v0, t1p)
        curve2 = cubic_hermite_curve(p2, t2, v0, t2p)
        return curve0, curve1, curve2  

      x0 = np.asarray([t0, t1, t2]).ravel()
    
  if method == 'smooth':
      def costfunc(c):
        curves = make_curves(c)
        return sum(curve.smoothness() / curve.length() for curve in curves)
  else:
      def costfunc(c):
        curves = make_curves(c)
        return sum(curve.length() for curve in curves)
  
  x = optimize.minimize(costfunc, x0, tol=1e-10).x
   
  curves = make_curves(x)

  ax.scatter(*np.stack([p0, p1, p2]).T)
  T = np.stack([t0, t1, t2])
  T = T * .01
  ax.quiver(*np.stack([p0, p1, p2]).T, *T.T, normalize=True, color='k')
  for curve in curves:
    curve.plot(ax=ax, show=False)
  
  if show:
    plt.show()
  
  return  (x[:3] if v0 is None else v0), x[:3], x[3:6], x[6:9] , fig, ax


def test():

  p0 = np.array([-1, 0, 0])
  p1 = np.array([1, 0, 0])
  p2 = np.array([.5, 1, 0])

  t0 = [.5, 1, .5]
  t1 = [-.5, 1, .5]
  t2 = [0, 1, 1]
  
  v0 = (p0 + p1 + p2) / 3

  v0, t0, t1, t2, fig, ax = optimize_smoothness(p0, t0, p1, t1, p2, t2, v0=v0, show=False)
  # fig, ax = optimize_smoothness(p0, t0, p1, t1, p2, t2, v0=v0, method='length', ax=ax, show=False)

  plt.show()


if __name__ == '__main__':
  test()
