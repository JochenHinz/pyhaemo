from splico.spl import UnivariateKnotVector, TensorKnotVector
from splico.mesh import rectilinear
from splico.util import np, flat_meshgrid

from typing import Sequence


def main(nelems: Sequence[int], verts: Sequence[np.ndarray], data: np.ndarray, **fitkwargs) -> None:
  """
    Example script for fitting a spline characterized by its knotvector
    to point data.

    Parameters
    ----------
    nelems : Sequence of :class:`int` of length between 1 and 3.
        The number of elements in each direction of the tensor-product spline.
    verts : Sequence of :class:`np.ndarray`, one-dimensional and monotone
            increasing
        The vertices in each direction at which the data points are assumed.
        The global vertices follow from an outer product.
    data : :class:`np.ndarray`
        The data points. Must satisfy data.shape[0] == np.prod(map(len, verts))
        if there are more axes, a multidimensional spline is fit to the data.
  """
  assert len(verts) <= 3, NotImplementedError
  assert data.shape[0] == np.prod(list(map(len, verts)))

  knotvector: TensorKnotVector = \
      np.prod([UnivariateKnotVector(np.linspace(0, 1, n)) for n in nelems])

  spline = knotvector.fit(verts, data, **fitkwargs)

  mesh = spline.sample_mesh(rectilinear(knotvector.refine(...).knots))

  mesh.plot()


if __name__ == '__main__':
  nelems = [3, 4, 5]
  verts = [np.linspace(0, 1, n) for n in (10, 11, 12)]
  x, y, z = flat_meshgrid(*verts, axis=0)
  data = np.stack([x, y, z + x + y], axis=1)

  main(nelems, verts, data)
