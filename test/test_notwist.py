from pyhaemo.geo.notwist import compute_notwistframe_from_spline, \
                                _compute_notwistframe_end_to_end
from pyhaemo.geo.disc import ellipse

from pyhaemo.mesh import rectilinear, mesh_union
from pyhaemo.spl import UnivariateKnotVector, TensorKnotVector
from pyhaemo.util import np, _

import unittest


class TestNoTwist(unittest.TestCase):

  def test_notwist(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 21))
    kv = TensorKnotVector([kv])
    radii = np.linspace(1, 2, 51)

    xi = np.linspace(0, 1, 101)
    centerline_points = np.stack([ 10 * xi, np.zeros_like(xi), 10 * xi**2 ], axis=1)
    X = kv.fit([xi], centerline_points)

    xi_notwist = np.linspace(0, 1, 51)
    Rs = compute_notwistframe_from_spline(X, xi_notwist)

    self.assertTrue( all( (np.abs(np.linalg.det(R)) - 1) < 1e-10 for R in Rs ) )

    disc = ellipse(1, 1, 4)
    print(disc)
    points = disc.arr[()].controlpoints.reshape(-1, 3)[:, :2]
    print(points.max())
    from matplotlib import pyplot as plt
    plt.scatter(*points.T)
    plt.show()
    rRs = kv.fit([xi_notwist], radii[:, _, _] * Rs)

    one = disc.one(disc.knotvector)

    rRs = rRs[_]
    vessel = (disc[:, _] @ rRs).sum(-1)
    vessel = vessel + (one @ X)[_]

    eval_mesh = rectilinear([5, 5, 101])

    mesh = mesh_union(*vessel.sample_mesh(eval_mesh), boundary=True)

    mesh.plot()

  def test_notwist_end_to_end(self):
    xi = np.linspace(0, 1, 101)

    kv = UnivariateKnotVector(np.linspace(0, 1, 11)).to_tensor()
    data = np.stack([xi, np.zeros(101), np.zeros(101)], axis=1)

    spl = kv.fit([xi], data, lam0=0)

    n0 = np.array([0, 1, 0]).astype(float)
    nfinal = np.array([0, .5, .5]).astype(float)

    frame = _compute_notwistframe_end_to_end(spl(xi, dx=1),
                                            n0,
                                            nfinal,
                                            1e-8)

    rRs = kv.fit([xi], frame, lam0=0)
    ell = ellipse(1, 1, 4)

    vessel = (ell[:, None] @ rRs).sum(-1) + ell.unity @ spl

    eval_mesh = rectilinear([5, 5, 10])
    mesh = mesh_union(*vessel.sample_mesh(eval_mesh), boundary=True)

    mesh.plot()

    # XXX: currently the test passes or fails by the `measure of eye`.
    #      Find a way to test whether the boundary union etc succeeds without
    #      having to look at it manually.

if __name__ == '__main__':
  unittest.main()
