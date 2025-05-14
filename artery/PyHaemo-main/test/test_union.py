from splico.mesh import rectilinear, mesh_union
from splico.mesh.bool import make_matching

import unittest

import numpy as np


class TestUnivariateKnotVector(unittest.TestCase):

  def test_boundary_union(self):
    xi = np.linspace(0, 1, 11)
    mesh0 = rectilinear([np.linspace(0, .5, 11), xi, xi])
    mesh1 = rectilinear([np.linspace(.5, 1, 11), xi, xi])

    points0 = mesh0.boundary.drop_points_and_renumber()
    points1 = mesh1.boundary.drop_points_and_renumber()

    # test matching
    matching = make_matching(points0, points1, 1e-7)

    self.assertEqual(len(matching), 121)

    mesh = mesh_union(mesh0, mesh1, boundary=True)
    mesh.plot()

    # XXX: test to make sure the boundary union gives the correct result


if __name__ == '__main__':
  unittest.main()
