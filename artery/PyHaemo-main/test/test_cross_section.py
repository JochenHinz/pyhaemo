from splico.geo import ellipse
from splico.mesh import rectilinear, mesh_union

import unittest


class TestDisc(unittest.TestCase):

  def test_eval(self):
    sol = ellipse(4, 4, 4)

    for n in (3, 5, 7, 11, 13, 17, 19):
      with self.subTest('Evaluating in {n} x {n} points'):
        rmesh = rectilinear([n, n])

        mesh = mesh_union(*sol.sample_mesh(rmesh), boundary=True)

        # 5 faces, 12 edges, 8 vertices
        npoints = 5 * (n-2)**2 + 12 * (n-2) + 8

        self.assertEqual(len(mesh.points), npoints)


if __name__ == '__main__':
  unittest.main()
