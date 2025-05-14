from splico.mesh import Mesh, rectilinear, Triangulation, mesh_union, PointMesh, \
                        LineMesh
from splico.util import global_precision

import unittest
from itertools import product

import numpy as np

_ = np.newaxis


def order_mesh(mesh: Mesh):
  """Sort mesh by points and then by lexsorting the elements."""
  points, reordering, inv = np.unique(mesh.points, axis=0, return_index=True, return_inverse=True)
  return mesh.__class__(inv[mesh.elements], points).lexsort_elements()


def unit_disc_triangulation(npoints=11, **kwargs):
  xi = np.linspace(0, 2 * np.pi, npoints)[:-1]
  points = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  return Triangulation.from_polygon(points, **kwargs)


class UnitSquare(unittest.TestCase):

  def test_rectilinear(self):
    points = [ np.linspace(0, 1, n) for n in (4, 5, 6) ]
    lengths = list(map(len, points))
    for i in range(1, 4):
      with self.subTest(f'Rectilinear of dimension {i}'):
        mesh = rectilinear(points[:i])

        with global_precision(8):
          element = np.asarray(list(product(*[range(2) for _ in range(i)]))).T
          indices = np.arange(np.multiply.reduce(lengths[:i])).reshape(*lengths[:i])
          ijk = np.stack(list(map(np.ravel, np.meshgrid(*(np.arange(n-1) for n in lengths[:i]), indexing='ij'))), axis=0)

          # 3, nelems, 8
          ijk = ijk[:, :, _] + element[:, _]
          elements = indices[tuple(ijk)]

          mypoints = np.stack(list(map(np.ravel, np.meshgrid(*points[:i], indexing='ij'))), axis=-1)

          self.assertTrue( (mesh.elements == elements).all() )
          self.assertTrue( np.allclose(mesh.points[:, :i], mypoints) )
          self.assertTrue( mesh.is_valid() )


class RefineAndGeometryMap(unittest.TestCase):

  def test_refine_structured(self):
    lengths = (4, 5, 6)

    for i in range(1, 4):
      with self.subTest(f'Structured refinement of dimension {i}.'):
        # make mesh and refine it
        mesh0 = rectilinear(lengths[:i])
        self.assertTrue(mesh0.is_valid())
        mesh1 = order_mesh(rectilinear([2 * n - 1 for n in lengths[:i]]))
        rmesh = order_mesh(mesh0.refine(1).drop_points_and_renumber())

        self.assertEqual(mesh1.elements.shape, rmesh.elements.shape)
        self.assertEqual(mesh1.points.shape, rmesh.points.shape)

        # sort and take unique to change to default ordering
        self.assertTrue( (np.unique(np.sort(mesh1.elements, axis=1), axis=0) == np.unique(np.sort(rmesh.elements, axis=1), axis=0)).all() )
        self.assertTrue( np.allclose(mesh1.points, rmesh.points) )

        nverts_r = mesh0.points.shape[0]
        while True:
          nverts_r += mesh0.elements.shape[0]  # for each submesh element, we get one additional point
          if isinstance(mesh0, LineMesh):
            break
          mesh0 = mesh0.submesh

        self.assertEqual(nverts_r, rmesh.points.shape[0])

  def test_refine_triangulation(self):
    mesh = unit_disc_triangulation()

    with global_precision(8):

      rmesh = mesh.refine(1)

      evalf = np.stack(list(map(np.ravel, np.meshgrid(*[np.array([0, 0.5, 1])]*2))), axis=1)
      evalf = evalf[ evalf.sum(1) < 1.000001 ]

      allpoints = np.round(mesh.eval_local(evalf).reshape(-1, 3), 8)

      u0 = np.unique(rmesh.points, axis=0)
      u1 = np.unique(allpoints, axis=0)

      self.assertTrue( u0.shape == u1.shape )
      self.assertTrue( np.allclose(u0, u1) )
      self.assertTrue( rmesh.is_valid() )


class TestBoundary(unittest.TestCase):
  """Boundary testing based on getting the expected points."""
  # XXX: also do an elements check

  def test_triangulation_boundary(self):
    npoints = 11
    mesh_size = 2 * 2 * np.pi / (npoints - 1)
    mesh = unit_disc_triangulation(mesh_size=mesh_size)

    dmesh = mesh.boundary.drop_points_and_renumber()

    xi = np.linspace(0, 2*np.pi, npoints)[:-1]
    circumference = np.stack([np.cos(xi), np.sin(xi)], axis=1)

    u0 = np.unique(circumference, axis=0)
    u1 = np.unique(dmesh.points[:, :2], axis=0)

    self.assertTrue( np.allclose(u0, u1) )
    self.assertTrue( dmesh.is_valid() )

  def test_boundary_structured(self):
    r"""
      Here we test by checking if the boundary meshes points shifted
      by -0.5 in all directions have at least one entry that is
      \pm 0.5.
    """
    npoints = 11
    for i in range(1, 4):
      with self.subTest(f'Taking boundary of {i}-dimensional rectilinear mesh.'):
        mesh = rectilinear((npoints,) * i)
        with global_precision(8):
          dmesh = mesh.boundary.drop_points_and_renumber()
          points_c = dmesh.points - np.array([.5] * dmesh.points.shape[1])[_]
          self.assertTrue( np.allclose(np.abs(points_c).max(axis=1), .5) )


class TestSubMesh(unittest.TestCase):

  def test_structured(self):
    mesh = rectilinear((4, 5, 6))
    for i in range(3):
      mesh = mesh.submesh
      self.assertTrue(mesh.is_valid())
    from splico.err import HasNoSubMeshError
    with self.assertRaises(HasNoSubMeshError):
      mesh.submesh

  def test_unstructured(self):
    mesh = unit_disc_triangulation()
    self.assertTrue(mesh.submesh.is_valid())


class TestUnion(unittest.TestCase):

  def test_union(self):
    npoints = (3, 3, 4)
    for i in range(1, 4):
      with self.subTest('Testing the mesh union in {i} spatial dimensions.'):
        with global_precision(8):
          meshes = [ rectilinear(points) for points in product(*([np.linspace(0, .5, n),
                                                                  np.linspace(.5, 1, n)] for n in npoints[:i])) ]
          mesh0 = order_mesh(rectilinear([2 * n - 1 for n in npoints[:i]]))
          mesh1 = order_mesh(mesh_union(*meshes))

          self.assertTrue( np.allclose(mesh0.points, mesh1.points) )
          self.assertTrue( mesh1.is_valid() )


class TestEval(unittest.TestCase):

  def test_eval(self):
    for i in range(1, 4):
      with self.subTest('Testing mesh.eval in {i} spatial dimensions.'):
        mesh = rectilinear([np.linspace(0, 1, 6) for j in range(i)])
        points = np.stack(list(map(np.ravel, np.meshgrid(*[np.linspace(0, 1, 2)] * i, indexing='ij'))), axis=1)

        self.assertTrue(np.allclose(mesh.eval_local(points), mesh.points[mesh.elements]))
        for j, dx in enumerate(np.eye(i)):
          self.assertTrue(np.allclose(mesh.eval_local(points, dx=dx)[..., j], 0.2))

    with self.subTest('Testing PointMesh'):
      # This one is important to test because PointMesh's local ordinances
      # have dimension 1, not 2.
      points = np.random.randn(20, 3)
      elements = np.arange(20)[:, None]
      mesh = PointMesh(elements, points)

      points = mesh._local_ordinances(1)
      self.assertTrue(np.allclose(mesh.eval_local(points),
                                  mesh.points[mesh.elements]))


if __name__ == '__main__':
  unittest.main()
