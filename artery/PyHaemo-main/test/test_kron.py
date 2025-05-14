from splico.util import np
from splico.spl.aux import greville_collocation_matrix
from splico.kron import Matrix, MatMul, KroneckerOperator
from splico.spl import UnivariateKnotVector

import unittest


class TestLinearOperator(unittest.TestCase):

  def test_Matrix(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11), 3)
    X = Matrix(greville_collocation_matrix(kv))

    a = np.random.randn(X.shape[0])

    self.assertTrue( np.allclose(X @ a, X.mat @ a) )
    self.assertTrue(X.inv.inv is X)

  def test_matmul(self):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, 6), 3)
    X0 = Matrix(greville_collocation_matrix(kv0))

    X0inv = X0.inv

    X1 = Matrix(2 * X0.mat)

    X1inv = X1.inv

    self.assertTrue( isinstance(X0 @ X1, Matrix) )
    self.assertTrue( isinstance(X0 @ X1inv, MatMul) )
    self.assertTrue( isinstance(X0inv @ X1inv, MatMul) )

    matmul = X1inv @ X0

    self.assertTrue( len((matmul @ matmul).operators) == 4 )

  def test_LU(self):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, 6), 3)
    X0 = Matrix(greville_collocation_matrix(kv0))

    inv = X0.inv

    self.assertTrue( np.allclose(X0.toarray(), inv.inv.toarray()) )


class TestKronecker(unittest.TestCase):

  def test_kronecker(self):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, 6), 3)
    kv1 = UnivariateKnotVector(np.linspace(0, 1, 10), 2)
    kv2 = UnivariateKnotVector(np.linspace(0, 1, 5), 4)

    mat = KroneckerOperator(list(map(greville_collocation_matrix, [kv0, kv1, kv2])))

    rhs = np.random.randn(mat.shape[0])
    self.assertTrue( np.allclose(mat @ rhs, mat.tocsr() @ rhs) )
    self.assertTrue( np.allclose(mat.toarray(), mat.inv.inv.toarray()) )
    self.assertTrue( np.allclose(mat.inv @ rhs, np.linalg.inv(mat.toarray()) @ rhs) )

  def test_empty(self):
    """
    Test if the behavior is as expected when the KroneckerOperator is empty.
    """

    mat = KroneckerOperator([])

    self.assertTrue( bool(mat) is False )
    self.assertTrue( mat.shape == (1, 1) )
    self.assertTrue( (mat @ np.array([5]))[0] == 5 )
    self.assertTrue( (mat.toarray() == np.eye(1)).all() )


if __name__ == '__main__':
  unittest.main()
