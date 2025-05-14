from pyhaemo.util import _
from pyhaemo.spl import UnivariateKnotVector, TensorKnotVector
from pyhaemo.mesh import rectilinear
from pyhaemo.geo import ellipse
from pyhaemo.spl import NDSpline, NDSplineArray

import operator
import unittest

import numpy as np


class TestNDSpline(unittest.TestCase):

  def test0D(self):
    knotvector = TensorKnotVector(())
    data = np.random.randn(4, 3)[_]

    spline = knotvector.fit([], data)

    self.assertTrue(spline().shape == (4, 3))

  def test1D(self):

    uknotvector = UnivariateKnotVector(np.linspace(0, 1, 51), 3)
    knotvector = TensorKnotVector([uknotvector])

    ndatapoints = 11
    xi = np.linspace(0, 1, ndatapoints) ** 2
    data = xi[:, _, _] + .4 * np.random.randn(*(ndatapoints,), 4, 3)
    spline = knotvector.fit([xi], data, lam0=1e-6)

    test = spline(xi)
    self.assertTrue(test.shape == (len(xi), 4, 3))

  def test2D(self):

    uknotvector = UnivariateKnotVector(np.linspace(0, 1, 21), 3)
    knotvector = uknotvector @ uknotvector

    ndatapoints = 11
    xi = np.linspace(0, 1, ndatapoints)
    data = (xi[:, _, _, _] + .2 * np.random.randn(*(ndatapoints,)*2, 4, 3)).reshape(-1, 4, 3)
    spline = knotvector.fit([xi, xi], data, lam0=1e-6)

    test = spline(xi, xi)
    self.assertTrue(test.shape == (len(xi), 4, 3))

  def test_arithmetic(self):
    uknotvector = UnivariateKnotVector(np.linspace(0, 1, 21), 3)
    knotvector = uknotvector @ uknotvector

    ndatapoints = 11
    xi = np.linspace(0, 1, ndatapoints)
    data = (xi[:, _, _, _] + .2 * np.random.randn(*(ndatapoints,)*2, 4, 3)).reshape(-1, 4, 3)
    spline = knotvector.fit([xi, xi], data, lam0=1e-6)

    for op in (operator.add, operator.mul, operator.truediv, operator.sub):
      # adding number doesn't change shape
      self.assertEqual(op(spline, 5).shape, spline.shape)

      # adding array that is shorter but not scalar should not change the shape
      self.assertEqual(op(spline, np.random.randn(1, 3)).shape, spline.shape)

      # adding array of shape spline.shape doesn't change array
      self.assertEqual(op(spline, np.random.randn(4, 3)).shape, spline.shape)

      # adding bigger array changes shape
      self.assertEqual(op(spline, np.random.randn(3, 4, 3)).shape, (3,) + spline.shape)

    self.assertTrue(np.allclose((spline + spline).controlpoints,
                                spline.controlpoints + spline.controlpoints))
    self.assertTrue(np.allclose((spline - spline).controlpoints,
                                spline.controlpoints - spline.controlpoints))

  def test_add(self):
    with self.subTest('Add KnotVectors with the same upper and lower bounds.'):
      kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
      kv = kv0 @ kv1

      spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))
      spl1 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

      self.assertTrue( np.allclose((spl0 + spl1).controlpoints, spl0.controlpoints + spl1.controlpoints) )

      # test if addition is added to each DOF individually
      offset = np.random.randn(3, 2, 4)
      self.assertTrue( np.allclose((spl0 + offset).controlpoints, spl0.controlpoints + offset[_]) )

    with self.subTest('Add knotvectors with unequal upper and lower bounds.'):
      kv0, kv1 = [UnivariateKnotVector(np.linspace(-i, 1, n)) for i, n in enumerate((5, 7))]

      spl0 = NDSpline(kv0, np.random.randn(kv0.dim, 3, 2, 4))
      spl1 = NDSpline(kv1, np.random.randn(kv1.dim, 3, 2, 4))
      spl = spl0 + spl1

      xi = np.linspace(-1, 1, 1001)

      self.assertTrue( np.allclose( spl0(xi) + spl1(xi), spl(xi) ) )

      spl0 = NDSpline(kv1 @ kv0, np.random.randn(kv0.dim * kv1.dim, 3, 2, 4))
      spl1 = NDSpline(kv0 @ kv1, np.random.randn(kv0.dim * kv1.dim, 3, 2, 4))
      spl = spl0 + spl1

      xi, eta = [np.linspace(-1, 1, 1001)]*2

      self.assertTrue( np.allclose( spl0(xi, eta, tensor=True)
                                    + spl1(xi, eta, tensor=True), spl(xi, eta, tensor=True) ) )

  def test_sum(self):
    kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
    kv = kv0 @ kv1

    spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

    # make sure but ways of summing are valid
    spl0_s = spl0.sum(axis=(0, 1))
    spl0_s = spl0.sum(0, 1)

    self.assertTrue(spl0_s.controlpoints.shape == (kv.ndofs, 4))
    self.assertTrue( np.allclose(spl0_s.controlpoints, spl0.controlpoints.sum((1, 2))) )

  def test_matmul(self):
    kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
    kv = kv0 @ kv1

    spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

    offset = np.random.randn(3, 2, 4)
    self.assertTrue( np.allclose((spl0 * offset).controlpoints, spl0.controlpoints * offset[_]) )

  def test_mul(self):
    # forthcoming
    pass

  def test_prolong(self):
    disc = ellipse(1, 1, 4).arr[()]
    kv = disc.knotvector.refine(...)
    disc_r = disc.prolong_to(kv)

    self.assertTrue(np.allclose(disc[0](*kv.knots),
                                disc_r[0](*kv.knots)))

    disc_r = disc.refine(...)

    self.assertTrue(np.allclose(disc[0](*kv.knots),
                                disc_r[0](*kv.knots)))

    self.assertTrue(all(np.allclose(kn0, kn1) for kn0, kn1 in zip(disc_r.knots, kv.knots)))

  def test_degree_elevate(self):
    # 1D
    kv = UnivariateKnotVector(np.linspace(0, 1, 5), 3).to_tensor()
    spl = kv.fit([np.linspace(0, 1, 11)], np.random.randn(11, 3))

    for i in range(4):

      myspl = spl.raise_multiplicities([0], [[2]], [i])
      splp = myspl.degree_elevate(1)

      xi = np.linspace(0, 1, 301)
      X, Y = spl(xi), splp(xi)

      self.assertTrue( np.allclose(X, Y) )

    # 2D
    disc = ellipse(1, 1, 4).arr[()][0]

    for i in range(4):
      mydisc = disc.raise_multiplicities([0, 1], [3, 2], [i, i])

      xi = disc.knots
      discp = mydisc.degree_elevate(1)

      X, Y = discp(*xi), mydisc(*xi)

      self.assertTrue( np.allclose(X, Y) )

  def test_split_join(self):
    # direction 0, position 4, amount 2
    spl = ellipse(1, 1, 10)[2].arr[()].raise_multiplicities([0], [4], [2])

    # split at position 4
    spl0, spl1 = spl.split(0, positions=4)

    # glue back together
    spl_ = spl0.join(spl1, 0)

    # make sure the spl with raised multiplicities is the same as the joined spl
    self.assertTrue(np.allclose(spl.controlpoints, spl_.controlpoints))

  def test_split_join_by_xvals(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 14), 3).to_tensor()
    spl = NDSpline(kv, np.random.randn(kv.ndofs, 3))

    # give him value that reaches machine precision
    # should be rounded to the global precision
    spl0, spl1 = spl.split( 0, xvals=[1/3] )
    spl = spl.add_knots(0, knotvalues=[1/3]).raise_multiplicities(0, positions=[5], amounts=[3])

    self.assertTrue( np.allclose(np.concatenate([spl0.controlpoints, spl1.controlpoints], axis=0),
                                 spl.controlpoints) )

  def test_split_join_multiple(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11), 3).to_tensor()
    X = kv.fit([np.linspace(0, 1, 11)], np.random.randn(11, 3))

    direction = 0
    xvals = .3, .6, .9

    Xs = X.split(direction, xvals=xvals)
    X = X.add_knots(direction, knotvalues=[xvals])
    X = X.raise_multiplicities(direction, positions=[np.searchsorted(X.knots[direction], xvals)],
                                          amounts=[2])

    self.assertTrue(np.allclose(X.controlpoints, Xs[0].join(Xs[1:], direction).controlpoints))

  # def test_tensor_call(self):
  #   ell = ellipse(1, 1, 4).to_ndim(1).arr[0]
  #   xi = [np.linspace(0, 1, i) for i in (5, 7)]
  #   Xi = list(map(np.ravel, np.meshgrid(*xi, indexing='ij')))
  #   X, Y = ell(*xi, tensor=True), ell(*Xi)

  #   self.assertTrue(np.allclose(X, Y))


class TestFitSample(unittest.TestCase):

  def test_fit_spline2D(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    kv = kv @ kv
    abscissae = np.linspace(0, 1, 21)
    x, y = map(np.ravel, np.meshgrid(abscissae, abscissae))
    z = np.zeros_like(x)
    spline = kv.fit([abscissae] * 2, np.stack([x, 1 + x + y, z], axis=1))
    mesh = rectilinear((21, 21))
    sampled_mesh = spline.sample_mesh( mesh )
    self.assertTrue( (np.abs(sampled_mesh.points[:, :2] - np.stack([x, 1 + x + y], axis=1)) < 1e-2).all() )

  def test_fit_spline3D(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    kv = kv @ kv @ kv
    abscissae = np.linspace(0, 1, 21)
    x, y, z = map(np.ravel, np.meshgrid(abscissae, abscissae, abscissae))
    spline = kv.fit([abscissae] * 3, np.stack([x, 1 + x + y, z], axis=1))
    mesh = rectilinear((21, 21, 21))
    sampled_mesh = spline.sample_mesh( mesh )
    self.assertTrue( (np.abs(sampled_mesh.points - np.stack([x, 1 + x + y, z], axis=1)) < 1e-2).all() )


class TestNDSplineArray(unittest.TestCase):

  def test_new(self):
    xi = np.linspace(0, 1, 11)
    kv0 = UnivariateKnotVector(np.linspace(0, 1, 6))
    kv1 = UnivariateKnotVector(xi)
    kv = kv0 @ kv1
    spline = NDSpline(kv, np.random.randn(kv.ndofs, 2, 3))
    spline0 = NDSpline(kv, np.random.randn(kv.ndofs, 2, 3, 4))

    arr = NDSplineArray([ [spline, spline], [spline, spline] ])
    self.assertTrue(arr(xi, xi).shape == (11, *arr.shape))

    with self.assertRaises(AssertionError):
      test = NDSplineArray([[spline, spline0], [spline, spline]])

    self.assertTrue( all(np.allclose( spl0.controlpoints, 2 * spl1.controlpoints ) for spl0, spl1 in zip((spline + spline).ravel(), spline.ravel())) )

  def test_ravel(self):
    disc = ellipse(1, 1, 4)
    self.assertTrue(disc.ravel().shape == (15,))

  def test_arithmetic(self):
    import random
    xi = np.linspace(0, 1, 11)
    B = ellipse(1, 1, 4)
    A = B.arr[()]
    for i in range(3):
      self.assertTrue( np.allclose(A(xi, xi), B(xi, xi)) )
      try:
        B = B.expand()
      except Exception:
        pass
    self.assertTrue(isinstance(A + B, NDSplineArray))
    self.assertTrue((A[:, _] + B[_]).shape == (5, 5, 3))

    B_ = B.contract_all()
    C_ = B.expand_all()

    for C, B in zip((C_, C_[0], C_[0, 0]), (B_, B_[0], B_[0, 0])):
      for j in range(2):
        for i in range(B.ndim, 10):
          # create numbers from o to i-1 and shuffle them
          no_none_indices = list(range(i))
          random.shuffle(no_none_indices)

          # keep only the first i-ndim indices
          no_none_indices = no_none_indices[:(i-B.ndim)]

          # item is given by slices only
          item = [slice(None)] * i

          # add i - 2 Nones in random places
          for ind in no_none_indices:
            item[ind] = None

          # convert to tuple
          item = tuple(item)

          # check if `B` which is partially expanded has the same shape as `C`
          # which is fully expanded give the same shape. The shape of `C.arr`
          # should always be correct because it is is a numpy array with thse shape
          # of B
          self.assertTrue(B[item].shape == C.arr[item].shape)

        B = B.expand(n=1 if B._elemdim else 0)

    B_ = NDSplineArray(A)
    for B in (B_, B_[0], B_[0, 0]):
      C = B.expand_all()
      final_shape = (4,) * 3 + B.shape
      for i in range(len(final_shape)):
        myshape = final_shape[::-1][:i][::-1]
        y = np.asarray(np.random.randn(*myshape))  # coerce to array in case float is returned
        shp = np.broadcast_shapes(y.shape, B.shape)
        sm = B + y
        self.assertTrue(sm.shape == shp)
        self.assertTrue( (np.array(C.arr + y) == sm.expand_all().arr).all() )

  def test_sum(self):
    import random
    kv = UnivariateKnotVector(np.linspace(0, 1, 5))
    for i in range(4):
      tkv = TensorKnotVector([kv] * i)
      for ndim in range(5):
        data = np.random.randn(*((tkv.ndofs,) + (3,) * ndim))
        spl = NDSpline(tkv, data)
        B = NDSplineArray(spl)
        C = B.expand_all()
        for j in range(ndim):
          for k in range(j):
            indices = list(range(ndim))
            random.shuffle(indices)
            indices = tuple(indices[:k])
            self.assertTrue((C.arr.sum(indices) == B.sum(indices).expand_all().arr).all())
            self.assertTrue(B.sum(indices).contract_all().arr.ravel()[0] == spl.sum(indices))
          B = B.expand()


class TestSymbolic(unittest.TestCase):
  """
  Test the routines that are used to perform symbolic operations between
  NDSpline and NDSplineArray objects.
  """

  def test_transpose_dependencies(self):
    with self.subTest('empty transpose_dependencies'):
      spl = NDSpline([], np.random.randn(1, 3, 4))

      # should not raise if `axes` is empty
      self.assertTrue(spl is spl.transpose_dependencies(()))

      # should raise because there are no axes to shuffle
      self.assertRaises(Exception, lambda: spl.transpose_dependencies((0, 1)))

    with self.subTest('Test transpose_dependencies normal input'):
      kv = TensorKnotVector([ UnivariateKnotVector(np.linspace(0, 1, i)) for i in (4, 5, 6) ])

      xi = [np.linspace(0, 1, i) for i in (11, 7, 21)]

      spl = NDSpline(kv, np.random.randn(kv.ndofs, 3, 4))
      splp = spl.transpose_dependencies((2, 0, 1))

      Xi = np.stack(list(map(np.ravel, np.meshgrid(*xi, indexing='ij'))), axis=1)
      Xip = Xi[:, [2, 0, 1]]
      self.assertTrue( np.allclose( spl(*Xi.T), splp(*Xip.T) ))

    # should be vectorized for NDSplineArray
    with self.subTest('Test transpose_dependencies NDSplineArray'):
      kv = TensorKnotVector([ UnivariateKnotVector(np.linspace(0, 1, i)) for i in (4, 5, 6) ])

      spl = NDSplineArray(NDSpline(kv, np.random.randn(kv.ndofs, 3, 4)))
      splp = spl.transpose_dependencies((2, 0, 1))

      self.assertTrue( ( np.asarray(spl.arr[()].tcontrolpoints.shape[:3])[[2, 0, 1]] ==
                         np.asarray(splp.arr[()].tcontrolpoints.shape[:3])).all())

  def test_insert_dependencies(self):
    with self.subTest('empty insert_dependencies'):
      spl = NDSpline([], np.random.randn(1, 3, 4))

      # special case: adding no dependencies should not change the spline
      self.assertTrue(spl is spl.insert_dependencies((), ()))

      # should raise because there are no axes to shuffle
      self.assertRaises(Exception, lambda: spl.insert_dependencies((0, 1), (0, 1)))

      # make sure adding dependencies to constant spline works
      new_spl = spl.insert_dependencies((0, 0), [UnivariateKnotVector(np.linspace(0, 1, 4))] * 2)
      self.assertTrue( new_spl.shape == spl.shape and new_spl.nvars == 2 )

    with self.subTest('Test insert_dependencies normal input'):
      kv = TensorKnotVector([ UnivariateKnotVector(np.linspace(0, 1, i)) for i in (4, 5, 6) ])

      xi = [np.linspace(0, 1, i) for i in (11, 7, 21)]

      spl = NDSpline(kv, np.random.randn(kv.ndofs, 3, 4))
      splp = spl.insert_dependencies((0,), (UnivariateKnotVector(np.linspace(0, 1, 2)),))

      Xi = np.stack(list(map(np.ravel, np.meshgrid(*xi, indexing='ij'))), axis=1)
      Xip = np.concatenate((Xi[:, [0]], Xi), axis=1)
      self.assertTrue( np.allclose( spl(*Xi.T), splp(*Xip.T) ))

    with self.subTest('Test insert_dependencies NDSplineArray'):
      kv = TensorKnotVector([ UnivariateKnotVector(np.linspace(0, 1, i)) for i in (4, 5, 6) ])

      spl = NDSplineArray(NDSpline(kv, np.random.randn(kv.ndofs, 3, 4)))
      splp = spl.insert_dependencies((0, 0), [UnivariateKnotVector(np.linspace(0, 1, i)) for i in (3, 4)])

      self.assertTrue(splp.nvars == 5 and splp.shape == spl.shape)


if __name__ == '__main__':
  unittest.main()
