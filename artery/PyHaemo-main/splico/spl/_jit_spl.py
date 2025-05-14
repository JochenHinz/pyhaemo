"""
JIT-compiled routines for spline evaluation.
All taken from the NURBS book.
@author: Jochen Hinz
"""


from ..util import np, _
from .._jit import arange_product, ravel_multi_index, unravel_multi_index

import multiprocessing

from numba import njit, float64, prange, config
from numba.typed import List

config.NUMBA_NUM_THREADS = multiprocessing.cpu_count()


# XXX: this scipt requires a formatting overhaul and some docstrings need
#      need to be adjusted to the default format.
#      Some functions' positional arguments are in a confusing order.


@njit(cache=True)
def _univariate_prolongation_matrix(kvold: np.ndarray, kvnew: np.ndarray, p: int):
  """
  NURBS-Book implementation.

  Parameters
  ----------
  kvold : :class:`np.ndarray`
      The old knotvector, including repeated knots.
  kvnew : :class:`np.ndarray`
      The new knotvector.
  p : :class:`int`
      The degree of the B-spline basis (must be the same for both knotvectors).

  Returns
  -------
  T : :class:`np.ndarray`
      The prolongation matrix, for now dense.
  """
  # XXX: implement for immediate sparse matrix output

  n, m = len(kvnew) - 1, len(kvold) - 1
  T = np.zeros((n, m), dtype=np.float64)

  for j in range(m):
    rows = np.where(np.logical_and(kvold[j] <= kvnew[:n], kvnew[:n] < kvold[j+1]))[0]
    T[rows, j] = 1

  for q in range(1, p + 1):
    T_new = np.zeros((n - q, m - q), dtype=np.float64)
    for i in range(n - q):
      for j in np.where(np.logical_or(T[i, : m - q] != 0,
                                      T[i, 1: m - q + 1] != 0))[0]:
        denom0 = kvold[j + q] - kvold[j]
        fac0 = (kvnew[i + q] - kvold[j]) / denom0 if denom0 != 0 else 0

        denom1 = kvold[j + q + 1] - kvold[j + 1]
        fac1 = (kvold[j + 1 + q] - kvnew[i + q]) / denom1 if denom1 != 0 else 0

        T_new[i, j] = fac0 * T[i, j] + fac1 * T[i, j + 1]

    T = T_new

  return T


@njit(cache=True)
def _pseudo_inverse(data: np.ndarray, rows: np.ndarray, cols: np.ndarray):
  """
  Compute a computationally inexpensive pseudo-inverse of a matrix given
  in COO format.
  Return in COO format.
  """
  shuffle = np.argsort(cols)
  data, rows, cols = data[shuffle], rows[shuffle], cols[shuffle]
  breaks = np.concatenate((np.array((0,), dtype=np.int64),
                           np.where(np.diff(cols) != 0)[0] + 1,
                           np.array((len(cols),), dtype=np.int64)))

  new_data = np.empty_like(data)
  for b0, b1 in zip(breaks, breaks[1:]):
    mydata = data[b0:b1]
    new_data[b0:b1] = mydata / np.abs(mydata).sum()

  return new_data, (cols, rows)


@njit(cache=True)
def position_in_knotvector(t, x):
  """
  Return the position of ``x`` in the knotvector ``t``.
  If x equals t[-1], return the position before the first
  occurence of x in t.

  Parameters
  ----------
  t : :class:`np.ndarray`
      The knotvector with repeated knots.
  x : :class:`np.ndarray`
      The vector of positions.

  Returns
  -------
  ret : :class:`np.ndarray` comprised of integers
      The positions in the knotvector. Has the same length as `x`.
      If entry is not found, defaults to -1.
  """
  ret = np.empty(len(x), dtype=np.int64)
  for i in range(len(x)):
    myx = x[i]
    if myx < t[0] or myx > t[-1]:
      ret[i] = -1
    elif myx == t[-1]:
      ret[i] = np.searchsorted(t, myx) - 1
    else:
      ret[i] = np.searchsorted(t, myx, side='right') - 1
  return ret


@njit(cache=True)
def nonzero_bsplines(mu, x, t, d):
  """
  Return the value of the d+1 nonzero basis
  functions at position ``x``.

  Parameters
  ----------
  mu : :class:`int`
      The position in `t` that contains `x`,
  x: :class:`float`
      The position,
  t: :class:`np.ndarray`
      The knotvector.
  d: :class:`int`
      The degree of the B-spline basis.

  Returns
  -------
  b : :class:`np.ndarray`
      The nonzero bsplines evalated in `x`
  """

  b = np.zeros(d + 1, dtype=float64)
  b[-1] = 1

  if x == t[-1]:
    return b

  for r in range(1, d + 1):

    k = mu - r + 1
    w2 = (t[k + r] - x) / (t[k + r] - t[k])
    b[d - r] = w2 * b[d - r + 1]

    for i in range(d - r + 1, d):
      k = k + 1
      w1 = w2
      w2 = (t[k + r] - x) / (t[k + r] - t[k])
      b[i] = (1 - w1) * b[i] + w2 * b[i + 1]

    b[d] = (1 - w2) * b[d]

  return b


@njit(cache=True)
def nonzero_bsplines_deriv(kv, p, x, dx):
  """
  Return the value of the d+1 nonzero basis
  functions and their derivatives up to order `dx` at position `x`.

  Parameters
  ----------
  kv : :class:`np.ndarray`
      The knotvector.
  p : :class:`int`
      The degree of the B-spline basis.
  x : :class:`float`
      The position.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ders : :class:`np.ndarray`
      The nonzero bsplines evalated in `x` and their derivatives up to order `dx`.
  """
  # Initialize variables
  span = position_in_knotvector(kv, np.array([x], dtype=np.float64))[0]
  left = np.ones((p + 1,), dtype=np.float64)
  right = np.ones((p + 1,), dtype=np.float64)
  ndu = np.ones((p + 1, p + 1), dtype=np.float64)

  for j in range(1, p + 1):
    left[j] = x - kv[span + 1 - j]
    right[j] = kv[span + j] - x
    saved = 0.0
    r = 0
    for r in range(r, j):
      # Lower triangle
      ndu[j, r] = right[r + 1] + left[j - r]
      temp = ndu[r, j - 1] / ndu[j, r]
      # Upper triangle
      ndu[r, j] = saved + (right[r + 1] * temp)
      saved = left[j - r] * temp
    ndu[j, j] = saved

  # Load the basis functions
  # ders = [[0.0 for _ in range(p + 1)] for _ in range((min(p, dx) + 1))]
  ders = np.zeros((min(p, dx) + 1, p + 1), dtype=np.float64)
  for j in range(0, p + 1):
    ders[0, j] = ndu[j, p]

  # Start calculating derivatives
  # a = [[1.0 for _ in range(p + 1)] for _ in range(2)]
  a = np.ones((2, p + 1), dtype=np.float64)
  # Loop over function index
  for r in range(0, p + 1):
    # Alternate rows in array a
    s1 = 0
    s2 = 1
    a[0, 0] = 1.0
    # Loop to compute k-th derivative
    for k in range(1, dx + 1):
      d = 0.0
      rk = r - k
      pk = p - k
      if r >= k:
        a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
        d = a[s2, 0] * ndu[rk, pk]
      if rk >= -1:
        j1 = 1
      else:
        j1 = -rk
      if (r - 1) <= pk:
        j2 = k - 1
      else:
        j2 = p - r
      for j in range(j1, j2 + 1):
        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
        d += (a[s2, j] * ndu[rk + j, pk])
      if r <= pk:
        a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
        d += (a[s2, k] * ndu[r, pk])
      ders[k, r] = d

      # Switch rows
      j = s1
      s1 = s2
      s2 = j

  # Multiply through by the the correct factors
  r = float(p)
  for k in range(1, dx + 1):
    for j in range(0, p + 1):
      ders[k, j] *= r
    r *= (p - k)

  # Return the basis function derivatives list
  return ders


@njit(cache=True)
def nonzero_bsplines_deriv_vectorized(kv, p, x, dx):
  """
  Vectorized (in x) version of `nonzero_bsplines_deriv`
  only returns the dx-th derivative though.

  Parameters are the same as in `nonzero_bsplines_deriv` but `x` is a vector.
  """
  ret = np.empty((len(x), p+1), dtype=np.float64)
  for i in prange(len(ret)):
    ret[i] = nonzero_bsplines_deriv(kv, p, x[i], dx)[dx]
  return ret


@njit(cache=True)
def _collocation_matrix(kv, p, x, dx):
  """
  Compute the collocation matrix for the univariate B-spline basis
  resulting from knotvector `kv` and degree `p` at the positions `x`.

  Return in COO format.
  """
  # Find the knot spans containing the x[i]
  pos = position_in_knotvector(kv, x)

  # The data has shape (len(x), p + 1)
  data = nonzero_bsplines_deriv_vectorized(kv, p, x, dx).ravel()

  # A value contained in knotspan number `i` gives a nonzero outcome for
  # all DOFS with indices `i - p.. i`
  i = (pos.reshape((-1, 1)) + np.arange(-p, 1, dtype=np.int64).reshape((1, -1))).ravel()
  j = np.repeat(np.arange(len(x), dtype=np.int64), p + 1)

  return data, (i, j)


@njit(cache=True)
def der_ith_basis_fun( kv, p, i, x, dx ):  # based on algorithm A2.5 from the NURBS-book
  """
  Return the N_ip(x) and its derivatives up to ``dx``,
  where N denotes the ``i``-th basis function of order
  ``p`` resulting from knotvector ``kv`` and x the position.

  Parameters
  ----------
  kv : :class:`np.ndarray`
      The knotvector.
  p : :class:`int`
      The degree of the B-spline basis.
  i : :class:`int`
      The index of the basis function.
  x : :class:`float`
      The position.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ders : :class:`np.ndarray`
      The basis function and its derivatives up to order `dx`.
  """

  if x == kv[-1]:
    x -= 1e-15

  basis_len = len(kv) - p - 1

  if x < kv[i] or x >= kv[i + p + 1]:
    if i != basis_len - 1 or x > kv[-1]:
      ''' x lies outside of the support of basis function or domain '''
      return np.zeros( (dx + 1, ), dtype=float64 )
    if i == basis_len - 1 and x == kv[-1]:
      '''
      special case: evaluation of the last basis function
      in the last point of the interval. Return a sequence
      (p / a_0) ** 0, (p / a_1) ** 1, ... (p / a_dx) ** dx
      '''
      # a = 1
      # ret = np.empty( (dx + 1, ), dtype=float64 )
      # for i in range( ret.shape[0] ):
      #     ret[i] = a
      #     if i != ret.shape[0] - 1:
      #         a *= p / ( kv[basis_len - 1 + p - i] - kv[basis_len - 1 - i] )
      # return ret
      x -= 1e-15

  ders = np.empty( (dx + 1, ), dtype=float64 )
  N = np.zeros( (p + 1, p + 1), dtype=float64 )

  for j in range(p + 1):
    if ( x >= kv[i + j] and x < kv[i + j + 1] ):
      N[j, 0] = 1.0
    else:
      N[j, 0] = 0.0

  for k in range(1, p + 1):
    saved = 0.0 if N[0, k - 1] == 0.0 else \
        (x - kv[i]) * N[0, k - 1] / (kv[i + k] - kv[i])
    for j in range(p - k + 1):
      Uleft, Uright = kv[i + j + 1], kv[i + j + k + 1]
      if N[j + 1, k - 1] == 0:
        N[j, k], saved = saved, 0
      else:
        temp = N[j + 1, k - 1] / (Uright - Uleft)
        N[j, k] = saved + (Uright - x) * temp
        saved = (x - Uleft) * temp

  ders[0] = N[0, p]
  ND = np.zeros( (k + 1, ), dtype=float64 )
  for k in range(1, dx + 1):
    for j in range(k + 1):
      ND[j] = N[j, p - k]
    for jj in range(1, k + 1):
      saved = 0.0 if ND[0] == 0.0 else ND[0] / (kv[i + p - k + jj] - kv[i])
      for j in range(k - jj + 1):
        # wrong in the NURBS book, -k is missing in Uright
        Uleft, Uright = kv[i + j + 1], kv[i + j + p - k + jj + 1]
        if ND[j + 1] == 0.0:
          ND[j], saved = (p - k + jj) * saved, 0.0
        else:
          temp = ND[j + 1] / (Uright - Uleft)
          ND[j] = (p - k + jj) * (saved - temp)
          saved = temp

    ders[k] = ND[0]
  return ders


@njit(cache=True)
def _call1D(xi, kv0, p0, x, dx):
  """
  Return function evaluations at positions xi.

  Parameters
  ----------
  xi : :class:`np.ndarray`
      The positions.
  kv0 : :class:`np.ndarray`
      The knotvector.
  p0 : :class:`int`
      The degree of the B-spline basis.
  x : :class:`np.ndarray`
      The control points. Unlike `_callND`, this is a 1D array.
  dx : :class:`int`
      The highest-order derivative.

  Returns
  -------
  ret : :class:`np.ndarray`
      The function evaluations at positions xi.

  This version is fully sequential.
  """

  # XXX:

  ret = np.zeros(xi.shape, dtype=float64)
  assert ret.ndim == 1
  element_indices0 = position_in_knotvector(kv0, xi)

  for i in range(len(xi)):
    xi_calls = nonzero_bsplines_deriv(kv0, p0, xi[i], dx)[dx]
    for j in range(p0 + 1):
      a = xi_calls[j]
      global_index = element_indices0[i] - p0 + j
      ret[i] += x[global_index] * a

  return ret


@njit(cache=True, parallel=True)
def _callND(Xi, list_of_knotvectors, degrees, controlpoints, derivatives, into):
  """
  Return function evaluations (or their derivatives) of a nD tensor product
  spline at positions Xi.

  Parameters
  ----------
  Xi : :class:`np.ndarray`
      The positions of shape (nentries, ncoords). Different coordinates in the columns.
  list_of_knotvectors : :class:`List`
      List containing the knotvectors of the ncoords directions.
  degrees : :class:`np.ndarray`
      The polynomial degrees in each direction.
  controlpoints : :class:`np.ndarray`
      The control points. Of shape (ndofs, naxes).
  derivatives : :class:`np.ndarray`
      The highest-order derivatives in each direction.
  into : :class:`np.ndarray`
      The array to store the results. Must be of shape (nentries, naxes).

  This version is parallelized along the `naxes` coordinate.
  """

  assert Xi.shape[0] == into.shape[0]
  nentries, naxes = into.shape
  assert controlpoints.shape[:1] == (naxes,)

  # make len(list_of_knotvectors) - length homogeneous container containing
  # temporary `into` arrays
  container = [np.zeros((nentries,), dtype=np.float64) for _ in range(naxes)]

  # make len(list_of_knotvectors) - shaped integer array with the ndofs per direction
  dims = np.empty(len(list_of_knotvectors), dtype=np.int64)
  for i, (kv, degree) in enumerate(zip(list_of_knotvectors, degrees)):
    dims[i] = kv.shape[0] - degree - 1

  # make an outer product flat meshgrid with aranges from 0 to p + 1
  inner_loop_indices = arange_product(degrees + 1)

  # make integer array containing the positions in the knotvectors of the univariate
  # contributions in `Xi`
  element_indices = np.empty((nentries, len(list_of_knotvectors)), dtype=np.int64)
  for i, (mykv, xi) in enumerate(zip(list_of_knotvectors, Xi.T)):
    element_indices[:, i] = position_in_knotvector(mykv, xi)

  for iaxis in prange(naxes):
    x = controlpoints[iaxis]
    myinto = container[iaxis]

    for i in range(nentries):

      # get all univariate local calls
      mycalls = [nonzero_bsplines_deriv(kv, p, xi, dx)[dx, :] for kv, p, xi, dx
                 in zip(list_of_knotvectors, degrees, Xi[i], derivatives)]

      for multi_index in inner_loop_indices:
        # global index in x results from the multi_index + the element_indices minus the degrees
        # and the `dims` vector
        global_index = ravel_multi_index(element_indices[i] + multi_index - degrees, dims)

        # add product of all evaluations times the weight to the corresponding
        # position
        myval = x[global_index]
        for j, myindex in enumerate(multi_index):
          myval = myval * mycalls[j][myindex]

        myinto[i] += myval

      into[:, iaxis] = myinto


def call(Xi,
         list_of_knotvectors,
         list_of_degrees,
         controlpoints,
         dx=None,
         into=None):
  """
  Return function evaluations at positions Xi.

  Optionally put them into a preallocated array ``into``.

  For a detailed docstring, see `_callND`.
  """

  nvars, = Xi.shape[1:]

  assert controlpoints.ndim == 2, "Please provide a 2D array of control points."

  # assert 1 <= nvars <= 3
  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * nvars

  degrees = np.asarray(list_of_degrees, dtype=int)
  dx = np.asarray(dx, dtype=int)

  assert len(list_of_knotvectors) == len(list_of_degrees) \
                                  == len(dx)

  if into is None:
    into = np.zeros(Xi.shape[:1] + controlpoints.shape[:1], dtype=np.float64)

  assert into.shape == Xi.shape[:1] + controlpoints.shape[:1]

  _callND(Xi,
          List(list_of_knotvectors),
          degrees,
          controlpoints, dx, into)

  return into


@njit(cache=True, parallel=True)
def _tensor_callND(list_of_xis, list_of_knotvectors, degrees, controlpoints, derivatives, into):
  """
  Return function evaluations at all positions
  (xi_i, eta_i) in the outer product of univariate
  positions ``xi`` and ``eta``.
  Optimized because the bases are only evaluated in
  ``xi`` and ``eta`` once.
  Returns a matrix instead of a flat vector.

  Parameters
  ----------
  list_of_xis : :class:`numba.typed.List` of :class:`np.ndarray`
      The positions in each direction.
  list_of_knotvectors : :class:`numba.typed.List` of :class:`np.ndarray`
      The knotvectors in each direction.
  degrees : :class:`np.ndarray`
      Integer array of the polynomial degrees in each direction.
  controlpoints : :class:`np.ndarray`
      The control points. Must be a 2D array. The first axis refers to the
      number of dofs. The second axis refers to the number of axes.
  derivatives : :class:`np.ndarray`
      The highest-order derivatives in each direction.
  into : :class:`np.ndarray`
      The array to store the results. Must be of shape (nentries, naxes).

  This version is parallelized along the `naxes` coordinate.
  """
  nentries, naxes = into.shape
  assert controlpoints.shape[:1] == (naxes,)

  # shape of the entries in each direction
  entry_shape = np.empty(len(list_of_xis), dtype=np.int64)
  for i, xi in enumerate(list_of_xis):
    entry_shape[i] = len(xi)

  # make a list containing the position of each xi in each direction in the
  # corresponding knotvector
  mus = []
  for kv, pos in zip(list_of_knotvectors, list_of_xis):
    mus.append(position_in_knotvector(kv, pos))

  # make len(list_of_knotvectors) - shaped integer array with the ndofs per direction
  dims = np.empty(len(list_of_knotvectors), dtype=np.int64)
  for i, (kv, degree) in enumerate(zip(list_of_knotvectors, degrees)):
    dims[i] = kv.shape[0] - degree - 1

  # total number of entries in the result
  nentries_ = 1
  for length in entry_shape:
    nentries_ *= length

  assert nentries_ == nentries

  # store the evaluations of the basis functions in each direction
  # for each evaluation point
  xi_evals = []
  for nelems, p, xi, dx in zip(degrees + 1, degrees, list_of_xis, derivatives):
    evals = np.empty( (len(xi), nelems), dtype=np.float64 )

    for i in range(len(xi)):
      evals[i] = nonzero_bsplines_deriv(kv, p, xi[i], dx)[dx]

    xi_evals.append(evals)

  # make an outer product flat meshgrid with aranges from 0 to p + 1
  inner_loop_indices = arange_product(degrees + 1)

  container = [np.zeros((nentries,), dtype=np.float64) for _ in range(naxes)]

  for iaxis in prange(naxes):
    x = controlpoints[iaxis]
    myinto = container[iaxis]

    for i in range(nentries):

      # get the multi-index of the current entry
      my_entry_index = unravel_multi_index(i, entry_shape)

      # make the element's multi-index
      # (position of the evaluation point in each knotvector in each direction)
      multi_index = -degrees
      for j, ind in enumerate(my_entry_index):
        multi_index[j] += mus[j][ind]

      for offset_index in inner_loop_indices:

        # get the global index of the control point, which is the element's
        # multi-index plus the offset index raveled
        global_index = ravel_multi_index(multi_index + offset_index, dims)

        # initialize the value of the basis function in the current point
        myval = x[global_index]

        # iterate over the contributions of all basis functions in each direction
        for ii, (j, offset) in enumerate(zip(my_entry_index, offset_index)):
          myval *= xi_evals[ii][j, offset]

        myinto[i] += myval

  # sequentially copy the results from the container into the `into` array
  for i in range(naxes):
    into[:, i] = container[i]


def tensor_call(list_of_abscissae,
                list_of_knotvectors,
                list_of_degrees,
                controlpoints,
                dx=None,
                into=None):
  # XXX: docstring
  raise NotImplementedError("This function is buggy and needs to be fixed.")
  nvars = len(list_of_abscissae)

  assert controlpoints.ndim == 2, "Please provide a 2D array of control points."

  # assert 1 <= nvars <= 3
  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * nvars

  degrees = np.asarray(list_of_degrees, dtype=int)
  dx = np.asarray(dx, dtype=int)

  assert len(list_of_abscissae) == len(list_of_knotvectors) \
                                == len(list_of_degrees) \
                                == len(dx)

  shape = tuple(map(len, list_of_abscissae))
  return_shape = (np.prod(shape, dtype=int),) + controlpoints.shape[:1]
  if into is None:
    into = np.zeros(return_shape, dtype=np.float64)

  assert into.shape == return_shape

  _tensor_callND(List(list_of_abscissae),
                 List(list_of_knotvectors),
                 degrees,
                 controlpoints, dx, into)

  return into
