from pyhaemo.util import np, clparam
from pyhaemo.spl import UnivariateKnotVector, TensorKnotVector, NDSpline
from scipy import sparse
from numpy.typing import NDArray


def solve_with_dirichlet_data(A, rhs, freezeindices, data, **kwargs):

  if not sparse.isspmatrix(A):
    A = sparse.csr_matrix(A)

  N = len(rhs)
  assert A.shape == (N, N)

  # convert the full rhs vector (i.e., the one that is tested against ALL test functions)
  # to an np.ndarray if not already.
  rhs = np.asarray(rhs)

  assert rhs.ndim in (1, 2)
  assert freezeindices.ndim == 1 and len(freezeindices) == data.shape[0]
  assert data.shape[1:] == rhs.shape[1:]

  freezemask = np.zeros(N, dtype=bool)
  freezemask[freezeindices] = True
  freemask = ~freezemask

  cons = np.zeros(rhs.shape)
  cons[freezeindices] = data

  rhs = (rhs - A @ cons)[freemask]

  A = A.tolil()[freemask][:, freemask].tocsr()

  # solve for u_0
  sol = sparse.linalg.spsolve(A, rhs, **kwargs)

  # cons is equal to the imposed Dirichlet data on the freezeindices
  # setting cons[dofindices] = sol will create a vector that is equal to
  # the Dirichlet data on the freezeindices and to u_0 on the dofindices.
  cons[freemask] = sol

  # return the result as an ordinary np.ndarray
  return cons


def fit_constrained(kv: TensorKnotVector,
                    absc: NDArray,
                    data: NDArray,
                    constrain_endpoints=True,
                    constrain_tangents=True,
                    lam0=1e-5, lam1=0):

  assert len(kv) == 1, NotImplementedError
  absc = np.asarray(absc, dtype=float)

  assert all(lam >= 0 for lam in (lam0, lam1))

  X, = kv.collocate(absc)
  M = X @ X.T

  if any( (lam0, lam1) ):
    if lam0 != 0:
      M += lam0 * kv.A
    if lam1 != 0:
      M += lam1 * kv.D

  _data = data.reshape((-1,) + (data.shape[1:] and (np.prod(data.shape[1:]),)))
  rhs = X @ _data

  freezeindices = []
  freeze_data = []

  if constrain_endpoints:
    freezeindices.extend([0, kv.ndofs-1])
    freeze_data.extend([_data[0], _data[-1]])

  if constrain_tangents:
    freezeindices.extend([1, kv.ndofs-2])
    knots, = kv.knots
    p, = kv.degree

    t0 = (_data[1] - _data[0]) / (absc[1] - absc[0])
    t1 = (_data[-1] - _data[-2]) / (absc[-1] - absc[-2])
    freeze_data.extend([_data[0] + t0 * (knots[1] - knots[0]) / p,
                        _data[-1] - t1 * (knots[-1] - knots[-2]) / p])

  freezeindices = np.array(freezeindices, dtype=int)
  freeze_data = np.asarray(freeze_data, dtype=float)

  controlpoints = solve_with_dirichlet_data(M,
                                            rhs,
                                            freezeindices,
                                            freeze_data)

  return NDSpline(kv, controlpoints.reshape(-1, *data.shape[1:]))
