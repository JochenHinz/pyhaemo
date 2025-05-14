from splico.util import np, frozen, _
from splico.types import NanVec, Immutable, Singleton, Int, lock
from splico.spl import NDSpline, NDSplineArray, UnivariateKnotVector
from splico.util import nutils_to_scipy
from splico.err import DuplicateOrientationError
from splico._jit import _apply_pairs
from splico.topo import Topology, as_topo, ZEROTOPO

from .mul import multipatch, basis_spline

from functools import cached_property, lru_cache
from itertools import product, count, repeat
from typing import Tuple, Optional, Dict

from numpy.typing import NDArray, ArrayLike
from nutils import function
from scipy.sparse import linalg as splinalg
from scipy import sparse


J = function.J
sl = slice(None)
concat = np.concatenate


def infer_topo(ndarr: NDSplineArray, tresh: float = 1e-5) -> Topology:
  """
  Infer the topology from the NDSplineArray.
  The topology is inferred by checking the distance between the
  control points of the patches along the last axis.

  Two points are considered equal if their distance is below the threshold.
  """

  n, m = ndarr.shape  # works only for 2D arrays for now

  ndarr = ndarr.to_ndim(1)
  nvars = ndarr.nvars

  if n == 0:
    return Topology(ZEROTOPO(nvars))

  topodim = 2 ** nvars  # no. entries per row

  if n == 1:
    return Topology([np.arange(topodim)])

  npoints = np.prod(ndarr.arr[0].shape)

  cpoints = np.array(list(product(*[(0, -1)] * nvars)), dtype=int)

  # shape (npatches, 2 ** nvars, npoints)
  points = np.asarray([cps[*cpoints.T] for cps
                       in (arr.tcontrolpoints for arr in ndarr.arr.ravel())], dtype=float)

  npatches = len(points)

  pairs = []
  for i, j in product(*[range(npatches)]*2):
    if j <= i: continue
    points0, points1 = points[i], points[j]

    # compute all distances
    distances = np.linalg.norm(points0[:, _] - points1[_], axis=-1)

    # compute all (i, j) pairs where j minimizes the distance to node i
    minij = np.stack([np.arange(len(points0)),
                      np.argmin(distances, axis=1)], axis=1)

    # keep the ones that are below the threshold and give it a global numbering
    pairs.append(minij[distances[*minij.T] < npoints * tresh]
                                + topodim * np.array([[i, j]], dtype=int))

  pairs = np.concatenate(pairs, axis=0)
  indices = _apply_pairs(np.arange(npatches * topodim, dtype=int), pairs)

  return Topology( np.unique(indices, return_inverse=True)[1]
                     .reshape(npatches, -1) ).orient()


class CSRMatrix(Immutable):
  """
  Wrapper around :class:`scipy.sparse.csr_matrix` to make it immutable
  and to add some convenience methods.
  """

  @classmethod
  def from_nutils(cls, mat: NDArray[np.float_], freemask=None):
    return cls.from_scipy(nutils_to_scipy(mat), freemask=freemask)

  @classmethod
  def from_scipy(cls, mat, **kwargs):
    mat = sparse.csr_matrix(mat)
    return cls(mat.data, mat.indices, mat.indptr, shape=mat.shape, **kwargs)

  def __init__(self, data: ArrayLike,
                     indices: ArrayLike,
                     indptr: ArrayLike,
                     shape: Optional[Tuple[int, int]] = None,
                     freemask: Optional[ArrayLike] = None) -> None:

    self.data = frozen(data, dtype=float)
    self.indices = frozen(indices, dtype=np.int32)
    self.indptr = frozen(indptr, dtype=np.int32)

    self.mat = sparse.csr_matrix((self.data, self.indices, self.indptr), shape=shape)

    if shape is None:
      shape = self.mat.shape

    self.shape = tuple(map(int, shape))
    assert self.shape == self.mat.shape

    if freemask is None:
      freemask = np.ones(self.shape[0], dtype=bool)

    self.freemask = frozen(freemask, dtype=bool)
    self.freezemask = frozen(~self.freemask, dtype=bool)

    assert self.freezemask.shape == (self.mat.shape[0],)

  def __matmul__(self, other):
    return self.mat @ other

  def __getattr__(self, attr: str):
    """
    If attribute lookup fails in the conventional way, try to get it from the
    underlying matrix (if already set).
    """
    try:
      # If we use `getattr(self.mat, attr)`, we get an infinite recursion because
      # self.mat may not be set yet and he will resort to this function again.

      # Make sure it's set by getting it in the convenitional way
      mat = object.__getattribute__(self, 'mat')
      return getattr(mat, attr)

    except AttributeError:
      raise AttributeError(f"'{self.__class__.__name__}' object has no "
                           f"attribute '{attr}'")

  @cached_property
  def submat(self):
    return CSRMatrix.from_scipy(self.tolil()[self.freemask][:, self.freemask])

  @cached_property
  def splu(self):
    return splinalg.splu(self.mat.tocsc())


@lru_cache(maxsize=32)
def _kv_dict(ndarr: 'NDSplineArray', topo: Topology) -> dict:
  map_edge_kv: Dict[Tuple[Int, Int], UnivariateKnotVector] = {}

  for (i, patch), dim in product(enumerate(topo.tensortopo), range(topo.ndim)):

    sides = [(0, 1)] * dim + [(sl,)] + [(0, 1)] * (ndarr.nvars - dim - 1)

    for side in product(*sides):
      edge = tuple(patch[side])
      kv = ndarr.knotvector[i][dim]

      if any( _kv != map_edge_kv.setdefault(_edge, _kv)
              for _edge, _kv in zip([edge, edge[::-1]], [kv, kv.flip()]) ):
        raise DuplicateOrientationError("Detected edge with different knotvectors.")

  return map_edge_kv


@lock
def kv_dict(arr: 'NDSplineArray', topo: Topology):
  """
  Check if the knotvectors associated with the same edges are equal.
  """
  if not arr.shape:
    if not topo: return {}
    raise ValueError("Topology and knotvectors do not match.")

  return _kv_dict(arr.to_ndim(1), topo)


@lru_cache(maxsize=8)
def _integrate(intf: 'NutilsInterface', intstr: str, degree=None):
  if degree is None:
    return _integrate(intf, intstr, degree=intf.degree)

  try:
    integrand = intstr @ intf.ns
  except Exception:
    integrand = intf.ns.eval_ij(intstr)

  return intf.domain.integrate(integrand, degree=degree or intf.degree)


class NutilsInterface(Singleton):

  def __init__(self, ndarr: NDSplineArray, topo: Tuple | Topology) -> None:
    # XXX: this __init__ is too long, find a way to split it up

    self.ndarr = NDSplineArray(ndarr)

    npatches, targetspace = self.ndarr.shape

    self.topo = as_topo(topo)
    if not self.topo.is_valid():
      raise ValueError("Topology is not valid. You can run topo.orient() to fix it.")

    assert (kvs := kv_dict(self.ndarr, self.topo)), \
      "Topology and knotvectors do not match."

    self.kvdict = kvs

    domain, pgeom, localgeom = multipatch(patches=self.topo.topo,
                                          knotvectors=kvs)

    basis = basis_spline(domain, kvs)
    basis_disc = basis_spline(domain, kvs, patchcontinuous=False)

    self.domain = domain
    self.basis = basis
    self.basis_disc = basis_disc
    self.localgeom = localgeom
    self.pgeom = pgeom

    reference = concat([ _ar.controlpoints.ravel()
                         for _ar in self.ndarr.to_ndim(1).arr.ravel() ])

    self.geom = basis_disc.vector(targetspace).dot(reference)
    self._reference_sol = frozen(reference)

    ns = function.Namespace()
    ns.basis = self.basis
    ns.basisd = self.basis_disc
    ns.x = self.geom

    J = self.geom.grad(self.localgeom)
    ns.J = J
    ns.G = function.matmat(J.T, J)
    ns.JGinv = function.matmat(J, function.inverse(ns.G))
    ns.g = function.determinant(ns.G) ** .5
    ns.dx = function.J(self.localgeom)

    # take the gradient of the basis with respect to the parametric domain
    dbasis = basis.grad(self.localgeom)
    ns.dbasis = dbasis

    self.ns = ns

    self.p = int(max(kv.degree for kv in kvs.values()))
    self.degree = int(2 * self.p + 1)
    self.segments = frozen(np.array([arr.knotvector.ndofs for arr in
                                     self.ndarr.to_ndim(1).arr.ravel()]).cumsum()[:-1])

    # get a boolean array that gives True for indices of basis functions that
    # are nonzero on the boundary
    freemask = np.isnan( domain.boundary
                               .project(1,
                                        onto=basis,
                                        geometry=self.geom,
                                        degree=self.degree) )

    self.freezemask = frozen(~freemask)

    # get the associated boolean array that is True for functions NOT on the boundary
    self.freemask = frozen(freemask)

    self.basisb = self.basis[self.freezemask]
    self.ns.basisb = self.basisb

  def integrate(self, intstr: str, *args, **kwargs):
    return _integrate(self, intstr, *args, **kwargs)

  @cached_property
  def A(self):
    dbasis = (self.ns.JGinv * self.ns.dbasis[:, _]).sum(-1)
    g, dx = self.ns.g, self.ns.dx
    integrand = (dbasis[:, _, :] * dbasis[_, :, :]).sum(-1) * g * dx
    return CSRMatrix.from_nutils(
      self.domain.integrate(integrand, degree=self.degree), freemask=self.freemask)

  @cached_property
  def CA(self):
    """
    Compute the Cholesky decomposition of the stiffness matrix restricted
    to the free indices (i.e. not on the boundary)
    """
    return self.A.submat.splu

  @cached_property
  def M(self):
    return CSRMatrix.from_nutils(self.integrate('basis_i basis_j g dx'),
                                 freemask=self.freemask)

  @cached_property
  def M_disc(self):
    return CSRMatrix.from_nutils(self.integrate('basisd_i basisd_j g dx'))

  @cached_property
  def T_disc(self):
    return CSRMatrix.from_nutils(self.integrate('basisd_i basis_j g dx'))

  @cached_property
  def CM_disc(self):
    return self.M_disc.splu

  @cached_property
  def CM(self):
    return self.M.splu

  @cached_property
  def Mb(self):
    g, dx = self.ns.g, self.ns.dx
    return CSRMatrix.from_nutils(
                    self.domain
                        .boundary
                        .integrate(function.outer(self.basisb) * g * dx,
                                                  degree=self.degree))

  @cached_property
  def CMb(self):
    return self.Mb.splu

  @cached_property
  def boundary_patches(self):
    """
    Return the patch, iside pairs that are on the boundary.
    Here, `iside` refers to the flat index of the interface on the patch
    interface. That is, for instance, in 2D `left => 0`, `right => 1`,
    `bottom => 2`, `top => 3`.
    Has the distinct advantage that it will work in any dimension.
    """
    mip = {}  # map interface: patch, side

    ndim = self.topo.ndim
    ninterfaces = 2 * ndim

    sl = slice(None),
    counter = count()

    for (i, patch), j in product(enumerate(self.topo.tensortopo), range(ndim)):
      for _sl in product(*([sl] * j + [(0, 1)] + [sl] * (ndim - j - 1))):
        mip.setdefault(tuple(np.sort(patch[_sl].ravel())), set()) \
           .add((i, next(counter) % ninterfaces))

    return tuple( val.pop() for val in mip.values() if len(val) == 1 )

  def break_apart(self, vec: np.ndarray, split: bool = False):
    ret = self.CM_disc.solve(self.T_disc @ vec)
    if split: return np.array_split(ret, self.segments, axis=0)
    return ret

  def project_boundary(self, bfuncs: function.Array | dict):
    """
    Project the data onto the boundary DOFs.

    Data may come in the form of a global function or a dictionary with
    (ipatch, iside) as keys and the corresponding function as values.
    """
    if isinstance(bfuncs, function.Array):
      bfuncs = dict(zip(self.boundary_patches, repeat(bfuncs)))

    assert set(bfuncs.keys()).issubset(set(self.boundary_patches))

    allfuncs = list(bfuncs.values())
    ndims = allfuncs[0].ndim
    assert ndims == 1, NotImplementedError
    assert all( bf.shape == allfuncs[0].shape for bf in allfuncs ), \
      "All functions must have the same shape."

    J = function.J(self.localgeom)

    basisb = self.basisb
    domain = self.domain

    vecs = [np.zeros(len(basisb), dtype=float)
                                        for i in range(allfuncs[0].shape[0])]

    for (ipatch, iside), func in bfuncs.items():
      myints = domain._topos[ipatch] \
                     .boundary \
                     ._topos[iside] \
                     .integrate([basisb * f * self.ns.g * J for f in func],
                                                          degree=self.degree)
      for vec0, vec1 in zip(vecs, myints): vec0 += vec1

    return tuple(map(self.CMb.solve, vecs))

  def harmonic_transform(self, bfunc: function.Array | dict):
    f = np.stack(self.project_boundary(bfunc), axis=0)

    A, CA = self.A, self.A.submat.splu

    solution = []

    for rhs in f:
      # constraint vector that is initially empty
      mycons = NanVec(A.shape[:1])

      # on the positions that are frozen from the dirichlet data, substitute the
      # Dirichlet data
      mycons[self.freezemask] = rhs

      # make the right hand side corresponding to the elimination of the
      # DOFs that are frozen from the Dirichlet BC
      rhs = -(self.A @ (mycons | 0))[self.freemask]

      # solve for the inner DOFs
      mysol = CA.solve(rhs)

      # fill the solution vector with the solution at the indices that
      # correspond to DOFs
      mycons[self.freemask] = mysol

      # add to solution list
      solution.append(mycons.view(np.ndarray))

    solution = self.break_apart(np.stack(solution, axis=1), split=True)

    return NDSplineArray(list(map(NDSpline,
                                  self.ndarr.to_ndim(1).knotvector.ravel(),
                                  solution))).contract_all()
