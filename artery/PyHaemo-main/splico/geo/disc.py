from splico.util import np, _
from splico.topo import Topology
from splico.nutils.mul import multipatch, basis_spline
from splico.spl import UnivariateKnotVector, NDSpline, NDSplineArray
from splico.types import Float, Int, Numeric

from itertools import product
from functools import lru_cache

from nutils import function, solver
import treelog as log


PATCHES = (2, 0, 3, 1), \
          (2, 4, 0, 6), \
          (2, 3, 4, 5), \
          (3, 1, 5, 7), \
          (4, 5, 6, 7)

KNOTVECTOR_EDGES = (0, 1), (0, 6), (0, 2)


ALL_SIDES = 'left', 'right', 'top', 'bottom'


def trampoline_template(inner_height: Float = .5,
                        inner_width: Float = .5):
  r"""
    1 - - - - - 7
    | \   D   / |
    |  3 - - 5  |
    |A |  C  | E|
    |  |     |  |
    |  2 - - 4  |
    | /   B   \ |
    0 - - - - - 6
  """
  assert 0 < inner_height < 1
  assert 0 < inner_width < 1

  h, w = (1 - inner_height) / 2, (1 - inner_width) / 2

  # the vertices of the multipatch topology numbered from 0 to 7
  patchverts = ( (0, 0), (0, 1),
                 (w, h), (w, 1 - h),
                 (1 - w, h), (1 - w, 1 - h),
                 (1, 0), (1, 1) )

  # a patch is characterised by its patch vertices
  # patch = (0, 1, 2, 3) means it is oriented like this:
  """
      1 ----- 3
      |       |
      |       |
      |       |
      0-------2
  """

  # so the left side of the patch is given by edge 0 --- 1
  # right by 2 --- 3
  # bottom by 0 --- 2
  # top by 1 --- 3

  # patches and patchverts are tuple. Immutable objects.

  return Topology(PATCHES), patchverts


@lru_cache(maxsize=32)
def _disc(kv0, kv1, kv2, reparam: bool = True) -> NDSplineArray:

  topo, patchverts = trampoline_template()
  opposites = topo.opposites

  kvs = {}
  for edge, kv in zip(KNOTVECTOR_EDGES, (kv0, kv1, kv2)):
    for _edge in (edge, *opposites[edge]):
      kvs[_edge] = kv
      kvs[_edge[::-1]] = kv.flip()

  degree = max( kv.degree for kv in kvs.values() )

  domain, geom, localgeom = multipatch(patches=topo.topo,
                                       knotvectors=kvs,
                                       patchverts=patchverts)

  basis = basis_spline(domain, kvs)
  basisd = basis_spline(domain, kvs, patchcontinuous=False)

  controlmap = make_unit_disc(domain, basis, geom, localgeom, topo.topo, reparam=reparam)

  controlpoints = domain.project(controlmap, onto=basisd.vector(2),
                                             geometry=geom,
                                             degree=2*degree).reshape(-1, 2)

  tkvs = [kvs[patch[::2]] * kvs[patch[:2]] for patch in PATCHES]
  segments = np.array([kv.ndofs for kv in tkvs]).cumsum()[:-1]

  spls = [NDSpline(kv, np.concatenate([cps, np.zeros(len(cps))[:, _]], axis=1))
          for kv, cps in zip(tkvs, np.array_split(controlpoints, segments, axis=0))]

  return NDSplineArray(spls)


def ellipse(a: Numeric, b: Numeric, kv0: Int | UnivariateKnotVector = 5,
                                    kv1: Int | UnivariateKnotVector | None = None,
                                    kv2: Int | UnivariateKnotVector | None = None,
                                    reparam: bool = True) -> NDSplineArray:
  """
  Create a five patch ellipse spline with major axes of length `a` and `b`.

  Parameters
  ----------
  a : Numeric
    First axis length.
  b : Numeric
    Second axis length.
  kv0 : Int | UnivariateKnotVector
    The knotvector assigned to edge (0, 1). If of type `Int`, it gets
    coerced into a `UnivariateKnotVector` with `degree=3` and `kv0` internal
    elements.
  kv1 : Int | UnivariateKnotVector | None
    The knotvector assigned to edge (0, 6). `Int` inputs get coerced and
    if `None`, defaults to `kv0`.
  kv2 : Int | UnivariateKnotVector | None
    The knotvector assigned to edge (0, 2). `Int` inputs get coerced and
    if `None`, defaults to `kv0`.
  reparam : bool
    Whether to suppress the disc's interface angles.
  """

  if isinstance(kv0, Int):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, kv0 + 1), degree=3)

  if isinstance(kv1, Int):
    kv1 = UnivariateKnotVector(np.linspace(0, 1, kv1 + 1), degree=3)
  elif kv1 is None:
    kv1 = kv0

  if isinstance(kv2, Int):
    kv2 = UnivariateKnotVector(np.linspace(0, 1, kv2 + 1), degree=3)
  elif kv2 is None:
    kv2 = kv0

  return _disc(kv0, kv1, kv2, reparam=reparam) * np.array([a, b, 0])[_]


def get_all_vertices(domain, geom, patches):

  # XXX: docstring

  verts = {}

  for i, patch in enumerate(patches):
    for ii, j in enumerate(patch):
      if j in verts: continue
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = domain._topos[i].boundary[side0] \
                                 .boundary[side1] \
                                 .sample('vertex', 0) \
                                 .eval(geom).ravel()

  return np.stack([verts[key] for key in sorted(verts.keys())], axis=0)


def multipatch_trace_penalty_stab(domain,
                                  basis,
                                  geom,
                                  localgeom,
                                  patches,
                                  stabmat=None,
                                  mu=800):

  """
  Nutils implementation for reparameterizing the default covering
  of the unit disc toward more interface regularity.

  The stabilization for avoiding singularities at the vertices is tuned by
  `mu`.
  """

  assert mu >= 0

  if stabmat is None:
    stabmat = lambda x: function.eye(2)

  Jmu = function.normalized(geom.grad(localgeom), axis=0)
  D = function.matmat(Jmu, Jmu.T)

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  stabmat = stabmat(x)

  allverts = get_all_vertices(domain, geom, patches)
  blending = sum( np.exp(-mu * ((vert - geom)**2).sum()) for vert in allverts )

  D = (1 - blending) * D + blending * stabmat
  Dx = function.matmat(x.grad(geom), D)

  integrand = (basis.vector(2).grad(geom) * Dx[None]).sum([1, 2])
  res = domain.integral( integrand * function.J(geom), ischeme='gauss6')

  cons = domain.boundary.project(geom, onto=basis.vector(2),
                                       geometry=geom,
                                       ischeme='gauss6')

  controlmap = basis.vector(2) \
                    .dot(solver.solve_linear('target', res, constrain=cons))

  return controlmap


def make_unit_disc(domain, basis, geom, localgeom, patches, reparam=False, **kwargs):

  """
  Create a multipatch covering of the unit disc from the topology.
  """

  if reparam is False and kwargs:
    log.warning('reparam is False so the keyword arguments will be ignored.')

  center = np.array([.5, .5])

  func = function.normalized(geom - center) / np.sqrt(2) + center
  cons = domain.boundary.project(func, geometry=geom,
                                       onto=basis.vector(2),
                                       ischeme='gauss6')

  for (i, patch), side in product(enumerate(domain._topos), ALL_SIDES):
    cons |= patch.boundary[side].project(geom, geometry=geom,
                                               onto=basis.vector(2),
                                               ischeme='gauss10')

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  J = x.grad(geom)
  G = function.matmat(J.T, J)

  costfunc = domain.integral(np.trace(G) * function.J(geom), degree=10)
  controlmap = basis.vector(2) \
                    .dot(solver.optimize('target', costfunc, constrain=cons))

  if not reparam:
    return (controlmap - center) * np.sqrt(2)

  return (multipatch_trace_penalty_stab(domain,
                                        basis,
                                        controlmap,
                                        localgeom,
                                        patches,
                                        **kwargs) - center) * np.sqrt(2)
