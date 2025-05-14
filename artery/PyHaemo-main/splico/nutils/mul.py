from splico.util import np, _
from splico.spl import UnivariateKnotVector
from splico._jit import _apply_pairs

import itertools
from typing import Tuple, Dict, List

from nutils import numeric, transform, transformseq, _util as util, \
                   topology, types, function
import treelog as log
from numpy.typing import ArrayLike


def reflect_kv_dict(kvdict: Dict) -> Dict:
  return {**kvdict, **{k[::-1]: -kv for k, kv in kvdict.items()}}


@log.withcontext
def multipatch(patches: ArrayLike,
               knotvectors: Dict[Tuple[int, int], UnivariateKnotVector],
               patchverts=None, space='X'):
  """
  Slight modification of the multipatch function from nutils.
  Create a multipatch topology from a list of patches and a dictionary of
  knotvectors that map each edge to a knotvector.

  Parameters
  ----------
  patches : array_like
    Array of shape (npatches,2,...,2) or (npatches,2*ndims) containing the
    indices of the vertices of each patch.  The number of vertices must be a
    power of two.
  knotvectors : dict
    Dictionary mapping each edge to a knotvector.  The keys are tuples
    containing the indices of the vertices of the edge, and the values are
    UnivariateKnotVector objects.
  patchverts : array_like, optional
    Array of shape (nverts,ndims) containing the coordinates of the vertices
    of the patches.  If not provided, the vertices are assumed to be in the
    unit hypercube.
  space : str, optional
    Name of the space to be used for the topology.  Default is 'X'.

  Returns
  -------
  tuple
    A tuple containing the topology, the geometry, and the local geometry of
    the multipatch.
  """
  knotvectors = reflect_kv_dict(knotvectors)

  patches = np.array(patches)
  if patches.dtype != int:
    raise ValueError('`patches` should be an array of ints.')
  if patches.ndim < 2 or patches.ndim == 2 and patches.shape[-1] % 2 != 0:
    raise ValueError('`patches` should be an array with shape'
                     ' (npatches,2,...,2) or (npatches,2*ndims).')
  elif patches.ndim > 2 and patches.shape[1:] != (2,) * (patches.ndim - 1):
    raise ValueError('`patches` should be an array with shape'
                     ' (npatches,2,...,2) or (npatches,2*ndims).')
  patches = patches.reshape(patches.shape[0], -1)

  # determine topological dimension of patches

  ndims = 0
  while 2**ndims < patches.shape[1]:
    ndims += 1
  if 2**ndims > patches.shape[1]:
    raise ValueError('Only hyperrectangular patches are supported: '
                     'number of patch vertices should be a power of two.')
  patches = patches.reshape([patches.shape[0]] + [2]*ndims)

  bnames: Tuple[Tuple[str, str], ...] = \
    ('left', 'right'), ('bottom', 'top'), ('front', 'back')

  if ndims > 3:
    bnames = bnames + tuple( (str(2*i), str(2*i+1)) for i in range(3, ndims) )

  # group all common patch edges (and/or boundaries?)

  # create patch topologies, geometries

  if patchverts is not None:
    patchverts = np.array(patchverts)
    indices = set(patches.flat)
    if tuple(sorted(indices)) != tuple(range(len(indices))):
      raise ValueError('Patch vertices in `patches` should be numbered'
                       ' consecutively, starting at 0.')
    if len(patchverts) != len(indices):
      raise ValueError('Number of `patchverts` does not equal number of'
                       ' vertices specified in `patches`.')
    if len(patchverts.shape) != 2:
      raise ValueError('Every patch vertex should be an array of dimension 1.')

  topos = []
  coords = []
  localcoords = []
  for i, patch in enumerate(patches):
    # find shape of patch and local patch coordinates
    shape = []
    for dim in range(ndims):
      nelems_sides = []
      sides: List[Tuple] = [(0, 1)]*ndims
      sides[dim] = slice(None),
      for side in itertools.product(*sides):
        # sideverts = frozenset(patch[side])
        sideverts = tuple(patch[side])
        nelems_sides.append(knotvectors[sideverts].knotvalues)
      if len(set(nelems_sides)) != 1:
        raise ValueError('duplicate number of elements specified for patch'
                         ' {} in dimension {}'.format(i, dim))
      shape.append(nelems_sides[0])

    # create patch topology
    topos.append(topology.StructuredTopology(
    space,
    root=transform.Index(ndims, i),
    axes=[transformseq.DimAxis(i=0, j=len(n)-1, mod=0, isperiodic=False) for n in shape],
    bnames=bnames[:ndims]))

    # compute patch geometry
    patchcoords = numeric.meshgrid(*shape).reshape(ndims, -1)
    localpatchcoords = numeric.meshgrid(*shape).reshape(ndims, -1)
    if patchverts is not None:
      patchcoords = np.array([
        sum(
          patchverts[j]*util.product(c if s else 1-c for c, s in zip(coord, side))
          for j, side in zip(patch.flat, itertools.product(*[[0, 1]]*ndims))
       )
        for coord in localpatchcoords.T
      ]).T
    coords.append(patchcoords)
    localcoords.append(localpatchcoords)

  # build patch boundary data

  topo = topology.MultipatchTopology(topos, patches)
  funcsp = topo.basis('spline', degree=1, patchcontinuous=False)
  geom = (funcsp * np.concatenate(coords, axis=1)).sum(-1)
  localgeom = (funcsp * np.concatenate(localcoords, axis=1)).sum(-1)

  return topo, geom, localgeom


def basis_spline(self, knotvectors: Dict[Tuple[int, ...], UnivariateKnotVector],
                 patchcontinuous: bool = True):
  """
  Spline from knotvectors
  Create a spline basis with degree ``degree`` per patch.  If
  ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
  interfaces.

  Parameters
  ----------
  knotvectors : dict
    Dictionary mapping each edge to a knotvector.  The keys are tuples
    containing the indices of the vertices of the edge, and the values are
    UnivariateKnotVector objects.
  patchcontinuous : bool, optional
    If true, the basis is $C^0$-continuous at patch interfaces.  Default is
    true.
  """

  knotvectors = reflect_kv_dict(knotvectors)

  knotvalues = {edge: kv.knotvalues for edge, kv in knotvectors.items()}
  knotmultiplicities = {edge: kv.knotmultiplicities for edge, kv in knotvectors.items()}
  degrees = {edge: kv.degree for edge, kv in knotvectors.items()}

  missing = object()

  coeffs = []
  dofmap: List = []
  dofcount = 0
  commonboundarydofs: Dict[Tuple[int, ...], List[np.ndarray]] = {}
  for ipatch, (topo, verts) in enumerate(zip(self._topos, self._connectivity)):
    # build structured spline basis on patch `patch.topo`
    patchknotvalues: List = []
    patchknotmultiplicities: List = []
    patchdegree: List = []
    for idim in range(self.ndims):
      left = tuple(0 if j == idim else slice(_) for j in range(self.ndims))
      right = tuple(1 if j == idim else slice(_) for j in range(self.ndims))
      dimknotvalues, dimknotmultiplicities, dimdegrees = set(), set(), set()
      for edge in zip(verts[left].flat, verts[right].flat):
        v = knotvalues[edge]
        m = knotmultiplicities[edge]
        d = degrees[edge]
        if v is missing:
          raise KeyError('missing edge')
        dimknotvalues.add(v)
        if m is missing:
          raise KeyError('missing edge')
        dimknotmultiplicities.add(m)
        if d is missing:
          raise KeyError('missing edge')
        dimdegrees.add(d)
      if len(dimknotvalues) != 1:
        raise ValueError('ambiguous knot values for patch {}, dimension {}'.format(ipatch, idim))
      if len(dimknotmultiplicities) != 1:
        raise ValueError('ambiguous knot multiplicities for patch {}, dimension {}'.format(ipatch, idim))
      if len(dimdegrees) != 1:
        raise ValueError('ambiguous degree for patch {}, dimension {}'.format(ipatch, idim))
      patchknotvalues.extend(dimknotvalues)
      patchknotmultiplicities.extend(dimknotmultiplicities)
      patchdegree.extend(dimdegrees)

    patchcoeffs, patchdofmap, patchdofcount = \
      topo._basis_spline(patchdegree, knotvalues=patchknotvalues,
                                      knotmultiplicities=patchknotmultiplicities)
    coeffs.extend(patchcoeffs)
    dofmap.extend(types.frozenarray(dofs+dofcount, copy=False) for dofs in patchdofmap)
    if patchcontinuous:
        # reconstruct multidimensional dof structure
        dofs = dofcount + np.arange(np.prod(patchdofcount), dtype=int).reshape(patchdofcount)
        for idim, iside, idx in self._iter_boundaries():
            # get patch boundary dofs and reorder to canonical form
            boundarydofs = dofs[idx].ravel()
            # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
            commonboundarydofs.setdefault(tuple(verts[idx].flat), []).append(boundarydofs)
    dofcount += np.prod(patchdofcount)

  merge = np.arange(dofcount)
  if patchcontinuous:
    # build merge mapping: merge common boundary dofs (from low to high)
    pairs = np.array(list(itertools.chain(*(zip(*dofs) for dofs
                     in commonboundarydofs.values() if len(dofs) > 1))))
    if pairs.shape[0]:
      merge = _apply_pairs(merge, pairs)
      # merge twice to remove annoying bug that occurs when an interior interface
      # has two different ids on the sides of 2 patches
    assert all(np.all(merge[a] == merge[b]) for a, *B in commonboundarydofs.values() for b in B), \
           'something went wrong is merging interface dofs; this should not have happened'
    # build renumber mapping: renumber remaining dofs consecutively, starting at 0
    remainder, renumber = np.unique(merge, return_inverse=True)
    # apply mappings
    dofmap = list(types.frozenarray(renumber[v], copy=False) for v in dofmap)
    dofcount = len(remainder)

  return function.PlainBasis(coeffs, dofmap, dofcount, self.f_index, self.f_coords)
