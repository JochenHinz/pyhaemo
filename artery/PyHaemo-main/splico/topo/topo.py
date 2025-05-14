from .util import all_permutations, tuple_convert
from splico.types import Immutable, Int, lock
from splico.util import np, frozen
from splico.err import DuplicateOrientationError

from itertools import product
from functools import lru_cache, cached_property
from typing import Tuple, Self

from numpy.typing import NDArray, ArrayLike
from matplotlib import pyplot as plt
import networkx


@lru_cache
def _orient_topo(topo: 'Topology') -> 'Topology':
  """
  Greedy algorithm for orienting the topology.
  """

  if len(topo) in (0, 1) or topo.is_valid():
    return topo

  positive_edges = set()
  new_topo = []

  for patch in topo:

    # loop over all permutations of the patch
    for perm in patch.permutations_iter():
      seen = set()  # keep track of all seen edges
      for ipatch, edge in perm.patch_edge_iter():

        # conflict found, move to the next perm
        if edge[::-1] in positive_edges: break
        seen.add(edge)

      else:  # no conflicts found, add to the new topology

        new_topo.append(perm.topo)
        positive_edges.update(seen)
        break  # break out of the for loop over all permutations

    else:  # no valid orientation found, raise an error
      raise DuplicateOrientationError("No valid orientation found.")

  return Topology(np.concatenate(new_topo))


ZEROTOPO = lambda n: np.zeros((0, 2**max(n, 0)), dtype=int)


class Topology(Immutable):
  """
  A topology induced by shared vertices.
  """

  def __init__(self, topo: ArrayLike) -> None:
    self.topo = frozen(topo)
    n, m = self.topo.shape
    assert (m & (m-1) == 0) and m != 0
    assert self.ndim >= 0

  @property
  def tensortopo(self):
    return self.topo.reshape((-1,) + ((2,) * self.ndim or (1,)))

  @cached_property
  def ndim(self):
    ndim = 0
    while 2 ** ndim != self.shape[1]:
      ndim += 1
    return ndim

  def permutations_iter(self):
    for perm in all_permutations(self.ndim):
      yield Topology(self.topo[:, perm])

  @property
  def shape(self):
    return self.topo.shape

  def __len__(self):
    return len(self.topo)

  def __bool__(self):
    return len(self) > 0

  def __repr__(self):
    if not self:
      return f"{self.__class__.__name__}({f"[], shape={self.shape}"})"
    return f"{self.__class__.__name__}({repr(self.topo)[6:-1]})"

  def __iter__(self):
    yield from (self.take([i]) for i in range(len(self)))

  def patch_edge_iter(self):
    """
    Iterator over patch index and the edge of the patch.
    """
    ndim = self.ndim
    zo, sl = [(0, 1)], [(slice(None),)]
    for (i, patch), j in product(enumerate(self.tensortopo), range(ndim)):
      for mysl in product(*(zo * j + sl + zo * (ndim - j - 1))):
        yield i, tuple_convert(patch[mysl])

  def patch_interface_iter(self):
    """
    Iterator over patch index and the interfaces of the patch.
    """
    ndim = self.ndim
    zo, sl = [(0, 1)], [(slice(None),)]
    for (i, patch), j in product(enumerate(self.tensortopo), range(ndim)):
      for mysl in product(*(sl * j + zo + sl * (ndim - j - 1))):
        yield i, tuple_convert(patch[mysl])

  def patch_interface_flat_iter(self):
    """
    Iterator over patch index and the interfaces of the patch.
    """
    for ipatch, intf in self.patch_interface_iter():
      yield ipatch, tuple_convert(np.ravel(intf))

  @cached_property
  @lock
  def map_edge_orientation(self):
    """
    Map edge to orientation.
    The orientation is 1 for the original edge and -1 for the flipped one.
    If conflicting orientations are found, raise DuplicateOrientationError.
    """
    map_edge_orientation = {}

    for ipatch, edge in self.patch_edge_iter():
      for myedge, ori in zip((edge, edge[::-1]), (1, -1)):
        if not map_edge_orientation.setdefault(myedge, ori) == ori:
          raise DuplicateOrientationError

    return map_edge_orientation

  def is_valid(self) -> bool:
    """
    Return True if no conflicting edge orientations are found.
    If the topology is empty, return True.
    """
    try:
      self.map_edge_orientation
      return True
    except DuplicateOrientationError:
      return False

  @cached_property
  @lock
  def map_interface_minterface(self):
    """
    Map each interface to the interface that minimizes all equivalent
    interfaces with the same digits but different order.
    Minimize means minimized in the lexicographic order.
    """
    # for each interface, sort the indices and keep only the one that is minimal
    find_minterface, minterface = {}, {}
    for ipatch, intf in self.patch_interface_flat_iter():
      mykey = tuple(sorted(intf))
      find_minterface[mykey] = min(intf, find_minterface.get(mykey, intf))

    # map each interface to the minimal interface
    for ipatch, intf in self.patch_interface_flat_iter():
      minterface[intf] = find_minterface[tuple(sorted(intf))]

    return minterface

  @cached_property
  def minterfaces(self) -> Tuple:
    return tuple(sorted(set(self.map_interface_minterface.values())))

  @cached_property
  @lock
  def map_interface_patch(self):
    """
    Map each minimal interface to the patches that contain it.
    Keys are in flat layout.
    """

    minterfaces = self.map_interface_minterface

    # map each minimal interface to the patches that border it
    map_interface_patch = {}
    for ipatch, intf in self.patch_interface_flat_iter():
      map_interface_patch.setdefault(minterfaces[intf], set()).add(ipatch)

    return {key: tuple(sorted(val)) for key, val in map_interface_patch.items()}

  @cached_property
  @lock
  def _immediate_opposites(self):
    """
    Map each interface to its immediate opposite.
    Do this for all orientations of the interface
    (and orient the opposite accordingly).
    In tensor layout.
    """
    ret = {}
    ttopo = self.tensortopo

    sl = slice(None), slice(None, None, -1)
    ndim = self.ndim

    for patch, j in product(ttopo, range(self.ndim)):
      for sl0, sl1 in zip(product(*([sl] * j + [(0,)] + [sl] * (ndim - j - 1))),
                          product(*([sl] * j + [(1,)] + [sl] * (ndim - j - 1)))):

        ind0 = tuple_convert(patch[sl0])
        ind1 = tuple_convert(patch[sl1])

        ret.setdefault(ind0, set()).add(ind1)
        ret.setdefault(ind1, set()).add(ind0)

    return {key: tuple(sorted(val)) for key, val in ret.items() }

  @cached_property
  @lock
  def opposites(self):
      """
      Map each interface to ALL its opposite interfaces and their opposites.
      As before, do this for all orientations of the interface.
      """
      opp = {key: set(val) for key, val in self._immediate_opposites.items()}
      while True:
          new_opp = {
              key: set.union(*[opp[i] for i in val], val) - {key}
              for key, val in opp.items()
          }
          if new_opp == opp: break
          opp = new_opp

      return {key: tuple(sorted(val)) for key, val in opp.items()}

  def orient(self):
    """
    Orient to Nutils layout.
    Take an initial, potentially invalid patch layout positively oriented
    in nutils layout and return a valid nutils layout that preserves
    the positive orientation.
    """

    return _orient_topo(self)

  @cached_property
  def interfaces(self) -> Self:
    """
    Get the interfaces of the topology and return as new topology.
    The function handles two special cases:
    1. If the topology is empty, return a topology with zero edges and lower
       the dimension by one (unless already zero).
    2. If the topology is one-dimensional but not empty,
       return a topology with zero edges.
    """
    if not self or not self.ndim:
      return self._edit(topo=ZEROTOPO(self.ndim - 1))

    return self._edit(topo=self.minterfaces)

  def extrude(self, axis: Int = -1):
    if not self:
      return self._edit(topo=ZEROTOPO(self.ndim + 1))

    axis = axis % (self.ndim + 1) + 1
    maxindex = self.topo.max() + 1
    topo = np.stack([self.tensortopo, self.tensortopo + maxindex], axis=axis)

    return self._edit( topo=topo.reshape(len(self), -1) )

  def take(self, elements: ArrayLike) -> Self:
    return self._edit(topo=np.atleast_2d(self.topo[np.asarray(elements)]))

  def plot_verts(self):
    assert self.ndim
    G = networkx.Graph()
    while self.ndim > 1:
      self = self.interfaces

    for edge in self.topo:
      G.add_edge(*edge)

    networkx.draw_planar(G, with_labels=True)
    plt.show()

  def plot(self, ax=None, **kwargs):
    G = networkx.Graph()
    G.add_nodes_from(range(len(self)))

    for i, patches in self.map_interface_patch.items():
      for j, ipatch in enumerate(patches):
        for k in patches[:j] + patches[j+1:]:
          G.add_edge(ipatch, k)

    networkx.draw_planar(G, with_labels=True)
    plt.show()


def as_topo(topo: Topology | ArrayLike) -> Topology:
  if isinstance(topo, Topology):
    return topo
  return Topology(topo)
