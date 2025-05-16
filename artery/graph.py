from pyhaemo.types import frozen, normalize, np, _, frozen_cached_property
from pyhaemo.types import Immutable
from pyhaemo.spl import NDSpline

from notwistframe import compute_notwistframe
from artery import Artery
from aux import compute_orientation, project_onto_plane

from typing import Sequence

from numpy.typing import NDArray


class Bifurcation(Immutable):

  @classmethod
  def from_points_tangents(cls, centerpoints, tangents, normal=None,
                                                        bifcenter=None):
    centerpoints = np.asarray(centerpoints, dtype=float)
    tangents = np.asarray(tangents, dtype=float)

    assert len(tangents) == len(centerpoints)

    if bifcenter is None:
      bifcenter = np.mean(centerpoints, axis=0)

    if normal is None:
      normal = compute_orientation(centerpoints - bifcenter)

    t0 = tangents[0]
    normal_proj = normalize(project_onto_plane(normal, t0))
    binormal = normalize(np.cross(t0, normal))

    frames = [np.stack((normal_proj, binormal, t0), axis=1)]

    for tangent in tangents[1:]:
      tangents_ = np.stack((t0, -tangent), axis=0)
      frame = compute_notwistframe(tangents_,
                                   n0=normal_proj)

      frames.append(frame[-1] @ np.diag([1, -1, -1]))

    return cls(centerpoints, frames, normal, bifcenter)

  def __init__(self, centerpoints: NDArray,
                     frames: NDArray,
                     normal: NDArray,
                     bifcenter: NDArray):

    self.centerpoints = frozen(centerpoints, dtype=float)
    self.frames = frozen(frames, dtype=float)
    self.normal = frozen(normal, dtype=float)
    self.bifcenter = frozen(bifcenter, dtype=float)

    assert all( arr.shape[-1:] == (3,) for arr in map(self.__getattr__,
                                                      self._field_names) )

    assert self.normal.ndim == self.bifcenter.ndim == 1
    assert self.centerpoints.shape[:-1] == (len(self.frames),)
    assert self.frames.shape[1:] == (3, 3)

  def __len__(self):
    return len(self.points)

  def orient(self, return_indices=True):
    ppoints = project_onto_plane(self.centerpoints, self.normal)
    vecs = normalize(ppoints - np.mean(ppoints, axis=0))

    cross = normalize(np.cross(vecs[0], vecs[1:]))
    dots = np.sum(vecs[:1] * vecs[1:], axis=1)
    orientation = np.sign((cross * self.normal[_]).sum(1))

    indices = np.argsort(np.concatenate([[0], np.arccos(dots) * orientation]))

    ret = self._edit( points=self.points[indices],
                      tangents=self.tangents[indices],
                      normal=self.normal,
                      bifcenter=self.bifcenter )

    if return_indices:
      return ret, tuple(indices)

    return ret


class FrameGraph(Immutable):

  def __init__(self, artery: Artery,
                     normals: NDArray):
    assert isinstance(artery, Artery)
    self.artery = artery
    self.normals = frozen(normals, dtype=float)
    assert self.normals.shape == (self.artery.nedges, 2, 3)

  @frozen_cached_property
  def endpoint_frames(self):
    tangents = np.stack([ cl(cl.knots[0][[0, -1]], dx=1)
                          for cl in self.artery.centerlines ], axis=0)
    binormals = normalize(np.cross(tangents, self.normals))

    return np.stack([self.normals, binormals, tangents], axis=-1)
