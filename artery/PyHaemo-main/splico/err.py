"""
Module defining various custom errors.
@author: Jochen Hinz
"""


""" For use in splico.mesh.mesh """


class MissingVertexError(Exception):
  """
  Raised when a vertex is not found in the mesh.
  """
  pass


class HasNoSubMeshError(Exception):
  """
  Raised when a mesh has no submesh.
  """
  pass


class HasNoBoundaryError(Exception):
  """
  Raised when a mesh has no boundary.
  """
  pass


class EmptyMeshError(Exception):
  """
  Deprecated. Raised when a mesh is empty. Empty meshes are now supported.
  """
  pass


""" Container-type  / sized-type operations """


class UnequalLengthError(Exception):
  """ For operations that require two iterables to have equal length. """
  pass


class EmptyContainerError(Exception):
  """
  For preventing instantiations of empty container-types.
  No longer used because empty :class:`NDSpline` objects are now supported.
  """
  pass


class DuplicateOrientationError(Exception):
  """ We throw this error when we detect a duplicate orientation. """
  pass


class CannotSetImmutableAttributeError(Exception):
  """
  Raised when trying to set an attribute that is immutable.
  """
  pass
