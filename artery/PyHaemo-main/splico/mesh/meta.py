"""
Module containing the :type:`MeshMeta` metaclass for use in
:class:`splico.mesh.Mesh`.

@author: Jochen Hinz
"""

from splico.types import ImmutableMeta


class MeshMeta(ImmutableMeta):
  """
  Metaclass for :class:`splico.mesh.Mesh` that ensures that the mesh class
  implements the necessary class-level attributes. The class-level attributes
  are copied from the reference element class to the mesh class. The reference
  element class must implement the following class-level attributes:
    `element_name`, `ndims`, `nverts`, `is_affine`
  as well as the `_local_ordinances` method.
  """

  def __new__(mcls, *args, **kwargs):
    """
    Copy the necessary class-level attributes from the reference element class
    to the mesh class.
    If the reference element class does not implement the necessary attributes,
    raise a :class:`TypeError` upon instantiation.
    """
    cls = super().__new__(mcls, *args, **kwargs)
    try:
      for attr in ('element_name', 'ndims', 'nverts', 'is_affine'):
        setattr(cls, attr, getattr(cls.reference_element, attr))
      cls._local_ordinances = \
          lambda self, *args, **kwargs: \
                 self.reference_element._local_ordinances(*args, **kwargs)
    except AttributeError:
      pass
    return cls

  def __call__(cls, *args, **kwargs):
    # make sure that class implements all necessary class-level attributes
    # before instantiation.
    try:  # hasattr evaluates to true because of class-level annotation
      cls.reference_element
    except AttributeError:
      raise TypeError("Cannot instantiate a mesh class that does not implement"
                      " the `reference_element` class-level attribute.")
    return ImmutableMeta.__call__(cls, *args, **kwargs)
