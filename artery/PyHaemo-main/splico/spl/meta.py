"""
Metaclasses specific to the `splico.spl` package.
@author: Jochen Hinz
"""

from ..types import ImmutableMeta


class NDSplineMeta(ImmutableMeta):
  """
  Metaclass for the NDSpline class.
  Is equivalent to :type:`ImmutableMeta` but automatically inherits a number
  of functions from :class:`splico.spl.TensorKnotVector` that create a new
  knotvector.
  The spline is then automatically prolonged to the newly created knotvector.
  Similarly, a number of properties are inherited. They can then directly be
  called from the spline rather than delegating them to the spline's knotvector.
  For example:

  >>> spl
  ... NDSpline<4, 5>
  >>> spl.knotvector.knots
  ... ([0, 0.1, 0.2, 0.4, 0.8, 1.0],)
  >>> spl.knots
  ... ([0, 0.1, 0.2, 0.4, 0.8, 1.0],)
  """
  def __new__(mcls, *args, **kwargs):
    cls = super().__new__(mcls, *args, **kwargs)

    def forward_and_refine(name):
      # create a new function that forwards input to the knotvector
      # and then prolongs to the new knotvector.
      def wrapper(self, *args, **kwargs):
        new_knotvector = getattr(self.knotvector, name)(*args, **kwargs)
        return self.prolong_to(new_knotvector)
      return wrapper

    def wrap_property(name):
      # create a new property that returns what the instantiation's knotvector
      # property of the same name would return.
      return property(lambda self: getattr(self.knotvector, name))

    # create the methods that are inherited from the knotvector
    # and subsequently prolonged to the new knotvector
    for name in ('refine', 'ref_by', 'add_knots', 'raise_multiplicities'):
      setattr(cls, name, forward_and_refine(name))

    # add the properties that are inherited from the knotvector
    for name in ('knots', 'km', 'degree', 'repeated_knots', 'greville'):
      setattr(cls, name, wrap_property(name))

    return cls
