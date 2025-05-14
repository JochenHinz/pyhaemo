"""
Module declaring various types that find applications throughout the library.
In particular, this module introduces :class:`Immutable`, a general purpose
base class for immutable and hashable classes. It also introduces the
:class:`Singleton` base class which is a variant of :class:`Immutable` that
can only create one instance per input.

In its current development phase, the ``SpliCo`` library prioritizes robustness
over performance. As such, we prioritize the use of immutable types to avoid
side effects and to ensure that objects can be hashed. The slight
re-instantiation overhead is negligible compared to the benefits of immutability.
The potential performance gains from using mutable classes are currently
negligible compared to the benefits of immutability, particularly as
performance-critical routines already leverage extensive Numpy vectorization
and JIT compilation in Numba.

While we currently discourage classes that forgo the advantages of immutability,
more permissive design patterns may be introduced in the future.
An example of this is the `splico.spl.NDSpline` class which may permit inplace
modifications to enhance interoperability with the Numpy API in the future.

@author: Jochen Hinz
"""


from .util import serialize_array, deserialize_array, serialized_array, \
                  serialize_objarr, deserialize_objarr, serialized_objarr, np
from .err import UnequalLengthError, CannotSetImmutableAttributeError
from .log import logger as log

from functools import wraps
from collections import ChainMap
from collections.abc import Hashable, Mapping
from abc import ABCMeta
from typing import Any, Self, Tuple, List, Dict, TypeVar, Sequence, \
                   Callable
from types import EllipsisType
from weakref import WeakValueDictionary
from inspect import signature, Signature, Parameter

from numpy.typing import NDArray


# Various fused types for type hinting

T = TypeVar('T')

Int = int | np.integer
Float = float | np.floating
Numeric = Int | Float

IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]
NumericArray = IntArray | FloatArray

NumpyIndex = Int | slice | Sequence[Int] | np.ndarray | None | EllipsisType
MultiNumpyIndex = Tuple[NumpyIndex, ...]

AnySequence = Sequence[T] | Tuple[T, ...]
AnyIntSeq = AnySequence[Int] | IntArray
AnyFloatSeq = AnySequence[Float] | FloatArray
AnyNumericSeq = AnySequence[Numeric] | NumericArray


# Decorators for use in classes' instance methods


def ensure_same_class(fn: Callable) -> Callable:
  """
  A decorator ensuring that a function of the form ``fn(self, other)``
  immediately returns ``NotImplemented`` in case
  ``self.__class__ is not other.__class__``.

  >>> @ensure_same_class
  ... def __le__(self, other):
  ...   return self.attr <= other.attr

  >>> @ensure_same_class
  ... def __eq__(self, other):
  ...   return self.attr == other.attr

  >>> a = MyClass(attr=5)  # derives from Immutable
  >>> b = MyClass(attr=5)
  >>> a <= b
  ... True
  >>> a == b
  ... True
  >>> c = MyClass(attr=4)
  >>> a <= c
  ... False
  >>> a <= 'foo'  # inequality not resolved
  ... ERROR
  >>> a == 'foo'  # equality not resolved
  ... False
  >>> c = MyOtherClass(attr=5)  # derives from Immutable
  >>> a <= c
  ... ERROR
  >>> a == c
  ... False
  """
  @wraps(fn)
  def wrapper(self, other: Any, *args, **kwargs):
    if self.__class__ is not other.__class__:
      return NotImplemented
    return fn(self, other, *args, **kwargs)
  return wrapper


def ensure_same_length(fn: Callable) -> Callable:
  """
  Decorator for making sure that sized objects have equal length.
  The types of ``self`` and ``other`` have to match.
  """
  @wraps(fn)
  def wrapper(self, other):
    if len(self) != len(other):
      raise UnequalLengthError("Can only perform operation on instances of"
                               " equal length.")
    return fn(self, other)
  return wrapper


# Various signature methods for use in metaclasses


def remove_self(signature: Signature) -> Signature:
  """
  Remove the ``self`` argument from a bound method signature.
  """
  assert next(iter(signature.parameters.keys())) == 'self'
  return Signature(list(signature.parameters.values())[1:])


def is_valid_signature(signature: Signature) -> bool:
  """
  Return ``True`` if an `__init__` signature neither contains `*args`
  nor `**kwargs`.
  """
  return bool(set(map(lambda x: x.kind, signature.parameters.values())) &
              {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}) is False


# Metaclasses and various base classes forming the core of most of the
# library's class structure, in particular the `splico.mesh.Mesh` and
# the `splico.spl.NDSpline` base class.


Im = TypeVar('Im', bound='Immutable')


class ImmutableMeta(ABCMeta):
  """
  Metaclass for immutable types. For use in the :class:`Immutable` base class.
  If a class does not implement the ``_field_names`` class-level attribute, it
  is inferred from the ``__init__``'s signature. The signature cannot be
  inferred if it contains ``*args`` or ``**kwargs``. If ``_field_names`` is not
  implemented or cannot be inferred from the base classes' signatures, the
  base class prevents instantiation.

  Example
  -------

  >>> class MyClass(metaclass=ImmutableMeta):
  ...   def __init__(self, a: float, b: float):
  ...     self.a = float(a)
  ...     self.b = float(b)

  >>> MyClass._field_names
  ... ('a', 'b')

  Similarly

  >>> class MyClass(metaclass=ImmutableMeta):
  ...   def __init__(self, *args, **kwargs):
  ...     pass

  >>> MyClass._field_names
  ... ERROR

  If a derived class has an `__init__` of the form `(self, *args, **kwargs)`
  and doesn't implement `_field_names`` explicitly, it is inferred from the
  parent class or its parent classes.

  >>> class MyDerivedClass(MyClass):  # MyClass._field_names == ('a', 'b')
  ...   def __init__(self, *args, **kwargs):
  ...     super().__init__(*args, **kwargs)

  >>> MyDerivedClass._field_names
  ... ('a', 'b')
  """

  def __new__(mcls, name, bases, attrs, *args, **kwargs):
    cls = super().__new__(mcls, name, bases, attrs, *args, **kwargs)

    # since we overwrite __call__, inspect cannot infer the correct signature
    # therefore we set it manually from the cls.__init__
    cls.__signature__ = remove_self(signature(cls.__init__))

    try:
      _field_names = attrs['_field_names']
      assert all(isinstance(element, str) for element in _field_names), \
          'Received invalid type for the _item class-level argument.'
      cls._field_names = tuple(_field_names)  # ensure it's converted to a tuple

      if is_valid_signature(cls.__signature__):
        if set(_field_names) != set(cls.__signature__.parameters.keys()):
          log.warning(f"Warning, the class `{cls.__name__}`'s signature does not"
                       " match its `_field_names` implementation which will lead"
                       " to errors when using `self._edit` or `hash(self)`.")
    except KeyError:
      # _field_names has not been implemented as a class-level attribute
      # -> try to infer  it first from the __init__ signature. The signature
      # may not contain  *args or **kwargs.
      # If this fails, try to infer the signature from the parent classes.
      # If that fails too and `_field_names` isn't implemented,
      # an error is thrown upon trying to instantiate the class.

      for pcls in cls.__mro__:
        if is_valid_signature((sig := remove_self(signature(pcls.__init__)))):
          cls._field_names = tuple(sig.parameters.keys())
          break

    return cls

  def __call__(cls, *args: Any, **kwargs: Any):
    """
    Overwrite the __call__ method to prevent instantiation if the class does
    not implement the `_field_names` class-level attribute.
    """
    # make sure that if _field_names is not implemented or inferred,
    # the class is not instantiated.
    if not hasattr(cls, '_field_names'):
      raise TypeError("The class's `_field_names` class-level attribute has "
                      "not been implemented or could not be inferred. Cannot "
                      "instantiate a class that does not implement this "
                      "attribute.")

    if len(args) == 1 and isinstance(args[0], cls):
      assert args[0]._is_initialized and not kwargs

      # if the first argument is of the same type as the class,
      # return the first argument
      if args[0].__class__ is cls:
        return args[0]

      # else try to coerce to current superclass via the _lib attribute
      return cls(**args[0]._lib)

    ret = type.__call__(cls, *args, **kwargs)

    # set private variable to prevent overwriting attributes
    ret._is_initialized = True
    return ret


class ArrayCoercionMixin:

  def __array__(self, dtype=None, copy=False):
    # avoid invoking `__iter__` on `self` which would return
    # an array of shape (ndim,) of UnivariateKnotVectors
    arr = np.empty((1,), dtype=object)
    arr[0] = self
    return arr.reshape(())


class Immutable(ArrayCoercionMixin, metaclass=ImmutableMeta):
  """
  Generic base class for immutable types.
  Has the ``_field_names`` class-level attribute. The ``_field_names``
  attribute is a tuple of strings where each string represents the name of a
  class attribute (typically set in ``__init__``) that contributes to the
  class's hash. If the class does not explicitly implement ``_field_names``,
  it is inferred from the ``__init__``'s signature. So, if a class stores the
  inputs it gets as attributes of the same name, it is possible to skip the
  implementation of ``_field_names``.
  Each element in ``_field_names`` needs to refer to a hashable type with the
  exception of :class:`np.ndarray` which is serialized using ``serialize_array``.

  The class then implements the ``__hash__`` and the ``__eq__`` dunder methods
  in the obvious way. For this, the ``__hash__` and ``__eq__`` dunder methods
  make use of the ``_tobytes`` cached property which serializes all relevant
  attributes and returns them as a tuple.

  The same Mixin implements the ``_lib`` method that returns a dictionary of
  all (relevant) attributes implemented by ``_field_names``, i.e.,
      {item: getattr(self, item) for item in self._field_names}.
  The ``_edit`` method then allows for editing single or several attributes
  while keeping all others intact and instantiates a new instance of the
  same class with the updated attributes.

  It is of paramount importance that the derived class is immutable.
  This means that all relevant attributes are immutable and hashable with
  the exception of :class:`np.ndarray`s which need to be frozen using
  ``util.frozen`` or ``util.freeze``.

  >>> class MyClass(Immutable):
  ...   _field_names = 'a', 'b'
  ...   def __init__(self, a: np.ndarray, b: Tuple[int, ...]):
  ...     self.a = frozen(a, dtype=float)
  ...     self.b = tuple(map(int, b))

  Or with signature inference:

  >>> class MyClass(Immutable):
  ...   def __init__(self, a: np.ndarray, b: Tuple[int, ...]):
  ...     self.a = frozen(a, dtype=float)
  ...     self.b = tuple(map(int, b))

  >>> MyClass._field_names
  ... ('a', 'b')

  >>> A = MyClass( np.linspace(0, 1, 11), (1, 2, 3) )
  >>> hash(A)
  ... 5236462403644277461

  >>> B = MyClass( np.linspace(0, 1, 11), (1, 2, 3) )
  >>> A == B
  ... True

  >>> B = MyClass( np.linspace(1, 2, 11), (1, 2, 3) )
  >>> A == B
  ... False

  >>> A._edit(a=np.linspace(1, 2, 11)) == B
  ... True
  """

  # class-level annotation to avoid mypy errors
  _field_names: Tuple[str, ...]
  _is_initialized: bool
  __signature__: Signature

  @property
  def _lib(self) -> dict:
    """
    Return a dictionary of all relevant attributes.
    """
    return {item: getattr(self, item) for item in self._field_names}

  def _edit(self, **kwargs) -> Self:
    """
    Edit single or multiple attributes of the class while keeping
    all others intact.
    """
    return self.__class__(**ChainMap(kwargs, self._lib))

  @property
  def _tobytes(self) -> Tuple[Hashable, ...]:
    """
    Serialize all relevant attributes and return them as a tuple.
    """
    ret: List[Hashable] = []
    for i, attr in enumerate(map(self.__getattribute__, self._field_names)):
      if isinstance(attr, np.ndarray):

        if attr.flags.writeable is True:
          log.warning("Warning, attempting to hash the attribute "
                      f"`{self._field_names[i]}` which is a writeable"
                      " `np.ndarray`. Ensure that all `np.ndarray` attributes"
                      " are read-only using `util.frozen` or `util.freeze`.")

        if attr.dtype == object:
          # serialize object arrays
          ret.append(serialize_objarr(attr))
        else:
          # serialize all other arrays
          ret.append(serialize_array(attr))
      elif isinstance(attr, Hashable):
        ret.append(attr)
      else:
        raise AssertionError(f"Attribute of type '{str(type(attr))}'"
                             " cannot be hashed.")
    return tuple(ret)

  def __repr__(self):
    rdict = repr({key: object.__repr__(getattr(self, key)) for key in self._field_names})[1:-1]
    return f'{self.__class__.__name__}({rdict})'

  def __setattr__(self, name: str, value: Any) -> None:
    """
    Prevent overwriting of attributes after initialization.
    """
    if name in self._field_names:
      try:
        object.__getattribute__(self, name)
        raise CannotSetImmutableAttributeError(
                              f"The {self.__class__.__name__}'s immutability"
                              " prohibits overwriting attributes in "
                              " `_field_names`.")
      except AttributeError:
        # attribute has not been set yet
        pass
    super().__setattr__(name, value)

  def __getstate__(self) -> Tuple[Hashable, ...]:
    """ For pickling. """
    return self._tobytes

  def __setstate__(self, state) -> None:
    """ For unpickling. """
    args = {}
    for key, item in zip(self._field_names, state):
      if isinstance(item, serialized_array):
        item = deserialize_array(item)
      elif isinstance(item, serialized_objarr):
        item = deserialize_objarr(item)
      args[key] = item
    self.__init__(**args)

  def __hash__(self) -> int:
    """
    Default implementation of ``__hash__`` for hashing of immutable objects.
    Once computed, the hash is cached for future use.
    """
    if not hasattr(self, '_hash'):
      self._hash = hash(self._tobytes)
    return self._hash

  def __eq__(self, other: Any) -> bool:
    """
    Default implementation of ``__eq__`` for comparison between same types.
    """
    if self.__class__ is not other.__class__:
      return NotImplemented  # Liskov substitution principle
    if self is other:
      return True
    if not hash(self) == hash(other):  # differing hashes means inequality
      return False
    for item0, item1 in zip(self._tobytes, other._tobytes):
      if item0 != item1: return False
    return True


class SingletonMeta(ImmutableMeta):
  """
  Singleton meta class.
  The same as :type:`ImmutableMeta` but adds a weakref cache to the class
  for memoizing class instances while avoiding memory leaks.
  Each separate class using this metaclass receives its own unique cache.
  """

  @staticmethod
  def _canonicalize_args(cls, *args: Hashable, **kwargs: Dict[str, Hashable]):
    """
    Bring input arguments along with defaults (if applicable)
    into one canonical form suitable for hashing. Avoids duplicates resulting
    from instantiating via positional or keyword arguments.
    """
    signature = cls.__signature__
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args, tuple(bound.kwargs.items())

  def __new__(mcls, *args, **kwargs):
    """
    Overwrite the __new__ method to add a weakref cache to the class.
    """
    cls = super().__new__(mcls, *args, **kwargs)
    cls._cache = WeakValueDictionary()
    return cls

  def __call__(cls, *args, **kwargs):
    """
    The __call__ method is overwritten to ensure that only one instance
    per set of inputs can exist at a time.
    """
    _args = cls._canonicalize_args(cls, *args, **kwargs)
    try:
      return cls._cache[_args]
    except KeyError:  # do the usual and add to weakref dictionary
      return cls._cache.setdefault(_args, super().__call__(*args, **kwargs))


class Singleton(Immutable, metaclass=SingletonMeta):
  """
  Singleton base class.
  For each set of inputs only one instance can exist at a time.
  Each instance may only take hashable arguments (for now) in order to be
  eligible for hashing.

  >>> class MySingleton(Singleton):
  ...   def __init__(self, a: float, b: float):
  ...     self.a = float(a)
  ...     self.b = float(b)

  >>> A = MySingleton(5, 3)
  >>> B = MySingleton(a=5.0, b=3.0)  # integer floats behave equivalent to ints
  >>> A is B
  ... True
  """

  _cache: WeakValueDictionary


class LockableDict(Mapping):
  """
  A dictionary that can be locked after initialization.
  If all values are hashable, the dictionary itself is hashable.
  Must be locked to be hashable.
  """

  _wrapped_dict: Mapping

  def __new__(cls, *args, locked=False, **kwargs):
    if args:
      assert not kwargs
      if len(args) > 1:
        raise TypeError(f'Expected at most 1 argument, got {len(args)}')
      args, = args
      if isinstance(args, LockableDict):
        return LockableDict(**args, locked=locked)
      kwargs = dict(args)
    self = super().__new__(cls)
    self._locked = False
    self._wrapped_dict = kwargs
    return self

  @staticmethod
  def _delegate_to_wrapped_dict(method: str) -> Callable:
    def wrapper(self, *args, **kwargs):
      return getattr(self._wrapped_dict, method)(*args, **kwargs)
    return wrapper

  def lock(self):
    """
    Lock the dictionary after initialization.
    """
    self._locked = True

  __getitem__ = _delegate_to_wrapped_dict('__getitem__')
  __iter__ = _delegate_to_wrapped_dict('__iter__')
  __len__ = _delegate_to_wrapped_dict('__len__')

  def __setitem__(self, name, value):
    if self._locked:
      raise KeyError(f'{self.__class__.__name__} is frozen')
    self._wrapped_dict.__setitem__(name, value)

  def __hash__(self):
    if not self._locked:
      raise TypeError(f'Cannot hash a {self.__class__.__name__}'
                      ' that is not locked')
    if not hasattr(self, '_hash'):
      self._hash = hash(tuple(sorted(self.items(), key=lambda x: x[0])))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, Mapping):
      return NotImplemented
    if self is other:
      return True
    return dict(self.items()) == dict(other.items())

  def __repr__(self):
    return f'{self.__class__.__name__}({self._wrapped_dict})'


def lock(f: Callable) -> Callable:
  """
  Decorator for locking a dictionary after initialization.
  The dictionary is initialized with the return value of the function.
  """
  @wraps(f)
  def wrapper(*args, **kwargs) -> LockableDict:
    ret = f(*args, **kwargs)
    return LockableDict(ret, locked=True)
  return wrapper


class NanVec(np.ndarray):
  """
  Vector of dtype float initilized to np.nan.
  Is used to implement Dirichlet boundary conditions.
  """

  @classmethod
  def from_indices_data(cls, length, indices, data):
    'Instantiate NanVec x of length ``length`` satisfying x[indices] = data.'
    vec = cls(length)
    vec[np.asarray(indices)] = np.asarray(data)
    return vec

  def __new__(cls, length):
    vec = np.empty(length, dtype=float).view(cls)
    vec[:] = np.nan
    return vec

  @property
  def where(self):
    """
    Return boolean mask ``mask`` of shape self.shape with mask[i] = True
    if self[i] != np.nan and False if self[i] == np.nan.
    >>> vec = NanVec(5)
    >>> vec[[1, 2, 3]] = 7
    >>> vec.where
    ... [False, True, True, True, False]
    """
    return ~np.isnan(self.view(np.ndarray))

  def __ior__(self, other):
    """
    Set self[~self.where] to other[~self.where].
    Other is either an array-like of self.shape or a scalar.
    If it is a scalar set self[~self.where] = other
    >>> vec = NanVec(5)
    >>> vec[[0, 4]] = 5
    >>> vec |= np.array([0, 1, 2, 3, 4])
    >>> print(vec)
    ... [5, 1, 2, 3, 5]

    >>> vec = NanVec(5)
    >>> vec[[0, 4]] = 5
    >>> vec |= 0
    >>> print(vec)
    ... [5, 0, 0, 0, 5]
    """
    wherenot = ~self.where
    self[wherenot] = other if np.isscalar(other) else other[wherenot]
    return self

  def __or__(self, other):
    """
    Same as self.__ior__ but the result is cast into a new vector, i.e.,
    z = self | other.
    """
    return self.copy().__ior__(other)
