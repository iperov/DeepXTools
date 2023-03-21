from __future__ import annotations

from typing import Callable, Generic, TypeVar

from .Disposable import Disposable
from .Event import Event1, IEvent_r
from .EventConnection import EventConnection

T = TypeVar('T')


class IProperty_r(Generic[T]):
    """Read-only interface of Property"""
    def listen(self, func : Callable[[T], None]) -> EventConnection: ...
    def reflect(self, func : Callable[[T], None]) -> EventConnection: ...
    def get(self) -> T: ...

class IProperty(IProperty_r[T]):
    """interface of Property"""
    def set(self, value : T): ...

class Property(Disposable, IProperty[T]):
    """holds T value. Listenable/reflectable, direct get/set."""
    def __init__(self, value : T):
        super().__init__()
        self._ev = Event1[T]().dispose_with(self)
        self._value = value

    def listen(self, func : Callable[[T], None]) -> EventConnection:
        return self._ev.listen(func)

    def reflect(self, func : Callable[[T], None]) -> EventConnection:
        conn = self._ev.listen(func)
        conn.emit(self._value)
        return conn

    def get(self) -> T:
        return self._value

    def set(self, value : T):
        self._value = value
        self._ev.emit(value)
        return self

class GetSetProperty(Disposable, IProperty[T]):
    """
    GetSetProperty built on getter, setter, and optional changed event.
    Does not hold value internally.
    """
    def __init__(self, getter, setter, changed_event : IEvent_r|None = None):
        super().__init__()
        self._ev = Event1[T]().dispose_with(self)
        self._getter = getter
        self._setter = setter

        if changed_event is not None:
            self._conn = changed_event.listen(lambda *_: self._ev.emit(self._getter())).dispose_with(self)
        else:
            self._conn = None

    def listen(self, func : Callable[[T], None]) -> EventConnection:
        return self._ev.listen(func)

    def reflect(self, func : Callable[[T], None]) -> EventConnection:
        conn = self._ev.listen(func)
        conn.emit(self._getter())
        return conn

    def get(self) -> T: return self._getter()

    def set(self, value : T):
        if (conn := self._conn) is not None:
            with conn.disabled_scope():
                self._setter(value)

        self._ev.emit(self._getter())


class DeferredProperty(Property[T]):
    """
    as Property,
    but `set()` is redirected to
    `defer(new_value, value, prop : DeferredProperty[T])`
    in which you should call (immediately or later) `prop.fset(v)` to set the value
    """
    def __init__(self, value : T, defer : Callable[ [T, T, DeferredProperty[T]], None ]):
        super().__init__(value)
        self._defer = defer

    def set(self, value : T):
        self._defer(value, self._value, self)
        return self

    def fset(self, value : T):
        """set value avoiding filter"""
        super().set(value)
        return self

class FilteredProperty(Property[T]):
    """
    as Property,
    but `set()` is filtered via `filter(new_value : T, value : T) -> T`
    """
    def __init__(self, value : T, filter : Callable[ [T, T], T ]):
        super().__init__(value)
        self._filter = filter

    def set(self, value : T):
        return super().set(self._filter(value, self._value))

    def fset(self, value : T):
        """set value avoiding filter"""
        super().set(value)
        return self

# class FilteredProperty(Disposable, IProperty_r[T]):
#     """
#     same as Property but also has a filter to control incoming new value.
#     """

#     def __init__(self, value : T, filter : Callable[ [T,T], T ] = ...):
#         """```
#             value           initial value

#             filter(...)     filter( new_value, old_value ) -> value
#                             Filter incoming value. Must return filtered value.

#                             ...:  identity filter

#                             None:   no filter.
#                                     In this case you have to attach to p_in manually,
#                                     and use p_out to post the value
#         ```"""
#         super().__init__() ###
#         p_in = self.__p_in = Event1[T]().dispose_with(self)
#         p_out = self.__p_out = Property[T](value).dispose_with(self)

#         if filter is not None:
#             if filter is Ellipsis:
#                 p_in.listen(lambda v: p_out.set(v)).dispose_with(self)
#             else:
#                 p_in.listen(lambda v: p_out.set(filter(v, p_out.get()))).dispose_with(self)

#     @property
#     def p_in(self) -> Event1[T]: return self.__p_in

#     @property
#     def p_out(self) -> Property[T]: return self.__p_out

#     def listen(self, func : Callable[[T], None]) -> EventConnection:
#         return self.__p_out.listen(func)

#     def reflect(self, func : Callable[[T], None]) -> EventConnection:
#         return self.__p_out.reflect(func)

#     def get(self) -> T:
#         return self.__p_out.get()

#     def set(self, value : T):
#         self.__p_in.emit(value)
#         return self

