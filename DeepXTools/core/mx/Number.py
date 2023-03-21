from __future__ import annotations

from dataclasses import dataclass
from numbers import Number as NumberType
from typing import Callable

from .Disposable import Disposable
from .Event import EventConnection
from .Property import DeferredProperty, FilteredProperty


@dataclass
class NumberConfig:
    min : NumberType|None = None
    max : NumberType|None = None
    step : NumberType = 1
    decimals : int = 0
    read_only : bool = False

    def filter(self, new_value : NumberType, value : NumberType) -> NumberType:
        if self.read_only:
            return value

        step = self.step
        if step != 1:
            new_value = round(new_value / step) * step

        if self.min is not None:
            new_value = max(self.min, new_value)

        if self.max is not None:
            new_value = min(new_value, self.max)

        return new_value

class INumber_r:
    """read-only interface of Number"""
    @property
    def config(self) -> NumberConfig:  ...
    def reflect(self, func : Callable[[NumberType], None]) -> EventConnection: ...
    def listen(self, func : Callable[[NumberType], None]) -> EventConnection: ...
    def get(self) -> NumberType: ...

class INumber(INumber_r):
    """interface of Number"""
    def set(self, value : NumberType): ...

class Number(Disposable, INumber):
    def __init__(self,  number : NumberType,
                        config : NumberConfig|None = None,

                        defer : Callable[ [NumberType, NumberType, Number], None ] = None,
                        filter : Callable[[NumberType, NumberType], NumberType] = None,
                 ):

        super().__init__()
        self.__config = config if config is not None else NumberConfig()

        if defer is not None:
            self.__prop = DeferredProperty[NumberType](number, defer=lambda n, o, prop: defer(config.filter(n, o), o, self)).dispose_with(self)
        elif filter is not None:
            self.__prop = FilteredProperty[NumberType](number, filter=lambda n, o: filter(config.filter(n, o), o)).dispose_with(self)
        else:
            self.__prop = FilteredProperty[NumberType](number, filter=lambda n, o: config.filter(n, o)).dispose_with(self)

    @property
    def config(self) -> NumberConfig:  return self.__config

    def reflect(self, func : Callable[[NumberType], None]) -> EventConnection:
        return self.__prop.reflect(func)

    def listen(self, func : Callable[[NumberType], None]) -> EventConnection:
        return self.__prop.listen(func)

    def get(self) -> NumberType: return self.__prop.get()

    def set(self, value : NumberType):
        self.__prop.set(value)
        return self

    def fset(self, value : NumberType):
        """set value avoiding filter"""
        self.__prop.fset(value)
        return self