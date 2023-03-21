from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from .Disposable import Disposable
from .Event import Event2, IEvent2_r

T = TypeVar('T')

class List(Disposable, Generic[T]):
    """List model"""
    def __init__(self):
        super().__init__()
        self._list = []

        self._mx_added = Event2[int, T]().dispose_with(self)
        self._mx_remove = Event2[int, T]().dispose_with(self)

    def __dispose__(self):
        self._list = None
        super().__dispose__()

    @property
    def mx_added(self) -> IEvent2_r[int, T]:
        """added value (idx, value)"""
        return self._mx_added

    @property
    def mx_remove(self) -> IEvent2_r[int, T]:
        """about to remove value (idx, value)"""
        return self._mx_remove

    def values(self) -> Sequence[T]:
        return self._list

    def append(self, value : T):
        self._list.append(value)
        idx = len(self._list)-1

        self._mx_added.emit(idx, value)

    def remove(self, value : T):
        """
        remove first occurence of value
        raise ValueError if value not present
        """
        idx = self._list.index(value)
        self._mx_remove.emit(idx, value)
        self._list.remove(value)
