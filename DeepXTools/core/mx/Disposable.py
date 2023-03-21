from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Set, TypeVar

DEBUG = False


class Disposable:
    """
    Base class for disposable objects.
    Not thread-safe.
    """

    TDisposable = TypeVar('TDisposable', bound='Disposable')

    def __init__(self):
        self.__disposed = 0
        self.__disposes_with : Deque[Disposable] = deque()
        self.__disposed_in : Set[Disposable] = set()

        if self.dispose.__func__ != Disposable.dispose:
            raise Exception(f'You must not to override {self.__class__.__qualname__}.dispose(), use __dispose__ instead.')

        if DEBUG:
            self.__dbg_parent = None
            self.__dbg_name = ''

    def __del__(self):
        if self.__disposed == 0:
            if DEBUG:
                name = deque()
                obj = self
                while obj is not None:
                    name.append(f'{obj}{obj.__dbg_name}')
                    obj = obj.__dbg_parent
                name = '/'.join(name)
            else:
                name = self

            print(f'WARNING. {name}.dispose() was not called.')

    def __dispose__(self):
        """
        inheritable at last.

        provide your disposition code here.
        """
        # Dispose "dispose_with" objects.
        disposes_with = self.__disposes_with
        while len(disposes_with) != 0:
            obj = disposes_with[0]

            if isinstance(obj, Disposable):
                obj.dispose()
            else:
                # Callable
                obj()
                disposes_with.popleft()

    def dispose(self) -> None:
        """"""
        #print(f"{' '*Disposable._indent}Dispose {self}")
        if self.__disposed == 0:
            self.__disposed = 1
        else:
            raise Exception(f'Disposing already disposed {self.__class__.__qualname__}')

        # Remove self from Disposable's where we should "dispose_with"
        disposables, self.__disposed_in = self.__disposed_in, None
        for disposable in disposables:
            disposable.__disposes_with.remove(self)

        Disposable._indent += 2
        self.__dispose__()
        Disposable._indent -= 2

        self.__disposed = 2

    def dispose_with(self : TDisposable, other : Disposable) -> TDisposable:
        """Dispose with other disposable. All disposables will be disposed in FILO order."""
        if other.__disposed == 2:
            raise Exception(f'{other} already disposed.')

        if other not in self.__disposed_in:
            self.__disposed_in.add(other)
            other.__disposes_with.appendleft(self)
        return self

    def dispose_and_new(self) -> Disposable:
        """disposes this Disposable and returns new Disposable()"""
        self.dispose()
        return Disposable()

    def call_on_dispose(self, func : Callable):
        """Will be called in FILO order together with other `dispose_with` obhects"""
        self.__disposes_with.appendleft(func)
        return self

    def inline(self : TDisposable, func : Callable[ [TDisposable], None ]) -> TDisposable:
        func(self)
        return self

    if DEBUG:
        def __setattr__(self, name: str, value) -> None:
            if isinstance(value, Disposable) and name != '_Disposable__dbg_parent':
                value.__dbg_parent = self
                value.__dbg_name = name
            super().__setattr__(name, value)

    def __repr__(self): return self.__str__()
    def __str__(self): return f"[{self.__class__.__qualname__}]{'[DISPOSED]' if self.__disposed == 2 else '[DISPOSING]' if self.__disposed == 1 else ''}"

    _indent = 0


class DisposableBag(Disposable):
    def __init__(self):
        super().__init__()
        self._items = deque()

    def dispose_items(self):
        """"""
        items = self._items
        while len(items) != 0:
            obj = items[0]
            obj.dispose()
