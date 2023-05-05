from typing import Callable, ParamSpec, TypeVar, Generic

T = TypeVar('T')
P = ParamSpec('P')


class CachedWrapper(Generic[P, T]):
    def __init__(self, func : Callable[P, T]):
        self._func = func
        self._cached = {}

    def __call__(self, *args : P.args, **kwargs : P.kwargs) -> T:
        key = (args, tuple(kwargs.items()))
        value = self._cached.get(key, Ellipsis)
        if value is Ellipsis:
            value = self._cached[key] = self._func(*args, **kwargs)
        return value

    def is_cached(self, *args : P.args, **kwargs : P.kwargs) -> bool:
        key = (args, tuple(kwargs.items()))
        return self._cached.get(key, Ellipsis) is not Ellipsis

    def get_cached(self, *args : P.args, **kwargs : P.kwargs) -> T:
        key = (args, tuple(kwargs.items()))
        return self._cached[key]


def cache(func : Callable[P, T]) -> CachedWrapper[P, T]:
    """
    Decorator.

    same as @functools.cache , but with access to cached dict and valid intellisense hint
    """
    return CachedWrapper[P, T](func)