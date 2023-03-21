from typing import Callable, Sequence, TypeVar

from .Property import FilteredProperty, IProperty, IProperty_r

T = TypeVar('T')


class ISingleChoice_r(IProperty_r[T]):
    @property
    def avail(self) -> Sequence[T]: ...

class ISingleChoice(ISingleChoice_r, IProperty[T]):
    ...

class SingleChoice(FilteredProperty[T], ISingleChoice):
    """
    SingleChoice is FilteredProperty[T] that is filtered by dynamic .avail values.
    """
    def __init__(self,  value : T,
                        avail : Callable[[], Sequence[T]],
                        filter : Callable[[T, T], T] = None
                 ):
        """
        ```
            avail       sequence of available T values.
                        Len must be >= 1
        ```
        Initial value is checked to be in avail.

        If value is not in avail, then first avail value is set.
        """
        a = avail()
        if value not in a:
            value = a[0]

        self.__avail = avail
        self.__filter = filter

        super().__init__(value, filter=self.__filter_func)

    @property
    def avail(self) -> Sequence[T]:
        """evaluate current dynamic avail values"""
        return tuple(dict.fromkeys(self.__avail()))

    def __filter_func(self, new_value : T, value : T):
        if new_value not in self.avail:
            new_value = value

        if self.__filter is not None:
            new_value = self.__filter(new_value, value)
        return new_value

