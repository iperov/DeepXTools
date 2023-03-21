from typing import Callable, Sequence, TypeVar

from .Property import FilteredProperty, IProperty, IProperty_r

T = TypeVar('T')

class IMultiChoice_r(IProperty_r[T]):
    @property
    def avail(self) -> Sequence[T]: ...

class IMultiChoice(IMultiChoice_r, IProperty[T]):
    def update(self): ...
    def update_added(self, v): ...
    def update_removed(self, v): ...

class MultiChoice(FilteredProperty[ Sequence[T] ], IMultiChoice):
    """
    MultiChoice is FilteredProperty[ Sequence[T] ] that is filtered by dynamic unique .avail values.
    """
    def __init__(self,  avail : Callable[[], Sequence[T]],
                        filter : Callable[[ Sequence[T], Sequence[T] ], Sequence[T] ] = None
                 ):
        self.__avail = avail
        self.__filter = filter
        super().__init__([], filter=self.__filter_func)

    @property
    def avail(self) -> Sequence[T]:
        """evaluate current dynamic unique avail values"""
        return tuple(dict.fromkeys(self.__avail()))

    def get(self) -> Sequence[T]:
        return super().get()

    def set(self, value : Sequence[T]):
        return super().set(value)

    def update_added(self, v):
        """"""
        self.set(self.get() + (v,))
        return self

    def update_removed(self, v):
        """"""
        self.set(tuple(x for x in self.get() if x != v) )
        return self
    
    def update(self): 
        """same as .set(.get())"""
        self.set(self.get())
        return self
        
    def __filter_func(self, new_value : Sequence[T], value : Sequence[T]):
        avail = self.avail

        new_value = tuple(dict.fromkeys(x for x in new_value if x in avail))

        if self.__filter is not None:
            new_value = self.__filter(new_value, value)
        return new_value

