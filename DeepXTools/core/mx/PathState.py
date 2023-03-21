from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .Disposable import Disposable
from .Property import IProperty_r, Property


@dataclass
class PathStateConfig:
    """```
        allow_open(True)  Allows opening existing file/dir.

        allow_new(False)  Allows new file/dir

        ^ if both are false, will work only as closeable control.

        dir_only(False)   Accepted directory only

        extensions      Accepted file extensions if not directory.
                        example ['jpg','png']

        desc            Description
                        example 'Video File'
                                'Sequence directory'
    ```"""
    allow_open : bool = True
    allow_new : bool = False
    dir_only : bool = False
    extensions : Sequence[str]|None = None
    desc : str|None = None

    def acceptable_extensions(self, path : Path) -> bool:
        return self.dir_only or self.extensions is None or path.suffix[1:] in self.extensions

    def openable(self, path : Path) -> bool:
        return (not self.dir_only or path.is_dir()) and self.acceptable_extensions(path) and path.exists()

    def newable(self, path : Path) -> bool:
        return self.dir_only or self.acceptable_extensions(path)

class IPathState_r:
    """read-only interface of PathState"""
    @property
    def config(self) -> PathStateConfig:  ...
    @property
    def mx_path(self) -> IProperty_r[Path|None]:  ...


class IPathState(IPathState_r):
    def close(self): ...
    def open(self, path : Path): ...
    def new(self, path : Path): ...

class PathState(Disposable, IPathState):
    def __init__(self,  config : PathStateConfig = None,
                        on_open : Callable[ [Path], bool] = None,
                        on_new : Callable[ [Path], bool] = None,
                        on_close : Callable[ [], None] = None ):
        """```
        Operate file/dir open/new/close

            allow_open(True)  Allows opening existing file/dir.

            allow_new(False)  Allows new file/dir

            ^ if both are false, will work only as closeable control.

        Auto closes on dispose.
        ```"""
        super().__init__()
        self.__config = config if config is not None else PathStateConfig()

        self.__on_open = on_open if on_open is not None else lambda p: True
        self.__on_new = on_new if on_new is not None else lambda p: True
        self.__on_close = on_close if on_close is not None else lambda: ...
        self.__mx_path = Property[Path|None](None).dispose_with(self)

    @property
    def config(self) -> PathStateConfig: return self.__config

    @property
    def mx_path(self) -> IProperty_r[Path|None]:
        """Indicates current opened Path"""
        return self.__mx_path

    def __dispose__(self):
        self.close()
        super().__dispose__()

    def close(self):
        """Close path"""
        if self.__mx_path.get() is not None:
            self.__on_close()
            self.__mx_path.set(None)

    def open(self, path : Path):
        """Open path if applicable by configuration."""
        if self.__config.openable(path):
            self.close()
            if self.__on_open(path):
                self.__mx_path.set(path)

    def new(self, path : Path):
        """New path if applicable by configuration."""
        if self.__config.newable(path):
            self.close()
            if self.__on_new(path):
                self.__mx_path.set(path)