from .Disposable import Disposable
from .Property import IProperty_r, Property


class IProgress_r:
    @property
    def mx_started(self) -> IProperty_r[bool]: ...
    @property
    def mx_progress(self) -> IProperty_r[int]: ...
    @property
    def mx_progress_max(self) -> IProperty_r[int]: ...

class Progress(Disposable, IProgress_r):
    """
    Int value progress control
    """

    def __init__(self):
        """"""
        super().__init__()
        self._mx_started = Property[bool](False).dispose_with(self)
        self._mx_progress = Property[int](0).dispose_with(self)
        self._mx_progress_max = Property[int](0).dispose_with(self)

    @property
    def mx_started(self) -> IProperty_r[bool]: return self._mx_started
    @property
    def mx_progress(self) -> IProperty_r[int]:
        return self._mx_progress
    @property
    def mx_progress_max(self) -> IProperty_r[int]:
        return self._mx_progress_max

    def start(self, i : int, max : int):
        self._mx_progress.set(i)
        self._mx_progress_max.set(max)
        self._mx_started.set(True)

    def progress(self, progress : int):
        self._mx_progress.set(progress)

    def inc(self):
        self._mx_progress.set(self._mx_progress.get()+1)

    def finish(self):
        self._mx_started.set(False)