from .. import mx
from ..lib import time as lib_time
from .QProgressBar import QProgressBar


class QProgressBarMxProgress(QProgressBar):

    def __init__(self, progress : mx.IProgress_r, hide_inactive = False):
        super().__init__()

        self._progress = progress
        self._hide_inactive = hide_inactive
        self._show_it_s = False

        self.set_minimum(0)

        self._sps = lib_time.SPSCounter()

        progress.mx_started.reflect(self._ref_started).dispose_with(self)
        progress.mx_progress.reflect(self._ref_progress).dispose_with(self)
        progress.mx_progress_max.reflect(self.set_maximum).dispose_with(self)

    def set_show_it_s(self, show : bool):
        self._show_it_s = show
        return self

    def _ref_started(self, started : bool):
        if started:
            self._sps.reset()
            if self._hide_inactive:
                self.show()
        else:
            if self._hide_inactive:
                self.hide()

    def _ref_progress(self, value : int):
        if self._show_it_s:
            self.set_format(f'%v / %m ({self._sps.step():.1f} it/s)')
        else:
            self.set_format(f'%v / %m')

        self.set_value(value)




