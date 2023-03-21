from typing import Callable

from .. import qt
from ._helpers import q_init
from .QEvent import QEvent0
from .QObject import QObject


class QTimer(QObject):
    def __init__(self, on_timeout : Callable, **kwargs):
        super().__init__(q_object=q_init('q_timer', qt.QTimer, **kwargs), **kwargs)

        q_timer = self.get_q_timer()
        QEvent0(q_timer.timeout).dispose_with(self).listen(on_timeout)

    def __dispose__(self):
        self.get_q_timer().stop()
        super().__dispose__()

    def get_q_timer(self) -> qt.QTimer: return self.get_q_object()

    def set_interval(self, msec : int):
        self.get_q_timer().setInterval(msec)
        return self

    def start(self):
        self.get_q_timer().start()
        return self