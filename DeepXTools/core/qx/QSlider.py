from .. import qt
from ._helpers import q_init
from .QAbstractSlider import QAbstractSlider


class QSlider(QAbstractSlider):
    TickPosition = qt.QSlider.TickPosition

    def __init__(self, **kwargs):
        super().__init__(q_abstract_slider=q_init('q_slider', qt.QSlider, **kwargs), **kwargs)

        self.set_tick_position(QSlider.TickPosition.NoTicks)

    def get_q_slider(self) -> qt.QSlider: return self.get_q_abstract_slider()

    def set_tick_position(self, tick_position : TickPosition):
        self.get_q_slider().setTickPosition(tick_position)
        return self
