from .. import qt
from ._helpers import q_init
from .QAbstractSlider import QAbstractSlider


class QScrollBar(QAbstractSlider):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_slider=q_init('q_scroll_bar', qt.QScrollBar, **kwargs), **kwargs)

    def get_q_scroll_bar(self) -> qt.QScrollBar: return self.get_q_abstract_slider()