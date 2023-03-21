from .. import qt
from ._helpers import q_init
from .QAbstractButton import QAbstractButton


class QCheckBox(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_button=q_init('q_checkbox', qt.QCheckBox, **kwargs), **kwargs)

    def get_q_checkbox(self) -> qt.QCheckBox: return self.get_q_abstract_button()
