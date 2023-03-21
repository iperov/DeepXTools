from .. import qt
from ._helpers import q_init
from .QAbstractButton import QAbstractButton


class QPushButton(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_button=q_init('q_pushbutton', qt.QPushButton, **kwargs), **kwargs)

    def get_q_pushbutton(self) -> qt.QPushButton: return self.get_q_abstract_button()





