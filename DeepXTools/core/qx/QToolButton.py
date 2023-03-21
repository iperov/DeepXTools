from .. import qt
from .QAbstractButton import QAbstractButton

from._constants import ArrowType
from ._helpers import q_init


class QToolButton(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_button=q_init('q_toolbutton', qt.QToolButton, **kwargs), **kwargs)

    def get_q_toolbutton(self) -> qt.QPushButton: return self.get_q_abstract_button()

    def set_arrow_type(self, arrow_type : ArrowType):
        self.get_q_toolbutton().setArrowType(arrow_type)
        return self