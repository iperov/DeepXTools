from .. import qt
from ._helpers import q_init
from .QWidget import QWidget


class QHeaderView(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_header_view', qt.QHeaderView, **kwargs), **kwargs)

    def get_q_header_view(self) -> qt.QHeaderView: return self.get_q_widget()

    def set_stretch_last_section(self, b : bool):
        self.get_q_header_view().setStretchLastSection(b)
        return self