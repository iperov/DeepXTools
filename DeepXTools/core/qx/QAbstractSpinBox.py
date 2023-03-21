from .. import mx, qt
from ._helpers import q_init
from .QEvent import QEvent0
from .QWidget import QWidget


class QAbstractSpinBox(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_abstract_spin_box', _QAbstractSpinBoxImpl, qt.QAbstractSpinBox, **kwargs), **kwargs)

        q_abstract_spin_box = self.get_q_abstract_spin_box()
        self._mx_editing_finished = QEvent0(q_abstract_spin_box.editingFinished).dispose_with(self)

        if isinstance(q_abstract_spin_box, _QAbstractSpinBoxImpl):
            ...

    @property
    def mx_editing_finished(self) -> mx.IEvent0_r: return self._mx_editing_finished

    def get_q_abstract_spin_box(self) -> qt.QAbstractSpinBox: return self.get_q_widget()

    def set_read_only(self, r : bool):
        self.get_q_abstract_spin_box().setReadOnly(r)
        return self


class _QAbstractSpinBoxImpl(qt.QAbstractSpinBox): ...