from typing import Tuple

from .. import lx, mx, qt
from ._constants import Size, icon_Size_to_int
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent0, QEvent1
from .QWidget import QWidget


class QAbstractButton(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_abstract_button', _QAbstractButtonImpl, qt.QAbstractButton, **kwargs))

        q_abstract_button = self.get_q_abstract_button()
        self._mx_clicked = QEvent0(q_abstract_button.clicked).dispose_with(self)
        self._mx_pressed = QEvent0(q_abstract_button.pressed).dispose_with(self)
        self._mx_released = QEvent0(q_abstract_button.released).dispose_with(self)
        self._mx_toggled = QEvent1[bool](q_abstract_button.toggled).dispose_with(self)

        if isinstance(q_abstract_button, _QAbstractButtonImpl):
            ...

        self.set_icon_size(Size.Default)

    @property
    def mx_clicked(self) -> mx.IEvent0_r: return self._mx_clicked
    @property
    def mx_pressed(self) -> mx.IEvent0_r: return self._mx_pressed
    @property
    def mx_released(self) -> mx.IEvent0_r: return self._mx_released
    @property
    def mx_toggled(self) -> mx.IEvent1_r[bool]: return self._mx_toggled

    def get_q_abstract_button(self) -> qt.QPushButton: return self.get_q_widget()
    def get_text(self) -> str: return self.get_q_abstract_button().text()
    def is_down(self) -> bool: return self.get_q_abstract_button().isDown()
    def is_checked(self) -> bool: return self.get_q_abstract_button().isChecked()

    def click(self):
        self.get_q_abstract_button().click()
        return self

    def toggle(self):
        self.get_q_abstract_button().toggle()
        return self

    def set_checkable(self, checkable : bool):
        self.get_q_abstract_button().setCheckable(checkable)
        return self

    def set_checked(self, checked : bool):
        self.get_q_abstract_button().setChecked(checked)
        return self

    def set_icon(self, icon : qt.QIcon):
        self.get_q_abstract_button().setIcon(icon)
        return self

    def set_icon_size(self, size : Tuple[int, int] | Size):
        if isinstance(size, Size):
            size = (icon_Size_to_int[size],)*2
        self.get_q_abstract_button().setIconSize(qt.QSize(*size))
        return self

    def set_text(self, text : str|None):
        if (disp := getattr(self, '_QAbstractButton_text_disp', None)) is not None:
            disp.dispose()
        self._QAbstractButton_text_disp = QApplication.instance().mx_language.reflect(lambda lang: self.get_q_abstract_button().setText(lx.L(text, lang))).dispose_with(self)
        return self


class _QAbstractButtonImpl(qt.QAbstractButton): ...








