from typing import Callable

from .. import mx, qt
from ._helpers import q_init
from .QEvent import QEvent0, QEvent1
from .QWidget import QWidget


class QLineEdit(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_line_edit', qt.QLineEdit, **kwargs), **kwargs)
        
        q_line_edit = self.get_q_line_edit()
        
        mx_editing_finished = self._mx_editing_finished = QEvent0(q_line_edit.editingFinished).dispose_with(self)
        mx_editing_finished.listen(lambda: self.clear_focus())

        self._mx_text = mx.GetSetProperty[str|None](self.get_text, self.set_text, QEvent1[str](q_line_edit.textChanged).dispose_with(self) ).dispose_with(self)

        self.set_placeholder_text(None)

    @property
    def mx_text(self) -> mx.IProperty[str|None]: return self._mx_text
    @property
    def mx_editing_finished(self) -> mx.IEvent0_r: return self._mx_editing_finished

    def get_q_line_edit(self) -> qt.QLineEdit: return self.get_q_widget()

    def get_text(self) -> str|None:
        if len(text := self.get_q_line_edit().text()) == 0:
            text = None
        return text

    def set_text(self, text : str|None):
        self.get_q_line_edit().setText(text)
        return self

    def set_placeholder_text(self, text : str|None):
        self.get_q_line_edit().setPlaceholderText(text if text is not None else '...')
        return self

    def set_read_only(self, read_only : bool):
        self.get_q_line_edit().setReadOnly(read_only)
        return self

    def set_filter(self, func : Callable[ [str], str ] ):
        """Filter string using callable func"""
        self.get_q_line_edit().setValidator(QFuncValidator(func))
        return self

class QFuncValidator(qt.QValidator):
    def __init__(self, func : Callable[ [str], str ]):
        super().__init__()
        self._func = func

    def fixup(self, s: str) -> str:
        return self._func(s)

    def validate(self, s: str, pos: int) -> object:
        if self._func(s) == s:
            return qt.QValidator.State.Acceptable
        return qt.QValidator.State.Invalid

