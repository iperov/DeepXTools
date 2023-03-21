from __future__ import annotations

from pathlib import Path
from typing import List

from .. import mx, qt
from ._helpers import q_init
from .QEvent import QEvent0
from .QWidget import QWidget


class QFileDialog(QWidget):
    Option = qt.QFileDialog.Option
    AcceptMode = qt.QFileDialog.AcceptMode
    FileMode = qt.QFileDialog.FileMode

    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_file_dialog', qt.QFileDialog, **kwargs), **kwargs)

        q_file_dialog = self.get_q_file_dialog()
        self._mx_accepted = mx.Event1[List[Path]]().dispose_with(self)

        QEvent0(q_file_dialog.accepted).dispose_with(self).listen( lambda: self._mx_accepted.emit( [Path(x) for x in q_file_dialog.selectedFiles()]) )

    @property
    def mx_accepted(self) -> mx.IEvent1_r[List[Path]]: return self._mx_accepted

    def get_q_file_dialog(self) -> qt.QFileDialog: return self.get_q_widget()

    def set_directory(self, dir : Path|None):
        if dir is not None:
            self.get_q_file_dialog().setDirectory(str(dir))
        return self

    def set_filter(self, filter : str):
        self.get_q_file_dialog().setNameFilter(filter)
        return self

    def set_option(self, option : Option, on : bool = None):
        self.get_q_file_dialog().setOption(option, on)
        return self

    def set_accept_mode(self, accept_mode : AcceptMode):
        self.get_q_file_dialog().setAcceptMode(accept_mode)
        return self

    def set_file_mode(self, file_mode : FileMode):
        self.get_q_file_dialog().setFileMode(file_mode)
        return self

    def reject(self):
        self.get_q_file_dialog().reject()
        return self

    def show(self):
        self.get_q_file_dialog().open()
        return self
