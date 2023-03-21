from __future__ import annotations

from pathlib import Path

from .. import qt
from .QBox import QHBox
from .QFontDB import Font
from .QIonIconDB import IonIcon, QIonIconDB
from .QLineEdit import QLineEdit
from .QPushButton import QPushButton
from .StyleColor import StyleColor


class QPathLabel(QHBox):
    def __init__(self, path : Path):
        """Read-only path with reveal button"""
        super().__init__()
        self._path = path

        line_edit = self._line_edit = QLineEdit()
        line_edit.set_font(Font.FixedWidth).set_read_only(True)
        line_edit.set_text(str(path))
        reveal_btn = self._reveal_btn = (QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.eye_outline, StyleColor.ButtonText))
                            .set_tooltip('@(Reveal_in_explorer)')
                            .inline(lambda btn: btn.mx_clicked.listen(lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))) )
                            .h_compact() )

        (self
            .add(reveal_btn.h_compact())
            .add(self._line_edit)
        )

