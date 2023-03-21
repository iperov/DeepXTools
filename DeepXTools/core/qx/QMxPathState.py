from __future__ import annotations

from pathlib import Path
from typing import List

from .. import mx, qt
from .QBox import QHBox
from .QFileDialog import QFileDialog
from .QFontDB import Font
from .QIonIconDB import IonIcon, QIonIconDB
from .QLineEdit import QLineEdit
from .QPushButton import QPushButton
from .StyleColor import StyleColor


class QMxPathState(QHBox):
    def __init__(self, path_state : mx.IPathState):
        """ViewController widget composition of model mx.PathState"""
        super().__init__()
        self._path_state = path_state

        config = path_state.config

        file_dlg = self._file_dlg = QFileDialog().set_parent(self)
        file_dlg.mx_accepted.listen(self._on_file_dlg_mx_accepted).dispose_with(path_state)

        line_edit = self._line_edit = QLineEdit()
        line_edit.set_font(Font.FixedWidth).set_read_only(True)
        if (desc := config.desc) is not None:
            line_edit.set_placeholder_text(desc)

        reveal_btn = self._reveal_btn = (QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.eye_outline, StyleColor.ButtonText))
                            .set_tooltip('@(Reveal_in_explorer)')
                            .inline(lambda btn: btn.mx_clicked.listen(lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path if path.is_dir() else path.parent))) if (path := path_state.mx_path.get()) is not None else ...) )
                            .h_compact() )

        (self
            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.close_outline, StyleColor.ButtonText) )
                                .set_tooltip('@(Close)')
                                .inline( lambda btn: btn.mx_clicked.listen(lambda: path_state.close()).dispose_with(path_state) )
                                .h_compact() )

            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.open_outline, StyleColor.ButtonText))
                                .set_tooltip('@(Open)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._show_dlg()).dispose_with(path_state) )
                                .h_compact() if config.allow_open else None )

            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.add_circle_outline, StyleColor.ButtonText))
                                .set_tooltip('@(New)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._show_dlg(is_new=True)).dispose_with(path_state) )
                                .h_compact() if config.allow_new else None )

            .add(reveal_btn.hide())
            .add(self._line_edit.h_expand())
        )

        path_state.mx_path.reflect(self._on_path_dlg_mx_path).dispose_with(self)

    def set_directory(self, dir : Path|None):
        self._file_dlg.set_directory(dir)
        return self

    def _on_path_dlg_mx_path(self, path : Path | None):
        if path is not None:
            self._file_dlg.set_directory( path if path.is_dir() else path.parent )
            self._line_edit.set_text(str(path))
            self._reveal_btn.show()
        else:
            self._line_edit.set_text(None )
            self._reveal_btn.hide()

    def _show_dlg(self, is_new = False):
        config = self._path_state.config
        file_dlg = self._file_dlg

        if config.dir_only:
            file_dlg.set_file_mode( QFileDialog.FileMode.Directory)
            file_dlg.set_option( QFileDialog.Option.ShowDirsOnly, True)
        else:
            if is_new:
                file_dlg.set_file_mode( QFileDialog.FileMode.AnyFile)
            else:
                file_dlg.set_file_mode( QFileDialog.FileMode.ExistingFile)

            if (extensions := config.extensions) is not None:

                if (desc := config.desc) is None:
                    desc = 'Accepted files'

                file_dlg.set_filter(f"{desc} ({' '.join([f'*.{ext}' for ext in extensions])})")

        if is_new and not config.dir_only:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptSave)
        else:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptOpen)

        self._is_new = is_new
        file_dlg.show()

    def _on_file_dlg_mx_accepted(self, files : List[Path]):
        if self._is_new:
            self._path_state.new(files[0])
        else:
            self._path_state.open(files[0])
