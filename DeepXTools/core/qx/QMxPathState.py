from __future__ import annotations

from enum import Enum, auto
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
        
        self._dlg_mode = None
        
        config = path_state.config

        file_dlg = self._file_dlg = QFileDialog().set_parent(self)
        file_dlg.mx_accepted.listen(self._on_file_dlg_mx_accepted).dispose_with(path_state)

        line_edit = self._line_edit = QLineEdit()
        line_edit.set_font(Font.FixedWidth).set_read_only(True)
        if (desc := config.desc) is not None:
            line_edit.set_placeholder_text(desc)
        

        (self
            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.close_outline, StyleColor.ButtonText) )
                                .set_tooltip('@(Close)')
                                .inline( lambda btn: btn.mx_clicked.listen(lambda: path_state.close()).dispose_with(path_state) )
                                .h_compact() )

            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.open_outline, StyleColor.ButtonText))
                                .set_tooltip('@(Open)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._show_dlg(QMxPathState._DlgMode.Open)).dispose_with(path_state) )
                                .h_compact() if config.allow_open else None )

            .add(QPushButton()  .set_icon( QIonIconDB.instance().icon(IonIcon.add_circle_outline, StyleColor.ButtonText))
                                .set_tooltip('@(New)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._show_dlg(QMxPathState._DlgMode.New)).dispose_with(path_state) )
                                .h_compact() if config.allow_new else None )
            
            .add( rename_btn := (QPushButton().hide()
                                .set_icon( QIonIconDB.instance().icon(IonIcon.pencil_outline, StyleColor.ButtonText))
                                .set_tooltip('@(Change)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: self._show_dlg(QMxPathState._DlgMode.Rename)).dispose_with(path_state) )
                                .h_compact()) if config.allow_rename else None )
                                
            .add( reveal_btn := (QPushButton().hide()
                                .set_icon( QIonIconDB.instance().icon(IonIcon.eye_outline, StyleColor.ButtonText))
                                .set_tooltip('@(Reveal_in_explorer)')
                                .inline(lambda btn: btn.mx_clicked.listen(lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path if path.is_dir() else path.parent))) if (path := path_state.mx_path.get()) is not None else ...) )
                                .h_compact()))           
            
            .add(self._line_edit.h_expand()))
        
        self._rename_btn = rename_btn
        self._reveal_btn = reveal_btn
        
        path_state.mx_path.reflect(self._on_path_dlg_mx_path).dispose_with(self)

    def set_directory(self, dir : Path|None):
        self._file_dlg.set_directory(dir)
        return self

    def _on_path_dlg_mx_path(self, path : Path | None):
        if path is not None:
            self._file_dlg.set_directory( path if path.is_dir() else path.parent )
            self._line_edit.set_text(str(path))
            if self._rename_btn is not None:
                self._rename_btn.show()
            self._reveal_btn.show()
        else:
            self._line_edit.set_text(None )
            if self._rename_btn is not None:
                self._rename_btn.hide()
            self._reveal_btn.hide()

    def _show_dlg(self, mode : QMxPathState._DlgMode):
        config = self._path_state.config
        file_dlg = self._file_dlg

        if config.dir_only:
            file_dlg.set_file_mode( QFileDialog.FileMode.Directory)
            file_dlg.set_option( QFileDialog.Option.ShowDirsOnly, True)
        else:
            if mode in [QMxPathState._DlgMode.New, QMxPathState._DlgMode.Rename]:
                file_dlg.set_file_mode( QFileDialog.FileMode.AnyFile)
            else:
                file_dlg.set_file_mode( QFileDialog.FileMode.ExistingFile)

            if (extensions := config.extensions) is not None:

                if (desc := config.desc) is None:
                    desc = 'Accepted files'

                file_dlg.set_filter(f"{desc} ({' '.join([f'*{ext}' for ext in extensions])})")

        if mode in [QMxPathState._DlgMode.New, QMxPathState._DlgMode.Rename] and not config.dir_only:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptSave)
        else:
            file_dlg.set_accept_mode(QFileDialog.AcceptMode.AcceptOpen)

        self._dlg_mode = mode
        file_dlg.show()

    def _on_file_dlg_mx_accepted(self, files : List[Path]):
        path = files[0]
        
        if self._dlg_mode == QMxPathState._DlgMode.Open:
            self._path_state.open(path)
        elif self._dlg_mode == QMxPathState._DlgMode.New:
            self._path_state.new(path)
        elif self._dlg_mode == QMxPathState._DlgMode.Rename:
            self._path_state.rename(path)
            
    class _DlgMode(Enum):
        Open = auto()
        New = auto()
        Rename = auto()