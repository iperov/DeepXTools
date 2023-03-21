from __future__ import annotations

from .. import lx, mx, qt
from ._helpers import q_init
from .QAction import QAction
from .QApplication import QApplication
from .QWidget import QWidget


class QMenu(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_menu', qt.QMenu, **kwargs), **kwargs)
        
        q_menu = self.get_q_menu()
        q_menu.aboutToShow.connect(lambda: self._mx_about_to_show.emit())

        self._mx_about_to_show = mx.Event0().dispose_with(self)

    @property
    def mx_about_to_show(self) -> mx.IEvent0_r: return self._mx_about_to_show

    def get_q_menu(self) -> qt.QMenu: return self.get_q_widget()

    def dispose_actions(self):
        for child in self.get_childs():
            if isinstance(child, QAction):
                child.dispose()
        return self

    def add_separator(self):
        action = QAction(q_action=self.get_q_menu().addSeparator()).set_parent(self)
        self.get_q_menu().addAction(action.get_q_action())
        return self

    def add(self, item : QMenu|QAction|None):
        if item is not None:
            if isinstance(item, QMenu):
                item.set_parent(self)
                self.get_q_menu().addMenu(item.get_q_menu())
            elif isinstance(item, QAction):
                item.set_parent(self)
                self.get_q_menu().addAction(item.get_q_action())
        return self

    def set_title(self, text : str|None):
        if (disp := getattr(self, '_QAction_title_disp', None)) is not None:
            disp.dispose()
        self._QAction_title_disp = QApplication.instance().mx_language.reflect(lambda lang: self.get_q_menu().setTitle(lx.L(text, lang))).dispose_with(self)
        return self

    def show(self):
        self.get_q_menu().popup(qt.QCursor.pos())
        return self



