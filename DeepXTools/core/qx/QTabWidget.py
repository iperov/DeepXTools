from __future__ import annotations

from typing import Callable

from .. import lx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QBox import QVBox
from .QEvent import QEvent1
from .QSettings import QSettings
from .QWidget import QWidget


class QTab(QVBox):
    def __init__(self, owner : QTabWidget, **kwargs):
        super().__init__(**kwargs)
        self._owner = owner

    def set_title(self, title : str):
        if (disp := getattr(self, '_QTab_title_disp', None)) is not None:
            disp.dispose()
        self._QTab_title_disp = QApplication.instance().mx_language.reflect(lambda lang:
                (q_tab_widget := self._owner.get_q_tab_widget()).setTabText(q_tab_widget.indexOf(self.get_q_widget()), lx.L(title, lang))
            ).dispose_with(self)

        return self

class QTabWidget(QWidget):
    TabPosition = qt.QTabWidget.TabPosition

    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_tab_widget', qt.QTabWidget, **kwargs), **kwargs)
        
        q_tab_widget = self.get_q_tab_widget()
        QEvent1[int](q_tab_widget.currentChanged).dispose_with(self).listen(self._on_current_changed)

        
        self.__settings = QSettings()

    def __dispose__(self):
        self.__settings = {}
        super().__dispose__()

    def get_q_tab_widget(self) -> qt.QTabWidget: return self.get_q_widget()

    def add_tab(self, inline : Callable[ [QTab], None]):
        tab = QTab(self).set_parent(self)
        self.get_q_tab_widget().addTab(tab.get_q_widget(), '')
        inline(tab)
        return self

    def set_tab_position(self, position : TabPosition):
        self.get_q_tab_widget().setTabPosition(position)
        return self

    def _on_current_changed(self, idx : int):
        self.__settings['current_index'] = idx
        
    def _settings_event(self, settings : QSettings):
        super()._settings_event(settings)
        self.__settings = settings
        
        if (current_index := settings.get('current_index', None)) is not None:
            self.get_q_tab_widget().setCurrentIndex(current_index)
            