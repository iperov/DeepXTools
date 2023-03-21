from typing import List

from .. import qt
from ._constants import Orientation
from ._helpers import q_init
from .QEvent import QEvent2
from .QSettings import QSettings
from .QWidget import QWidget


class QSplitter(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_splitter', qt.QSplitter, **kwargs), **kwargs)
        
        self.__settings = QSettings()
        self.__default_sizes = []
        
        q_splitter = self.get_q_splitter()

        QEvent2[int, int](q_splitter.splitterMoved).dispose_with(self).listen(self._on_splitter_moved)

    def get_q_splitter(self) -> qt.QSplitter: return self.get_q_widget()

    def add(self, widget : QWidget|None):
        if widget is not None:
            widget.set_parent(self)
            self.get_q_splitter().addWidget(widget.get_q_widget())
        return self

    def set_default_sizes(self, sizes : List[int]):
        self.__default_sizes = sizes
        return self

    def set_orientation(self, orientation : Orientation):
        self.get_q_splitter().setOrientation(orientation)
        return self

    def _settings_event(self, settings : QSettings):
        super()._settings_event(settings)
        self.__settings = settings

    def _show_event(self, ev: qt.QShowEvent):
        super()._show_event(ev)

        q_splitter = self.get_q_splitter()

        sizes = self.__settings.get('sizes', self.__default_sizes)[:len(q_splitter.sizes())]
        if len(sizes) != 0:
            q_splitter.setSizes(sizes)

    def _on_splitter_moved(self, pos, index):
        if self.is_visible():
            self.__settings['sizes'] = self.get_q_splitter().sizes()

