from typing import Tuple

from .. import qt
from ._constants import Size, icon_Size_to_int
from .QWidget import QWidget


class QIconWidget(QWidget):
    """draws QIcon centered in provided area"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qp = qt.QPainter()
        self._icon : qt.QIcon = None

        icon_size = self.get_style().pixelMetric(qt.QStyle.PixelMetric.PM_ButtonIconSize)
        self._icon_size = qt.QSize(icon_size, icon_size)

    def set_icon(self, icon : qt.QIcon|None):
        self._icon = icon
        self._update()
        self.update()
        return self

    def set_icon_size(self, size : Tuple[int, int] | Size):
        if isinstance(size, Size):
            size = (icon_Size_to_int[size],)*2
        self._icon_size = qt.QSize(*size)
        self._update()
        self.update()
        return self

    def _update(self):
        rect = self.rect()

        if (icon := self._icon) is not None:
            size = icon.actualSize(self._icon_size)
            self._pixmap_rect = qt.QRect_center_in(qt.QRect(0,0, size.width(), size.height()), rect)
        else:
            self._pixmap_rect = rect

    def _minimum_size_hint(self) -> qt.QSize:
        return self._icon_size.grownBy(qt.QMargins(2,2,2,2))

    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._update()

    def _paint_event(self, ev: qt.QPaintEvent):
        if (icon := self._icon) is not None:
            qp = self._qp
            qp.begin(self.get_q_widget())
            pixmap = icon.pixmap(self._icon_size)
            qp.drawPixmap(self._pixmap_rect, pixmap, pixmap.rect())
            qp.end()
