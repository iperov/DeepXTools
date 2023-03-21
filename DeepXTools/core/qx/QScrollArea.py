from .. import qt
from ._helpers import q_init
from .QTimer import QTimer
from .QWidget import QWidget


class QVScrollArea(QWidget):
    def __init__(self, min_width_from_widget=True, **kwargs):
        super().__init__(q_widget=q_init('q_scroll_area', qt.QScrollArea, **kwargs),  **kwargs)

        q_scroll_area = self.get_q_scroll_area()

        self._min_width_from_widget = min_width_from_widget
        self._widget_min_size = qt.QSize(0,0)

        q_scroll_area.setWidgetResizable(True)
        q_scroll_area.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        q_scroll_area.setVerticalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        QTimer(on_timeout=self._on_timer).set_interval(200).start().dispose_with(self)

    def get_q_scroll_area(self) -> qt.QScrollArea: return self.get_q_widget()

    def set_widget(self, widget : QWidget):
        widget.set_parent(self)
        self.get_q_scroll_area().setWidget(widget.get_q_widget())
        return self

    def _on_timer(self):
        if (widget := self.get_q_scroll_area().widget()) is not None:
            if self._widget_min_size != widget.minimumSizeHint():
                self.update_geometry()

    def _minimum_size_hint(self) -> qt.QSize:
        min_size = super()._minimum_size_hint()

        if self._min_width_from_widget:
            if (widget := self.get_q_scroll_area().widget()) is not None:

                widget_min_size = self._widget_min_size = widget.minimumSizeHint()
                min_size.setWidth(widget_min_size.width() + self.get_style().pixelMetric(qt.QStyle.PixelMetric.PM_ScrollBarExtent)+2 )

        return min_size




