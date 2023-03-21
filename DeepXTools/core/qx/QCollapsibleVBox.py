from .. import qt
from .QAbstractButton import QAbstractButton
from .QBox import QVBox
from .QSettings import QSettings
from .StyleColor import StyleColor


class QCollapsibleBarButton(QAbstractButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__hover = False
        self.set_checkable(True).set_checked(True)

    def _minimum_size_hint(self) -> qt.QSize:
        fm = self.get_q_widget().fontMetrics()
        return qt.QSize(super()._minimum_size_hint().width(), fm.size(0, 'XXXX').height() + 8)

    def _enter_event(self, ev: qt.QEnterEvent):
        super()._enter_event(ev)
        self.__hover = True
        self.update()

    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self.__hover = False
        self.update()

    def _paint_event(self, ev: qt.QPaintEvent):
        rect = self.rect()
        icon_size = rect.height()
        q_widget = self.get_q_widget()
        fm = q_widget.fontMetrics()
        font = self.get_font()
        style = self.get_style()
        text_rect = qt.QRect(icon_size, 0, rect.width()-icon_size, rect.height())

        opt = qt.QStyleOption()
        opt.initFrom(q_widget)
        opt.rect = qt.QRect(0,0, icon_size,icon_size)

        qp = qt.QPainter(q_widget)


        qp.fillRect(rect, StyleColor.Midlight if self.__hover else StyleColor.Mid)

        style.drawPrimitive(qt.QStyle.PrimitiveElement.PE_IndicatorArrowDown if self.is_checked() else qt.QStyle.PrimitiveElement.PE_IndicatorArrowRight, opt, qp)

        qp.setFont(font)
        qp.drawText(text_rect, qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignVCenter,
                    fm.elidedText(self.get_text(), qt.Qt.TextElideMode.ElideRight, text_rect.width()))

        qp.end()


class QCollapsibleVBox(QVBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__settings = QSettings()

        content_vbox = self.__content_vbox = QVBox()

        bar_btn = self._bar_btn = QCollapsibleBarButton()
        bar_btn.mx_toggled.listen(lambda checked: (self.__settings.set('opened', checked), content_vbox.set_visible(checked)))

        self.set_spacing(1).add(bar_btn.v_compact()).add(content_vbox)

    @property
    def content_vbox(self) -> QVBox:
        return self.__content_vbox

    def is_opened(self) -> bool: return self._bar_btn.is_checked()

    def open(self):
        if not self._bar_btn.is_checked():
            self._bar_btn.set_checked(True)
        return self
    
    def close(self):
        if self._bar_btn.is_checked():
            self._bar_btn.set_checked(False)
        return self
    
    def toggle(self):
        self.open() if not self.is_opened() else self.close()
        return self
    
    def set_text(self, text : str|None):
        self._bar_btn.set_text(text)
        return self

    def _settings_event(self, settings : QSettings):
        super()._settings_event(settings)
        self.__settings = settings
        
        opened = settings.get('opened', None)
        if opened is not None:
            if opened:
                self.open() 
            else:
                self.close()
