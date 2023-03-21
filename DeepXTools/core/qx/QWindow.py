from .. import lx, mx, qt
from ._constants import WindowType
from .QApplication import QApplication
from .QBox import QVBox
from .QSettings import QSettings


class QWindow(QVBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__settings = QSettings()
        self.__q_window = None

        self.__mx_window_key_press = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self.__mx_window_key_release = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self.__mx_window_leave = mx.Event0().dispose_with(self)

    @property
    def mx_window_key_press(self) -> mx.IEvent1_r[qt.QKeyEvent]: return self.__mx_window_key_press
    @property
    def mx_window_key_release(self) -> mx.IEvent1_r[qt.QKeyEvent]: return self.__mx_window_key_release
    @property
    def mx_window_leave(self) -> mx.IEvent0_r: return self.__mx_window_leave

    def get_window_handle(self) -> qt.QWindow | None: return self.__q_window

    def activate(self):
        self.get_q_widget().activateWindow()
        return self

    def set_title(self, title : str|None = None):
        if (disp := getattr(self, '_QWindow_title_disp', None)) is not None:
            disp.dispose()
        self._QWindow_title_disp = QApplication.instance().mx_language.reflect(lambda lang: self.get_q_widget().setWindowTitle(lx.L(title, lang))).dispose_with(self)
        return self

    def set_window_icon(self, icon : qt.QIcon):
        self.get_q_widget().setWindowIcon(icon)
        return self

    def set_window_size(self, width : int, height : int):
        self.get_q_widget().setFixedWidth(width)
        self.get_q_widget().setFixedHeight(height)
        return self

    def set_window_flags(self, wnd_type : WindowType):
        self.get_q_widget().setWindowFlags(wnd_type)
        return self

    def _event_filter(self, object: qt.QObject, ev: qt.QEvent) -> bool:
        r = super()._event_filter(object, ev)
        ev_type = ev.type()
        if ev_type == ev.Type.KeyPress:
            self.__mx_window_key_press.emit(ev)
        elif ev_type == ev.Type.KeyRelease:
            self.__mx_window_key_release.emit(ev)
        elif ev_type == ev.Type.Leave:
            self.__mx_window_leave.emit()
        return r

    def _settings_event(self, settings : QSettings):
        super()._settings_event(settings)
        self.__settings = settings

    def _show_event(self, ev: qt.QShowEvent):
        if self.get_q_object().parent() is not None:
            raise Exception(f'{self} must have no parent')

        if self.__q_window is None:
            self.__q_window = q_window = self.get_q_widget().windowHandle()
            q_window.installEventFilter(self.get_q_object())

        super()._show_event(ev)
        pos = self.__settings.get('geometry.pos', None)
        size = self.__settings.get('geometry.size', (640,480))
        if size is not None:
            self.get_q_widget().resize( qt.QSize(*size) )

        if pos is not None:
            self.get_q_widget().move( qt.QPoint(*pos) )
        else:
            # Center on screen
            app : qt.QGuiApplication = qt.QApplication.instance()
            screen_size = app.primaryScreen().size()
            widget_width, widget_height = self.get_q_widget().size().width(), self.get_q_widget().size().height()
            self.get_q_widget().move( (screen_size.width() - widget_width) // 2,  (screen_size.height() - widget_height) // 2 )

    def _hide_event(self, ev: qt.QHideEvent):
        super()._hide_event(ev)
        if self.__q_window is not None:
            self.__q_window.removeEventFilter(self.get_q_object())
            self.__q_window = None

    def _move_event(self, ev : qt.QMoveEvent):
        super()._move_event(ev)
        if self.is_visible():
            self.__settings['geometry.pos'] = self.get_q_widget().pos().toTuple()

    def _resize_event(self, ev : qt.QResizeEvent):
        super()._resize_event(ev)
        if self.is_visible():
            self.__settings['geometry.size'] = self.get_q_widget().size().toTuple()

