from typing import TypeVar

from .. import lx, mx, qt
from ._constants import LayoutDirection
from ._helpers import q_init
from .QApplication import QApplication
from .QFontDB import Font, QFontDB
from .QFuncWrap import QFuncWrap
from .QObject import QObject


class QWidget(QObject):
    _Self = TypeVar('_Self', bound='QWidget')

    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_widget', qt.QWidget, **kwargs), **kwargs)

        self._compact_width = None
        self._compact_height = None

        self._mx_show = mx.Event1[qt.QShowEvent]().dispose_with(self)
        self._mx_hide = mx.Event1[qt.QHideEvent]().dispose_with(self)
        self._mx_close = mx.Event1[qt.QCloseEvent]().dispose_with(self)
        self._mx_resize = mx.Event1[qt.QResizeEvent]().dispose_with(self)
        self._mx_focus_in = mx.Event1[qt.QFocusEvent]().dispose_with(self)
        self._mx_focus_out = mx.Event1[qt.QFocusEvent]().dispose_with(self)
        self._mx_enter = mx.Event1[qt.QEnterEvent]().dispose_with(self)
        self._mx_leave = mx.Event1[qt.QEvent]().dispose_with(self)
        self._mx_move = mx.Event1[qt.QMoveEvent]().dispose_with(self)
        self._mx_key_press = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self._mx_key_release = mx.Event1[qt.QKeyEvent]().dispose_with(self)
        self._mx_mouse_move = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_mouse_press = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_mouse_release = mx.Event1[qt.QMouseEvent]().dispose_with(self)
        self._mx_wheel = mx.Event1[qt.QWheelEvent]().dispose_with(self)
        self._mx_paint = mx.Event1[qt.QPaintEvent]().dispose_with(self)

        self._minimumSizeHint_wrap = QFuncWrap(q_widget := self.get_q_widget(), 'minimumSizeHint', lambda *args, **kwargs: self._minimum_size_hint(*args, **kwargs)).dispose_with(self)
        self._sizeHint_wrap = QFuncWrap(q_widget, 'sizeHint', lambda *args, **kwargs: self._size_hint(*args, **kwargs)).dispose_with(self)
        self._showEvent_wrap = QFuncWrap(q_widget, 'showEvent', lambda *args, **kwargs: self._show_event(*args, **kwargs)).dispose_with(self)
        self._hideEvent_wrap = QFuncWrap(q_widget, 'hideEvent', lambda *args, **kwargs: self._hide_event(*args, **kwargs)).dispose_with(self)
        self._closeEvent_wrap = QFuncWrap(q_widget, 'closeEvent', lambda *args, **kwargs: self._close_event(*args, **kwargs)).dispose_with(self)
        self._resizeEvent_wrap = QFuncWrap(q_widget, 'resizeEvent', lambda *args, **kwargs: self._resize_event(*args, **kwargs)).dispose_with(self)
        self._focusInEvent_wrap = QFuncWrap(q_widget, 'focusInEvent', lambda *args, **kwargs: self._focus_in_event(*args, **kwargs)).dispose_with(self)
        self._focusOutEvent_wrap = QFuncWrap(q_widget, 'focusOutEvent', lambda *args, **kwargs: self._focus_out_event(*args, **kwargs)).dispose_with(self)
        self._enterEvent_wrap = QFuncWrap(q_widget, 'enterEvent', lambda *args, **kwargs: self._enter_event(*args, **kwargs)).dispose_with(self)
        self._leaveEvent_wrap = QFuncWrap(q_widget, 'leaveEvent', lambda *args, **kwargs: self._leave_event(*args, **kwargs)).dispose_with(self)
        self._moveEvent_wrap = QFuncWrap(q_widget, 'moveEvent', lambda *args, **kwargs: self._move_event(*args, **kwargs)).dispose_with(self)
        self._keyPressEvent_wrap = QFuncWrap(q_widget, 'keyPressEvent', lambda *args, **kwargs: self._key_press_event(*args, **kwargs)).dispose_with(self)
        self._keyReleaseEvent_wrap = QFuncWrap(q_widget, 'keyReleaseEvent', lambda *args, **kwargs: self._key_release_event(*args, **kwargs)).dispose_with(self)
        self._mouseMoveEvent_wrap = QFuncWrap(q_widget, 'mouseMoveEvent', lambda *args, **kwargs: self._mouse_move_event(*args, **kwargs)).dispose_with(self)
        self._mousePressEvent_wrap = QFuncWrap(q_widget, 'mousePressEvent', lambda *args, **kwargs: self._mouse_press_event(*args, **kwargs)).dispose_with(self)
        self._mouseReleaseEvent_wrap = QFuncWrap(q_widget, 'mouseReleaseEvent', lambda *args, **kwargs: self._mouse_release_event(*args, **kwargs)).dispose_with(self)
        self._wheelEvent_wrap = QFuncWrap(q_widget, 'wheelEvent', lambda *args, **kwargs: self._wheel_event(*args, **kwargs)).dispose_with(self)
        self._paintEvent_wrap = QFuncWrap(q_widget, 'paintEvent', lambda *args, **kwargs: self._paint_event(*args, **kwargs)).dispose_with(self)
        
        if not self._is_wrap_mode():
            self.h_normal()
            self.v_normal()

    @property
    def mx_show(self) -> mx.IEvent1_r[qt.QShowEvent]: return self._mx_show
    @property
    def mx_hide(self) -> mx.IEvent1_r[qt.QHideEvent]: return self._mx_hide
    @property
    def mx_close(self) -> mx.IEvent1_r[qt.QCloseEvent]: return self._mx_close
    @property
    def mx_resize(self) -> mx.IEvent1_r[qt.QResizeEvent]: return self._mx_resize
    @property
    def mx_focus_in(self) -> mx.IEvent1_r[qt.QFocusEvent]: return self._mx_focus_in
    @property
    def mx_focus_out(self) -> mx.IEvent1_r[qt.QFocusEvent]: return self._mx_focus_out
    @property
    def mx_enter(self) -> mx.IEvent1_r[qt.QEnterEvent]: return self._mx_enter
    @property
    def mx_leave(self) -> mx.IEvent1_r[qt.QEvent]: return self._mx_leave
    @property
    def mx_move(self) -> mx.IEvent1_r[qt.QMoveEvent]: return self._mx_move
    @property
    def mx_key_press(self) -> mx.IEvent1_r[qt.QKeyEvent]: return self._mx_key_press
    @property
    def mx_key_release(self) -> mx.IEvent1_r[qt.QKeyEvent]: return self._mx_key_release
    @property
    def mx_mouse_move(self) -> mx.IEvent1_r[qt.QMouseEvent]: return self._mx_mouse_move
    @property
    def mx_mouse_press(self) -> mx.IEvent1_r[qt.QMouseEvent]: return self._mx_mouse_press
    @property
    def mx_mouse_release(self) -> mx.IEvent1_r[qt.QMouseEvent]: return self._mx_mouse_release
    @property
    def mx_wheel(self) -> mx.IEvent1_r[qt.QWheelEvent]: return self._mx_wheel
    @property
    def mx_paint(self) -> mx.IEvent1_r[qt.QPaintEvent]: return self._mx_paint

    def __dispose__(self):
        if not self._is_wrap_mode():
            self.get_q_widget().hide()
        super().__dispose__()

    def get_q_widget(self) -> qt.QWidget: return self.get_q_object()
    def get_font(self) -> qt.QFont: return self.get_q_widget().font()
    def get_style(self) -> qt.QStyle: return self.get_q_widget().style()
    def get_palette(self) -> qt.QPalette: return self.get_q_widget().palette()
    def is_visible(self) -> bool: return self.get_q_widget().isVisible()
    def rect(self) -> qt.QRect: return self.get_q_widget().rect()
    def size(self) -> qt.QSize: return self.get_q_widget().size()
    def map_to_global(self, point : qt.QPoint) -> qt.QPoint: return self.get_q_widget().mapToGlobal(point)

    def h_compact(self, width : int = None):
        """Fix widget's width at minimum. Specify `width` to override widget's default minimum."""
        self._compact_width = width
        self.get_q_widget().setSizePolicy(qt.QSizePolicy.Policy.Fixed, self.get_q_widget().sizePolicy().verticalPolicy())
        return self

    def v_compact(self, height : int = None):
        """Fix widget's height at minimum. Specify `height` to override widget's default minimum."""
        self._compact_height = height
        self.get_q_widget().setSizePolicy(self.get_q_widget().sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Fixed)
        return self

    def h_normal(self):
        """set widget's regular width"""
        self._compact_width = None
        self.get_q_widget().setSizePolicy(qt.QSizePolicy.Policy.Preferred, self.get_q_widget().sizePolicy().verticalPolicy())
        return self

    def v_normal(self):
        """set widget's regular height"""
        self._compact_height = None
        self.get_q_widget().setSizePolicy(self.get_q_widget().sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Preferred)
        return self

    def h_expand(self):
        """set widget's maximum width"""
        self._compact_width = None
        self.get_q_widget().setSizePolicy(qt.QSizePolicy.Policy.Expanding, self.get_q_widget().sizePolicy().verticalPolicy())
        return self

    def v_expand(self):
        """set widget's maximum height"""
        self._compact_height = None
        self.get_q_widget().setSizePolicy(self.get_q_widget().sizePolicy().horizontalPolicy(), qt.QSizePolicy.Policy.Expanding)
        return self

    def show(self):
        self.set_visible(True)
        return self

    def hide(self):
        self.set_visible(False)
        return self

    def enable(self):
        self.set_enabled(True)
        return self

    def disable(self):
        self.set_enabled(False)
        return self

    def clear_focus(self):
        self.get_q_widget().clearFocus()
        return self

    def set_focus(self):
        self.get_q_widget().setFocus()
        return self

    def set_font(self, font : qt.QFont | Font):
        if isinstance(font, Font):
            font = QFontDB.instance().get(font)
        self.get_q_widget().setFont(font)
        return self

    def set_cursor(self, cursor : qt.QCursor|qt.Qt.CursorShape|qt.QPixmap):
        self.get_q_widget().setCursor(cursor)
        return self

    def unset_cursor(self):
        self.get_q_widget().unsetCursor()
        return self

    def set_layout_direction(self, layout_direction : LayoutDirection):
        self.get_q_widget().setLayoutDirection(layout_direction)
        return self

    def set_mouse_tracking(self, b : bool):
        self.get_q_widget().setMouseTracking(b)
        return self

    def set_visible(self, visible : bool):
        self.get_q_widget().setVisible(visible)
        return self

    def set_enabled(self, enabled : bool):
        self.get_q_widget().setEnabled(enabled)
        return self

    def set_tooltip(self, tooltip : str):
        if (disp := getattr(self, '_QWidget_tooltip_disp', None)) is not None:
            disp.dispose()
        self._QWidget_tooltip_disp = QApplication.instance().mx_language.reflect(lambda lang: self.get_q_widget().setToolTip(lx.L(tooltip, lang))).dispose_with(self)
        return self

    def repaint(self):
        self.get_q_widget().repaint()
        return self

    def update(self):
        self.get_q_widget().update()
        return self

    def update_geometry(self):
        self.get_q_widget().updateGeometry()
        return self

    def _minimum_size_hint(self) -> qt.QSize:
        """inheritable/overridable"""
        return self._minimumSizeHint_wrap.get_super()()

    def _size_hint(self) -> qt.QSize:
        """inheritable/overridable"""
        size = self._sizeHint_wrap.get_super()()
        if (compact_width := self._compact_width) is not None:
            size = qt.QSize(compact_width, size.height())
        if (compact_height := self._compact_height) is not None:
            size = qt.QSize(size.width(), compact_height)
        return size

    def _show_event(self, ev : qt.QShowEvent):
        """inheritable"""
        self._showEvent_wrap.get_super()(ev)
        self._mx_show.emit(ev)

    def _hide_event(self, ev : qt.QHideEvent):
        """inheritable"""
        self._hideEvent_wrap.get_super()(ev)
        self._mx_hide.emit(ev)

    def _close_event(self, ev : qt.QHideEvent):
        """inheritable"""
        self._closeEvent_wrap.get_super()(ev)
        self._mx_close.emit(ev)

    def _resize_event(self, ev : qt.QResizeEvent):
        """inheritable"""
        self._resizeEvent_wrap.get_super()(ev)
        self._mx_resize.emit(ev)

    def _focus_in_event(self, ev : qt.QFocusEvent):
        """inheritable"""
        self._focusInEvent_wrap.get_super()(ev)

    def _focus_out_event(self, ev : qt.QFocusEvent):
        """inheritable"""
        self._focusOutEvent_wrap.get_super()(ev)

    def _enter_event(self, ev : qt.QEnterEvent):
        """inheritable"""
        self._enterEvent_wrap.get_super()(ev)
        self._mx_enter.emit(ev)

    def _leave_event(self, ev : qt.QEvent):
        """inheritable"""
        self._leaveEvent_wrap.get_super()(ev)
        self._mx_leave.emit(ev)

    def _move_event(self, ev : qt.QMoveEvent):
        """inheritable"""
        self._moveEvent_wrap.get_super()(ev)
        self._mx_move.emit(ev)

    def _key_press_event(self, ev : qt.QKeyEvent):
        """inheritable"""
        #if ev.key() != qt.Qt.Key.Key_Tab:
        self._keyPressEvent_wrap.get_super()(ev)
        self._mx_key_press.emit(ev)

    def _key_release_event(self, ev : qt.QKeyEvent):
        """inheritable"""
        self._keyReleaseEvent_wrap.get_super()(ev)
        self._mx_key_release.emit(ev)

    def _mouse_move_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mouseMoveEvent_wrap.get_super()(ev)
        self._mx_mouse_move.emit(ev)

    def _mouse_press_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mousePressEvent_wrap.get_super()(ev)
        self._mx_mouse_press.emit(ev)

    def _mouse_release_event(self, ev : qt.QMouseEvent):
        """inheritable"""
        self._mouseReleaseEvent_wrap.get_super()(ev)
        self._mx_mouse_release.emit(ev)

    def _wheel_event(self, ev : qt.QWheelEvent):
        """inheritable"""
        self._wheelEvent_wrap.get_super()(ev)
        self._mx_wheel.emit(ev)

    def _paint_event(self, ev : qt.QPaintEvent):
        """inheritable"""
        self._paintEvent_wrap.get_super()(ev)
        self._mx_paint.emit(ev)
