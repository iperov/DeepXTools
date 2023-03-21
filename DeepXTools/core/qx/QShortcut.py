from .. import mx, qt
from .QApplication import QApplication
from .QObject import QObject
from .QWidget import QWidget
from .QWindow import QWindow


class QShortcut(mx.Disposable):
    def __init__(self, keycomb : qt.QKeyCombination, anchor : QWidget):
        """
        Shortcut.

        Will be disposed with anchor.
        """
        super().__init__()
        self._keycomb = keycomb
        self._anchor = anchor
        self._typing_focused = False
        self._pressed = False
        self._window_disp_bag : mx.Disposable = None
        self._app_disp_bag : mx.Disposable = None

        self._mx_press = mx.Event0().dispose_with(self)
        self._mx_release = mx.Event0().dispose_with(self)

        anchor.mx_visible_to_parent.listen(self._owner_visible_to_parent, replay=True).dispose_with(self)
        anchor.mx_invisible_to_parent.listen(self._owner_invisible_to_parent).dispose_with(self)
        anchor.mx_hide.listen(self._on_owner_hide).dispose_with(self)

        self.dispose_with(anchor)

    def __dispose__(self):
        self.release()
        super().__dispose__()

    @property
    def mx_press(self) -> mx.IEvent0_r: return self._mx_press
    @property
    def mx_release(self) -> mx.IEvent0_r: return self._mx_release

    def press(self):
        self.release()
        if not self._pressed:
            self._pressed = True
            self._mx_press.emit()

    def release(self):
        if self._pressed:
            self._pressed = False
            self._mx_release.emit()

    def _on_app_mx_focus_widget(self, widget : qt.QWidget):
        self._typing_focused = False

        if isinstance(widget, (qt.QLineEdit, qt.QTextEdit)):
            if not widget.isReadOnly():
                # Disable while focused on typing widgets
                self._typing_focused = True

        if self._typing_focused:
            self.release()

    def _owner_visible_to_parent(self, parent : QObject):
        if isinstance(parent, QWindow):
            if self._window_disp_bag is not None:
                self._window_disp_bag = self._window_disp_bag.dispose()
            window_disp_bag = self._window_disp_bag = mx.Disposable().dispose_with(self)
            parent.mx_window_key_press.listen(self._on_window_key_press).dispose_with(window_disp_bag)
            parent.mx_window_key_release.listen(self._on_window_key_release).dispose_with(window_disp_bag)
            parent.mx_window_leave.listen(self._on_window_leave).dispose_with(window_disp_bag)

        elif isinstance(parent, QApplication):
            app_disp_bag = self._app_disp_bag = mx.Disposable().dispose_with(self)
            parent.mx_focus_widget.reflect(self._on_app_mx_focus_widget).dispose_with(app_disp_bag)

    def _owner_invisible_to_parent(self, parent : QObject):
        if isinstance(parent, QWindow):
            if self._window_disp_bag is not None:
                self._window_disp_bag = self._window_disp_bag.dispose()

        elif isinstance(parent, QApplication):
            if self._app_disp_bag is not None:
                self._app_disp_bag = self._app_disp_bag.dispose()

    def _on_window_key_press(self, ev : qt.QKeyEvent):
        if self._anchor.is_visible() and not self._typing_focused:
            if not ev.isAutoRepeat():
                if ev.key() in [qt.Qt.Key.Key_Control, qt.Qt.Key.Key_Shift, qt.Qt.Key.Key_Alt]:
                    keycomb = ev.keyCombination()
                else:
                    # Using native virtual key in order to ignore keyboard language
                    keycomb = qt.QKeyCombination(ev.modifiers(), qt.Qt.Key(ev.nativeVirtualKey()))
                if self._keycomb == keycomb:
                    self.press()

    def _on_window_key_release(self, ev : qt.QKeyEvent):
        if not ev.isAutoRepeat():
            if ev.key() in [qt.Qt.Key.Key_Control, qt.Qt.Key.Key_Shift, qt.Qt.Key.Key_Alt]:
                keycomb = ev.keyCombination()
            else:
                # Using native virtual key in order to ignore keyboard language
                keycomb = qt.QKeyCombination(ev.modifiers(), qt.Qt.Key(ev.nativeVirtualKey()))

            if self._keycomb.key() == keycomb.key():
                # Release by key, not modifier.
                self.release()

    def _on_window_leave(self):
        self.release()

    def _on_owner_hide(self, ev):
        self.release()