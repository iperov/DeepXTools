from __future__ import annotations

import pickle
from pathlib import Path

from .. import ax, mx, qt
from ..lib import os as lib_os
from ._constants import ProcessPriority
from .QClipboard import QClipboard
from .QDarkFusionStyle import QDarkFusionStyle
from .QEvent import QEvent2
from .QFontDB import QFontDB
from .QIonIconDB import QIonIconDB
from .QObject import QObject
from .QSettings import QSettings
from .QTimer import QTimer


class QApplication(QObject):


    @staticmethod
    def instance() -> QApplication:
        if QApplication._instance is None:
            raise Exception('No QApplication instance.')
        return QApplication._instance

    def __init__(self, app_name : str = None, settings_path : Path = None):
        if QApplication._instance is not None:
            raise Exception('QApplication instance already exists.')
        QApplication._instance = self
        QObject._QApplication = QApplication

        q_app = qt.QApplication.instance() or qt.QApplication()
        if not isinstance(q_app, qt.QApplication):
            raise ValueError('q_app must be an instance of QApplication')
        self.__q_app = q_app

        super().__init__(q_object=q_app)
        
        self.__q_clipboard = QClipboard(q_clipboard=q_app.clipboard(), wrap_mode=True).dispose_with(self)

        self.__save_tg = ax.TaskGroup().dispose_with(self)

        self.__settings_path = settings_path
        state = {}
        try:
            if settings_path is not None and settings_path.exists():
                state = pickle.loads(settings_path.read_bytes())
        except: ...
        self.__settings = state.get('settings', {})

        QFontDB().dispose_with(self)
        QIonIconDB().dispose_with(self)

        q_app.deleteLater = lambda *_: ... # Supress QApplication object deletion
        q_app.setQuitOnLastWindowClosed(False)
        q_app.setApplicationName(app_name or 'QApplication')
        q_app.setFont(QFontDB.instance().default())
        q_app.setStyle(QDarkFusionStyle())
        self.__mx_focus_widget = mx.Property[qt.QWidget|None](None).dispose_with(self)

        self.__mx_language = mx.Property[str]( state.get('language', 'en') ).dispose_with(self)
        self.__mx_language.listen(lambda _: self._deferred_save_settings())

        QEvent2(q_app.focusChanged).dispose_with(self).listen(lambda old, new: self.__mx_focus_widget.set(new))

        self.__mx_process_priority = mx.SingleChoice[ProcessPriority]( lib_os.ProcessPriority.NORMAL,
                                                                                    avail=lambda: ProcessPriority,
                                                                                    filter=self._flt_mx_process_priority ).dispose_with(self)
        self.__mx_process_priority.set( lib_os.ProcessPriority(state.get('process_priority', lib_os.ProcessPriority.NORMAL.value)) )
        self.__mx_process_priority.listen(lambda _: self._deferred_save_settings())

        self.__timer_counter = 0
        self._t = QTimer(self._on_timer).set_interval(0).start().dispose_with(self)

    def __dispose__(self):
        super().__dispose__()
        self.__q_app = None
        QApplication._instance = None

    @property
    def mx_language(self) -> mx.IProperty[str]:
        """Language state"""
        return self.__mx_language
    @property
    def mx_process_priority(self) -> mx.ISingleChoice[ProcessPriority]:
        """"""
        return self.__mx_process_priority
    @property
    def mx_focus_widget(self) -> mx.IProperty_r[qt.QWidget|None]:
        """Current focus widget."""
        return self.__mx_focus_widget

    def exec(self):
        self.__q_app.exec()
        self.save_settings()

    def quit(self):
        self.__q_app.quit()
        
    def get_clipboard(self) -> QClipboard: return self.__q_clipboard 

    def get_settings(self, key) -> QSettings:
        """get lifetime mutable settings dict of specified key. Settings are saved/loaded."""
        d = self.__settings.get(key, None)
        if d is None:
            d = self.__settings[key] = {}

        return QSettings(d, save_func=self._deferred_save_settings)

    def save_settings(self):
        """Save settings to file."""
        if self.__settings_path is not None:
            state = {}
            state['language'] = self.__mx_language.get()
            state['process_priority'] = self.__mx_process_priority.get()
            state['settings'] = self.__settings

            self.__settings_path.write_bytes(pickle.dumps(state))

    def reset_settings(self):
        self.__settings = {}
        self._deferred_save_settings()

    def set_override_cursor(self, cursor : qt.QCursor | qt.Qt.CursorShape):
        if isinstance(cursor, qt.QCursor):
            cursor = cursor._get_q_cursor()
        self.__q_app.setOverrideCursor(cursor)

    def restore_override_cursor(self):
        self.__q_app.restoreOverrideCursor()

    def _flt_mx_process_priority(self, new_prio : ProcessPriority, prio):
        lib_os.set_process_priority(new_prio)
        return new_prio

    @ax.task
    def _deferred_save_settings(self):
        yield ax.attach_to(self.__save_tg, cancel_all=True)
        yield ax.sleep(1.0)
        self.save_settings()

    def _on_timer(self):
        ax.get_current_thread().execute_tasks_once()

        if self.__timer_counter % 10 == 0:
            lib_os.sleep_precise(0.001)

        self.__timer_counter += 1
    _instance : QApplication = None


    # def _on_timer(self):
    #     ax.get_current_thread().execute_tasks_once()
    #     if self.__timer_counter % 10 == 0:
    #         lib_os.sleep_precise(1)

    #     self.__timer_counter += 1

    # _instance : QApplication = None
