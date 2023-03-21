from __future__ import annotations

from ..lx import allowed_langs
from ._constants import Align, ProcessPriority
from .QAction import QAction
from .QApplication import QApplication
from .QAxMonitorWindow import QAxMonitorWindow
from .QBox import QHBox, QVBox
from .QFrame import QHFrame
from .QMenu import QMenu
from .QMenuBar import QMenuBar
from .QTextBrowser import QTextBrowser
from .QWindow import QWindow


class QAppWindow(QWindow):
    def __init__(self):
        """
        Main application window.

        Provides base functionality and menus.

        Allowed only single instance. Automatically parented to QApplication.
        """
        if QAppWindow._instance is not None:
            raise Exception('Only one QAppWindow can exist.')
        QAppWindow._instance = self
        super().__init__()
        
        app = QApplication.instance()
        self.set_parent(app)
        self.mx_close.listen(lambda _: app.quit())

        menu_bar = self._menu_bar = (QMenuBar()
            .add(QMenu().set_title('@(QAppWindow.Application)')
                    .add(QMenu().set_title('@(QAppWindow.Process_priority)')
                         .inline(lambda menu: menu.mx_about_to_show.listen(lambda me=menu:
                                    me.dispose_actions()
                                        .add( QAction()
                                                .set_text(f"{'-> ' if app.mx_process_priority.get() == ProcessPriority.NORMAL else ''}@(QAppWindow.Process_priority.Normal)")
                                                .inline(lambda act: act.mx_triggered.listen(lambda: app.mx_process_priority.set(ProcessPriority.NORMAL))))

                                        .add( QAction()
                                                .set_text(f"{'-> ' if app.mx_process_priority.get() == ProcessPriority.IDLE else ''}@(QAppWindow.Process_priority.Lowest)")
                                                .inline(lambda act: act.mx_triggered.listen(lambda: app.mx_process_priority.set(ProcessPriority.IDLE)))))))

                    .add(QAction().set_text('@(QAppWindow.Reset_UI_settings)').inline(lambda act: act.mx_triggered.listen(lambda: app.reset_settings())))
                    .add(QAction().set_text('@(QAppWindow.AsyncX_monitor)').inline(lambda act: act.mx_triggered.listen(lambda: self._on_open_ax_monitor())))
                    .add(QAction().set_text('@(QAppWindow.Quit)').inline(lambda act: act.mx_triggered.listen(lambda: app.quit()))))

            .add(QMenu().set_title('@(QAppWindow.Language)')
                    .inline(lambda menu:
                                [ menu.add(QAction().set_text(name).inline(lambda act: act.mx_triggered.listen(lambda lang=lang: app.mx_language.set(lang))))
                                    for lang, name in allowed_langs.items() ]))
                    
            .add(QMenu().set_title('@(QAppWindow.Help)')
                .add(QAction().set_text('@(QAppWindow.About)').inline(lambda act: act.mx_triggered.listen(lambda: self._on_open_about())))))
    
        self._central_vbox = QVBox()
        self._top_bar_hbox = QHBox()

        (self
            .add(QHBox()
                .add(menu_bar.h_compact(), align=Align.CenterV)
                .add(QHFrame().add(self._top_bar_hbox, align=Align.LeftF))
                .v_compact())
            .add_spacer(4)
            .add(self._central_vbox)
        )

    def __dispose__(self):
        QAppWindow._instance = None
        super().__dispose__()

    def get_menu_bar(self) -> QMenuBar: return self._menu_bar
    def get_top_bar_hbox(self) -> QHBox: return self._top_bar_hbox
    def get_central_vbox(self) -> QVBox: return self._central_vbox


    def _on_open_ax_monitor(self):
        if (wnd := getattr(self, '_ax_monitor_wnd', None)) is not None:
            wnd.activate()
        else:
            wnd = self._ax_monitor_wnd = QAxMonitorWindow().dispose_with(self)
            wnd.call_on_dispose(lambda: setattr(self, '_ax_monitor_wnd', None) )
            wnd.mx_close.listen(lambda ev: wnd.dispose())
            wnd.show()

    def _on_open_about(self):
        if (wnd := getattr(self, '_about_wnd', None)) is not None:
            wnd.activate()
        else:
            wnd = self._about_wnd = QAboutWindow().dispose_with(self)
            wnd.call_on_dispose(lambda: setattr(self, '_about_wnd', None) )
            wnd.mx_close.listen(lambda ev: wnd.dispose())
            wnd.show()

    _instance : QAppWindow = None


class QAboutWindow(QWindow):
    def __init__(self):
        super().__init__()
        
        app = QApplication.instance()
        self.set_parent(app)
        
        self.set_window_size(320,200).set_title('@(QAppWindow.About)')
         
        te = QTextBrowser().set_open_external_links(True)
        te.set_html("""
<html><body>

<table width="100%" height="100%">
<tr>
<td valign='middle' align='center'>

<span style='font-size:14.0pt'>DeepXTools</span>
<br>
<span style='font-size:10.0pt'><a href="https://iperov.github.io/DeepXTools">https://iperov.github.io/DeepXTools</a></span>
<br><br>
<span style='font-size:8.0pt'>Free open source software under GPL-3 license.
<br>
Designed and developed from scratch by <a href="https://github.com/iperov">iperov</a><br>
</span>
<br>
<br>

</td>
</tr>
</table>

</body></html>    
""")
        self.add(te)