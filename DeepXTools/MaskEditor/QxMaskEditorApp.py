from pathlib import Path

from core import qt, qx

from .QxMaskEditor import QxMaskEditor


class QxMaskEditorApp(qx.QApplication):
    def __init__(self, settings_path = None):
        super().__init__(app_name='Mask Editor App', settings_path=settings_path)

        app_wnd = self._app_wnd = qx.QAppWindow().set_title('Mask Editor')
        app_wnd.set_window_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'app_icon.png')))
        app_wnd.get_central_vbox().add(
            QxMaskEditor(app_wnd.get_menu_bar()),
        )
        app_wnd.show()





