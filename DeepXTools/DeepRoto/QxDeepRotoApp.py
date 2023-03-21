from pathlib import Path

from core import qt, qx

from .MxDeepRoto import MxDeepRoto
from .QxDeepRoto import QxDeepRoto


class QxDeepRotoApp(qx.QApplication):
    def __init__(self, deep_roto : MxDeepRoto, settings_path = None):#
        super().__init__(app_name='Deep Roto App', settings_path=settings_path)

        self._deep_roto = deep_roto
        
        app_wnd = self._app_wnd = qx.QAppWindow().set_title('Deep Roto')
        app_wnd.set_window_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'app_icon.png')))
        app_wnd.get_central_vbox().add(QxDeepRoto(deep_roto))
        app_wnd.show()




