from pathlib import Path

from core import qt, qx

from .MxDeepFake import MxDeepFake
from .QxDeepFake import QxDeepFake


class QxDeepFakeApp(qx.QApplication):
    def __init__(self, deep_fake : MxDeepFake, settings_path = None):
        super().__init__(app_name='Deep Fake App', settings_path=settings_path)

        self._deep_fake = deep_fake

        app_wnd = self._app_wnd = qx.QAppWindow().set_title('Deep Fake')
        app_wnd.set_window_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'app_icon.png')))
        app_wnd.get_central_vbox().add(QxDeepFake(deep_fake))
        app_wnd.show()




