from typing import Union

import numpy as np

from .. import mx, qx, qt
from .QPixmapWidget import QPixmapWidget


class QPixmapWidgetRxNumpyState(QPixmapWidget):
    def __init__(self, np_state : mx.NumpyNoneableState, q_widget = None):
        super().__init__(q_widget=q_widget)
        self._np_state = np_state
        
        np_state.reflect(self._on_np_state).dispose_with(self)
        
    def _on_np_state(self, ar : Union[np.ndarray, None]):
        if ar is not None:
            q_pixmap = qt.QPixmap_from_np(ar)
            self.set_pixmap(q_pixmap)
        else:
            self.set_pixmap(None)