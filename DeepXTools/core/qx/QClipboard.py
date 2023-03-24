from .. import qt
from ._helpers import q_init
from .QObject import QObject


class QClipboard(QObject):
    def __init__(self, **kwargs):
        super().__init__(q_object=q_init('q_clipboard', None, qt.QClipboard, **kwargs), **kwargs)

        q_clipboard = self.get_q_clipboard()

    def get_q_clipboard(self) -> qt.QClipboard: return self.get_q_object()

    def get_q_mime_data(self) -> qt.QMimeData:
        return self.get_q_clipboard().mimeData()
    
    def get_image(self) -> qt.QImage|None:
        mime_data = self.get_q_mime_data()
        if mime_data.hasImage():        
            image = self.get_q_clipboard().image()
            return image if not image.isNull() else None
        return None
    
    def set_image(self, image : qt.QImage):
        self.get_q_clipboard().setImage(image)
        return self