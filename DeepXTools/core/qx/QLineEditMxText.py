from .. import mx
from .QLineEdit import QLineEdit


class QLineEditMxText(QLineEdit):
    def __init__(self, text : mx.IText, **kwargs):
        super().__init__(**kwargs)
        self._text = text

        self._conn = self.mx_text.listen(lambda s: text.set(s if s is not None else ''))
        text.reflect(self._ref_text).dispose_with(self)

    def _ref_text(self, text):
        with self._conn.disabled_scope():
            self.set_text(text)
