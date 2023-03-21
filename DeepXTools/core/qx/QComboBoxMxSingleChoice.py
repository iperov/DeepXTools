from typing import Any, Callable

from .. import mx
from .QComboBox import QComboBox


class QComboBoxMxSingleChoice(QComboBox):
    def __init__(self,  sc : mx.ISingleChoice,
                        stringifier : Callable[ [Any], str ] = None,
                        **kwargs):
        super().__init__(**kwargs)
        self._sc = sc
        self._sc_avail = None

        if stringifier is None:
            stringifier = lambda val: '' if val is None else str(val)
        self._stringifier = stringifier

        self._conn = self.mx_current_index.listen(lambda idx: self._sc.set(self._sc_avail[idx]))
        sc.reflect(lambda _: self.update_items()).dispose_with(self)

    def update_items(self):
        avail = self._sc_avail = self._sc.avail
        value = self._sc.get()

        with self._conn.disabled_scope():
            self.clear()
            try:
                idx = avail.index(value)
            except:
                idx = None

            for x in avail:
                self.add_item(self._stringifier(x))

            if idx is not None:
                self.set_current_index(idx)

    def show_popup(self) -> None:
        self.update_items()
        super().show_popup()
