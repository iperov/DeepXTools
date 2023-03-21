from typing import Any, Callable

from .. import mx
from .QBox import QHBox, QVBox
from .QCheckBox import QCheckBox


class QCheckBoxMxMultiChoice(QVBox):
    def __init__(self,  mc : mx.IMultiChoice,
                        stringifier : Callable[ [Any], str ] = None,
                        **kwargs):
        super().__init__(**kwargs)
        self._mc = mc
        self._mc_avail = None

        self._holder = QHBox()
        self.add(self._holder)

        if stringifier is None:
            stringifier = lambda val: '' if val is None else str(val)
        self._stringifier = stringifier

        mc.reflect(lambda _: self.update_items()).dispose_with(self)


    def update_items(self):
        avail = self._mc_avail = self._mc.avail
        values = self._mc.get()

        self._holder.dispose_childs()
        for v in avail:
            q_combobox = QCheckBox().set_text(self._stringifier(v)).set_checked(v in values)
            q_combobox.mx_toggled.listen(lambda checked, v=v: (self._mc.update_added(v) if checked else self._mc.update_removed(v)) )

            self._holder.add(q_combobox)

