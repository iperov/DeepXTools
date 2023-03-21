from __future__ import annotations

from .. import mx
from .QAction import QAction
from .QMenu import QMenu


class QMenuMxMenu(QMenu):
    def __init__(self, menu : mx.IMenu, **kwargs):
        super().__init__(**kwargs)
        self._mx_menu = menu
        self.mx_about_to_show.listen(self._on_about_to_show)

    def _on_about_to_show(self):
        self.dispose_actions()

        for choice in self._mx_menu.avail_choices:
            self.add(QAction()  .set_text(str(choice) )
                                .inline(lambda act: act.mx_triggered.listen(lambda choice=choice: self._mx_menu.choose(choice)).dispose_with(act)))


