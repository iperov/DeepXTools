from pathlib import Path

from core import qt, qx

from .ImageDSInfo import ImageDSInfo
from .MxImageDSRefList import MxImageDSRefList, MxImageDSRefListItem


class QxImageDSRefListItem(qx.QHBox):
    def __init__(self, item : MxImageDSRefListItem):
        super().__init__()
        self._item = item

        (self
            .add(qx.QMxPathState(item.mx_image_ds_path))
            .add((holder := qx.QHBox())))

        item.mx_image_ds_path.mx_path.reflect(lambda path: self._ref_path(path, holder) ).dispose_with(self)

    def _ref_path(self, path : Path|None, holder : qx.QHBox):
        holder.dispose_childs()
        if path is not None:
            holder.add(qx.QComboBoxMxSingleChoice(self._item.mx_mask_type))
            holder.add(qx.QPushButton().set_tooltip('@(Reveal_in_explorer)').set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.eye_outline, qx.StyleColor.ButtonText))
                                        .inline(lambda btn: btn.mx_clicked.listen(lambda:
                                                qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(ImageDSInfo(path).get_mask_dir_path(mask_type)))) \
                                                    if ((mask_type := self._item.mx_mask_type.get()) is not None) and
                                                       ((path := self._item.mx_image_ds_path.mx_path.get()) is not None) else ...)))

class QxImageDSRefList(qx.QVBox):
    def __init__(self, list : MxImageDSRefList):
        super().__init__()
        self._list = list

        self._q_item_by_item = {}

        list.mx_added.listen(self._on_item_added).dispose_with(self)
        list.mx_remove.listen(self._on_item_remove).dispose_with(self)


        items_l = self._items_l = qx.QVBox()
        (self
            .add(items_l)

            .add(qx.QPushButton()
                        .set_text('@(Add_item)')
                        .inline(lambda btn: btn.mx_clicked.listen(lambda: list.new() )))
        )

        for idx, item in enumerate(self._list.values()):
            self._on_item_added(idx, item)

    def _on_item_added(self, idx, item ):
        q_item = self._q_item_by_item[item] = (qx.QHBox()
            .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.remove, qx.StyleColor.ButtonText))
                    .set_tooltip('@(Remove_item)')
                    .inline(lambda btn: btn.mx_clicked.listen(lambda: self._list.remove(item)))
                    .h_compact())
            .add(QxImageDSRefListItem(item))
        )

        self._items_l.insert(idx, q_item)


    def _on_item_remove(self, idx, item):
        q_widget = self._items_l.widget_at(idx)
        q_widget.dispose()