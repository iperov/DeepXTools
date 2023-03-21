from .. import qt
from ._helpers import q_init
from .QAbstractItemModel import QAbstractItemModel
from .QAbstractItemView import QAbstractItemView


class QTreeView(QAbstractItemView):
    def __init__(self, **kwargs):
        super().__init__(q_abstract_item_view=q_init('q_tree_view', qt.QTreeView, **kwargs), **kwargs)

    def get_q_tree_view(self) -> qt.QTreeView: return self.get_q_abstract_item_view()

    def set_model(self, model : QAbstractItemModel):
        self.get_q_tree_view().setModel(model.get_q_abstract_item_model())
        return self