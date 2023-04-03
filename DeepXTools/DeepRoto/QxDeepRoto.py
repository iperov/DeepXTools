from common.FileStateManager import MxFileStateManager, QxFileStateManager
from core import qx

from .MxDeepRoto import MxDeepRoto
from .QxDataGenerator import QxDataGenerator
from .QxExport import QxExport
from .QxModel import QxModel
from .QxModelTrainer import QxModelTrainer
from .QxPreview import QxPreview


class QxDeepRoto(qx.QVBox):
    def __init__(self, deep_roto : MxDeepRoto):
        super().__init__()
        self._deep_roto = deep_roto

        file_state_manager = deep_roto.mx_file_state_manager
        file_state_manager.mx_state.reflect(lambda state: self._ref_file_state_manager_state(state)).dispose_with(self)

    def _ref_file_state_manager_state(self, state : MxFileStateManager.State):
        self.dispose_childs()
        
        q_file_state_manager = QxFileStateManager(self._deep_roto.mx_file_state_manager)

        if state == MxFileStateManager.State.Initialized:
            self.add(qx.QSplitter().set_orientation(qx.Orientation.Vertical)
                    .add(qx.QVScrollArea().set_widget(
                            qx.QVBox().set_spacing(4)
                                .add(qx.QCollapsibleVBox().set_text('@(File_state_manager)').inline(lambda c: c.content_vbox.add(q_file_state_manager)).v_compact())
                                .add(qx.QCollapsibleVBox().set_text('@(Data_generator)').inline(lambda c: c.content_vbox.add(QxDataGenerator(self._deep_roto.mx_data_generator))).v_compact())
                                .add(qx.QCollapsibleVBox().set_text('@(Model)').inline(lambda c: c.content_vbox.add(QxModel(self._deep_roto.mx_model))).v_compact())
                                .add(qx.QCollapsibleVBox().set_text('@(Trainer)').inline(lambda c: c.content_vbox.add(QxModelTrainer(self._deep_roto.mx_model_trainer))).v_compact())
                                .add(qx.QCollapsibleVBox().set_text('@(Export)').inline(lambda c: c.content_vbox.add(QxExport(self._deep_roto.mx_export))).v_compact())
                                .add(qx.QWidget())
                            ))

                    .add(QxPreview(self._deep_roto.mx_preview)))
        else:
            self.add(q_file_state_manager.v_compact(), align=qx.Align.TopE)