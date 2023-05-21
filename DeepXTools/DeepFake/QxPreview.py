from pathlib import Path

from common.SSI import MxSSI, QxSSI
from core import qt, qx

from .MxPreview import MxPreview


class QxPreview(qx.QVBox):
    def __init__(self, preview : MxPreview):
        super().__init__()
        self._preview= preview

        sheet_holder = qx.QHBox()
        (self
            .add(sheet_holder)
            .add(qx.QMsgNotifyMxTextEmitter(preview.mx_error).set_title("<span style='color: red;'>@(Error)</span>"))
            .add(qx.QHBox().v_compact()
                .add(qx.QLabel().set_text('@(QxPreview.Source)').h_compact())
                .add(qx.QComboBoxMxSingleChoice( preview.mx_source_type,
                                                stringifier=lambda val: {MxPreview.SourceType.DataGenerator : '@(QxPreview.Data_generator)',
                                                                         MxPreview.SourceType.Directory : '@(QxPreview.Directory)',
                                                                        }[val],
                                                ).h_compact())

                .add(source_type_holder := qx.QVBox())

                , align=qx.Align.CenterH)
        )

        preview.mx_ssi_sheet.reflect(lambda ssi_sheet: self._ref_sheet(ssi_sheet, sheet_holder)).dispose_with(self)
        preview.mx_source_type.reflect(lambda source_type: self._ref_source_type(source_type, source_type_holder)).dispose_with(self)

    def _ref_source_type(self, source_type : MxPreview.SourceType, holder : qx.QVBox):
        holder.dispose_childs()
        if source_type == MxPreview.SourceType.DataGenerator:
            holder.add( qx.QPushButton().set_text('@(QxPreview.Generate)').inline(lambda btn: btn.mx_clicked.listen(lambda: self._preview.generate_one() )) )
        elif source_type == MxPreview.SourceType.Directory:
            holder.add( qx.QHBox()
                            .add(qx.QMxPathState(self._preview.mx_directory_path))
                            .add(sub_holder := qx.QHBox().set_spacing(4) )
                        )

            self._preview.mx_directory_path.mx_path.reflect(lambda path: self._ref_directory_path(path, sub_holder)).dispose_with(qx.QObject().set_parent(holder))

    def _ref_directory_path(self, path : Path|None, holder : qx.QHBox):
        holder.dispose_childs()

        if path is not None:
            (holder .add( qx.QHBox()
                            .add(qx.QLabel().set_text('@(QxPreview.Image_index)'))
                            .add(qx.QDoubleSpinBoxMxNumber(self._preview.mx_directory_image_idx)))
                            .add(qx.QPushButton().set_tooltip('@(Reload)').set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.reload, qx.StyleColor.ButtonText)).inline(lambda btn: btn.mx_clicked.listen(lambda: self._preview.update_directory_sample())))
                            )

    def _ref_sheet(self, ssi_sheet : MxSSI.Sheet, holder : qx.QHBox ):
        holder.dispose_childs().add(QxSSI.Sheet(ssi_sheet))

