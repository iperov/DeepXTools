from pathlib import Path

from core import qt, qx

from .MxPreview import MxPreview


class QxPreview(qx.QVBox):
    def __init__(self, preview : MxPreview):
        super().__init__()
        self._preview= preview

        sample_holder = qx.QHBox()
        (self
            .add(sample_holder)
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

        preview.mx_sample.reflect(lambda sample: self._ref_sample(sample, sample_holder)).dispose_with(self)
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

                    .add( qx.QCheckBoxMxFlag(self._preview.mx_patch_mode).set_text('@(QxPreview.Patch_mode)'))
                    .add( patch_mode_holder := qx.QHBox()))

            self._preview.mx_patch_mode.reflect(lambda patch_mode: self._ref_patch_mode(patch_mode, patch_mode_holder))

    def _ref_patch_mode(self, patch_mode, holder : qx.QHBox):
        holder.dispose_childs()
        if patch_mode:
            holder.add( qx.QHBox()
                            .add(qx.QLabel().set_text('@(QxPreview.Sample_count)'))
                            .add(qx.QDoubleSpinBoxMxNumber(self._preview.mx_sample_count))
                            .add_spacer(4)
                            .add( qx.QCheckBoxMxFlag(self._preview.mx_fix_borders).set_text('@(QxPreview.Fix_borders)')))

    def _ref_sample(self, sample : MxPreview.Sample|None, holder : qx.QHBox ):

        holder.dispose_childs()
        if sample is not None:

            holder.add(qx.QGrid().set_row_stretch(0,1,1,0)
                    .row(0)
                        .add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.image_np.HWC())) )
                        .add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.target_mask_np.HWC())) if sample.target_mask_np is not None else None)
                        .add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.pred_mask_np.HWC())) )
                    .next_row()
                        .add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text(sample.name), align=qx.Align.CenterH )
                        .add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxPreview.Target_mask)') if sample.target_mask_np is not None else None, align=qx.Align.CenterH)
                        .add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxPreview.Predicted_mask)'), align=qx.Align.CenterH)
                    .grid())
