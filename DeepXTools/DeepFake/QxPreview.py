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

        preview.mx_sample_dict.reflect(lambda sample_dict: self._ref_samples(sample_dict, sample_holder)).dispose_with(self)
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
                            .add(qx.QDoubleSpinBoxMxNumber(self._preview.mx_directory_image_idx))))

    def _ref_samples(self, sample_dict : MxPreview.SampleDict|None, holder : qx.QHBox ):
        holder.dispose_childs()
        if sample_dict is not None:
            tab_widget = qx.QTabWidget().set_tab_position(qx.QTabWidget.TabPosition.South)

            for name, sample in sample_dict.samples.items():
                if sample is not None:
                    grid = qx.QGrid().set_row_stretch(0,1,1,0)

                    sample_row = grid.row(0)
                    title_row = grid.row(1)

                    if sample.image_np is not None:
                        sample_row.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.image_np.HWC())) )
                        title_row.add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text(sample.image_name), align=qx.Align.CenterH )

                    if sample.target_mask_np is not None:
                        sample_row.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.target_mask_np.HWC())) )
                        title_row.add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxPreview.Target_mask)'), align=qx.Align.CenterH)

                    if sample.pred_image_np is not None:
                        sample_row.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.pred_image_np.HWC())) )
                        title_row.add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxPreview.Predicted_image)'), align=qx.Align.CenterH)

                    if sample.pred_mask_np is not None:
                        sample_row.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(sample.pred_mask_np.HWC())) )
                        title_row.add( qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxPreview.Predicted_mask)'), align=qx.Align.CenterH)

                    tab_widget.add_tab(lambda tab: tab.set_title(name).add(grid))

            holder.add(tab_widget)

