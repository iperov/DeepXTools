import itertools

from common.ImageDS import QxImageDSRefList
from core import ax, qt, qx

from .MxDataGenerator import MxDataGenerator


class QxDataGenerator(qx.QVBox):
    def __init__(self, data_gen : MxDataGenerator):
        super().__init__()
        self._data_gen = data_gen
        self._tg = tg = ax.TaskGroup().dispose_with(self)

        holder = qx.QVBox()

        (self
            .add(QxImageDSRefList(data_gen.mx_image_ds_ref_list))
            .add(qx.QPushButton().set_text('@(QxDataGenerator.Reload)')
                                 .inline(lambda btn: btn.mx_clicked.listen(lambda: data_gen.reload())))
            .add(qx.QMsgNotifyMxTextEmitter(data_gen.mx_error).set_title("<span style='color: red;'>@(Error)</span>"))
            .add_spacer(4)
            .add(qx.QHBox()
                    .add(qx.QVBox()

                            .add(qx.QGrid().set_spacing(1).row(0)
                                    .next_row()
                                    .add(qx.QLabel().set_text('@(Mode)'), align=qx.Align.RightF)
                                    .add(qx.QComboBoxMxSingleChoice(data_gen.mx_mode,
                                                            stringifier=lambda val: {MxDataGenerator.Mode.Fit   : '@(QxDataGenerator.Mode.Fit)',
                                                                                     MxDataGenerator.Mode.Patch : '@(QxDataGenerator.Mode.Patch)'}[val]))
                                    .grid())

                            .add(qx.QHBox()
                                .add(qx.QVBox()
                                        .add(transform_holder := qx.QVBox(), align=qx.Align.RightF))

                                .add_spacer(8)
                                .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('@(QxDataGenerator.Random)'), align=qx.Align.CenterF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_flip)
                                                            .set_text('@(QxDataGenerator.Flip)'), align=qx.Align.LeftF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_levels_shift)
                                                            .set_text('@(QxDataGenerator.Levels_shift)'), align=qx.Align.LeftF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_sharpen_blur)
                                                            .set_text('@(QxDataGenerator.Sharpen_blur)'), align=qx.Align.LeftF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_glow_shade)
                                                            .set_text('@(QxDataGenerator.Glow_shade)'), align=qx.Align.LeftF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_resize)
                                                            .set_text('@(QxDataGenerator.Resize)'), align=qx.Align.LeftF)
                                    .next_row()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_rnd_jpeg_artifacts)
                                                            .set_text('@(QxDataGenerator.JPEG_artifacts)'), align=qx.Align.LeftF)
                                    .grid().v_compact()))
                                    
                            .add_spacer(8)
                                    
                            # .add(qx.QHBox()
                            #                     .add(qx.QLabel().set_text('@(QxDataGenerator.Output_type)'), align=qx.Align.RightF)
                            #                     .add(qx.QComboBoxMxSingleChoice(data_gen.mx_output_type,
                            #                                             stringifier=lambda val: {MxDataGenerator.OutputType.Image_n_Mask : '@(QxDataGenerator.Image_n_Mask)',
                            #                                                                      MxDataGenerator.OutputType.Image_n_ImageGrayscaled : '@(QxDataGenerator.Image_n_ImageGrayscaled)'}[val]))
                                             
                            #                     , align=qx.Align.RightF)      
                                    

                        , align=qx.Align.CenterV)

                    .add_spacer(8)
                    .add(qx.QVBox()
                            .add(holder.v_compact())
                            .add(qx.QPushButton().set_checkable(True)
                                        .set_text('@(QxDataGenerator.Generate_preview)')
                                        .inline(lambda btn: btn.mx_toggled.listen(lambda checked: self._gen_preview_task(tg, holder) if checked else tg.cancel_all() ) )
                                        .v_compact())
                            .h_compact(),
                        align=qx.Align.CenterV
                        ),
                align=qx.Align.CenterF)
        )

        data_gen.mx_mode.reflect(lambda mode: self._ref_mx_mode(mode, transform_holder)).dispose_with(self)

    def _ref_mx_mode(self, mode : MxDataGenerator.Mode, holder : qx.QVBox ):
        data_gen = self._data_gen

        holder.dispose_childs()
        holder.add(grid := qx.QGrid().set_spacing(1).v_compact(), align=qx.Align.RightF)

        row = grid.row(0)
        row.add(None)
        if mode == MxDataGenerator.Mode.Fit:
            row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxDataGenerator.Offset)'), align=qx.Align.BottomF)
        row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxDataGenerator.Random)'), align=qx.Align.BottomF)

        if mode == MxDataGenerator.Mode.Fit:
            row = row.next_row()
            (row    .add(qx.QLabel().set_text('@(QxDataGenerator.Translation_X)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_tx) )
                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_tx_var) ))
        if mode == MxDataGenerator.Mode.Fit:
            row = row.next_row()
            (row    .add(qx.QLabel().set_text('@(QxDataGenerator.Translation_Y)'), align=qx.Align.RightF)
                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_ty))
                    .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_ty_var)))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(QxDataGenerator.Scale)'), align=qx.Align.RightF)
        if mode == MxDataGenerator.Mode.Fit:
            row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_scale))
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_scale_var))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(QxDataGenerator.Rotation)'), align=qx.Align.RightF)
        if mode == MxDataGenerator.Mode.Fit:
            row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_rot_deg))
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_rot_deg_var))

        holder.add_spacer(8)
        holder.add( qx.QGrid().set_spacing(1).v_compact().row(0)
                        .add(qx.QLabel().set_text('@(QxDataGenerator.Transform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_transform_intensity))
                    .next_row()
                        .add(qx.QLabel().set_text('@(QxDataGenerator.Image_deform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_image_deform_intensity))
                    .next_row()
                        .add(qx.QLabel().set_text('@(QxDataGenerator.Mask_deform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_mask_deform_intensity))
                    .grid(), align=qx.Align.RightF )


    @ax.task
    def _gen_preview_task(self, tg : ax.TaskGroup, holder : qx.QVBox):
        yield ax.attach_to(tg, cancel_all=True)

        W = H = 224

        for i in itertools.count():
            yield ax.wait(t := self._data_gen.generate(1, W=W, H=H))
            
            if i == 0:
                holder.dispose_childs()
                image_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)
                mask_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)

                holder.add(qx.QTabWidget().set_tab_position(qx.QTabWidget.TabPosition.South)
                            .add_tab(lambda tab: tab.set_title('@(QxDataGenerator.Image)').add(image_pixmap_widget))
                            .add_tab(lambda tab: tab.set_title('@(QxDataGenerator.Mask)').add(mask_pixmap_widget)))

            if t.succeeded:
                result = t.result

                image_pixmap_widget.set_pixmap( qt.QPixmap_from_np(result.image_np[0].HWC()) )
                mask_pixmap_widget.set_pixmap( qt.QPixmap_from_np(result.target_mask_np[0].HWC()) )
            else:
                error = t.error
                holder.dispose_childs()
                holder.add( qx.QLabel().set_text("<span style='color: red;'>@(Error)</span>").v_compact(), align=qx.Align.CenterF)
                holder.add( qx.QTextEdit().h_compact(W).v_compact(H).set_font(qx.Font.FixedWidth).set_read_only(True).set_plain_text(str(error)) )
                
                yield ax.cancel(error)

            yield ax.sleep(0.1)

