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
            .add(qx.QPushButton().set_text('@(Reload)')
                                 .inline(lambda btn: btn.mx_clicked.listen(lambda: data_gen.reload())))
            .add(qx.QMsgNotifyMxTextEmitter(data_gen.mx_error).set_title("<span style='color: red;'>@(Error)</span>"))
            .add_spacer(4)
            .add(qx.QHBox()
                    .add(qx.QVBox()

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
                                    .grid().v_compact()))

                            .add_spacer(8)


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

        transform_holder.add(grid := qx.QGrid().set_spacing(1).v_compact(), align=qx.Align.RightF)

        row = grid.row(0)
        row.add(None)
        row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxDataGenerator.Offset)'), align=qx.Align.BottomF)
        row.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxDataGenerator.Random)'), align=qx.Align.BottomF)

        row = row.next_row()
        (row    .add(qx.QLabel().set_text('@(QxDataGenerator.Translation_X)'), align=qx.Align.RightF)
                .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_tx) )
                .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_tx_var) ))
        row = row.next_row()
        (row    .add(qx.QLabel().set_text('@(QxDataGenerator.Translation_Y)'), align=qx.Align.RightF)
                .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_ty))
                .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_ty_var)))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(QxDataGenerator.Scale)'), align=qx.Align.RightF)
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_scale))
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_scale_var))

        row = row.next_row()
        row.add(qx.QLabel().set_text('@(QxDataGenerator.Rotation)'), align=qx.Align.RightF)
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_offset_rot_deg))
        row.add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_rnd_rot_deg_var))

        transform_holder.add_spacer(8)
        transform_holder.add( qx.QGrid().set_spacing(1).v_compact().row(0)
                        .add(qx.QLabel().set_text('@(QxDataGenerator.Transform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_transform_intensity))
                    .next_row()
                        .add(qx.QLabel().set_text('@(QxDataGenerator.Image_deform_intensity)'), align=qx.Align.RightF, col_span=1)
                        .add(qx.QDoubleSpinBoxMxNumber(data_gen.mx_image_deform_intensity))
                    .grid(), align=qx.Align.RightF )

        transform_holder.add_spacer(8)
        transform_holder.add( qx.QGrid().set_spacing(1).v_compact().row(0)
                                .add(qx.QLabel().set_text('@(QxDataGenerator.Border_type)'), align=qx.Align.RightF, col_span=1)
                                .add(qx.QComboBoxMxSingleChoice(data_gen.mx_border_type,
                                                                stringifier=lambda val: {MxDataGenerator.BorderType.CONSTANT: '@(QxDataGenerator.Border_type.CONSTANT)',
                                                                                         MxDataGenerator.BorderType.REFLECT: '@(QxDataGenerator.Border_type.REFLECT)',
                                                                                         MxDataGenerator.BorderType.REPLICATE: '@(QxDataGenerator.Border_type.REPLICATE)',
                                                                                         }[val] ))
                            .next_row()
                                .add(qx.QLabel().set_text('@(QxDataGenerator.Decrease_chance_similar)'), align=qx.Align.RightF, col_span=1)
                                .add( qx.QHBox()
                                        .add(qx.QCheckBoxMxFlag(data_gen.mx_dcs))
                                        .add(computing_lbl := qx.QLabel().set_text('@(Computing)...').hide() ))
                            .grid(), align=qx.Align.RightF )


        data_gen.mx_dcs_computing.reflect(lambda b: computing_lbl.set_visible(b) )

    @ax.task
    def _gen_preview_task(self, tg : ax.TaskGroup, holder : qx.QVBox):
        yield ax.attach_to(tg, cancel_all=True)

        W = H = 224

        for i in itertools.count():
            yield ax.wait(t := self._data_gen.generate(1, W=W, H=H))

            if i == 0:
                holder.dispose_childs()
                image_deformed_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)
                image_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)
                mask_pixmap_widget = qx.QPixmapWidget().h_compact(W).v_compact(H)

                holder.add(qx.QTabWidget().set_tab_position(qx.QTabWidget.TabPosition.South)
                            .add_tab(lambda tab: tab.set_title('Image_deformed').add(image_deformed_pixmap_widget))
                            .add_tab(lambda tab: tab.set_title('@(QxDataGenerator.Image)').add(image_pixmap_widget))
                            .add_tab(lambda tab: tab.set_title('@(QxDataGenerator.Mask)').add(mask_pixmap_widget)))

            if t.succeeded:
                result = t.result

                image_deformed_pixmap_widget.set_pixmap( qt.QPixmap_from_np(result.image_deformed_np[0].HWC()) )
                image_pixmap_widget.set_pixmap( qt.QPixmap_from_np(result.image_np[0].HWC()) )

                mask = result.target_mask_np[0]
                if mask is not None:
                    mask_pixmap_widget.set_pixmap( qt.QPixmap_from_np(result.target_mask_np[0].HWC()) )
                else:
                    mask_pixmap_widget.set_pixmap(None)
            else:
                error = t.error
                holder.dispose_childs()
                holder.add( qx.QLabel().set_text("<span style='color: red;'>@(Error)</span>").v_compact(), align=qx.Align.CenterF)
                holder.add( qx.QTextEdit().h_compact(W).v_compact(H).set_font(qx.Font.FixedWidth).set_read_only(True).set_plain_text(str(error)) )

                yield ax.cancel(error)

            yield ax.sleep(0.1)

