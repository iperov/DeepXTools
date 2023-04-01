from core import ax, qx

from .MxExport import MxExport


class QxExport(qx.QVBox):
    def __init__(self, export : MxExport):
        super().__init__()
        self._export = export
        self._tg = ax.TaskGroup().dispose_with(self)

        self._export_task = None

        (self
            .add(qx.QMsgNotifyMxTextEmitter(export.mx_error).set_title("<span style='color: red;'>@(Error)</span>"))

            .add(qx.QGrid()
                .row(0)
                    .add(qx.QLabel().set_text('@(QxExport.Input)'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_input_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(QxExport.Output)'), align=qx.Align.RightF)
                    .add(qx.QMxPathState(export.mx_output_path))
                .next_row()
                    .add(qx.QLabel().set_text('@(QxExport.Patch_mode)'), align=qx.Align.RightF)
                    .add( qx.QHBox()
                            .add(qx.QCheckBoxMxFlag(export.mx_patch_mode))
                            .add(patch_mode_holder := qx.QHBox() ), align=qx.Align.LeftF)

                .grid())

            .add(qx.QHBox()
                    .add( (export_btn := qx.QPushButton()).set_text('@(QxExport.Export)')
                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: export.start()).dispose_with(self)))

                    .add( (cancel_btn := qx.QPushButton()).set_text('@(Cancel)')
                                    .inline(lambda btn: btn.mx_clicked.listen(lambda: export.stop()).dispose_with(self)))

                    .add(qx.QProgressBarMxProgress(export.mx_progress, hide_inactive=True).set_show_it_s(True).set_font(qx.Font.FixedWidth).h_expand())))


        export.mx_progress.mx_started.reflect(lambda started:
            (export_btn.hide(),
             cancel_btn.show()) if started else
            (export_btn.show(),
             cancel_btn.hide())
        ).dispose_with(self)

        export.mx_patch_mode.reflect(lambda patch_mode: self._ref_patch_mode(patch_mode, patch_mode_holder))


    def _ref_patch_mode(self, patch_mode, holder : qx.QHBox):
        holder.dispose_childs()
        if patch_mode:
            holder.add( qx.QHBox()
                            .add(qx.QLabel().set_text('@(QxExport.Sample_count)'))
                            .add(qx.QDoubleSpinBoxMxNumber(self._export.mx_sample_count))
                            .add_spacer(4)
                            .add(qx.QCheckBoxMxFlag(self._export.mx_fix_borders).set_text('@(QxExport.Fix_borders)'))
                            )
