from common.Graph import QxGraph
from core import qx

from .MxModelTrainer import MxModelTrainer


class QxModelTrainer(qx.QVBox):
    def __init__(self, trainer : MxModelTrainer):
        super().__init__()
        self._trainer = trainer

        (self
            .add(qx.QMsgNotifyMxTextEmitter(trainer.mx_error).set_title("<span style='color: red;'>@(Error)</span>"))

            .add(qx.QHBox().set_spacing(8)

                    .add(qx.QVBox()
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('@(QxModelTrainer.Batch_size)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_batch_size))
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(QxModelTrainer.Batch_acc)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_batch_acc))
                                    .next_row()
                                        .add(qx.QLabel().set_text('@(QxModelTrainer.Learning_rate)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_learning_rate))
                                        .add(qx.QLabel().set_font(qx.Font.FixedWidth).set_text('* 1e-6'), align=qx.Align.LeftF)
                                     .grid())
                        ,align=qx.Align.CenterV)

                    .add(qx.QVBox()
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('MSE @(QxModelTrainer.power)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_mse_power))
                                    .next_row()
                                        .add(qx.QLabel().set_text('DSSIM/4 @(QxModelTrainer.power)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_dssim_x4_power))
                                    .next_row()
                                        .add(qx.QLabel().set_text('DSSIM/8 @(QxModelTrainer.power)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_dssim_x8_power))
                                    .next_row()
                                        .add(qx.QLabel().set_text('DSSIM/16 @(QxModelTrainer.power)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_dssim_x16_power))
                                    .next_row()
                                        .add(qx.QLabel().set_text('DSSIM/32 @(QxModelTrainer.power)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(trainer.mx_dssim_x32_power))

                                    .grid())
                        ,align=qx.Align.CenterV)

                , align=qx.Align.CenterH )

            .add_spacer(8)

            .add(qx.QGrid().set_spacing(1)
                 .row(0)
                    .add(qx.QLabel().set_text('@(QxModelTrainer.Iteration_time)'), align=qx.Align.RightF)
                    .add(qx.QLabel().set_font(qx.Font.Digital)
                            .inline(lambda lbl: trainer.mx_iteration_time.reflect(lambda time: lbl.set_text(f'{time:.3f}')).dispose_with(lbl)))

                    .add(qx.QLabel().set_font(qx.Font.FixedWidth).set_text('@(QxModelTrainer.second)'), align=qx.Align.LeftF)
                    .grid(),align=qx.Align.CenterH)


            .add(qx.QOnOffPushButtonMxFlag(trainer.mx_training)
                    .inline(lambda btn: btn.off_button.set_text('@(QxModelTrainer.Start_training)').mx_clicked.listen(lambda: trainer.set_training(True)))
                    .inline(lambda btn: btn.on_button.set_text('@(QxModelTrainer.Stop_training)').mx_clicked.listen(lambda: trainer.set_training(False))))

            .add_spacer(4)

            .add(qx.QCollapsibleVBox().set_text('@(QxModelTrainer.Metrics)').inline(lambda c: c.content_vbox.add(QxGraph(trainer.mx_metrics_graph).v_compact(200)))))


