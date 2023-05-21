from core import qx

from .MxModel import MxModel


class QxModel(qx.QVBox):
    def __init__(self, model : MxModel):
        super().__init__()
        self._model = model

        (self
            .add(qx.QMsgNotifyMxTextEmitter(model.mx_info).set_title("@(Info)"))

            .add(qx.QGrid().set_spacing(1)
                    .row(0)
                        .add(qx.QLabel().set_text('@(QxModel.Device)'), align=qx.Align.RightF)
                        .add(qx.QComboBoxMxSingleChoice(model.mx_device).set_font(qx.Font.FixedWidth))
                    .grid(),
                align=qx.Align.CenterH)

            .add_spacer(8)

            .add(qx.QVBox()
                    .add(qx.QVBox()
                        .add(qx.QGrid().set_spacing(1)
                                .row(0)
                                    .add(qx.QLabel().set_text('@(QxModel.Stage)'), align=qx.Align.RightF)
                                    .add(qx.QComboBoxMxSingleChoice(model.mx_stage, stringifier=lambda v: {MxModel.Stage.AutoEncoder : 'AutoEncoder',
                                                                                                        MxModel.Stage.Enhancer : 'Enhancer',}[v]).set_font(qx.Font.FixedWidth))
                                .grid()
                                , align=qx.Align.CenterH)

                        .add_spacer(8)

                        .add(qx.QHBox()
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('@(QxModel.Resolution)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_resolution))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Encoder @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_encoder_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('AE @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_ae_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Decoder @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_decoder_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Decoder mask @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_decoder_mask_dim))
                                    .grid()
                                    , align=qx.Align.CenterV)
                            .add(qx.QGrid().set_spacing(1)
                                    .row(0)
                                        .add(qx.QLabel().set_text('Enhancer depth'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_depth))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Enhancer @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Enhancer upscale factor'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_upscale_factor))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Enhancer GAN @(dimension)'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_gan_dim))
                                    .next_row()
                                        .add(qx.QLabel().set_text('Enhancer GAN patch size'), align=qx.Align.RightF)
                                        .add(qx.QDoubleSpinBoxMxNumber(model.mx_enhancer_gan_patch_size))
                                    .grid()
                                    , align=qx.Align.CenterV)
                                , align=qx.Align.CenterH)

                        .add_spacer(8)

                        .add(qx.QGrid().set_spacing(1)
                                .row(0)
                                    .add(qx.QLabel().set_text('Mixed precision'), align=qx.Align.RightF)
                                    .add(qx.QCheckBoxMxFlag(model.mx_mixed_precision))
                                .grid()
                                , align=qx.Align.CenterH)

                        .add_spacer(8)

                        .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(QxModel.Current_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.revert_model_settings())))
                            .add(qx.QPushButton().set_text('@(QxModel.Apply_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.apply_model_settings()))))
                        , align=qx.Align.CenterH)

                    .add_spacer(4)

                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) encoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_encoder())))
                        .add(qx.QPushButton().set_text('@(Reset) inter_src').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_inter_src())))
                        .add(qx.QPushButton().set_text('@(Reset) inter_dst').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_inter_dst())))
                        .add(qx.QPushButton().set_text('@(Reset) decoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder())))
                        .add(qx.QPushButton().set_text('@(Reset) decoder_mask').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder_mask())))
                        .add(qx.QPushButton().set_text('@(Reset) enhancer').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_enhancer())))

                        , align=qx.Align.CenterH)

                    , align=qx.Align.CenterH) )