from core import qx

from .MxModel import MxModel


class QxModel(qx.QVBox):
    def __init__(self, model : MxModel):
        super().__init__()
        self._model = model


        (self
            .add(qx.QGrid().set_spacing(1)
                    .row(0)
                        .add(qx.QLabel().set_text('@(QxModel.Device)'), align=qx.Align.RightF)
                        .add(qx.QComboBoxMxSingleChoice(model.mx_device).set_font(qx.Font.FixedWidth))
                    .grid(),
                align=qx.Align.CenterH)

            .add(qx.QVBox()#.set_spacing(8)
                    .add(qx.QVBox()
                        .add(qx.QGrid().set_spacing(1)
                                .row(0)
                                    .add(qx.QLabel().set_text('@(QxModel.Input)'), align=qx.Align.RightF)
                                    
                                    .add(qx.QComboBoxMxSingleChoice(model.mx_input_type,
                                                            stringifier=lambda val: {MxModel.InputType.Color     : '@(QxModel.InputType.Color)',
                                                                                     MxModel.InputType.Luminance : '@(QxModel.InputType.Luminance)'}[val]))
                                    
                                .next_row()
                                    .add(qx.QLabel().set_text('@(QxModel.Resolution)'), align=qx.Align.RightF)
                                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_resolution))
                                .next_row()
                                    .add(qx.QLabel().set_text('@(QxModel.Base_dimension)'), align=qx.Align.RightF)
                                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_base_dim))
                                .grid()
                                , align=qx.Align.CenterH)
                        .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(QxModel.Current_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.revert_model_settings())))
                            .add(qx.QPushButton().set_text('@(QxModel.Apply_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.apply_model_settings()))))
                        , align=qx.Align.CenterH)

                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(Reset) Encoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_encoder())))
                        .add(qx.QPushButton().set_text('@(Reset) Decoder').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_decoder())))
                        , align=qx.Align.CenterH)
                    , align=qx.Align.CenterH) )