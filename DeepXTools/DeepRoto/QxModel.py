import urllib.parse

from core import qx

from .MxModel import MxModel


class QxModel(qx.QVBox):
    def __init__(self, model : MxModel):
        super().__init__()
        self._model = model
        
        q_download_menu = qx.QMenuMxMenu(model.mx_url_download_menu,
                                         stringifier=lambda url: urllib.parse.urlparse(url).path.split('/')[-1].split('.')[0] ).set_parent(self)
            
        
        export_model_dlg = ( qx.QFileDialog().set_parent(self)
                            .set_accept_mode(qx.QFileDialog.AcceptMode.AcceptSave)
                            .set_file_mode(qx.QFileDialog.FileMode.AnyFile)
                            .set_filter(f"Deep Roto model (*.dxrm)")
                            .inline(lambda dlg: dlg.mx_accepted.listen(lambda p: model.export_model(p[0]))))
        
        import_model_dlg = ( qx.QFileDialog().set_parent(self)
                            .set_accept_mode(qx.QFileDialog.AcceptMode.AcceptOpen)
                            .set_file_mode(qx.QFileDialog.FileMode.ExistingFile)
                            .set_filter(f"Deep Roto model (*.dxrm)")
                            .inline(lambda dlg: dlg.mx_accepted.listen(lambda p: model.import_model(p[0]))))
        
        (self
            .add(qx.QMsgNotifyMxTextEmitter(model.mx_info).set_title("@(Info)"))
            
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
                                .next_row()
                                    .add(qx.QLabel().set_text('@(QxModel.Generalization_level)'), align=qx.Align.RightF)
                                    .add(qx.QDoubleSpinBoxMxNumber(model.mx_generalization_level))
                                .grid()
                                , align=qx.Align.CenterH)
                        .add(qx.QHBox()
                            .add(qx.QPushButton().set_text('@(QxModel.Current_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.revert_model_settings())))
                            .add(qx.QPushButton().set_text('@(QxModel.Apply_settings)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.apply_model_settings()))))
                        , align=qx.Align.CenterH)
                        
                    .add_spacer(4)

                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(QxModel.Reset_model)').inline(lambda btn: btn.mx_clicked.listen(lambda: model.reset_model())))
                        .add(qx.QPushButton().set_text('@(QxModel.Import_model)').inline(lambda btn: btn.mx_clicked.listen(lambda: import_model_dlg.show())))
                        .add(qx.QPushButton().set_text('@(QxModel.Export_model)').inline(lambda btn: btn.mx_clicked.listen(lambda: export_model_dlg.show())))
                        , align=qx.Align.CenterH)
                
                    .add_spacer(4)
                     
                    .add(qx.QHBox()
                        .add(qx.QPushButton().set_text('@(QxModel.Download_pretrained_model)...').inline(lambda btn: btn.mx_clicked.listen(lambda: q_download_menu.show())))
                        , align=qx.Align.CenterH)
                        
                    , align=qx.Align.CenterH) )