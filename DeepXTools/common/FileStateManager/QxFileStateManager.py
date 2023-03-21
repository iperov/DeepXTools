from core import qt, qx

from .MxFileStateManager import MxFileStateManager


class QxFileStateManager(qx.QVBox):

    def __init__(self, mgr : MxFileStateManager):
        super().__init__()
        self._mgr = mgr

        holder = qx.QVBox()
        self.add(holder).v_compact()

        mgr.mx_state.reflect(lambda state: self._ref_mgr_state(state, mgr, holder)).dispose_with(self)

    def _ref_mgr_state(self, state : MxFileStateManager.State, mgr : MxFileStateManager, holder : qx.QVBox):
        holder.dispose_childs()
        holder_null_child = qx.QObject().set_parent(holder)

        load_menu = qx.QMenuMxMenu(mgr.mx_load_menu).set_parent(holder).set_font(qx.Font.FixedWidth)
        
        
        (holder
            .add(qx.QHBox().set_spacing(1)
                .add((grid_col := qx.QGrid().set_spacing(4).col(0)).grid())))
                    
        if state == MxFileStateManager.State.Loading:
            grid_col.add(qx.QProgressBarMxProgress(mgr.mx_load_progress))
        elif state != MxFileStateManager.State.Uninitialized:
            (grid_col       
                    .add(qx.QPushButton().set_text('@(New)').inline(lambda btn: btn.mx_clicked.listen(lambda: mgr.reset())).v_compact())        
                .next_col()
                    .add(qx.QVBox()
                            .add(load_btn := qx.QPushButton().set_text('@(Load)...').inline(lambda btn: btn.mx_clicked.listen(lambda: load_menu.show())).v_compact())
                            .add(qx.QProgressBarMxProgress(mgr.mx_load_progress, hide_inactive=True))
                            .inline(lambda _: mgr.mx_load_progress.mx_started.reflect(lambda started: load_btn.set_visible(not started)).dispose_with(holder_null_child)))
                    .add(qx.QPathLabel(mgr.get_state_path()), align=qx.Align.TopE))            

        if state == MxFileStateManager.State.Error:
            (holder
                .add(qx.QVBox()
                    .add(qx.QHBox()
                        .add(qx.QIconWidget().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.alert_circle_outline, qt.QColor(255,0,0))))
                        .add(qx.QLabel().set_text("<span style='color: red;'>@(Error)</span>").set_align(qx.Align.CenterE))
                        , align=qx.Align.CenterH)

                    .add(qx.QTextEdit() .set_font(qx.Font.FixedWidth).set_read_only(True)
                                        .set_plain_text(str(exc) if (exc := mgr.mx_error.get()) is not None else 'unknown')
                                        .v_compact(64))))
        elif state == MxFileStateManager.State.Initialized:
            (grid_col.next_col()
                .add(qx.QVBox()
                    .add(save_btn := qx.QPushButton().set_text('@(Save)').inline(lambda btn: btn.mx_clicked.listen(lambda: mgr.save())))
                    .add(qx.QProgressBarMxProgress(mgr.mx_save_progress, hide_inactive=True))
                    .inline(lambda _: mgr.mx_save_progress.mx_started.reflect(lambda started: save_btn.set_visible(not started)).dispose_with(holder_null_child)))
                .add(qx.QGrid().set_spacing(4).row(0)
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.Every)'), align=qx.Align.RightF )
                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_autosave) )
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.minutes)'), align=qx.Align.LeftF )
                    .grid(), align=qx.Align.TopF)
            .next_col()
                .add(qx.QVBox()
                    .add(backup_btn := qx.QPushButton().set_text('@(QxFileStateManager.Backup)').inline(lambda btn: btn.mx_clicked.listen(lambda: mgr.save(backup=True))))
                    .add(qx.QProgressBarMxProgress(mgr.mx_backup_progress, hide_inactive=True))
                    .inline(lambda _: mgr.mx_backup_progress.mx_started.reflect(lambda started: backup_btn.set_visible(not started)).dispose_with(holder_null_child)))
                .add(qx.QGrid().set_spacing(4).row(0)
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.Every)'), align=qx.Align.RightF )
                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_autobackup) )
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.minutes)'), align=qx.Align.LeftF )
                    .next_row()
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.Maximum)'), align=qx.Align.RightF )
                        .add( qx.QDoubleSpinBoxMxNumber(mgr.mx_backup_count) )
                        .add( qx.QLabel().set_text('@(QxFileStateManager.Save.backups)'), align=qx.Align.LeftF )
                    .grid(), align=qx.Align.TopF)
                    )
            
            (holder
                .add_spacer(8)
                .add(qx.QCollapsibleVBox().set_text('@(QxFileStateManager.Notes)').close().inline(lambda c: c.content_vbox.add(qx.QTextEditMxText(mgr.mx_notes).v_compact(64) )).v_compact()))
            