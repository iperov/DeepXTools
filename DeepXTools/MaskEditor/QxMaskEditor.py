import itertools
import re
from pathlib import Path

from common.ImageDS import ImageDS
from core import ax, mx, qt, qx

from .QxMaskEditorCanvas import QxMaskEditorCanvas


class QxMaskEditor(qx.QVBox):
    def __init__(self, root_path : Path, menu_bar : qx.QMenuBar):
        super().__init__()
        self._root_path = root_path
        self._menu_bar = menu_bar

        self._copied_mask = None
        self._q_mask_editor : QxMaskEditorCanvas = None
        self._q_mask_editor_hash = None

        mx_path_state = self._mx_path_state = \
                    mx.PathState( config=mx.PathStateConfig(dir_only=True, desc='Sequence of images'),
                                      on_open=lambda path: self._mx_path_dlg_open(path, holder_mx_image_ds, holder_top_bar),
                                      on_close=lambda: self._mx_path_dlg_close(holder_mx_image_ds, holder_top_bar)).dispose_with(self)

        (self   .add(qx.QHBox()
                        .add(qx.QMxPathState(mx_path_state).set_directory(root_path))
                        .add(holder_top_bar := qx.QHBox()).v_compact() )
                .add(holder_mx_image_ds := qx.QVBox()) )

    def __dispose__(self):
        self._mx_path_state.close()
        super().__dispose__()

    def _mx_path_dlg_close(self, holder : qx.QVBox, holder_top_bar : qx.QHBox):
        self._save()
        holder.dispose_childs()
        holder_top_bar.dispose_childs()

    def _mx_path_dlg_open(self, path : Path, holder : qx.QVBox, holder_top_bar : qx.QHBox) -> bool:
        holder.dispose_childs()
        holder_top_bar.dispose_childs()

        holder_disp = qx.QObject().set_parent(holder)

        try:
            image_ds = self._image_ds = ImageDS.open(path)
        except Exception as e:
            holder.add(qx.QLabel().set_align(qx.Align.CenterE).set_text(f'@(Error): {str(e)}'))
            return False

        mx_mask_type = self._mx_mask_type = mx.SingleChoice[str|None](
                                                avail_mask_types[0] if len(avail_mask_types := image_ds.get_mask_types()) != 0 else None,
                                                avail=lambda: [None]+image_ds.get_mask_types()).dispose_with(holder_disp)

        q_tape = self._q_tape = (qx.QTapeCachedItemView(task_get_item_pixmap=lambda idx, rect: self._tape_get_item_pixmap(idx, rect, image_ds))
                                    .set_item_size(64, 64+16+1))

        q_tape_scrollbar = self._q_tape_scrollbar = qx.QScrollBar().set_minimum(0)
        self._q_tape_scrollbar_value_conn = q_tape_scrollbar.mx_value.listen(lambda idx: self._save_and_goto(idx) )


        tg = ax.TaskGroup().dispose_with(q_tape)
        prev_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_A), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next(tg, -1)),
                                shortcut.mx_release.listen(lambda: tg.cancel_all())))

        next_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_D), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next(tg, 1)),
                                shortcut.mx_release.listen(lambda: tg.cancel_all())))

        prev_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_A), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next_mask(tg, forward=False)),
                                shortcut.mx_release.listen(lambda: tg.cancel_all())))

        next_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_D), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next_mask(tg, forward=True)),
                                shortcut.mx_release.listen(lambda: tg.cancel_all())))

        force_save_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_S), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._save(force=True)))

        remove_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_X), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._delete_mask() ))

        copy_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_C), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._copy_mask()))

        paste_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_V), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._paste_mask()))


        self._menu_bar.add( qx.QMenu().dispose_with_childs(holder)
                                .set_title('@(QxMaskEditor.Thumbnail_size)')
                                .inline(lambda menu:
                                        [menu.add( qx.QAction().set_text(str(32*i)).inline(lambda act: act.mx_triggered.listen(lambda i=i: q_tape.set_item_size(32*i, 32*i+16+1)) ))
                                            for i in range(1, 9) ]))

        holder_top_bar.add(qx.QHBox()
                            .add( qx.QLabel().set_text('@(QxMaskEditor.Mask_type)'), align=qx.Align.RightF)
                            .add( qx.QComboBoxMxSingleChoice(mx_mask_type) )
                            .add( qx.QPushButton().set_tooltip('@(Reveal_in_explorer)').set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.eye_outline, qx.StyleColor.ButtonText))
                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(image_ds.get_mask_dir_path(mask_type))))
                                                    if (mask_type := mx_mask_type.get()) is not None else ...)))
                            .add( qx.QHBox()
                                    .add( btn_new_mask := qx.QPushButton().set_tooltip('@(New)').set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.add_circle_outline, qx.StyleColor.ButtonText))
                                                            .inline(lambda btn: btn.mx_clicked.listen(lambda:  (btn_new_mask.hide(),
                                                                                                                holder_new_mask.show(),
                                                                                                                le_mask_name.set_focus(),
                                                                                                                le_mask_name.set_text(None)))))
                                    .add( holder_new_mask := qx.QHBox().hide()
                                                .add(qx.QLabel().set_text('@(QxMaskEditor.Mask_name)'))

                                                .add(le_mask_name := qx.QLineEdit().h_compact().set_filter(lambda s: re.sub('\W', '', s)))

                                                .add(qx.QPushButton().h_compact().set_text('@(Ok)')
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda:  (image_ds.add_mask_type(le_mask_name.get_text()),
                                                                                                            mx_mask_type.set(le_mask_name.get_text()),
                                                                                                            btn_new_mask.show(),
                                                                                                            holder_new_mask.hide()))))
                                                .add(qx.QPushButton().h_compact().set_text('@(Cancel)')
                                                        .inline(lambda btn: btn.mx_clicked.listen(lambda: (btn_new_mask.show(), holder_new_mask.hide())))))))

        (holder .add(qx.QSplitter().set_orientation(qx.Orientation.Vertical).set_default_sizes([9999,1])
                        .add(qx.QVBox()
                                .add(holder_mask_editor := qx.QVBox()))
                        .add(qx.QVBox()
                                .add(q_tape)
                                .add(q_tape_scrollbar.v_compact())
                                .add(qx.QHBox().v_compact()

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_skip_back, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Save_prev_img_mask) (CTRL+A)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: prev_mask_shortcut.press()), btn.mx_released.listen(lambda: prev_mask_shortcut.release()))))
                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_back, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Save_prev_img) (A)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: prev_shortcut.press()), btn.mx_released.listen(lambda: prev_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.copy, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Copy_mask) (CTRL+S)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: copy_mask_shortcut.press()), btn.mx_released.listen(lambda: copy_mask_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.clipboard, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Paste_mask) (CTRL+X)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: paste_mask_shortcut.press()), btn.mx_released.listen(lambda: paste_mask_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.save, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Force_save) (CTRL+S)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: force_save_shortcut.press()), btn.mx_released.listen(lambda: force_save_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.trash, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Delete_mask) (CTRL+X)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: remove_mask_shortcut.press()), btn.mx_released.listen(lambda: remove_mask_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_forward, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Save_next_img) (D)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: next_shortcut.press()), btn.mx_released.listen(lambda: next_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_skip_forward, qx.StyleColor.ButtonText)).set_tooltip('@(QxMaskEditor.Save_next_img_with_mask) (CTRL+D)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: next_mask_shortcut.press()), btn.mx_released.listen(lambda: next_mask_shortcut.release()))))

                                        ))))

        q_tape.set_item_count(image_count := image_ds.image_count)
        q_tape_scrollbar.set_maximum(image_count-1)

        q_tape.mx_current_idx.listen(lambda _: rebuild_canvas())
        mx_mask_type.listen(lambda _: (rebuild_canvas(), self._q_tape.update_items()))

        rebuild_canvas = self._rebuild_canvas = lambda override_mask=None: self._on_rebuild_canvas(holder_mask_editor, override_mask=override_mask)
        rebuild_canvas()

        return True

    def _on_rebuild_canvas(self, holder : qx.QVBox, override_mask = None):
        self._save()
        holder.dispose_childs()
        self._q_mask_editor = None

        if (idx := self._q_tape.get_current_idx()) is not None:
            if (mask_type := self._mx_mask_type.get()) is not None:

                with self._q_tape_scrollbar_value_conn.disabled_scope():
                    self._q_tape_scrollbar.set_value(idx)

                err = None
                try:
                    image = self._image_ds.load_image(idx)

                    if override_mask is not None:
                        mask = override_mask
                    else:
                        if self._image_ds.has_mask(idx, mask_type):
                            mask = self._image_ds.load_mask(idx, mask_type)
                        else:
                            mask = None
                except Exception as e:
                    err = e

                if err is None:
                    self._q_mask_editor      = q_mask_editor = QxMaskEditorCanvas(image, mask=mask)
                    self._q_mask_editor_hash = q_mask_editor.get_state_hash() + (0 if override_mask is None else 1)

                    holder.add(q_mask_editor)
                else:
                    holder.add(qx.QLabel().set_font(qx.Font.FixedWidth).set_align(qx.Align.CenterF).set_text(f'@(Error) {err}'))
            else:
                holder.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxMaskEditor.No_mask_selected)'))
        else:
            holder.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxMaskEditor.No_image_selected)'))

    def _copy_mask(self):
        if (q_mask_editor := self._q_mask_editor) is not None:
            self._copied_mask = q_mask_editor.get_mask()

    def _paste_mask(self):
        if (copied_mask := self._copied_mask) is not None:
            self._rebuild_canvas(override_mask=copied_mask)

    def _delete_mask(self):
        if (idx := self._q_tape.get_current_idx()) is not None:
            if (mask_type := self._mx_mask_type.get()) is not None:
                try:
                    self._image_ds.delete_mask(idx, mask_type)
                except Exception as e:
                    self._popup_error(str(e))

                self._q_tape.update_item(idx)

    def _save(self, force = False) -> bool:
        """returns True if success"""
        if (idx := self._q_tape.get_current_idx()) is not None:
            if (mask_type := self._mx_mask_type.get()) is not None:
                if (q_mask_editor := self._q_mask_editor) is not None:

                    if self._q_mask_editor_hash != (state_hash := q_mask_editor.get_state_hash()) or force:
                        # Save if something changed or force

                        try:
                            self._image_ds.save_mask(idx, mask_type, q_mask_editor.get_mask())
                        except Exception as e:
                            self._popup_error(str(e))
                            return False

                        self._q_mask_editor_hash = state_hash
                        self._q_tape.update_item(idx)
        return True

    def _save_and_goto(self, idx : int):
        if self._save():
            self._q_tape.set_current_idx(idx)

    def _popup_error(self, info : str):
        ((wnd := qx.QWindow()).dispose_with(self)
                .set_window_size(300, 300)
                .set_window_flags(qx.WindowType.Window | qx.WindowType.WindowTitleHint | qx.WindowType.CustomizeWindowHint |  qx.WindowType.WindowStaysOnTopHint)
                .set_window_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.alert_circle_outline, qt.QColor(255,0,0)))
                .set_title('@(Error)')

                .add(qx.QTextEdit() .set_font(qx.Font.FixedWidth).set_read_only(True)
                                    .set_plain_text(info))
                .add(qx.QPushButton().v_compact().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.checkmark_done, qt.QColor(100,200,0)))
                        .inline(lambda btn: btn.mx_clicked.listen(lambda: wnd.dispose())))

                .show())

    @ax.task
    def _save_and_next(self, tg : ax.TaskGroup, diff : int):
        yield ax.attach_to(tg, cancel_all=True)


        for i in itertools.count():
            if (current_idx := self._q_tape.get_current_idx()) is not None:
                self._save_and_goto(current_idx+diff)

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)

    @ax.task
    def _save_and_next_mask(self, tg : ax.TaskGroup, forward : bool):
        yield ax.attach_to(tg, cancel_all=True)

        for i in itertools.count():

            if (idx := self._q_tape.mx_current_idx.get()) is not None:
                if (mask_type := self._mx_mask_type.get()) is not None:
                    if (next_idx := self._image_ds.get_next_image_id_with_mask(idx, mask_type, forward)) is not None:
                        self._save_and_goto(next_idx)

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)


    @ax.task
    def _tape_get_item_pixmap(self, idx : int, size : qt.QSize, image_ds : ImageDS) -> qt.QPixmap:
        yield ax.sleep(0)

        w, h = size.width(),size.height()

        caption_bg_color = qx.StyleColor.Midlight
        try:
            pixmap = qt.QPixmap_from_np(image_ds.load_image(idx).u8().HWC())

            if (mask_type := self._mx_mask_type.get()) is not None and \
                image_ds.has_mask(idx, mask_type):
                caption_bg_color = qt.QColor(0,50,100)

        except Exception as e:
            pixmap = qx.QIonIconDB.instance().pixmap(qx.IonIcon.alert_circle_outline, qt.QColor(255,0,0))

        caption = image_ds.get_image_name(idx)

        out_pixmap = qt.QPixmap(w, h)
        out_pixmap.fill(qt.QColor(0,0,0,0))
        qp = qt.QPainter(out_pixmap)


        image_rect = qt.QRect(0, 0, w, w)
        cap_rect = qt.QRect(0, h-16, w, 16)
        qp.fillRect(image_rect, qx.StyleColor.Midlight)
        qp.fillRect(cap_rect, caption_bg_color)

        fitted_image_rect = qt.QRect_fit_in(pixmap.rect(), image_rect)
        qp.drawPixmap(fitted_image_rect, pixmap)

        font = qx.QFontDB.instance().fixed_width()
        fm = qt.QFontMetrics(font)
        qp.setFont(font)
        qp.setPen(qx.StyleColor.Text)
        caption_text = fm.elidedText(caption, qt.Qt.TextElideMode.ElideLeft, cap_rect.width())
        qp.drawText(cap_rect, qt.Qt.AlignmentFlag.AlignCenter, caption_text)


        qp.end()

        return out_pixmap

