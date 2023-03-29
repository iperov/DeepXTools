from __future__ import annotations

import itertools
import re
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import numpy as np

from common.ImageDS import ImageDS
from core import ax, mx, qt, qx
from core.lib.image import LSHash64, NPImage

from .QxMaskEditorCanvas import QxMaskEditorCanvas


class QxMaskEditor(qx.QVBox):
    def __init__(self, q_menu_bar : qx.QMenuBar):
        super().__init__()
        # L0
        self._q_menu_bar = q_menu_bar
        self._q_holder = qx.QVBox()
        self._q_holder_top_bar = qx.QHBox()

        mx_path_state = self._mx_path_state = \
                    mx.PathState(   config=mx.PathStateConfig(dir_only=True, desc='Sequence of images'),
                                    on_open=lambda path: self._mx_path_dlg_open(path),
                                    on_close=lambda: self._mx_path_dlg_close()).dispose_with(self)

        # L1
        self._L1_initialized = False
        self._image_ds : ImageDS = None

        self._tape_tg : ax.TaskGroup = None
        self._sort_tg : ax.TaskGroup = None

        self._f_idx_to_ds_idx : Callable = None
        self._f_ds_idx_to_idx : Callable = None

        self._mx_mask_type : mx.SingleChoice[str|None] = None

        self._q_tape : qx.QTapeCachedItemView = None
        self._q_tape_scrollbar : qx.QScrollBar = None
        self._q_tape_scrollbar_value_conn : mx.EventConnection = None
        self._q_sort_progress_bar : qx.QProgressBar = None
        self._q_holder_me : qx.QVBox = None
        self._q_keep_view : qx.QCheckBox = None
        self._f_rebuild_canvas : Callable = None

        # L2
        self._L2_initialized = False
        self._q_me_canvas : QxMaskEditorCanvas = None
        self._q_me_canvas_hash : int = None
        self._q_me_view_scale : float|None = None
        self._q_me_view_look_img_pt : np.ndarray|None = None

        #
        (self   .add(qx.QHBox()
                        .add(qx.QMxPathState(mx_path_state))
                        .add(self._q_holder_top_bar).v_compact() )
                .add(self._q_holder) )

    def __dispose__(self):
        # L0
        self._mx_path_state.close()
        super().__dispose__()

    def _mx_path_dlg_close(self):
        # L1
        self._save()
        self._q_holder.dispose_childs()
        self._q_holder_top_bar.dispose_childs()

    def _mx_path_dlg_open(self, path : Path) -> bool:
        # L0
        self._q_holder.dispose_childs()
        self._q_holder_top_bar.dispose_childs()

        # + L1
        try:
            image_ds = self._image_ds = ImageDS.open(path)
        except Exception as e:
            self._q_holder.add(qx.QLabel().set_align(qx.Align.CenterE).set_text(f'@(Error): {str(e)}'))
            return False

        L1_disp = qx.QObject().set_parent(self._q_holder).call_on_dispose(lambda: setattr(self, '_L1_initialized', False))

        self._tape_tg = ax.TaskGroup().dispose_with(L1_disp)
        self._sort_tg = ax.TaskGroup().dispose_with(L1_disp)

        self._f_idx_to_ds_idx = lambda idx: idx
        self._f_ds_idx_to_idx = lambda idx: idx

        self._mx_mask_type = mx_mask_type = mx.SingleChoice[str|None](
                                                avail_mask_types[0] if len(avail_mask_types := image_ds.get_mask_types()) != 0 else None,
                                                avail=lambda: [None]+image_ds.get_mask_types()).dispose_with(L1_disp)

        self._q_tape = q_tape = (qx.QTapeCachedItemView(task_get_item_pixmap=lambda idx, rect: self._tape_get_item_pixmap(idx, rect, image_ds))
                                    .set_item_size(64, 64+16+1))

        self._q_tape_scrollbar = q_tape_scrollbar = qx.QScrollBar().set_minimum(0)
        self._q_tape_scrollbar_value_conn = q_tape_scrollbar.mx_value.listen(lambda idx: self._save_and_goto(idx) )
        self._q_sort_progress_bar = qx.QProgressBar()
        self._q_holder_me = qx.QVBox()
        self._q_keep_view = qx.QCheckBox().set_text('@(QxMaskEditor.Keep_view)')

        prev_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_A), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next(-1)),
                                shortcut.mx_release.listen(lambda: self._tape_tg.cancel_all())))

        next_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.Key.Key_D), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next(1)),
                                shortcut.mx_release.listen(lambda: self._tape_tg.cancel_all())))

        prev_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_A), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next_mask(forward=False)),
                                shortcut.mx_release.listen(lambda: self._tape_tg.cancel_all())))

        next_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_D), q_tape).inline(lambda shortcut: (
                                shortcut.mx_press.listen(lambda: self._save_and_next_mask(forward=True)),
                                shortcut.mx_release.listen(lambda: self._tape_tg.cancel_all())))

        force_save_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_S), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._save(force=True)))

        delete_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_X), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._delete_mask() ))

        copy_image_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier|qt.Qt.KeyboardModifier.ShiftModifier, qt.Qt.Key.Key_C), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._copy_image()))

        copy_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_C), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._copy_mask()))

        paste_mask_shortcut = qx.QShortcut(qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_V), q_tape).inline(lambda shortcut:
                                shortcut.mx_press.listen(lambda: self._paste_mask()))


        self._q_menu_bar.add( qx.QMenu().dispose_with(L1_disp)
                                .set_title('@(QxMaskEditor.Thumbnail_size)')
                                .inline(lambda menu:
                                        [menu.add( qx.QAction().set_text(str(32*i)).inline(lambda act: act.mx_triggered.listen(lambda i=i: q_tape.set_item_size(32*i, 32*i+16+1)) ))
                                            for i in range(1, 9) ]))

        self._q_holder_top_bar.add(qx.QHBox()
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
        icon_size = qx.Size.L

        (self._q_holder.add(qx.QSplitter().set_orientation(qx.Orientation.Vertical).set_default_sizes([9999,1])
                        .add(qx.QVBox()
                                .add(self._q_holder_me))
                        .add(qx.QVBox()
                                .add(q_tape)
                                .add(q_tape_scrollbar.v_compact())
                                .add(self._q_sort_progress_bar.hide())



                                .add(qx.QHBox().v_compact()
                                        .add(qx.QHBox().v_compact()

                                            .add(qx.QLabel().set_text('@(QxMaskEditor.Sort_by)'))

                                            .add((q_sort_by := qx.QComboBox())
                                                .inline(lambda c: [
                                                    c.add_item({ QxMaskEditor._SortBy.Name : '@(QxMaskEditor._SortBy.Name)',
                                                                 QxMaskEditor._SortBy.PerceptualDissimilarity : '@(QxMaskEditor._SortBy.PerceptualDissimilarity)',
                                                                }[x], data=x) for x in QxMaskEditor._SortBy]))
                                            .add_spacer(4)
                                            .add(self._q_keep_view), align=qx.Align.CenterH)

                                        .add_spacer(16)

                                        .add(qx.QPushButton().set_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'copy_image.png'))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Copy_image) (CTRL+SHIFT+С)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: copy_image_shortcut.press()), btn.mx_released.listen(lambda: copy_image_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'copy_mask.png'))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Copy_mask) (CTRL+С)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: copy_mask_shortcut.press()), btn.mx_released.listen(lambda: copy_mask_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'paste_mask.png'))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Paste_mask) (CTRL+X)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: paste_mask_shortcut.press()), btn.mx_released.listen(lambda: paste_mask_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'save_mask.png'))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Force_save_mask) (CTRL+S)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: force_save_shortcut.press()), btn.mx_released.listen(lambda: force_save_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qt.QIcon(str(Path(__file__).parent / 'assets' / 'icons' / 'delete_mask.png'))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Delete_mask) (CTRL+X)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen( lambda: delete_mask_shortcut.press()), btn.mx_released.listen(lambda: delete_mask_shortcut.release()))))

                                        .add_spacer(16)

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_skip_back, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Save_prev_img_mask) (CTRL+A)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: prev_mask_shortcut.press()), btn.mx_released.listen(lambda: prev_mask_shortcut.release()))))
                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_back, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Save_prev_img) (A)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: prev_shortcut.press()), btn.mx_released.listen(lambda: prev_shortcut.release()))))
                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_forward, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Save_next_img) (D)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: next_shortcut.press()), btn.mx_released.listen(lambda: next_shortcut.release()))))

                                        .add(qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.play_skip_forward, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditor.Save_next_img_with_mask) (CTRL+D)')
                                                            .inline(lambda btn: (btn.mx_pressed.listen(lambda: next_mask_shortcut.press()), btn.mx_released.listen(lambda: next_mask_shortcut.release()))))


                                        ))))

        q_tape.set_item_count(image_count := image_ds.image_count)
        q_tape.mx_current_idx.listen(lambda _: self._f_rebuild_canvas())
        q_tape_scrollbar.set_maximum(image_count-1)
        mx_mask_type.listen(lambda _: (self._f_rebuild_canvas(), self._q_tape.update_items()))

        rebuild_canvas = self._f_rebuild_canvas = lambda override_mask=None: self._rebuild_canvas(override_mask=override_mask)
        self._L1_initialized = True

        rebuild_canvas()

        q_sort_by.mx_current_index.reflect(lambda idx: self._sort_by(q_sort_by.get_item_data(idx)) if idx is not None else ...)

        return True

    def _rebuild_canvas(self, override_mask : NPImage = None):
        # L1
        self._save()
        holder = self._q_holder_me.dispose_childs()
        L2_disp = qx.QObject().set_parent(holder).call_on_dispose(lambda: setattr(self, '_L2_initialized', False))

        if (idx := self._q_tape.get_current_idx()) is not None:
            if (mask_type := self._mx_mask_type.get()) is not None:

                with self._q_tape_scrollbar_value_conn.disabled_scope():
                    self._q_tape_scrollbar.set_value(idx)

                err = None
                try:
                    act_idx = self._f_idx_to_ds_idx(idx)

                    image = self._image_ds.load_image(act_idx)

                    if override_mask is not None:
                        mask = override_mask
                    else:
                        if self._image_ds.has_mask(act_idx, mask_type):
                            mask = self._image_ds.load_mask(act_idx, mask_type)
                        else:
                            mask = None
                except Exception as e:
                    err = e

                if err is None:
                    # + L2

                    self._q_me_canvas      = q_me_canvas = QxMaskEditorCanvas(image, mask=mask)
                    self._q_me_canvas_hash = q_me_canvas.get_state_hash() + (0 if override_mask is None else 1)

                    if self._q_keep_view.is_checked():
                        if (view_scale := self._q_me_view_scale) is not None:
                            q_me_canvas.set_view_scale(view_scale)

                        if (view_look_img_pt := self._q_me_view_look_img_pt) is not None:
                            q_me_canvas.set_view_look_img_pt(view_look_img_pt)

                    self._L2_initialized = True

                    holder.add(q_me_canvas)
                else:
                    holder.add(qx.QLabel().set_font(qx.Font.FixedWidth).set_align(qx.Align.CenterF).set_text(f'@(Error) {err}'))
            else:
                holder.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxMaskEditor.No_mask_selected)'))
        else:
            holder.add(qx.QLabel().set_align(qx.Align.CenterF).set_text('@(QxMaskEditor.No_image_selected)'))

    @ax.task
    def _sort_by(self, sort_by : QxMaskEditor._SortBy ):
        # L1
        yield ax.attach_to(self._sort_tg, cancel_all=True)

        if sort_by == QxMaskEditor._SortBy.Name:
            self._f_idx_to_ds_idx = lambda idx: idx
            self._f_ds_idx_to_idx = lambda idx: idx
        elif sort_by == QxMaskEditor._SortBy.PerceptualDissimilarity:
            progress_bar = self._q_sort_progress_bar

            image_ds = self._image_ds

            progress_bar.show()
            progress_bar.set_minimum(0).set_maximum(image_ds.image_count-1).set_format(f'%v / %m')

            err = None
            hashes = []
            try:
                for i in range(image_ds.image_count):
                    npi = image_ds.load_image(i)

                    hashes.append(npi.get_ls_hash64())

                    progress_bar.set_value(i)
                    yield ax.sleep(0)
            except Exception as e:
                err = e

            if err is None:
                sorted_idxs = LSHash64.sorted_by_dissim(hashes)
                rev_sorted_idxs = { sorted_idxs[idx] : idx for idx in range(len(sorted_idxs)) }

                self._f_idx_to_ds_idx = lambda idx: sorted_idxs[idx]
                self._f_ds_idx_to_idx = lambda idx: rev_sorted_idxs[idx]

            progress_bar.hide()

        self._q_tape.update_items()
        self._f_rebuild_canvas()

    def _copy_image(self):
        # L1
        if self._L2_initialized:
            qx.QApplication.instance().get_clipboard().set_image( qt.QImage_from_np(self._q_me_canvas.get_image().bgr().HWC()) )

    def _copy_mask(self):
        # L1
        if self._L2_initialized:
            qx.QApplication.instance().get_clipboard().set_image( qt.QImage_from_np(self._q_me_canvas.get_mask().grayscale().HWC()) )

    def _paste_mask(self):
        # L1
        if (image := qx.QApplication.instance().get_clipboard().get_image()) is not None:
            image = NPImage(qt.QImage_to_np(image, qt.QImage.Format.Format_Grayscale8 ))
            self._f_rebuild_canvas(override_mask=image)

    def _delete_mask(self):
        # L1
        if (idx := self._q_tape.get_current_idx()) is not None:
            if (mask_type := self._mx_mask_type.get()) is not None:
                try:
                    self._image_ds.delete_mask(self._f_idx_to_ds_idx(idx), mask_type)
                except Exception as e:
                    self._popup_error(str(e))

                self._q_tape.update_item(idx)

    def _save(self, force = False) -> bool:
        """returns True if success"""
        # L1
        if self._L2_initialized:
            q_me_canvas = self._q_me_canvas
            self._q_me_view_scale = q_me_canvas.get_view_scale()
            self._q_me_view_look_img_pt = q_me_canvas.get_view_look_img_pt()

            if (idx := self._q_tape.get_current_idx()) is not None:
                if (mask_type := self._mx_mask_type.get()) is not None:


                    if self._q_me_canvas_hash != (state_hash := q_me_canvas.get_state_hash()) or force:
                        # Save if something changed or force
                        try:
                            self._image_ds.save_mask(self._f_idx_to_ds_idx(idx), mask_type, q_me_canvas.get_mask())
                        except Exception as e:
                            self._popup_error(str(e))
                            return False

                        self._q_me_canvas_hash = state_hash

                        self._q_tape.update_item(idx)
        return True

    def _save_and_goto(self, idx : int):
        # L1
        if self._save():
            self._q_tape.set_current_idx(idx)

    def _popup_error(self, info : str):
        # L0
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
    def _save_and_next(self, diff : int):
        # L1
        yield ax.attach_to(self._tape_tg, cancel_all=True)

        for i in itertools.count():
            if (current_idx := self._q_tape.get_current_idx()) is not None:
                self._save_and_goto(current_idx+diff)

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)

    @ax.task
    def _save_and_next_mask(self, forward : bool):
        # L1
        yield ax.attach_to(self._tape_tg, cancel_all=True)
        image_ds = self._image_ds

        for i in itertools.count():

            if (idx := self._q_tape.mx_current_idx.get()) is not None:
                if (mask_type := self._mx_mask_type.get()) is not None:

                    for next_idx in range(idx+1, image_ds.image_count) if forward else \
                                    range(idx-1, -1,-1):
                        if image_ds.has_mask(self._f_idx_to_ds_idx(next_idx), mask_type):
                            self._save_and_goto(next_idx)
                            break

            if i == 0:
                yield ax.sleep(0.5)
            else:
                yield ax.sleep(0.05)



    @ax.task
    def _tape_get_item_pixmap(self, idx : int, size : qt.QSize, image_ds : ImageDS) -> qt.QPixmap:
        # L1
        yield ax.sleep(0)

        w, h = size.width(),size.height()

        caption_bg_color = qx.StyleColor.Midlight

        act_idx = self._f_idx_to_ds_idx(idx)
        try:
            pixmap = qt.QPixmap_from_np(image_ds.load_image(act_idx).u8().HWC())

            if (mask_type := self._mx_mask_type.get()) is not None and \
                image_ds.has_mask(act_idx, mask_type):
                caption_bg_color = qt.QColor(0,50,100)

        except Exception as e:
            pixmap = qx.QIonIconDB.instance().pixmap(qx.IonIcon.alert_circle_outline, qt.QColor(255,0,0))

        caption = image_ds.get_image_name(act_idx)

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

    class _SortBy(Enum):
        Name = auto()
        PerceptualDissimilarity = auto()