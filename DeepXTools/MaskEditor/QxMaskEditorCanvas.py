from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path

import numpy as np

from core import ax, qt, qx
from core.lib.image import NPImage

from .FMaskEditor import FMaskEditor


class QxMaskEditorCanvas(qx.QHBox):
    def __init__(self, image : NPImage, mask : NPImage = None):
        """
            image       HWC
            mask        HW1
        """
        super().__init__()
        self.__settings = qx.QSettings()

        H,W,_ = image.shape

        if mask is None:
            mask = NPImage(np.zeros((H,W), np.float32))
            
        mask = mask.grayscale().resize(W, H)

        self._mask       = mask.f32().HWC()
        self._mask_uint8 = mask.u8().HWC()

        self._image_pixmap = qt.QPixmap_from_np(image.bgr().HWC())

        self._buffer_overlay_image = np.zeros((H,W,4), np.uint8)
        self._buffer_overlay_mask = np.zeros((H,W,1), np.uint8)
        self._q_buffer_overlay_image = qt.QImage_from_np(self._buffer_overlay_image)

        self._bw_mode = False
        self._overlay_q_color = qt.QColor(0,255,0)
        self._overlay_opacity = 0.25

        self._cursor_base_image = qt.QImage(str(Path(__file__).parent / 'assets' / 'cursors' / 'cross_base.png'))
        self._cursor_overlay_image = qt.QImage(str(Path(__file__).parent / 'assets' / 'cursors' / 'cross_overlay.png'))

        self._fme = FMaskEditor().set_img_size(W, H)
        self._fme_undo = deque([self._fme])
        self._fme_redo = deque()

        self._qfme_qp = qt.QPainter()
        qfme = self._qfme = qx.QWidget()
        qfme.set_mouse_tracking(True)
        qfme.mx_resize.listen(lambda ev: self._fme_result(self._fme.set_cli_size(ev.size().width(), ev.size().height())))
        qfme.mx_mouse_move.listen(lambda ev: self._fme_result(self._fme.mouse_move(ev.pos().x(), ev.pos().y())))
        qfme.mx_mouse_press.listen(lambda ev: self._fme_result( self._fme.mouse_lbtn_down(ev.pos().x(), ev.pos().y()) ) if ev.button()==qt.Qt.MouseButton.LeftButton else
                                              self._fme_result( self._fme.mouse_mbtn_down(ev.pos().x(), ev.pos().y()) ) if ev.button()==qt.Qt.MouseButton.MiddleButton else
                                              ...)
        qfme.mx_mouse_release.listen(lambda ev: self._fme_result( self._fme.mouse_lbtn_up(ev.pos().x(), ev.pos().y()) ) if ev.button()==qt.Qt.MouseButton.LeftButton else
                                                self._fme_result( self._fme.mouse_mbtn_up(ev.pos().x(), ev.pos().y()) ) if ev.button()==qt.Qt.MouseButton.MiddleButton else
                                                ...)
        qfme.mx_wheel.listen(lambda ev: self._fme_result(self._fme.mouse_wheel(ev.position().x(), ev.position().y(), ev.angleDelta().y())))
        qfme.mx_paint.listen(self._qfme_paint_event)

        bw_mode_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Agrave), self)
        bw_mode_shortcut.mx_press.listen( lambda: self._set_bw_mode(not self._bw_mode))
        red_overlay_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_1), self)
        red_overlay_shortcut.mx_press.listen( lambda: self._set_overlay_color(qt.QColor(255,0,0)))
        green_overlay_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_2), self)
        green_overlay_shortcut.mx_press.listen( lambda: self._set_overlay_color(qt.QColor(0,255,0)))
        blue_overlay_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_3), self)
        blue_overlay_shortcut.mx_press.listen( lambda: self._set_overlay_color(qt.QColor(0,0,255)))
        opacity_0_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_4), self)
        opacity_0_shortcut.mx_press.listen( lambda: self._set_overlay_opacity(0.0))
        opacity_25_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_5), self)
        opacity_25_shortcut.mx_press.listen( lambda: self._set_overlay_opacity(0.25))
        opacity_50_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_6), self)
        opacity_50_shortcut.mx_press.listen( lambda: self._set_overlay_opacity(0.50))
        opacity_75_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_7), self)
        opacity_75_shortcut.mx_press.listen( lambda: self._set_overlay_opacity(0.75))

        icon_size = qx.Size.L
        overlay_pixmap = qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'overlay.png')
        overlay_bw_mode_pixmap = qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'overlay_bw_mode.png')

        bw_mode_btn = qx.QPushButton().set_icon(qt.QIcon(overlay_bw_mode_pixmap)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.BW_mode) (`)')
        bw_mode_btn.mx_pressed.listen(lambda: bw_mode_shortcut.press())
        bw_mode_btn.mx_released.listen(lambda: bw_mode_shortcut.release())

        red_overlay_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(255,0,0)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Red_overlay) (1)')
        red_overlay_btn.mx_pressed.listen(lambda: red_overlay_shortcut.press())
        red_overlay_btn.mx_released.listen(lambda: red_overlay_shortcut.release())

        green_overlay_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(0,255,0)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Green_overlay) (2)')
        green_overlay_btn.mx_pressed.listen(lambda: green_overlay_shortcut.press())
        green_overlay_btn.mx_released.listen(lambda: green_overlay_shortcut.release())

        blue_overlay_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(0,0,255)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Blue_overlay) (3)')
        blue_overlay_btn.mx_pressed.listen(lambda: blue_overlay_shortcut.press())
        blue_overlay_btn.mx_released.listen(lambda: blue_overlay_shortcut.release())

        opacity_0_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(255,255,255,0)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Opacity) 0% (4)')
        opacity_0_btn.mx_pressed.listen(lambda: opacity_0_shortcut.press())
        opacity_0_btn.mx_released.listen(lambda: opacity_0_shortcut.release())

        opacity_25_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(255,255,255,64)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Opacity) 25% (5)')
        opacity_25_btn.mx_pressed.listen(lambda: opacity_25_shortcut.press())
        opacity_25_btn.mx_released.listen(lambda: opacity_25_shortcut.release())

        opacity_50_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(255,255,255,128)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Opacity) 50% (6)')
        opacity_50_btn.mx_pressed.listen(lambda: opacity_50_shortcut.press())
        opacity_50_btn.mx_released.listen(lambda: opacity_50_shortcut.release())

        opacity_75_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(overlay_pixmap, qt.QColor(255,255,255,196)))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Opacity) 75% (7)')
        opacity_75_btn.mx_pressed.listen(lambda: opacity_75_shortcut.press())
        opacity_75_btn.mx_released.listen(lambda: opacity_75_shortcut.release())

        undo_redo_tg = ax.TaskGroup().dispose_with(self)
        undo_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Z), self)
        undo_shortcut.mx_press.listen(lambda: self._undo_fme(undo_redo_tg))
        undo_shortcut.mx_release.listen(lambda: undo_redo_tg.cancel_all())

        redo_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier|qt.Qt.KeyboardModifier.ShiftModifier, qt.Qt.Key.Key_Z), self)
        redo_shortcut.mx_press.listen(lambda: self._redo_fme(undo_redo_tg))
        redo_shortcut.mx_release.listen(lambda: undo_redo_tg.cancel_all())

        edit_points_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.KeyboardModifier.ControlModifier, qt.Qt.Key.Key_Control), self)
        edit_points_shortcut.mx_press.listen(lambda: ( (conn := edit_points_btn_toggled_conn).disable(), edit_points_btn.set_checked(True), conn.enable(), self._fme_result(self._fme.set_edit_poly_mode(FMaskEditor.EditPolyMode.PT_ADD_DEL))))
        edit_points_shortcut.mx_release.listen(lambda: ( (conn := edit_points_btn_toggled_conn).disable(), edit_points_btn.set_checked(False), conn.enable(), self._fme_result(self._fme.set_edit_poly_mode(FMaskEditor.EditPolyMode.PT_MOVE))))

        edit_points_btn = qx.QPushButton().set_checkable(True).set_checked(False).set_icon(qt.QIcon(qt.QPixmap_colorized(qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'edit_points.png'), qx.StyleColor.ButtonText))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Add_delete_points). (@(Hold) CTRL)')
        edit_points_btn_toggled_conn = edit_points_btn.mx_toggled.listen(lambda toggled: edit_points_shortcut.press() if toggled else edit_points_shortcut.release() )

        undo_btn = qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.arrow_undo, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Undo_action) (Ctrl-Z)')
        undo_btn.mx_pressed.listen(lambda: undo_shortcut.press())
        undo_btn.mx_released.listen(lambda: undo_shortcut.release())

        redo_btn = qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.arrow_redo, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Redo_action) (Shift-Ctrl-Z)')
        redo_btn.mx_pressed.listen(lambda: redo_shortcut.press())
        redo_btn.mx_released.listen(lambda: redo_shortcut.release())

        apply_fill_poly_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_Q), self)
        apply_fill_poly_shortcut.mx_press.listen( lambda: self._fme_result( self._fme.apply_state_poly(FMaskEditor.PolyApplyType.INCLUDE) ))
        apply_cut_poly_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_W), self)
        apply_cut_poly_shortcut.mx_press.listen( lambda: self._fme_result( self._fme.apply_state_poly(FMaskEditor.PolyApplyType.EXCLUDE) ))
        delete_state_poly_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_E), self)
        delete_state_poly_shortcut.mx_press.listen(lambda: self._fme_result( self._fme.delete_state_poly() ))


        apply_fill_poly_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'include_poly.png'), qx.StyleColor.ButtonText))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Apply_fill_poly) (Q)')
        apply_fill_poly_btn.mx_pressed.listen( lambda: apply_fill_poly_shortcut.press())
        apply_fill_poly_btn.mx_released.listen( lambda: apply_fill_poly_shortcut.release())
        apply_cut_poly_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'exclude_poly.png'), qx.StyleColor.ButtonText))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Apply_cut_poly) (W)')
        apply_cut_poly_btn.mx_pressed.listen( lambda: apply_cut_poly_shortcut.press())
        apply_cut_poly_btn.mx_released.listen( lambda: apply_cut_poly_shortcut.release())
        delete_state_poly_btn = qx.QPushButton().set_icon(qt.QIcon(qt.QPixmap_colorized(qt.QPixmap(Path(__file__).parent / 'assets' / 'icons' / 'delete_poly.png'), qx.StyleColor.ButtonText))).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Delete_poly) (E)')
        delete_state_poly_btn.mx_pressed.listen( lambda: delete_state_poly_shortcut.press())
        delete_state_poly_btn.mx_released.listen( lambda: delete_state_poly_shortcut.release())

        center_on_cursor_shortcut = qx.QShortcut( qt.QKeyCombination(qt.Qt.Key.Key_C), self)
        center_on_cursor_shortcut.mx_press.listen(lambda: self._fme_result( self._fme.center_on_cursor() ))
        center_on_cursor_btn = qx.QPushButton().set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.move, qx.StyleColor.ButtonText)).set_icon_size(icon_size).set_tooltip('@(QxMaskEditorCanvas.Center_at_cursor) (C)')
        center_on_cursor_btn.mx_pressed.listen( lambda: center_on_cursor_shortcut.press())
        center_on_cursor_btn.mx_released.listen( lambda: center_on_cursor_shortcut.release())

        (self
            .add(qx.QVBox()
                    .add(undo_btn).add(redo_btn).add_spacer(8)
                    .add(center_on_cursor_btn).add_spacer(8)
                    .add(edit_points_btn).add(apply_fill_poly_btn)
                    .add(apply_cut_poly_btn).add(delete_state_poly_btn)
                    .v_compact().h_compact())
            .add(qfme)
            .add(qx.QVBox()
                    .add(bw_mode_btn).add_spacer(8)
                    .add(red_overlay_btn).add(green_overlay_btn).add(blue_overlay_btn).add_spacer(8)
                    .add(opacity_0_btn).add(opacity_25_btn).add(opacity_50_btn).add(opacity_75_btn)
                    .v_compact().h_compact())
        )

        self._update_cursor()

    def get_mask(self) -> NPImage:
        """returns current HW1 mask"""
        return NPImage(self._fme.bake_polys(self._mask.copy(), [1.0], [0.0]))

    def get_state_hash(self) -> int:
        return id(self._fme_undo[-1])

    def _set_bw_mode(self, bw_mode : bool):
        if self._bw_mode != bw_mode:
            self._bw_mode = bw_mode
            self.__settings['bw_mode'] = bw_mode
            self._qfme.update()

    def _set_overlay_color(self, color : qt.QColor):
        if self._overlay_q_color != color:
            self._overlay_q_color = color
            self.__settings['overlay_color'] = color.getRgb()
            self._update_cursor()
            self._qfme.update()

    def _set_overlay_opacity(self, opacity : float):
        if self._overlay_opacity != opacity:
            self._overlay_opacity = opacity
            self.__settings['overlay_opacity'] = opacity
            self._qfme.update()

    def _settings_event(self, settings : qx.QSettings):
        super()._settings_event(settings)
        self.__settings = settings

        if (bw_mode := settings.get('bw_mode', None)) is not None:
           self._set_bw_mode(bw_mode)
        if (overlay_color := settings.get('overlay_color', None)) is not None:
            self._set_overlay_color(qt.QColor(*overlay_color))
        if (overlay_opacity := settings.get('overlay_opacity', None)) is not None:
            self._set_overlay_opacity(overlay_opacity)


    def _update_cursor(self):
        fme = self._fme

        if fme.get_drag_type() == FMaskEditor.DragType.VIEW_LOOK:
            cursor = qt.Qt.CursorShape.ClosedHandCursor
        else:
            cursor = qt.QPixmap(self._cursor_base_image)
            qp = qt.QPainter(cursor)

            qp.drawImage(0,0, qt.QImage_colorized(self._cursor_overlay_image, self._overlay_q_color))
            qp.end()

            cursor = qt.QCursor(cursor)

        self._qfme.set_cursor(cursor)

    @ax.task
    def _undo_fme(self, tg : ax.TaskGroup):
        yield ax.attach_to(tg, cancel_all=True)

        for i in itertools.count():
            if len(self._fme_undo) > 1:
                fme = self._fme

                undo_fme = self._fme_undo.pop()
                self._fme_redo.appendleft(fme)

                self._fme = (undo_fme   .set_view_scale(fme.get_view_scale())
                                        .set_view_look_img_pt(fme.get_view_look_img_pt())
                                        .set_mouse_cli_pt(fme.get_mouse_cli_pt()) )
                self._update_cursor()
                self._qfme.update()

            yield ax.sleep(0.4 if i == 0 else 0.1)

    @ax.task
    def _redo_fme(self, tg : ax.TaskGroup):
        yield ax.attach_to(tg, cancel_all=True)

        for i in itertools.count():
            if len(self._fme_redo) != 0:
                fme = self._fme

                redo_fme = self._fme_redo.popleft()
                self._fme_undo.append(fme)

                self._fme = (redo_fme   .set_view_scale(fme.get_view_scale())
                                        .set_view_look_img_pt(fme.get_view_look_img_pt())
                                        .set_mouse_cli_pt(fme.get_mouse_cli_pt()) )
                self._update_cursor()
                self._qfme.update()

            yield ax.sleep(0.4 if i == 0 else 0.1)

    def _fme_result(self, new_fme :FMaskEditor):
        fme, self._fme = self._fme, new_fme

        upd = False
        upd = upd or new_fme.is_changed_cli_size(fme) \
                  or new_fme.is_changed_img_size(fme) \
                  or (new_fme.is_changed_mouse_cli_pt(fme) and (new_fme.get_state() in [FMaskEditor.State.DRAW_POLY,FMaskEditor.State.EDIT_POLY])) \
                  or new_fme.is_changed_view_look_img_pt(fme) \
                  or new_fme.is_changed_view_scale(fme) \
                  or new_fme.is_changed_polys(fme) \
                  or new_fme.is_changed_state(fme) \
                  or new_fme.is_changed_state_poly(fme) \
                  or new_fme.is_changed_edit_poly_mode(fme)

        if new_fme.is_changed_view_scale(fme) or \
           new_fme.is_activated_center_on_cursor(fme):
            qt.QCursor.setPos ( self._qfme.map_to_global(qt.QPoint_from_np(new_fme.get_mouse_cli_pt())) )

        if new_fme.is_changed_state(fme) or \
           new_fme.is_changed_drag_type(fme):
            self._update_cursor()

        if new_fme.is_changed_for_undo(fme):
            self._fme_undo.append(fme)
            self._fme_redo = deque()

        if upd:
            self._qfme.update()


    def _qfme_paint_event(self, ev : qt.QPaintEvent):
        fme = self._fme

        mouse_cli_pt = fme.get_mouse_cli_pt()
        state = fme.get_state()


        overlay_bw_mode = self._bw_mode
        overlay_q_color = overlay_image_q_color = self._overlay_q_color
        overlay_image_opacity = self._overlay_opacity
        if overlay_bw_mode:
            overlay_image_q_color = qt.QColor(255,255,255)
            overlay_image_opacity = 1.0

        overlay_mask_buffer = self._buffer_overlay_mask
        overlay_mask_buffer[...] = self._mask_uint8
        fme.bake_polys(overlay_mask_buffer, [255], [0])

        overlay_image_buffer = self._buffer_overlay_image
        overlay_mask_buffer = (overlay_mask_buffer*overlay_image_opacity).astype(np.uint8)
        overlay_image_buffer[...,0:3] = overlay_image_q_color.getRgb()[2::-1]
        overlay_image_buffer[...,3:4] = overlay_mask_buffer

        qp = self._qfme_qp
        qp.begin(self._qfme.get_q_widget())
        qp.setRenderHint(qt.QPainter.RenderHint.SmoothPixmapTransform)

        mat = fme.get_img_to_cli_mat()
        qp.setTransform(qt.QTransform(  mat[0,0], mat[1,0], 0,
                                        mat[0,1], mat[1,1], 0,
                                        mat[0,2], mat[1,2], 1))

        if overlay_bw_mode:
            qp.fillRect(self._image_pixmap.rect(), qt.QColor(0,0,0))
        else:
            qp.drawPixmap(0,0, self._image_pixmap)

        qp.drawImage(0,0, self._q_buffer_overlay_image)

        # Overlays
        qp.resetTransform()
        qp.setRenderHint(qt.QPainter.RenderHint.Antialiasing)

        if state in [FMaskEditor.State.DRAW_POLY, FMaskEditor.State.EDIT_POLY]:
            poly = fme.get_state_poly()

            poly_line_path = qt.QPainterPath()
            overlay_path = qt.QPainterPath()
            overlay_black_path = qt.QPainterPath()


            poly_line_path.addPolygon(qt.QPolygon([ qt.QPoint_from_np(pt) for pt in fme.map_img_to_cli(poly.points)]))
            if state == FMaskEditor.State.DRAW_POLY:
                ## Line from last point to mouse
                poly_line_path.lineTo( qt.QPoint_from_np(mouse_cli_pt) )

            if poly.points_count >= 2:
                # Closing poly line
                poly_line_path.lineTo( qt.QPointF_from_np(fme.map_img_to_cli(poly.points[0])))


            if state == FMaskEditor.State.DRAW_POLY:
                if poly.points_count >= 3 and fme.get_mouse_state_poly_pt_id() == 0:
                    # Circle around first poly point
                    overlay_path.addEllipse(qt.QPoint_from_np(fme.map_img_to_cli(poly.points[0])), 10, 10)
                    overlay_black_path.addEllipse(qt.QPoint_from_np(fme.map_img_to_cli(poly.points[0])), 11, 11)
            elif state == FMaskEditor.State.EDIT_POLY:
                mouse_at_state_poly = fme.is_mouse_at_state_poly()
                mouse_state_poly_pt_id = fme.get_mouse_state_poly_pt_id()


                pt_select_rad = fme.get_pt_select_radius()

                for pt_id, pt in enumerate(poly.points):
                    cli_pt = fme.map_img_to_cli(pt)

                    if fme.get_edit_poly_mode() == FMaskEditor.EditPolyMode.PT_ADD_DEL:
                        if pt_id == mouse_state_poly_pt_id:
                            poly_line_path.moveTo( qt.QPoint_from_np(cli_pt + np.float32([-pt_select_rad*2,pt_select_rad*2])) )
                            poly_line_path.lineTo( qt.QPoint_from_np(cli_pt + np.float32([pt_select_rad*2,-pt_select_rad*2])) )
                            poly_line_path.moveTo( qt.QPoint_from_np(cli_pt + np.float32([-pt_select_rad*2,-pt_select_rad*2])) )
                            poly_line_path.lineTo( qt.QPoint_from_np(cli_pt + np.float32([pt_select_rad*2,pt_select_rad*2])) )

                    if mouse_at_state_poly or mouse_state_poly_pt_id == pt_id:
                        overlay_path.addEllipse(qt.QPoint_from_np(fme.map_img_to_cli(pt)), 10, 10)
                        overlay_black_path.addEllipse(qt.QPoint_from_np(fme.map_img_to_cli(pt)), 11, 11)

                if mouse_state_poly_pt_id is None and fme.get_edit_poly_mode() == FMaskEditor.EditPolyMode.PT_ADD_DEL:

                    if (edge_id_cli_pt := fme.get_mouse_state_poly_edge_id_cli_pt()) is not None:
                        edge_id, cli_pt = edge_id_cli_pt
                        overlay_path.addEllipse(qt.QPoint_from_np(cli_pt), 10, 10)
                        overlay_black_path.addEllipse(qt.QPoint_from_np(cli_pt), 11, 11)
                        overlay_path.moveTo( qt.QPoint_from_np(cli_pt + np.float32([0,-pt_select_rad])) )
                        overlay_path.lineTo( qt.QPoint_from_np(cli_pt + np.float32([0,pt_select_rad])) )
                        overlay_path.moveTo( qt.QPoint_from_np(cli_pt + np.float32([-pt_select_rad,0])) )
                        overlay_path.lineTo( qt.QPoint_from_np(cli_pt + np.float32([pt_select_rad,0])) )

                fme.get_edit_poly_mode()

            qp.setBrush(qt.QBrush())
            qp.setPen(qt.QPen(overlay_q_color, 2.0))
            qp.drawPath(poly_line_path)
            qp.setPen(overlay_q_color)
            qp.drawPath(overlay_path)
            qp.setPen(qt.QColor(0,0,0))
            qp.drawPath(overlay_black_path)

        qp.end()










