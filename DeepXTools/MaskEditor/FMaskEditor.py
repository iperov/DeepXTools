from __future__ import annotations

from enum import Enum, auto
from typing import Sequence, Tuple

import cv2
import numpy as np
import numpy.linalg as npla

from core.lib.math import Affine2DMat, sd


class FMaskEditor:
    class State(Enum):
        NONE = auto()
        DRAW_POLY = auto()
        EDIT_POLY = auto()
        #VIEW_BAKED = auto()
        #VIEW_XSEG_MASK = auto()

    class DragType(Enum):
        NONE = auto()
        VIEW_LOOK = auto()
        STATE_POLY_PT = auto()

    class EditPolyMode(Enum):
        NONE = auto()
        PT_MOVE = auto()
        PT_ADD_DEL = auto()

    class PolyApplyType(Enum):
        INCLUDE = auto()
        EXCLUDE = auto()

    def __init__(self, other : FMaskEditor = None):
        """Functional core of Mask Editor"""
        self._pt_select_radius = other._pt_select_radius if other is not None else 8

        self._cli_size : np.ndarray     = other._cli_size if other is not None else np.array([320,240], np.int32)
        self._img_size : np.ndarray     = other._img_size if other is not None else np.array([320,240], np.int32)
        self._mouse_cli_pt : np.ndarray = other._mouse_cli_pt if other is not None else np.array([0,0], np.int32)

        self._view_look_img_pt : np.ndarray = other._view_look_img_pt if other is not None else None
        self._view_scale : float            = other._view_scale if other is not None else None

        self._drag_type : FMaskEditor.DragType     = other._drag_type if other is not None else FMaskEditor.DragType.NONE
        self._drag_start_cli_pt : np.ndarray       = other._drag_start_cli_pt if other is not None else None
        self._drag_start_view_look_pt : np.ndarray = other._drag_start_view_look_pt if other is not None else None
        self._drag_state_poly_pt_id : int          = other._drag_state_poly_pt_id if other is not None else None
        self._drag_state_poly_pt : np.ndarray      = other._drag_state_poly_pt if other is not None else None

        self._polys : Tuple[FPoly]          = other._polys if other is not None else ()
        self._state : FMaskEditor.State = other._state if other is not None else FMaskEditor.State.NONE
        self._state_poly : FPoly            = other._state_poly if other is not None else None
        self._edit_poly_mode : FMaskEditor.EditPolyMode = other._edit_poly_mode if other is not None else FMaskEditor.EditPolyMode.NONE

        self._center_on_cursor_cnt : int = other._center_on_cursor_cnt if other is not None else 0

    def bake_polys(self, out : np.ndarray, include_color, exclude_color) -> np.ndarray:
        """Bake polys into out of img_size"""
        for poly in self.get_polys():
            if poly.points_count != 0:
                cv2.fillPoly(out, [ poly.points.astype(np.int32) ], include_color if poly.fill_type == FPoly.FillType.INCLUDE else exclude_color )
        return out

    def is_changed_cli_size(self, other : FMaskEditor): return (self._cli_size != other._cli_size).any()
    def get_cli_size(self) -> np.ndarray: return self._cli_size
    def set_cli_size(self, width, height) -> FMaskEditor:
        model = FMaskEditor(self)
        model._cli_size = np.array([width, height], np.int32)
        return model

    def is_changed_img_size(self, other : FMaskEditor): return (self._img_size != other._img_size).any()
    def get_img_size(self) -> np.ndarray: return self._img_size
    def set_img_size(self, width, height) -> FMaskEditor:
        model = FMaskEditor(self)
        model._img_size = np.array([width, height], np.int32)
        return model

    def is_changed_mouse_cli_pt(self, other : FMaskEditor): return (self._mouse_cli_pt != other._mouse_cli_pt).any()
    def get_mouse_cli_pt(self) -> np.ndarray: return self._mouse_cli_pt
    def set_mouse_cli_pt(self, mouse_cli_pt : np.ndarray) -> FMaskEditor:
        if (mouse_cli_pt == self._mouse_cli_pt).all():
            return self

        model = FMaskEditor(self)
        model._mouse_cli_pt = mouse_cli_pt

        if model._drag_type == FMaskEditor.DragType.VIEW_LOOK:
            # Drag view look
            delta_img_pt = model.get_mouse_img_pt() - model.map_cli_to_img(model._drag_start_cli_pt)
            model = model.set_view_look_img_pt(model._drag_start_view_look_pt - delta_img_pt)

        elif model._drag_type == FMaskEditor.DragType.STATE_POLY_PT:
            if model._state == FMaskEditor.State.EDIT_POLY:
                delta_img_pt = model.get_mouse_img_pt() - model.map_cli_to_img(model._drag_start_cli_pt)

                model = FMaskEditor(model)
                model._state_poly = model.get_state_poly().replace_pt(model._drag_state_poly_pt_id, model._drag_state_poly_pt + delta_img_pt )

        return model

    def is_changed_view_look_img_pt(self, other : FMaskEditor): return (self.get_view_look_img_pt() != other.get_view_look_img_pt()).any()
    def get_view_look_img_pt(self) -> np.ndarray:
        return self._view_look_img_pt if self._view_look_img_pt is not None else self.get_img_size() / 2
    def set_view_look_img_pt(self, view_look_img_pt : np.ndarray) -> FMaskEditor:
        model = FMaskEditor(self)
        model._view_look_img_pt = view_look_img_pt
        return model

    def is_changed_view_scale(self, other : FMaskEditor): return (self._view_scale or 0.0) != (other._view_scale or 0.0)
    def get_view_scale(self) -> float:
        return self._view_scale if self._view_scale is not None else (self._cli_size.min() / self._img_size.max())
    def set_view_scale(self, view_scale : float) -> FMaskEditor:
        model = FMaskEditor(self)
        model._view_scale = np.clip(view_scale, min(1.0, (self._cli_size.min() / self._img_size.max())), 20.0)
        return model

    def is_changed_drag_type(self, other : FMaskEditor): return self._drag_type != other._drag_type
    def get_drag_type(self) -> DragType:
        return self._drag_type

    def is_changed_polys(self, other : FMaskEditor): return not (self._polys is other._polys)
    def get_polys(self) -> Tuple[FPoly]: return self._polys
    def add_poly(self, poly : FPoly) -> FMaskEditor:
        model = FMaskEditor(self)
        model._polys = model._polys + (poly,)
        return model
    def update_poly(self, poly : FPoly, new_poly : FPoly) -> FMaskEditor:
        model = FMaskEditor(self)
        poly_idx = model._polys.index(poly)
        model._polys = model._polys[:poly_idx] + (new_poly,) + model._polys[poly_idx+1:]
        return model

    def is_changed_state(self, other : FMaskEditor): return self._state != other._state
    def get_state(self) -> State: return self._state

    def _set_state(self, state : State) -> FMaskEditor:
        if self._state == state:
            return self
        model = self

        # Exit from
        if model._state == FMaskEditor.State.DRAW_POLY:
            ...
        elif model._state == FMaskEditor.State.EDIT_POLY:
            if model._drag_type == FMaskEditor.DragType.STATE_POLY_PT:
                model = FMaskEditor(model)
                model._drag_type = FMaskEditor.DragType.NONE

            model = model.set_edit_poly_mode(FMaskEditor.EditPolyMode.NONE)

        model = FMaskEditor(model)
        model._state = state

        # Enter to
        if model._state == FMaskEditor.State.EDIT_POLY:
            model = model.set_edit_poly_mode(FMaskEditor.EditPolyMode.PT_MOVE)

        return model

    def is_changed_state_poly(self, other : FMaskEditor): return not (self.get_state_poly() is other.get_state_poly())
    def get_state_poly(self) -> FPoly|None:
        return self._state_poly if self.get_state() in [FMaskEditor.State.DRAW_POLY, FMaskEditor.State.EDIT_POLY] else None

    def is_changed_edit_poly_mode(self, other : FMaskEditor) -> bool: return self._edit_poly_mode != other._edit_poly_mode
    def get_edit_poly_mode(self) -> FMaskEditor.EditPolyMode:
        return self._edit_poly_mode
    def set_edit_poly_mode(self, edit_poly_mode : FMaskEditor.EditPolyMode) -> FMaskEditor:
        model = self
        if model._state == FMaskEditor.State.EDIT_POLY:
            model = FMaskEditor(model)
            model._edit_poly_mode = edit_poly_mode
        return model

    def is_changed_for_undo(self, old : FMaskEditor) -> bool:
        r = self.is_changed_state(old) or \
               self.is_changed_polys(old)

        if self.get_drag_type() == FMaskEditor.DragType.STATE_POLY_PT:
            r = r or self.is_changed_drag_type(old)
        else:
            r = r or self.is_changed_state_poly(old)
        return r

    # Evaluators
    def get_pt_select_radius(self) -> int: return self._pt_select_radius

    def is_mouse_at_state_poly(self) -> bool:
        if (state_poly := self.get_state_poly()) is not None:
            if (dist := state_poly.dist(self.map_cli_to_img(self._mouse_cli_pt))) is not None:
                return dist <= self._pt_select_radius
        return False

    def get_mouse_img_pt(self) -> np.ndarray: return self.map_cli_to_img(self._mouse_cli_pt)

    def get_mouse_state_poly_pt_id(self) -> int|None:
        if (state_poly := self.get_state_poly()) is not None:
            x = np.argwhere ( npla.norm ( self.get_mouse_cli_pt() - self.map_img_to_cli( state_poly.points ), axis=1 ) <= self._pt_select_radius )
            return None if len(x) == 0 else x[-1][0]
        return None

    def get_mouse_state_poly_edge_id_cli_pt(self) -> Tuple[int, np.ndarray] | None:
        if (state_poly := self.get_state_poly()) is not None:
            if (edge_id_cli_pt := self.map_poly_to_cli(state_poly).nearest_edge_id_pt(self._mouse_cli_pt)) is not None:
                edge_id, cli_pt = edge_id_cli_pt
                if npla.norm (self._mouse_cli_pt-cli_pt) <= self._pt_select_radius:
                    return edge_id, cli_pt
        return None

    def get_img_to_cli_mat(self) -> Affine2DMat:
        cli_size = self.get_cli_size()
        view_scale = self.get_view_scale()
        view_look_img_pt = self.get_view_look_img_pt()
        return (Affine2DMat()   .translated(cli_size[0]/2, cli_size[1]/2)
                                .scaled(view_scale, view_scale)
                                .translated(-view_look_img_pt[0], -view_look_img_pt[1])    )

    def map_img_to_cli(self, pts : np.ndarray):
        return (pts - self.get_view_look_img_pt() ) * self.get_view_scale() + self.get_cli_size()/2.0

    def map_cli_to_img(self, pts : np.ndarray):
        return (pts - self.get_cli_size()/2.0 ) / self.get_view_scale() + self.get_view_look_img_pt()

    def map_poly_to_cli(self, poly : FPoly) -> FPoly:
        return FPoly(self.map_img_to_cli(poly.points), poly.fill_type)

    # Actions
    def is_activated_center_on_cursor(self, old : FMaskEditor) -> bool:
        return self._center_on_cursor_cnt > old._center_on_cursor_cnt

    def center_on_cursor(self) -> FMaskEditor:
        model = self
        model = model.set_view_look_img_pt(model.get_mouse_img_pt())

        model = model.set_mouse_cli_pt( model.map_img_to_cli(model.get_view_look_img_pt()) )
        model = FMaskEditor(model)
        model._center_on_cursor_cnt += 1
        return model

    def mouse_move(self, x, y) -> FMaskEditor:
        return self.set_mouse_cli_pt(np.array([x,y], np.int32))

    def mouse_lbtn_down(self, x, y) -> FMaskEditor:
        model = self.mouse_move(x, y)

        mouse_img_pt = model.get_mouse_img_pt()

        if (state := model.get_state()) == FMaskEditor.State.NONE:
            # Pressing in NO OPERATION mode

            # if self.mouse_wire_poly is not None:
            #     # Click on wire on any poly -> switch to EDIT_MODE
            #     self.set_op_mode(OpMode.EDIT_POLY, op_poly=self.mouse_wire_poly)
            # else:

            # Click on empty space -> create new poly with single point
            model = model._set_state(FMaskEditor.State.DRAW_POLY)
            model = FMaskEditor(model)
            model._state_poly = FPoly().add_pt(mouse_img_pt)

        elif state == FMaskEditor.State.DRAW_POLY:
            # Pressing in DRAW_POLY mode
            if model.get_mouse_state_poly_pt_id() == 0 and model._state_poly.points_count >= 3:
                # Click on first point -> close poly and switch to edit mode
                model = model._set_state(FMaskEditor.State.EDIT_POLY)
            else:
                # Click on empty space -> add point to current poly
                model = FMaskEditor(model)
                model._state_poly = model._state_poly.add_pt(mouse_img_pt)
        elif state == FMaskEditor.State.EDIT_POLY:
            # Pressing in EDIT_POLY mode

            if (mouse_state_poly_pt_id := model.get_mouse_state_poly_pt_id()) is not None:
                # Click on point of state_poly



                if model.get_edit_poly_mode() == FMaskEditor.EditPolyMode.PT_ADD_DEL:
                    # in mode 'delete point'
                    state_poly = model._state_poly.remove_pt(mouse_state_poly_pt_id)

                    if state_poly.points_count >= 3:
                        model = FMaskEditor(model)
                        model._state_poly = state_poly
                    else:
                        # not enough points after delete -> exit EDIT_POLY
                        model = model._set_state(FMaskEditor.State.NONE)
                else:

                    if model.get_drag_type() == FMaskEditor.DragType.NONE:
                        # otherwise -> start drag
                        model = FMaskEditor(model)
                        model._drag_type = FMaskEditor.DragType.STATE_POLY_PT
                        model._drag_start_cli_pt = model._mouse_cli_pt
                        model._drag_state_poly_pt_id = mouse_state_poly_pt_id
                        model._drag_state_poly_pt    = model.get_state_poly().points[mouse_state_poly_pt_id]
            else:

                if model.get_edit_poly_mode() == FMaskEditor.EditPolyMode.PT_ADD_DEL:
                    if (edge_id_cli_pt := model.get_mouse_state_poly_edge_id_cli_pt()) is not None:
                        edge_id, cli_pt = edge_id_cli_pt

                        img_pt = model.map_cli_to_img(cli_pt)

                        model = FMaskEditor(model)
                        model._state_poly = model._state_poly.insert_pt(edge_id+1, img_pt)



                    #self._mouse_cli_pt
                    #model.get_poly_edge_id_pt_at_pt(model._state_poly)


            # elif self.mouse_op_poly_edge_id is not None:
            #     # Click on edge of op_poly
            #     if self.pt_edit_mode == PTEditMode.ADD_DEL:
            #         # in mode 'insert new point'
            #         edge_img_pt = self.cli_to_img_pt(self.mouse_op_poly_edge_id_pt)
            #         self.op_poly.insert_pt (self.mouse_op_poly_edge_id+1, edge_img_pt)
            #         self.update()
            #     else:
            #         # Otherwise do nothing
            #         pass

        return model

    def mouse_lbtn_up(self, x, y) -> FMaskEditor:
        model = self.mouse_move(x, y)

        if model.get_state() == FMaskEditor.State.EDIT_POLY:
            if model.get_drag_type() == FMaskEditor.DragType.STATE_POLY_PT:
                model = FMaskEditor(model)
                model._drag_type = FMaskEditor.DragType.NONE

        return model

    def mouse_mbtn_down(self, x, y) -> FMaskEditor:
        model = self.mouse_move(x, y)
        if model._drag_type == FMaskEditor.DragType.NONE:
            model = FMaskEditor(model)
            model._drag_type = FMaskEditor.DragType.VIEW_LOOK
            model._drag_start_cli_pt = model.get_mouse_cli_pt()
            model._drag_start_view_look_pt = model.get_view_look_img_pt()
        return model

    def mouse_mbtn_up(self, x, y) -> FMaskEditor:
        model = self.mouse_move(x, y)
        if model._drag_type == FMaskEditor.DragType.VIEW_LOOK:
            model = FMaskEditor(model)
            model._drag_type = FMaskEditor.DragType.NONE
        return model

    def mouse_wheel(self, x, y, delta : int) -> FMaskEditor:
        model = self.mouse_move(x, y)

        if model._drag_type == FMaskEditor.DragType.NONE:
            prev_img_pt = model.get_mouse_img_pt()
            sign = np.sign(delta)
            model = model.set_view_scale( model.get_view_scale() + (sign*0.2 + sign*model.get_view_scale()/10.0) )

            if sign > 0:
                model = model.set_view_look_img_pt( model.get_view_look_img_pt() + (prev_img_pt - model.get_mouse_img_pt()) )
            else:
                model = model.set_mouse_cli_pt( model.map_img_to_cli(prev_img_pt) )

        return model


    def apply_state_poly(self, type : PolyApplyType) -> FMaskEditor:
        """works in DRAW_POLY|EDIT_POLY"""
        model = self

        if model.get_state() == FMaskEditor.State.DRAW_POLY:
            if model._state_poly.points_count >= 3:
                model = model._set_state(FMaskEditor.State.EDIT_POLY)
            else:
                model = model._set_state(FMaskEditor.State.NONE)

        if model.get_state() == FMaskEditor.State.EDIT_POLY:
            state_poly = model._state_poly.set_fill_type(FPoly.FillType.INCLUDE if type == FMaskEditor.PolyApplyType.INCLUDE else FPoly.FillType.EXCLUDE)

            model = model.add_poly(state_poly)._set_state(FMaskEditor.State.NONE)

        return model

    def delete_state_poly(self) -> FMaskEditor:
        """works in DRAW_POLY|EDIT_POLY"""
        model = self
        if model.get_state() in [FMaskEditor.State.DRAW_POLY, FMaskEditor.State.EDIT_POLY]:
            model = FMaskEditor(model)
            model = model._set_state(FMaskEditor.State.NONE)

        return model


class FPoly():
    class FillType(Enum):
        NONE = auto()
        INCLUDE = auto()
        EXCLUDE = auto()

    def __init__(self, pts : np.ndarray = None, fill_type : FillType = FillType.NONE):
        self._fill_type = fill_type
        self._pts = pts.astype(np.float32) if pts is not None else np.empty( (0,2), dtype=np.float32 )

    def add_pt(self, pt : np.ndarray) -> FPoly:
        return FPoly(np.concatenate([self._pts, pt.astype(np.float32)[None,:]], axis=0), self._fill_type)

    def insert_pt(self, n, pt : np.ndarray) -> FPoly:
        return FPoly(np.concatenate([self._pts[:n], pt.astype(np.float32)[None,:], self._pts[n:]], axis=0), self._fill_type)

    def remove_pt(self, n) -> FPoly:
        return FPoly( np.concatenate([self._pts[:n], self._pts[n+1:]], axis=0), self._fill_type)

    def replace_pt(self, n, pt : np.ndarray) -> FPoly:
        return FPoly(np.concatenate([self._pts[:n], pt.astype(np.float32)[None,:], self._pts[n+1:]], axis=0), self._fill_type)

    def set_fill_type(self, fill_type : FillType) -> FPoly:
        return FPoly(self._pts, fill_type)

    def dist(self, pt : np.ndarray) -> float|None:
        if self.points_count >= 3:
            return -cv2.pointPolygonTest(self._pts[None,...], pt, True)
        return None

    def dists_to_edges(self, pt) -> Tuple[ Sequence[float], np.ndarray ] | None:
        if self.points_count >= 2:
            return sd.dist_to_edges(self.points, pt, is_closed=True)
        return None

    def nearest_edge_id_pt(self, pt):
        if (dists_projs := self.dists_to_edges(pt)) is not None:
            dists, projs = dists_projs
            if len(dists) > 0:
                edge_id = np.argmin(dists)
                return edge_id, projs[edge_id]
        return None

    @property
    def fill_type(self) -> FillType:
        return self._fill_type

    @property
    def last_point(self):
        return self._pts[self._n-1].copy()

    @property
    def points(self) -> np.ndarray:
        return self._pts.copy()

    @property
    def points_count(self):
        return self._pts.shape[0]

