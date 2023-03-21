import math

from .. import mx, qt
from ..qt import QPoint, QRect, QSize
from .QWidget import QWidget


class QTapeItemView(QWidget):

    def __init__(self):
        super().__init__()

        self._item_count = 0
        self._item_w = 0
        self._item_h = 0
        self._spacing = 4

        self._col_width = -1
        self._row_height = -1
        self._v_col_count = -1
        self._v_row_count = -1
        self._table_l_offset = 0
        self._current_idx = None

        self._v_grid_idx_start = 0
        self._v_grid_idx_end = 0
        self._v_grid_start_x = 0
        self._v_grid_start_y = 0


        self._v_idx_start = 0
        self._v_idx_count = 0

        self._mouse_down_pt = None
        self.__qp = qt.QPainter()
        self._mx_current_idx = mx.DeferredProperty[int|None](self._current_idx, defer=lambda new_value, value, prop: self.set_current_idx(new_value)).dispose_with(self)

    @property
    def mx_current_idx(self) -> mx.IProperty[int|None]:
        """None only if item_count==0"""
        return self._mx_current_idx

    def get_item_size(self) -> qt.QSize: return qt.QSize(self._item_w, self._item_h)
    def set_item_size(self, w, h): self._update(new_item_w=w, new_item_h=h); return self

    def get_item_count(self) -> int: return self._item_count
    def set_item_count(self, item_count : int): self._update(new_item_count=item_count); return self
    def get_current_idx(self) -> int|None:
        """None only if item_count==0"""
        return self._current_idx
    def set_current_idx(self, idx : int): self._update(new_current_idx=idx); return self

    def _minimum_size_hint(self) -> QSize: return QSize(self._item_w+self._spacing*2, self._item_h+self._spacing*2)

    def _resize_event(self, ev : qt.QResizeEvent):
        super()._resize_event(ev)
        self._update(upd_geo=True)

    def _paint_event(self, ev : qt.QPaintEvent):
        rect = self.rect()
        qp = self.__qp
        qp.begin(self.get_q_widget())

        current_idx = self.get_current_idx()

        pal = self.get_q_widget().palette()
        mid_color = pal.color(qt.QPalette.ColorRole.Text)
        if self._item_count != 0:
            for idx in range(self._v_idx_start, self._v_idx_start+self._v_idx_count):

                item_rect = self._get_item_rect(idx)

                if current_idx == idx:
                    select_rect = item_rect.adjusted(-self._spacing, -self._spacing, self._spacing, self._spacing)
                    qp.fillRect(select_rect, mid_color)


                self._on_paint_item(idx, qp, item_rect)

                #qp.fillRect(item_rect, 'white')

                #qp.drawText(item_rect, 0, f'{idx}')


                # if self.is_selected(idx):
                #     item_outter_rect = item_rect.marginsAdded( qt.QMargins(1,1,1,1))
                #     qp.fillRect(item_outter_rect, selection_frame_q_color)

                    # pen_item_selected = QPen( Qt.white )
                    # pen_item_selected.setWidth(self._spacing)
                    # qp.setPen(pen_item_selected)
                    # qp.drawRect(item_outter_rect)

        else:
            qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, 'no items')

        qp.end()

    def _on_paint_item(self, id : int, qp : qt.QPainter, rect : QRect):
        """overridable. Paint item content in given rect."""
        raise NotImplementedError()

    def _get_item_rect(self, idx) -> QRect:
        diff_idx = idx-self._v_grid_idx_start
        x = self._v_grid_start_x + int(math.fmod(diff_idx, self._v_col_count)) * self._col_width
        y = self._v_grid_start_y + math.trunc(float(diff_idx)/self._v_col_count) * self._row_height
        return QRect(QPoint(x,y), QSize(self._item_w, self._item_h))


    def _update(self,   new_item_count = None,
                        new_item_w = None,
                        new_item_h = None,
                        new_spacing = None,
                        new_current_idx = None,
                        upd_geo = False):
        rect = self.rect()

        w = rect.width()
        h = rect.height()

        item_count = new_item_count if new_item_count is not None else self._item_count
        item_w     = new_item_w if new_item_w is not None else self._item_w
        item_h     = new_item_h if new_item_h is not None else self._item_h
        spacing    = new_spacing if new_spacing is not None else self._spacing

        current_idx = new_current_idx if new_current_idx is not None else self._current_idx

        if item_count == 0:
            current_idx = None
        else:
            if current_idx is None:
                current_idx = 0
            current_idx = max(0, min(current_idx, item_count-1))

        upd = False

        if self._item_count != item_count:
            self._item_count = item_count
            upd = True

        if self._item_w != item_w:
            self._item_w = item_w
            upd = True
            upd_geo = True

        if self._item_h != item_h:
            self._item_h = item_h
            upd = True
            upd_geo = True

        if self._spacing != spacing:
            self._spacing = spacing
            upd = True
            upd_geo = True

        if self._current_idx != current_idx or new_current_idx is not None:
            self._current_idx = current_idx
            self._mx_current_idx.fset(current_idx)

            upd = True
            upd_geo = True

        if upd_geo:
            if item_count != 0:
                col_width = self._col_width = item_w + spacing
                row_height = self._row_height = item_h + spacing

                v_col_count = max(1, w // col_width)
                if v_col_count % 2 == 0:
                    v_col_count -= 1
                self._v_col_count = v_col_count

                v_row_count = max(1, int(math.ceil(h / row_height)))
                if v_row_count % 2 == 0:
                    v_row_count -= 1

                half = (v_row_count*v_col_count) // 2
                v_grid_idx_start = self._v_grid_idx_start = current_idx-half
                v_grid_idx_end   = self._v_grid_idx_end = current_idx+half
                v_grid_start_x = self._v_grid_start_x = w//2 - (v_col_count//2)*col_width - item_w//2
                v_grid_start_y = self._v_grid_start_y = h//2 - (v_row_count//2)*row_height - item_h//2

                v_idx_start = self._v_idx_start = max(0, min(item_count-1, v_grid_idx_start))
                v_idx_end = self._v_idx_end = max(0, min(item_count-1, v_grid_idx_end))
                v_idx_count = self._v_idx_count = v_idx_end - v_idx_start + 1

            # print('--------')
            # print('w', w)
            # print('h', h)
            # print('v_col_count', v_col_count)
            # print('v_row_count', v_row_count)
            # print('current_idx', current_idx)
            # print('v_grid_idx_start', v_grid_idx_start)
            # print('v_grid_idx_end', v_grid_idx_end)
            # print('v_grid_start_x', v_grid_start_x)
            # print('v_grid_start_y', v_grid_start_y)
            # print('v_idx_start', v_idx_start)
            # print('v_idx_count', v_idx_count)

        if upd:
            self.update()
        if upd_geo:
            self.update_geometry()




