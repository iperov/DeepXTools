import colorsys
from typing import Sequence, Tuple

import numba as nb
import numpy as np

from core import ax, qt, qx, lx

from .MxGraph import MxGraph


class QxGraph(qx.QVBox):
    def __init__(self, graph : MxGraph):
        super().__init__()
        self._graph = graph
        self._main_thread = ax.get_current_thread()

        self._tg = ax.TaskGroup().dispose_with(self)
        self._update_task_tg = ax.TaskGroup().dispose_with(self)
        self._graph_changed_tg = ax.TaskGroup().dispose_with(self)
        
        self._settings = qx.QSettings()
        
        self._q_graph = q_graph = QGraph()
        self._q_slider = q_slider = qx.QHRangeDoubleSlider()
        
        (self
            .add( qx.QCheckBoxMxMultiChoice(graph.mx_names).v_compact(), align=qx.Align.CenterF )      
            .add(q_graph.v_expand())
            .add(qx.QHBox().v_compact()
                    .add(q_slider.set_decimals(4).set_range(0.0, 1.0) 
                                .inline(lambda slider:
                                            (slider.mx_values.set((0.0, 1.0)),
                                             slider.mx_values.listen(lambda v: self._on_slider_values(v)))))
                
                    .add(qx.QPushButton().h_compact().set_icon_size(qx.Size.S).set_icon(qx.QIonIconDB.instance().icon(qx.IonIcon.cut_sharp, qx.StyleColor.Text))
                            .inline(lambda btn: btn.mx_clicked.listen(lambda: (graph.trim(*q_slider.mx_values.get()),
                                                                               q_slider.mx_values.set((0.0,1.0)))  ))))
                                                                         

            )
        
        
        graph.mx_data.reflect(lambda data: self._ref_data(data)).dispose_with(self)

    def _on_slider_values(self, values : Tuple[float, float]):
        self._settings['slider_values'] = values
        self._q_graph.set_view_range(*values)

    def _settings_event(self, settings : qx.QSettings):
        super()._settings_event(settings)
        self._settings = settings
        
        if (slider_values := settings.get('slider_values', None)) is not None:
            self._q_slider.mx_values.set(slider_values)

    @ax.task
    def _ref_data(self, data : MxGraph.Data):
        names = data.names
        names_color = [ qt.QColor.fromRgbF(*colorsys.hsv_to_rgb( n*(1.0/len(names)), 0.5, 1.0 )[::-1]) for n in range(len(names)) ]
        
        self._q_graph.set_data(data.array, names, names_color)
        
        

class QGraph(qx.QWidget):
    def __init__(self):
        super().__init__()

        self._main_thread = ax.get_current_thread()
        self._sub_thread = ax.Thread().dispose_with(self)
        self._tg = ax.TaskGroup().dispose_with(self)

        self._qp = qt.QPainter()

        self._data : np.ndarray = None
        self._names : Sequence[str] = ()
        self._colors : Sequence[qt.QColor] = ()

        self._view_start_f = 0.0
        self._view_end_f = 1.0

        self._p_pixmap = None
        self._p_data = None
        self._p_data_C_start = None
        self._p_data_C_per_W = None
        self._p_names = None
        self._p_colors = None
        
        self._mouse_pt = qt.QPoint(0,0)
        self._mouse_in = False
        self._select_pt_start = None
        
        self.set_mouse_tracking(True)
        self.set_cursor(qt.Qt.CursorShape.BlankCursor)
        
        qx.QApplication.instance().mx_language.reflect(self._ref_lang).dispose_with(self)
        
    def _ref_lang(self, lang : str):
        self._lang = lang
        self._update_data()
    
    def clear_data(self):
        self._data = None
        self._colors = ()
        self._update_data()
        return self
    
    def set_data(self, data : np.ndarray, names : Sequence[str], colors : Sequence[qt.QColor]):
        """
        update/set new data
        """
        self._data = data
        self._colors = colors
        self._names = names
        self._update_data()
        return self

    def set_view_range(self, start_f : float, end_f : float):
        """
        uniform float values.

        Example 0.9, 1.0
        """
        if self._view_start_f != start_f or self._view_end_f != end_f:
            self._view_start_f = start_f
            self._view_end_f = end_f
            self._update_data()

        return self

    @ax.task
    def _update_data(self, instant = False):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)
        
        if not instant:
            yield ax.time_barrier(0.1, max_task=1)
            
        data = self._data
        names = self._names
        colors = self._colors
        view_start_f = self._view_start_f
        view_end_f = self._view_end_f

        rect = self.rect()
        W, H = rect.width(), rect.height()

        yield ax.switch_to(self._sub_thread)
        # Queue mode
        p_pixmap = None 
        p_data = None
        p_data_C_start = None
        p_data_C_per_W = None
        p_names = None
        p_colors = None
        
        img = _draw_bg(W,H)
        if data is not None:
            N, C = data.shape
            if N > 0:
                C_view_start = int(C*view_start_f)
                C_view_end   = int(C*view_end_f)
                if C_view_end - C_view_start > 0:
                    data = data[:,C_view_start:C_view_end]
                    
                    p_data_C_start = C_view_start
                    p_data_C_per_W = max(1.0, data.shape[1] / W)

                    data, g_min, g_max = _preprocess_data(W, data)
                    
                    # import code
                    # code.interact(local=dict(globals(), **locals()))

                    _overlay_graph(img, data, g_min, g_max, colors=np.float32([ color.getRgbF()[2::-1] for color in colors ]) )
                    
                    p_colors = colors
                    p_data = data
                    p_names = [ lx.L(name, lang=self._lang) for name in names]
        
        p_pixmap = qt.QPixmap_from_np(img)

        yield ax.switch_to(self._main_thread)

        self._p_pixmap = p_pixmap
        self._p_data = p_data
        self._p_data_C_start = p_data_C_start
        self._p_data_C_per_W = p_data_C_per_W
        self._p_names = p_names
        self._p_colors = p_colors
        
        self.update()

    def _enter_event(self, ev: qt.QEnterEvent):
        super()._enter_event(ev)
        self._mouse_in = True
        self.update()
        
    def _leave_event(self, ev: qt.QEvent):
        super()._leave_event(ev)
        self._mouse_in = False
        self.update()

    def _mouse_press_event(self, ev: qt.QMouseEvent):
        pt = ev.pos()
        self._select_pt_start = pt
        self.update()
        
    def _mouse_move_event(self, ev: qt.QMouseEvent):
        pt = ev.pos()
        self._mouse_pt = pt
        self.update()
        
    def _mouse_release_event(self, ev: qt.QMouseEvent):
        pt = ev.pos()
        self._select_pt_start = None
        self.update()
        
    def _resize_event(self, ev: qt.QResizeEvent):
        super()._resize_event(ev)
        self._update_data()

    def _minimum_size_hint(self) -> qt.QSize:
        return qt.QSize(64, 64)

    def _paint_event(self, ev: qt.QPaintEvent):
        super()._paint_event(ev)
        
        rect = self.rect()
        W = rect.width()
        H = rect.height()
        
        qp = self._qp
        qp.begin(self.get_q_widget())
        qp.fillRect(rect, qx.StyleColor.Shadow)
        
        
        
        if (pixmap := self._p_pixmap) is not None:
            qp.drawPixmap(0,0, pixmap)
        
        if self._mouse_in:
            mouse_pt = self._mouse_pt
            mouse_x = mouse_pt.x()
            mouse_y = mouse_pt.y()
            
            select_pt_start = self._select_pt_start
            
            if select_pt_start is not None:
                sel_start = select_pt_start.x()
                sel_end = mouse_x
                if sel_end < sel_start:
                    sel_start, sel_end = sel_end, sel_start
            else:
                sel_start = sel_end = mouse_x 
            
            pen_color = qt.QColor(qx.StyleColor.Text)
            pen_color.setAlpha(64)
            qp.setPen(pen_color)
            
            qp.fillRect( qt.QRect(sel_start, 0, sel_end-sel_start+1, H), pen_color )
            
            pen_color = qt.QColor(qx.StyleColor.Text)
            pen_color.setAlpha(127)
            qp.setPen(pen_color)
            
            qp.drawLine(0, mouse_y, W, mouse_y)
            
            if (data := self._p_data) is not None:
                N,C,_ = data.shape                
                
                sel_end   = max(0, min(sel_end, C-1)) 
                sel_start = max(0, min(sel_start, sel_end, C-1)) 
                
                sel_data = data[:,sel_start:sel_end+1,0].mean(-1)
                
                if sel_data.max() != 0:
                    p_names = self._p_names
                    p_colors = self._p_colors
                    
                    p_data_C_start = self._p_data_C_start
                    p_data_C_per_W = self._p_data_C_per_W
                    data_C_start = p_data_C_start + int( sel_start * p_data_C_per_W)
                    data_C_end   = p_data_C_start + int( (sel_end+1) * p_data_C_per_W) - 1
                    
                    font = qx.QFontDB.instance().fixed_width()
                    fm = qt.QFontMetrics(font)
        
                    text_lines = []
                    text_sizes = []
                    text_colors = []
                    
                    text_lines.append(f"{lx.L('@(QxGraph.Average_for)', self._lang)} {data_C_start} - {data_C_end}")
                    text_sizes.append(fm.boundingRect(text_lines[-1]).size())
                    text_colors.append(qx.StyleColor.Text)
                    
                    for n in range(N):
                        v = sel_data[n]
                        if v != 0.0:
                            text_lines.append(text := f'{v:.4f} — {p_names[n]}')
                            text_sizes.append(fm.boundingRect(text).size())
                            text_colors.append(p_colors[n])
                    
                    tooltip_width  = max(size.width() for size in text_sizes) 
                    tooltip_height = sum(size.height() for size in text_sizes)
                    
                    tooltip = qt.QPixmap(tooltip_width, tooltip_height)
                    tooltip.fill(qt.QColor(0,0,0,0))
                    tooltip_qp = qt.QPainter(tooltip)
                    tooltip_qp.setFont(font)
                    
                    text_y = 0
                    for text, color, size in zip(text_lines, text_colors, text_sizes):
                        text_height = size.height()
                        tooltip_qp.setPen(color)
                        tooltip_qp.drawText(0, text_y, tooltip_width, text_height, qt.Qt.AlignmentFlag.AlignVCenter, text )
                        text_y += text_height
                    tooltip_qp.end()
                    
                    tooltip_pad = 4
                    tooltip_rect = tooltip.rect()
                    draw_at_left = mouse_x >= (tooltip_rect.width()+tooltip_pad*2)
                    draw_at_top  = mouse_y >= (tooltip_rect.height()+tooltip_pad*2)
                    
                    qp.fillRect  (  tooltip_rect.translated(mouse_x+ ( (-tooltip_rect.width() -tooltip_pad*2) if draw_at_left else 1),
                                                            mouse_y+ ( (-tooltip_rect.height()-tooltip_pad*2) if draw_at_top else 1) )
                                                .adjusted(0,0,tooltip_pad*2,tooltip_pad*2), 
                                    qt.QColor(0,0,0,176))
                    
                    qp.drawPixmap(  tooltip_rect.translated(mouse_x+ ( (-tooltip_rect.width()-tooltip_pad)  if draw_at_left else 1+tooltip_pad), 
                                                            mouse_y+ ( (-tooltip_rect.height()-tooltip_pad) if draw_at_top else 1+tooltip_pad) ),  
                                    tooltip)
        qp.end()


@nb.njit(nogil=True)
def _preprocess_data(W, graph : np.ndarray):
    N, C = graph.shape
    
    C_per_W = max(1.0, float(C) / float(W) )
    
    g_min = np.inf
    g_max = -np.inf
    
    out = np.zeros( (N,W,3), np.float32 )
    for n in range(N):
        for w in range(min(W,C)):
            w_start = int(    w*C_per_W) 
            w_end   = int((w+1)*C_per_W)
            
            a_v = 0.0
            if (w_end - w_start) != 0:
                v_min = np.inf
                v_max = -np.inf
                for i in range(w_start, w_end):
                    v = graph[n,i]
                    
                    v_min = min(v_min, v)
                    v_max = max(v_max, v)
                    g_min = min(g_min, v)
                    g_max = max(g_max, v)
                    
                    a_v += v
                a_v /= w_end - w_start
            else:
                v_min = 0
                v_max = 0
                
            out[n,w] = [a_v, v_min, v_max]
            
            
    return out, g_min, g_max        


@nb.njit(nogil=True)
def _draw_bg(W, H):
    img = np.empty( (H,W,4), np.float32)
    for h in range(H):
        
        for w in range(W):
            img[h,w] = [1.0, 1.0, 1.0, 0.05 if h % 8 < 2 else 0]
    return img
        
                
@nb.njit(nogil=True)
def _overlay_graph(img : np.ndarray, graph : np.ndarray, g_min, g_max, colors : np.ndarray):
    H,W,_ = img.shape
    N, _, _ = graph.shape
        
    g_d = g_max-g_min
    if g_d != 0:
        for n in range(N):
            b,g,r = colors[n]
            for w in range(W):
                a_v, v_min, v_max = graph[n,w]
                
                if v_min != 0 or v_max != 0:
                        
                    if v_min == v_max:
                        if w > 0:
                            prev_a_v,_,_ = graph[n, w-1]
                            if a_v >= prev_a_v:
                                v_min = prev_a_v
                                v_max = a_v
                            else:
                                v_min = a_v
                                v_max = prev_a_v
                        
                    a_v_f   = (a_v-g_min)   / g_d
                    v_min_f = (v_min-g_min) / g_d
                    v_max_f = (v_max-g_min) / g_d
                    
                    h_a_v = max(0, min( int((H-1)*a_v_f  ), H-1))
                    h_min = max(0, min( int((H-1)*v_min_f), H-1))
                    h_max = max(0, min( int((H-1)*v_max_f), H-1))
                    
                    for h in range(h_min, h_max+1):
                        
                        a = 1
                        if h >= h_a_v:
                            d = (h_a_v-h_max)
                            if d != 0:
                                a = float(h - h_max) / d
                        else:
                            d = (h_a_v-h_min)
                            if d != 0:
                                a = float(h - h_min) / d
                            
                        
                        img[H-1-h, w] = [b,g,r,a]
                    