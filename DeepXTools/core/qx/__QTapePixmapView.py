from dataclasses import dataclass
from typing import Callable, Dict

from .. import ax, qt
from .QTapeItemView import QTapeItemView
from .StyleColor import StyleColor


class QTapePixmapView(QTapeItemView):
    
    @dataclass
    class ItemInfo:
        pixmap : qt.QPixmap|None
        caption : str|None
        caption_bg_color : qt.QColor|None

    def __init__(self,  task_get_item_info : Callable[ [int], ax.Task[ItemInfo] ] = None,
                    ):
        super().__init__()

        self._task_get_item_info = task_get_item_info
        self._cap_spacing = 1
        self._cap_h = 16
        self._cached_ids = set()
        self._cached_ids_task : Dict[int, ax.Task[QTapePixmapView.ItemInfo] ] = {}
        self._cached_ids_item_info : Dict[int, QTapePixmapView.ItemInfo]= {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._tg_cache = ax.TaskGroup().dispose_with(self)

        self._caption_font = self.get_font()

        self.set_item_size(128, 128)
        self.set_item_count(0)
        
        self._bg_task()

    def __dispose__(self):
        self._cached_ids = None
        self._cached_ids_task = None
        self._cached_ids_item_info = None
        super().__dispose__()

    def set_item_size(self, w, h):
        return super().set_item_size(w, h + self._cap_spacing + self._cap_h)

    def set_caption_font(self, font : qt.QFont):
        self._caption_font = font
        return self
        
    def update_items(self):
        self._tg_cache.cancel_all()
        self._cached_ids = set()
        self._cached_ids_task = {}
        self._cached_ids_item_info = {}
        self.update()
        
    def update_item(self, id : int):
        self._cache_item(id, force=True)
        
    @ax.task
    def _bg_task(self):
        yield ax.attach_to(self._tg)
        
        while True:
            
            # Uncaching not visible items
            idx_start = self._v_idx_start
            idx_count = self._v_idx_count

            if len(self._cached_ids) > idx_count*2:
                for id in self._cached_ids:
                    if id < idx_start or id >= idx_start+idx_count:
                        break
                    id = None

                if id is not None:
                    self._uncache_item(id)
                    
            yield ax.sleep(0.1)
    
    
        
    @ax.task
    def _cache_item(self, id : int, force = False):
        if not force and id in self._cached_ids:
            return
            
        if (task_get_item_info := self._task_get_item_info) is None:
            return
            
        self._uncache_item(id)
    
        yield ax.attach_to(self._tg)
        yield ax.attach_to(self._tg_cache)

        self._cached_ids.add(id)
        self._cached_ids_task[id] = task = task_get_item_info(id)
        self._cached_ids_item_info[id] = None

        yield ax.wait(task)

        if task.succeeded:
            self._cached_ids_item_info[id] = task.result
            self.update()
            
    def _uncache_item(self, id):
        if id in self._cached_ids:
            self._cached_ids.remove(id)
            self._cached_ids_task.pop(id).cancel()
            self._cached_ids_item_info.pop(id)

    def _on_paint_item(self, id : int, qp : qt.QPainter, rect : qt.QRect):
        """overridable"""

        font = self._caption_font
        #font_color = self._font._get_q_color()

        fm = qt.QFontMetrics(font)

        image_bg_color = StyleColor.Midlight
        
        text_color = StyleColor.Text
        
        caption_bg_color = None
        
        qp.setFont(font)
        qp.setPen(text_color)

        item_info = self._cached_ids_item_info.get(id, None)
        if item_info is not None:
            text = item_info.caption
            if text is None:
                text = ''
            pixmap = item_info.pixmap
            caption_bg_color = item_info.caption_bg_color
        else:
            self._cache_item(id, force=False)

            text = f'{id}...'
            pixmap = None
        
        image_rect = rect.adjusted(0,0,0,-self._cap_spacing - self._cap_h)
        cap_rect = rect.adjusted(0,rect.height() - self._cap_h,0, 0)

        qp.fillRect(image_rect, image_bg_color)
        qp.fillRect(cap_rect, caption_bg_color or StyleColor.Midlight)

        if pixmap is not None:

            size = pixmap.size()
            if size.width() == 0 or size.height() == 0:
                qp.drawText(image_rect,  qt.Qt.AlignmentFlag.AlignCenter, 'Unviewable')
            else:
                fitted_image_rect = qt.QRect_fit_in( qt.QRect( qt.QPoint(0,0), pixmap.size() ), image_rect)
                qp.drawPixmap(fitted_image_rect, pixmap)
        else:
            #
            qp.drawText(image_rect,  qt.Qt.AlignmentFlag.AlignCenter, '...')

        elided_text = fm.elidedText(text, qt.Qt.TextElideMode.ElideRight, cap_rect.width())

        qp.drawText(cap_rect, qt.Qt.AlignmentFlag.AlignCenter, elided_text)


