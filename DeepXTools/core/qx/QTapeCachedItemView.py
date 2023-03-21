from typing import Callable, Dict

from .. import ax, qt
from .QTapeItemView import QTapeItemView


class QTapeCachedItemView(QTapeItemView):
    def __init__(self,  task_get_item_pixmap : Callable[ [int, qt.QSize], ax.Task[qt.QPixmap] ] = None,
                    ):
        super().__init__()
        self._task_get_item_pixmap = task_get_item_pixmap
        self._cached_ids : Dict[int, ax.Task[qt.QPixmap]] = {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._tg_cache = ax.TaskGroup().dispose_with(self)

        self._bg_task()

    def set_item_count(self, item_count: int):
        super().set_item_count(item_count)
        self.update_items()
        return self

    def set_item_size(self, w, h):
        super().set_item_size(w, h)
        self.update_items()
        return self

    def update_items(self):
        self._tg_cache.cancel_all()
        self._cached_ids = {}
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

        if (task_get_item_pixmap := self._task_get_item_pixmap) is None:
            return

        self._uncache_item(id)

        yield ax.attach_to(self._tg)
        yield ax.attach_to(self._tg_cache)

        self._cached_ids[id] = task = task_get_item_pixmap(id, self.get_item_size())

        yield ax.wait(task)

        if task.succeeded:
            self.update()

    def _uncache_item(self, id):
        if id in self._cached_ids:
            self._cached_ids.pop(id).cancel()

    def _on_paint_item(self, id : int, qp : qt.QPainter, rect : qt.QRect):
        """overridable"""
        pixmap_task = self._cached_ids.get(id, None)

        if pixmap_task is not None and pixmap_task.finished and pixmap_task.succeeded:
            pixmap = pixmap_task.result

            fitted_rect = qt.QRect_fit_in(pixmap.rect(), rect)
            qp.drawPixmap(fitted_rect, pixmap)
        else:
            self._cache_item(id, force=False)
            qp.drawText(rect, qt.Qt.AlignmentFlag.AlignCenter, '...')

