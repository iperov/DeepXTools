import threading
from typing import Generic, TypeVar

from .. import mx
from .g_log import g_log
from .Task import get_current_task
from .Thread import get_current_thread

T = TypeVar('T')

class TaskGroup(mx.Disposable, Generic[T]):

    def __init__(self, name : str = None):
        """
        Provides a set of Tasks used to control execution.
        Task detaches from his parent task when added to TaskGroup.
        """
        super().__init__()
        self._name = name
        self._lock = threading.RLock()
        self._tasks = set()

    def __dispose__(self):
        """
        Dispose TaskGroup. Cancel all tasks.
        New Tasks that are added to TaskGroup using yield will be automatically cancelled.
        """
        if g_log.get_level() >= 2:
            print(f"{('Dispose'):12} {self}")

        with self._lock:
            tasks, self._tasks = self._tasks, None

        for task in tasks:
            task.cancel()

        super().__dispose__()

    @property
    def count(self) -> int:
        """Amount of registered tasks."""
        tasks = self._tasks
        return 0 if tasks is None else len(tasks)

    @property
    def is_empty(self) -> bool: return self.count() == 0

    @property
    def name(self) -> str: return self._name

    def cancel_all(self):
        """
        Cancel all current active tasks in TaskGroup.
        If called from Task which is inside TaskGroup,
        then task will not be cancelled.
        """
        with self._lock:
            tasks, self._tasks = self._tasks, set()

        if g_log.get_level() >= 2 and len(tasks) != 0:
            print(f'Cancelling {len(tasks)} tasks in {self._name} TaskGroup in {get_current_thread()}')

        cur_task = get_current_task()

        for task in tasks:
            if task != cur_task:
                task.cancel()


    def __repr__(self): return self.__str__()
    def __str__(self):
        s = '[TaskGroup]'
        if self._name is not None:
            s += f'[{self._name}]'
        tasks = self._tasks
        if tasks is not None:
            s += f'[{len(self._tasks)} tasks]'
        else:
            s += '[FINALIZED]'
        return s


