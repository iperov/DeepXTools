from typing import Generic, Set, TypeVar

from .Task import Task

T = TypeVar('T')

class TaskSet(Generic[T]):
    """
    Simple Set of Tasks.

    Non thread-safe.

    Useful when need to check the status of multiple tasks periodically in local func.
    """

    def __init__(self):
        self._tasks : Set[Task[T]] = set()

    @property
    def count(self) -> int: return len(self._tasks)
    @property
    def empty(self) -> bool: return len(self._tasks) == 0

    def add(self, task : Task[T]):     self._tasks.add(task)
    def remove(self, task : Task[T]):  self._tasks.remove(task)

    def fetch(self, finished=None, succeeded=None) -> Set[Task[T]]:
        """
        fetch Task's from TaskSet with specific conditions

            finished(None)  None : don't check
                        True : task is finished
                        False : task is not finished

            succeeded(None)  None : don't check
                           True : task is finished and succeeded
                           False : task is finished and not succeeded

        if both args None, fetches all tasks.
        """
        out_tasks = set()

        for task in self._tasks:
            if (finished is None or finished==task.finished) and \
               (succeeded is None or (task.finished and succeeded==task.succeeded)):
                out_tasks.add(task)

        self._tasks.difference_update(out_tasks)

        return out_tasks

    def cancel_all(self, remove=True):
        """Cancel all current tasks in TaskSet"""
        if len(self._tasks) != 0:
            for task in self._tasks:
                task.cancel()
            if remove:
                self._tasks = set()

    def __repr__(self): return self.__str__()
    def __str__(self):
        s = "[TaskSet]"
        if self._name is not None:
            s += f"[{self._name}]"
        s += f'[contains {len(self._tasks)} tasks]'
        return s
