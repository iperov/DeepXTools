import threading
from datetime import datetime


class TimeBarrier:
    def __init__(self):
        self._lock = threading.RLock()
        self._task_count = 0
        self._activation_time = None

    def try_pass(self, interval : float, max_task : int) -> float|bool:
        """
        returns < 0 if fail to pass.
        returns float time(sec) needed to wait to pass the barrier.
        """
        with self._lock:
            time = datetime.now().timestamp()

            if self._task_count != 0:
                if time - self._activation_time >= interval:
                    self._task_count = 0

            if self._task_count >= max_task:
                return -1

            if self._task_count == 0:
                self._activation_time = time
            self._task_count += 1

            return (self._activation_time + interval) - time

