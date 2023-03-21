"""
AsyncX library. Designed and developed from scratch by github.com/iperov

Work with tasks in multiple threads easily.
```
Task    .finished  True:   .succeeded   True:   .result:      task result
                    |                       |
                    |                       |
                    |                       False:  mean cancelled
                    |                               .exception():   None        - no error provided
                    False:  mean active                             |
                                                                    Exception   - class of Exception error
```


"""
import sys as _sys

if not (_sys.version_info.major >= 4 or
       (_sys.version_info.major >= 3 and _sys.version_info.minor >= 10)):
    raise Exception("AsyncX requires Python 3.10+")

from .clear import clear
from .g_debug import g_debug
from .g_log import g_log
from .Task import (Task, TaskFinishError, attach_to, cancel, detach,
                   get_current_task, propagate, protected_task, sleep, success,
                   switch_to, task, time_barrier, wait)
from .TaskGroup import TaskGroup
from .TaskSet import TaskSet
from .Thread import Thread, get_current_thread
from .ThreadPool import ThreadPool
