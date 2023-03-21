from __future__ import annotations

import threading
import traceback
from collections import deque
from datetime import datetime
from types import GeneratorType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, Iterable, ParamSpec,
                    Set, TypeVar)

from .g_log import g_log
from .g_debug import g_debug
from .Thread import Thread, get_current_thread
from .ThreadPool import ThreadPool
from .TimeBarrier import TimeBarrier

if TYPE_CHECKING:
    from .TaskGroup import TaskGroup

T = TypeVar('T')
P = ParamSpec('P')
class Task(Generic[T]):

    def __init__(self, name : str = None):
        super().__init__()
        self._name = name

        self._state = 0 # 0 - active. 1 - succeeded. 2 - cancelled

        self._lock = threading.RLock()
        self._finish_lock = threading.RLock()

        self._on_finish_funcs = deque()
        #^ accessed inside Task._finish_lock only

        self._child_tasks : Set[Task] = set()
        #^ added - inside Task._lock and Task execution only
        #  remove - no lock

        self._tgs : Set[TaskGroup] = set()
        #^ accessed inside Task._lock only
        
        self._creation_time = datetime.now().timestamp()

        # Execution variables
        self._gen_tms : Dict = None
        self._gen = None
        self._gen_next = True
        self._gen_thread = current_thread = get_current_thread()
        self._gen_yield = None

        exec_stack = current_thread.get_tls()._task_exec_stack
        if len(exec_stack) != 0:
            # Task created inside execution of other Task in current thread
            # connect parent-child
            parent_task = self._parent = exec_stack[-1]
            parent_task._child_tasks.add(self)
        else:
            self._parent : Task = None
        
        g_debug._on_task_created(self)
        
        #Task._active_tasks.add(self)

        if g_log.get_level() >= 2:
            print(f"{('Starting'):12} {self}")

    @property
    def name(self) -> str: return self._name

    @property
    def creation_time(self) -> float: return self._creation_time
    
    @property
    def alive_time(self) -> float: return datetime.now().timestamp() - self._creation_time
    
    @property
    def finished(self) -> bool:  return self._state > 0

    @property
    def succeeded(self) -> bool:
        """result. Avail only if finished"""
        if self._state == 0:
            raise Exception(f'{self} task is not finished to check succeeded.')
        return self._state == 1

    @property
    def result(self) -> T:
        """result. Avail only if succeeded"""
        if self._state != 1:
            raise Exception(f'{self} no result for non-succeeded task.')
        return self._result

    @property
    def error(self) -> Exception|None:
        """error. Avail only if not succeeded"""
        if self._state != 2:
            raise Exception(f'{self} no error for non-cancelled task.')
        return self._error

    def attach_to(self, tg : TaskGroup, detach_parent : bool) -> bool:
        """
        Try to attach to TaskGroup.

        Task will be detached from current parent task.

        """
        with self._lock:
            if self._state == 0:
                if tg not in self._tgs:

                    with tg._lock:
                        tg_tasks = tg._tasks
                        if tg_tasks is not None and self not in tg_tasks:
                            tg_tasks.add(self)

                            self._tgs.add(tg)

                            if detach_parent:
                                if self._parent is not None:
                                    self._parent._child_tasks.remove(self)
                                    self._parent = None

                            if g_log.get_level() >= 2:
                                print(f"{('Attached'):12} {self} to {tg}")
                            return True
        return False

    def _detach(self):
        """
        Detach Task from all task registries or from parent task.
        Useful to prevent error while cancelling running Task from TaskGroup.
        """
        with self._lock:
            for tg in self._tgs:
                if (tasks := tg._tasks) is not None:
                    try:
                        tasks.remove(self)
                    except: ...
            self._tgs.clear()

            if self._parent is not None:
                self._parent._child_tasks.remove(self)
                self._parent = None

    def cancel(self, error : Exception = None):
        """
        Finish task in blocking mode.
        Stop its execution and set Task to cancelled state with optional error(default None).
        If Task is already finished, nothing will happen.
        """
        self._finish(False, error=error)
        return self

    def call_on_finish(self, func : Callable[ [Task], None ], in_this_thread=True):
        """
        Call func when Task is finished.
        If Task is already finished, func will be called immediately.
        """
        with self._finish_lock:
            if self._state == 0:
                if in_this_thread:
                    func = lambda task, current_thread=get_current_thread(), func=func: current_thread.call_asap(lambda: func(task))
                self._on_finish_funcs.append(func)
                return self
        func(self)
        return self

    def wait(self):
        """
        Block execution and wait Task in current (or automatically registered) ax.Thread

        raises Exception if calling wait() inside Task.
        """
        if get_current_task() != None:
            raise Exception('Unable to .wait() inside Task. Use yield ax.wait(task)')

        get_current_thread().execute_tasks_loop(exit_if=lambda: self.finished)
        return self

    def _success(self, result : Any = None):
        """
        Finish task, stop its execution and set Task to succeeded state optional result(default None).
        All tasks created during execution of this task will be cancelled.
        If it is already finished, nothing will happen.
        """
        self._finish(True, result)

    def _finish(self, success : bool, result = None, error : Exception = None):
        if self._state == 0:
            on_finish_funcs = ()
            with self._lock, self._finish_lock:
                if self._state == 0:

                    if g_log.get_level() >= 2:
                        child_tasks_len = len(self._child_tasks)
                        if child_tasks_len != 0:
                            print(f"{('Finishing'):12} {self} Child tasks:[{child_tasks_len}]")
                        else:
                            print(f"{('Finishing'):12} {self}")

                    if success:
                        self._result = result
                        self._state = 1
                    else:
                        self._error = error
                        self._state = 2

                    if self._gen is not None:
                        try:
                            self._gen.throw( TaskFinishError(self) )
                        except Exception as e:
                            ...
                        self._gen.close()
                        self._gen = None

                    for child_task in tuple(self._child_tasks):
                        child_task.cancel()

                    if self._parent is not None:
                        self._parent._child_tasks.remove(self)
                        self._parent = None

                    taskrings, self._tgs = self._tgs, None
                    for taskring in taskrings:
                        try:
                            tasks = taskring._tasks
                            if tasks is not None:
                                tasks.remove(self)
                        except:
                            ...
                    
                    g_debug._on_task_finished(self)
                    #Task._active_tasks.remove(self)

                    if g_log.get_level() >= 2:
                        print(f"{('Finish'):12} {self}")

                    on_finish_funcs, self._on_finish_funcs = self._on_finish_funcs, None

            for func in on_finish_funcs:
                func(self)

    def _exec(self):
        while True:
            if self._lock.acquire(timeout=0.005):
                break
            if self._state != 0:
                # Task not in active state
                # may be in cancelling state in other Thread
                # don't wait and return
                return

        # Acquired lock
        if self._state == 0:
            # Task in active state
            exec_stack = get_current_thread().get_tls()._task_exec_stack

            # Execute Task until interruption yield-command
            while self._state == 0:

                if self._gen_next:
                    self._gen_next = False

                    # add Task to ThreadLocalStorage execution stack
                    exec_stack.append(self)
                    try:
                        self._gen_yield = next(self._gen)
                    except StopIteration as e:
                        # Method returns value directly
                        exec_stack.pop()
                        self._success(e.value)
                        break
                    except Exception as e:
                        # Unhandled exception
                        if g_log.get_level() >= 1:
                            print(f'Unhandled exception {e} occured during execution of task {self}. Traceback:\n{traceback.format_exc()}')
                        exec_stack.pop()
                        self.cancel(error=e)
                        break
                    exec_stack.pop()

                else:
                    # Process yield value

                    gen_yield = self._gen_yield
                    gen_yield_t = type(gen_yield)

                    if gen_yield_t is wait:
                        if gen_yield.is_finished():
                            self._gen_next = True

                    elif gen_yield_t is switch_to:
                        thread = gen_yield._thread_or_pool

                        if isinstance(thread, ThreadPool):
                            thread = thread._get_next_thread()

                        if thread is not None:
                            gen_yield._thread_or_pool = thread

                            if self._gen_thread.ident == thread.ident:
                                self._gen_next = True
                            else:
                                self._gen_thread = thread

                    elif gen_yield_t is sleep:
                        if gen_yield._sec > 0:
                            if gen_yield.is_finished():
                                self._gen_next = True
                        else:
                            gen_yield._ticks += 1
                            if gen_yield._ticks > 1:
                                self._gen_next = True

                    elif gen_yield_t is attach_to:
                        if gen_yield._cancel_all:
                            gen_yield._tg.cancel_all()

                        if self.attach_to(gen_yield._tg, detach_parent=gen_yield._detach_parent):
                            self._gen_next = True
                        else:
                            self.cancel()
                            break
                        
                    elif gen_yield_t is time_barrier:
                        tms = self._gen_tms
                        
                        key = self._gen.gi_frame.f_lineno
                        if (tb := tms.get(key, None)) is None:
                            tb = tms[key] = TimeBarrier()
                            
                        if (r := tb.try_pass(gen_yield._interval, gen_yield._max_task)) < 0:
                            self.cancel()
                            break

                        self._gen_yield = sleep(r)
                    


                    # elif gen_yield_t is try_pass:
                    #     if (r := gen_yield._tb.try_pass()) < 0:
                    #         self.cancel()
                    #         break

                    #     self._gen_yield = sleep(r)

                    elif gen_yield_t is cancel:
                        self.cancel(gen_yield._error)
                        break

                    elif gen_yield_t is success:
                        self._success(result=gen_yield._result)
                        break

                    elif gen_yield_t is detach:
                        self._detach()
                        self._gen_next = True
                    elif gen_yield_t is propagate:
                        gen_yield._task.call_on_finish (lambda other_task: (
                                    self._success(other_task.result) if other_task.succeeded else
                                    self.cancel(error=other_task.error)), in_this_thread=False)
                        break

                    else:
                        print(f'{self} Unknown type of yield value: {gen_yield}')
                        self.cancel()
                        break

                    if not self._gen_next:
                        if not self._gen_thread._add_task(self):
                            self.cancel()
                        break
        self._lock.release()

    def __repr__(self): return self.__str__()
    def __str__(self):
        s = f'[Task][{self.name}]'

        if self.finished:
            if self.succeeded:
                s += f'[SUCCEEDED][Result: {type(self.result).__name__}]'
            else:
                s += f'[CANCELLED]'
                error = self.error
                if error is not None:
                    s += f'[Exception:{error}]'
        else:
            s += f'[ACTIVE]'

        return s

    #_active_tasks = set()


def get_current_task() -> Task|None:
    current_thread = get_current_thread()
    tls = current_thread.get_tls()
    if len(tls._task_exec_stack) != 0:
        return tls._task_exec_stack[-1]
    return None

class wait:
    """Stop execution until task_or_list will be entered to finished state."""
    def __init__(self, task_or_list : Task|Iterable[Task]):
        if not isinstance(task_or_list, Iterable):
            task_or_list = (task_or_list,)

        self._lock = threading.Lock()
        self._count = len(task_or_list)

        for task in task_or_list:
            task.call_on_finish(self._on_task_finish, in_this_thread=False)

        self._task_list = task_or_list

    def _on_task_finish(self, task):
        with self._lock:
            self._count -=1

    def is_finished(self):
        return self._count == 0

class switch_to:
    """
    Switch thread of this Task.
    If Task already in Thread, execution will continue immediately.
    If Thread is disposed, Task will be cancelled.
    """
    def __init__(self, thread_or_pool : Thread|ThreadPool):
        self._thread_or_pool = thread_or_pool

class sleep:
    """
    Sleep execution of this Task.

    `sleep(0)` sleep single tick, i.e. minimum possible amount of time between two executions of task loop.
    """
    def __init__(self, sec : float):
        self._time = datetime.now().timestamp()
        self._sec = sec
        self._ticks = 0

    def is_finished(self):
        return (datetime.now().timestamp() - self._time) >= self._sec

class attach_to:
    """```
    Attach Task to TaskGroup.
    If TaskGroup is disposed, current Task will be cancelled without exception immediatelly.

        cancel_all(False)   cancel all tasks from TaskGroup before addition.

        detach_parent(True)     detach from parent Task, thus if parent Task is cancelled,
                                this Task will not be cancelled.
    ```"""
    def __init__(self, tg : TaskGroup, cancel_all : bool = False, detach_parent : bool = True):
        self._tg = tg
        self._cancel_all = cancel_all
        self._detach_parent = detach_parent

class time_barrier:
    """
    pass `max_task` in `interval` otherwise cancel
    """
    def __init__(self, interval : float, max_task : int = 1):
        self._interval = interval
        self._max_task = max_task

class success:
    """
    Same as `return <result>`, but:
    yield.success inside `try except ax.TaskFinishError`: will be caught.

    All tasks created during execution of this task and if they are not in TaskGroup's (child tasks) - will be cancelled.
    """
    def __init__(self, result = None):
        self._result = result

class cancel:
    """
    Finish task execution and mark this Task as cancelled with optional error.

    yield.cancel inside `try except ax.TaskFinishError`: will be caught.

    All tasks created during execution of this task and if they are not in TaskGroup's (child tasks) - will be cancelled.
    """
    def __init__(self, error : Exception = None):
        self._error = error

class propagate:
    """Wait Task and returns it's result as result of this Task"""
    def __init__(self, task : Task):
        self._task = task

class detach:
    """
    Detach Task from all TaskRegistry or from parent task.

    Useful to prevent error while cancelling TaskGroup in which current Task exist.
    """

class TaskFinishError(Exception):
    """
    an Exception to catch task finish.

    Thread is undetermined.

    After this exception any `yield` will stop execution and will have no effect.
    """
    def __init__(self, task : Task):
        self._task = task

    def __str__(self): return f'TaskFinishError of {self._task}'
    def __repr__(self): return self.__str__()

def task(func : Callable[P, T]) -> Callable[P, Task[T] ]:
    """decorator.

    Turns func to ax.Task.

    Decorated func always returns Task object with type hint of func return value.

    available yields inside task:

    ```
        yield ax.wait

        yield ax.sleep

        yield ax.switch_to

        yield ax.attach_to

        yield ax.try_pass

        yield ax.success

        yield ax.cancel

        yield ax.propagate

        yield ax.detach
    ```

    """
    tms = {}
    
    def wrapper(*args : P.args, **kwargs : P.kwargs) -> Task[T]:
        task = Task(name=f'{func.__qualname__}')

        if isinstance(ret := func(*args, **kwargs), GeneratorType):
            task._gen_tms = tms
            task._gen = ret
            task._exec()
        else:
            task._success(ret)

        return task

    return wrapper


def protected_task(func : Callable[P, T]) -> Callable[P, Task[T] ]:
    """Same as @task but wraps with wrapper_task which progates the result of this task.

    Used to protect Task from cancellation using TaskGroup

    Example
    ```
    @ax.protected_task
    def task(*args):
        yield ax.attach_to(tg) # <- should be used
        yield ax.sleep(999)

    t = task()
    t.cancel() # cancelled wrapper Task, original Task is working
    ```
    """
    p = task(func)
    def wrapper(*args : P.args, **kwargs : P.kwargs) -> Task[T]:
        t = _wrapper_task(p, *args, **kwargs)
        t._name = f'{func.__qualname__}(protected)'
        return t
    return wrapper

@task
def _wrapper_task(func, *args, **kwargs):
    yield propagate(func(*args, **kwargs))
