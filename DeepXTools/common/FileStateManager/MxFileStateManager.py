from __future__ import annotations

import math
import os
import pickle
import shutil
import time
from collections import deque
from enum import Enum, auto
from numbers import Number
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from core import ax, mx
from core.lib import path as lib_path


class MxFileStateManager(mx.Disposable):
    """
    Manages state dict stored in file.
    """

    class State(Enum):
        Uninitialized = auto()
        Loading = auto()
        Error = auto()
        Initialized = auto()

    def __init__(self,  file_suffix : str,
                        on_close : Callable[ [], ax.Task ] = None,
                        task_on_load : Callable[ [Dict], ax.Task ] = None,
                        task_get_state : Callable[ [], ax.Task[Dict] ] = None ):
        """```

            on_close            called from main thread

            task_on_load        called from main thread

            task_get_state      called from main thread
        ```

        Automatically closes on dispose.

        State must contain only basic python types, include numpy types.
        """

        super().__init__()
        self._state_path : Path = None

        self._on_close = on_close
        self._task_on_load = task_on_load
        self._task_get_state = task_get_state

        self._rel_path : Path = None

        self._mx_state = mx.Property[MxFileStateManager.State](MxFileStateManager.State.Uninitialized).dispose_with(self)
        self._mx_error = mx.Property[Exception|None](None).dispose_with(self)

        self._mx_path = mx.PathState(mx.PathStateConfig(allow_open=True, allow_new=True, allow_rename=True, extensions=[file_suffix], desc=f'*{file_suffix}'),
                                     on_close=self._on_path_close,
                                     on_open=self._on_path_open,
                                     on_new=self._on_path_new,
                                     on_rename=self._on_path_rename,
                                     ).dispose_with(self)

        self._mx_load_progress = mx.Progress().dispose_with(self)
        self._mx_controls_bag = mx.Disposable().dispose_with(self)

        self._main_thread = ax.get_current_thread()
        self._io_thread = ax.Thread(name='IO').dispose_with(self)
        self._tg = ax.TaskGroup().dispose_with(self)
        self._save_tg = ax.TaskGroup().dispose_with(self)


    def __dispose__(self):
        self._mx_path.close()
        super().__dispose__()

    def _on_path_close(self):
        self._tg.cancel_all()

        if self._mx_state.get() == MxFileStateManager.State.Initialized:
            self._mx_state.set(MxFileStateManager.State.Uninitialized)
            if (on_close := self._on_close) is not None:
                on_close()

        self._state_path = None

    def _on_path_open(self, path : Path) -> bool:
        self._state_path = path
        self._reinitialize(path, rel_path=path.parent)
        return True

    def _on_path_new(self, path : Path) -> bool:
        self._state_path = path
        self._reinitialize(rel_path=path.parent)
        return True

    def _on_path_rename(self, path : Path) -> bool:
        self._state_path = path
        return True

    @property
    def mx_state(self) -> mx.IProperty_r[State]:
        return self._mx_state
    @property
    def mx_error(self) -> mx.IProperty_r[Exception|None]|None:
        """Avail when .mx_state == Error"""
        return self._mx_error
    @property
    def mx_path(self) -> mx.IPathState:
        return self._mx_path

    @property
    def mx_load_progress(self) -> mx.IProgress_r:
        """Avail when .mx_state == Loading"""
        return self._mx_load_progress
    @property
    def mx_save_progress(self) -> mx.IProgress_r|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_save_progress
    @property
    def mx_backup_progress(self) -> mx.IProgress_r|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_backup_progress
    @property
    def mx_autosave(self) -> mx.INumber|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_autosave
    @property
    def mx_autobackup(self) -> mx.INumber|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_autobackup
    @property
    def mx_backup_count(self) -> mx.INumber|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_backup_count
    @property
    def mx_notes(self) -> mx.IText|None:
        """Avail when .mx_state == Initialized"""
        return self._mx_notes

    def get_state_path(self) -> Path|None: return self._state_path


    @ax.task
    def _reinitialize(self, state_path : Path|None = None, rel_path : Path = None):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._tg, cancel_all=True)

        self._mx_state.set(MxFileStateManager.State.Loading)
        self._mx_load_progress.start(50, 100)
        self._mx_controls_bag = mx_controls_bag = self._mx_controls_bag.dispose_and_new()

        yield ax.switch_to(self._io_thread)

        state = {}
        err = None
        if state_path is not None and state_path.exists():
            try:
                with open(state_path, 'rb') as file:
                    state = pickle.load(file)



                state = self._repack_traverse(state, is_pack=False)
            except Exception as e:
                err = e

        yield ax.switch_to(self._main_thread)

        if err is None:
            self._rel_path = rel_path

            if (task_on_load := self._task_on_load) is not None:
                yield ax.wait(load_t := task_on_load(state.get('user_state', {})))

                if not load_t.succeeded:
                    # task_on_load must work fine
                    # raise unhandled developer error
                    raise load_t.error

        self._mx_load_progress.finish()

        if err is not None:
            self._mx_error.set(err)
            self._mx_state.set(MxFileStateManager.State.Error)
            self._mx_path.close()
        else:
            self._last_save_time = time.time()
            self._last_backup_time = time.time()
            self._mx_autosave = mx.Number(state.get('autosave', 25), config=mx.NumberConfig(min=0, max=3600), filter=self._flt_mx_autosave).dispose_with(mx_controls_bag)
            self._mx_autobackup = mx.Number(state.get('autobackup', 0), config=mx.NumberConfig(min=0, max=3600), filter=self._flt_mx_autobackup).dispose_with(mx_controls_bag)
            self._mx_backup_count = mx.Number(state.get('backup_count', 8), config=mx.NumberConfig(min=1, max=32)).dispose_with(mx_controls_bag)
            self._mx_notes = mx.Text(state.get('notes', '')).dispose_with(mx_controls_bag)

            self._mx_save_progress = mx.Progress().dispose_with(mx_controls_bag)
            self._mx_backup_progress = mx.Progress().dispose_with(mx_controls_bag)

            self._mx_state.set(MxFileStateManager.State.Initialized)

            self._run_auto_save_task()

    def _flt_mx_autosave(self, new_mins : Number, mins : Number):
        self._last_save_time = time.time()
        return new_mins

    def _flt_mx_autobackup(self, new_mins : Number, mins : Number):
        self._last_backup_time = time.time()
        return new_mins

    @ax.task
    def _run_auto_save_task(self) -> ax.Task:
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        while True:
            if (autosave := self._mx_autosave.get()) != 0 and \
               (time.time() - self._last_save_time) >= autosave*60:
                yield ax.wait(self.save())

            if (autobackup := self._mx_autobackup.get()) != 0 and \
               (time.time() - self._last_backup_time) >= autobackup*60:
                yield ax.wait(self.save(backup=True))

            yield ax.sleep(1)

    @ax.protected_task
    def save(self, as_name : str = None, backup=False):
        """
        Save task.
        Avail in state==Initialized, otherwise cancelled.
        Task will be cancelled if other save task is already running.
        """
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        if self._save_tg.count != 0:
            # Already saving
            yield ax.cancel()
        yield ax.attach_to(self._save_tg)

        if self._mx_state.get() != MxFileStateManager.State.Initialized:
            # Nothing to save
            yield ax.cancel()

        root_path = self._state_path.parent
        state_path = self._state_path
        if as_name is not None:
            state_path = root_path / f'{state_path.stem} — {as_name}.state'

        state_path_part = state_path.parent / (state_path.name + '.part')

        self._last_save_time = time.time()
        self._mx_save_progress.start(0, 100)

        if backup:
            self._last_backup_time = self._last_save_time
            self._mx_backup_progress.start(0, 100)

        # Collect state
        user_state = {}
        if (task_get_state := self._task_get_state) is not None:
            yield ax.wait(user_state_t := task_get_state())

            if user_state_t.succeeded:
                user_state = user_state_t.result
            else:
                # _task_get_state must work fine
                # raise unhandled developer error
                raise user_state_t.error

        yield ax.switch_to(self._io_thread)

        state = {'user_state' : user_state,
                 'autosave' : self._mx_autosave.get(),
                 'autobackup' : self._mx_autobackup.get(),
                 'backup_count' : self._mx_backup_count.get(),
                 'notes' : self._mx_notes.get(),
                   }

        # Prepare state dump
        state = self._repack_traverse(state, is_pack=True)

        mv = memoryview(pickle.dumps(state))
        mv_size = len(mv)
        chunks_count = 50
        mv_chunk_size = math.ceil(mv_size / chunks_count)

        # Trying to save .part file
        err = None
        file = None
        try:
            file = open(state_path_part, 'wb')

            for i in range(chunks_count):
                chunk_end = min( (i+1)*mv_chunk_size, mv_size )
                file.write( mv[i*mv_chunk_size: chunk_end])
                file.flush()
                os.fsync(file.fileno())

                yield ax.switch_to(self._main_thread)

                self._mx_save_progress.progress( progress := min(99, int(i / (chunks_count-1)*100)) )
                self._mx_backup_progress.progress( progress // 2 )

                yield ax.switch_to(self._io_thread)


        except Exception as e:
            err = e
        finally:
            if file is not None:
                file.close()
                file = None
            if err is not None:
                if state_path_part.exists():
                    state_path_part.unlink()

        yield ax.switch_to(self._main_thread)

        if err is None:
            if state_path.exists():
                state_path.unlink()
            state_path_part.rename(state_path)

            if backup:
                try:
                    yield ax.switch_to(self._io_thread)

                    backup_count = self._mx_backup_count.get()



                    # Delete redundant backups
                    for filepath in lib_path.get_files_paths(root_path):
                        if filepath.suffix == state_path.suffix:
                            if len(splits := filepath.stem.split(f'{state_path.stem} — bckp — ')) == 2:
                                backup_id = int(splits[1])
                                if backup_id > backup_count:
                                    filepath.unlink()

                    # Renaming existing backups to free backup slot 01
                    for i in range(backup_count-1,0,-1):
                        p1 = root_path / f'{state_path.stem} — bckp — {i:02}{state_path.suffix}'
                        p2 = root_path / f'{state_path.stem} — bckp — {i+1:02}{state_path.suffix}'
                        if p2.exists():
                            p2.unlink()
                        if p1.exists():
                            p1.rename(p2)

                    # Copy saved state file to backup slot 01
                    shutil.copy(state_path, root_path / f'{state_path.stem} — bckp — 01{state_path.suffix}')

                except Exception as e:
                    err = e

        yield ax.switch_to(self._main_thread)

        self._mx_save_progress.finish()
        if backup:
            self._mx_backup_progress.finish()

        if err is not None:
            # Something goes wrong.
            # Close and go to error state in order not to waste comp time.
            if self._mx_state.get() == MxFileStateManager.State.Initialized:
                if (on_close := self._on_close) is not None:
                    on_close()

            self._mx_error.set(err)
            self._mx_state.set(MxFileStateManager.State.Error)



    def _repack_traverse(self, value, is_pack : bool = True):
        # Like a copy, but traverse nested containers, checks and changes some types

        if isinstance(value, Path):
            out_value = lib_path.relpath(value, self._rel_path) if is_pack else lib_path.abspath(value, self._rel_path)
        elif isinstance(value, dict):
            out_value = { k : self._repack_traverse(v, is_pack=is_pack) for k, v in value.items() }
        elif isinstance(value, list):
            out_value = [ self._repack_traverse(v, is_pack=is_pack) for v in value ]
        elif isinstance(value, deque):
            out_value = deque(self._repack_traverse(v, is_pack=is_pack) for v in value)
        elif isinstance(value, tuple):
            out_value = tuple(self._repack_traverse(v, is_pack=is_pack) for v in value)
        elif isinstance(value, set):
            out_value = set(self._repack_traverse(v, is_pack=is_pack) for v in value)
        elif value is None or isinstance(value, (int, float, str, bytes, bytearray, np.ndarray, np.generic)):
            out_value = value
        else:
            raise ValueError(f'Value of type {type(value)} is not allowed. Use basic python types (include numpy types) and collections to save the data.')

        return out_value
