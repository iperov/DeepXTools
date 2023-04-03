from __future__ import annotations

from pathlib import Path

from core import ax, mx
from core.lib import path as lib_path
from core.lib.image import NPImage

from .MxModel import MxModel


class MxExport(mx.Disposable):
    def __init__(self, model : MxModel, state : dict = None):
        super().__init__()
        state = state or {}

        self._model = model
        self._imagespaths = []

        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._export_thread = ax.Thread().dispose_with(self)
        self._export_thread_pool = ax.ThreadPool().dispose_with(self)
        self._export_tg = ax.TaskGroup().dispose_with(self)

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_progress = mx.Progress().dispose_with(self)
        self._mx_input_path  = mx.PathState(config=mx.PathStateConfig(dir_only=True), on_open=self._on_input_path_open).dispose_with(self)
        self._mx_output_image_path = mx.PathState(config=mx.PathStateConfig(dir_only=True, allow_open=False, allow_new=True)).dispose_with(self)
        self._mx_output_mask_path  = mx.PathState(config=mx.PathStateConfig(dir_only=True, allow_open=False, allow_new=True)).dispose_with(self)

        if (input_path := state.get('input_path', None)) is not None:
            self._mx_input_path.open(input_path)

        if (output_image_path := state.get('output_image_path', None)) is not None:
            self._mx_output_image_path.new(output_image_path)

        if (output_mask_path := state.get('output_mask_path', None)) is not None:
            self._mx_output_mask_path.new(output_mask_path)

    @property
    def mx_model(self) -> MxModel: return self._model
    @property
    def mx_error(self) -> mx.ITextEmitter_r:
        return self._mx_error
    @property
    def mx_progress(self) -> mx.IProgress_r:
        return self._mx_progress
    @property
    def mx_input_path(self) -> mx.IPathState:
        return self._mx_input_path
    @property
    def mx_output_image_path(self) -> mx.IPathState:
        return self._mx_output_image_path
    @property
    def mx_output_mask_path(self) -> mx.IPathState:
        return self._mx_output_mask_path

    def _on_input_path_open(self, path : Path):
        self._mx_output_image_path.new( path.parent / (path.name + '_swap') )
        self._mx_output_mask_path.new( path.parent / (path.name + '_swap_mask') )
        return path

    def get_state(self) -> dict:
        return {'input_path' : self._mx_input_path.mx_path.get(),
                'output_image_path' : self._mx_output_image_path.mx_path.get(),
                'output_mask_path' : self._mx_output_mask_path.mx_path.get(),
                }

    @ax.protected_task
    def stop(self):
        """
        Stop current export.

        Avail when mx_progress.mx_start == True
        """
        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_tg, cancel_all=True)
        yield ax.switch_to(self._main_thread)
        self._mx_progress.finish()


    @ax.protected_task
    def start(self):
        """
        Start export.

        Avail when mx_directory_path.mx_path is not None
        """
        yield ax.switch_to(self._main_thread)

        if ((input_path  := self._mx_input_path.mx_path.get()) is None or
            (output_image_path := self._mx_output_image_path.mx_path.get()) is None or
            (output_mask_path := self._mx_output_mask_path.mx_path.get()) is None):
            yield ax.cancel()

        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_tg, cancel_all=True)

        err = None
        try:
            output_image_path.mkdir(parents=True, exist_ok=True)
            output_mask_path.mkdir(parents=True, exist_ok=True)
            imagespaths = lib_path.get_files_paths(input_path, extensions=NPImage.avail_image_suffixes())
        except Exception as e:
            err=e

        yield ax.switch_to(self._main_thread)

        self._mx_progress.start(0, len(imagespaths))

        yield ax.switch_to(self._export_thread)

        if err is None:
            infer_tasks = ax.TaskSet()
            it = iter(imagespaths)
            while True:
                # Fill infer_tasks buffer
                while infer_tasks.count < self._export_thread_pool.count*2:
                    if it is None:
                        break

                    try:
                        imagepath = next(it)
                        infer_task = self._infer_path(imagepath, output_image_path / (imagepath.stem+'.png'), output_mask_path / (imagepath.stem+'.png'))
                        infer_task._imagepath = imagepath
                        infer_tasks.add(infer_task)
                    except StopIteration:
                        it = None

                if infer_tasks.count == 0:
                    # Done
                    break

                # Process finished infer_tasks
                for t in infer_tasks.fetch(finished=True):
                    yield ax.switch_to(self._main_thread)
                    if not t.succeeded:
                        self._mx_error.emit(f'{t._imagepath} : {str(t.error)}')
                    else:
                        self._mx_progress.inc()
                    yield ax.switch_to(self._export_thread)

                yield ax.sleep(0)

        yield ax.switch_to(self._main_thread)

        self._mx_progress.finish()

        if err is not None:
            self._mx_error.emit(str(err))

    @ax.task
    def _infer_path(self, image_path : Path, output_image_path : Path, output_mask_path : Path) -> NPImage:
        yield ax.attach_to(self._tg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        model = self._model

        yield ax.switch_to(self._export_thread_pool)

        err = None
        try:
            image_np = NPImage.from_file(image_path)
        except Exception as e:
            err = e

        if err is None:

            yield ax.wait(step_task := model.step(MxModel.StepRequest(dst_image_np=[image_np], pred_swap_image=True, pred_swap_mask=True)))

            if step_task.succeeded:
                pred_swap_image_np = step_task.result.pred_swap_image_np[0]
                pred_swap_mask_np = step_task.result.pred_swap_mask_np[0]
            else:
                err = step_task.error

        if err is None:
            try:
                H, W, _ = image_np.shape

                pred_swap_image_np.resize(W, H, interp=NPImage.Interp.LANCZOS4).save(output_image_path)
                pred_swap_mask_np.resize(W, H, interp=NPImage.Interp.LANCZOS4).save(output_mask_path)
            except Exception as e:
                err = e

        if err is not None:
            yield ax.cancel(err)

