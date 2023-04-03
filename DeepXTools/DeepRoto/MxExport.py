from __future__ import annotations

from pathlib import Path

from core import ax, mx
from core.lib import path as lib_path
from core.lib.image import NPImage, Patcher

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
        self._mx_output_path = mx.PathState(config=mx.PathStateConfig(dir_only=True, allow_open=False, allow_new=True)).dispose_with(self)

        self._mx_patch_mode = mx.Flag(state.get('patch_mode', False)).dispose_with(self)
        self._mx_sample_count = mx.Number(state.get('sample_count', 2), mx.NumberConfig(min=1, max=4)).dispose_with(self)
        self._mx_fix_borders = mx.Flag(state.get('fix_borders', False)).dispose_with(self)

        if (input_path := state.get('input_path', None)) is not None:
            self._mx_input_path.open(input_path)

        if (output_path := state.get('output_path', None)) is not None:
            self._mx_output_path.new(output_path)

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
    def mx_output_path(self) -> mx.IPathState:
        return self._mx_output_path
    @property
    def mx_patch_mode(self) -> mx.IFlag:
        return self._mx_patch_mode
    @property
    def mx_sample_count(self) -> mx.INumber:
        return self._mx_sample_count
    @property
    def mx_fix_borders(self) -> mx.IFlag:
        return self._mx_fix_borders
    
    def _on_input_path_open(self, path : Path):
        self._mx_output_path.new( path.parent / (path.name + '_trained_mask') )
        return path

    def get_state(self) -> dict:
        return {'patch_mode' : self._mx_patch_mode.get(),
                'sample_count' : self._mx_sample_count.get(),
                'fix_borders' : self._mx_fix_borders.get(),
                'input_path' : self._mx_input_path.mx_path.get(),
                'output_path' : self._mx_output_path.mx_path.get(),
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
            (output_path := self._mx_output_path.mx_path.get()) is None):
            yield ax.cancel()

        yield ax.switch_to(self._export_thread)
        yield ax.attach_to(self._export_tg, cancel_all=True)

        err = None
        try:
            output_path.mkdir(parents=True, exist_ok=True)
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
                        infer_task = self._infer_path(imagepath, output_path / (imagepath.stem+'.png'))
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
    def _infer_path(self, image_path : Path, output_path : Path) -> NPImage:
        yield ax.attach_to(self._tg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        model = self._model
        model_res = model.get_input_resolution()

        patch_mode = self._mx_patch_mode.get()
        if patch_mode:
            sample_count = self._mx_sample_count.get()
            fix_borders = self._mx_fix_borders.get()
            
        yield ax.switch_to(self._export_thread_pool)

        err = None
        try:
            image_np = NPImage.from_file(image_path)
        except Exception as e:
            err = e

        if err is None:
            if patch_mode:
                H, W, _ = image_np.shape
                if H >= model_res and W >= model_res:

                    patcher = Patcher(image_np, model_res, sample_count=sample_count, use_padding=fix_borders)
                    for i in range(patcher.patch_count):
                        
                        
                        yield ax.wait(step_task := model.step(MxModel.StepRequest(image_np=[patcher.get_patch(i)], pred_mask=True)))

                        if step_task.succeeded:
                            patcher.merge_patch(i, step_task.result.pred_mask_np[0] )
                        else:
                            err = step_task.error
                            break

                    if err is None:
                        pred_mask_np = patcher.get_merged_image()
                else:
                    err = Exception('Image size is less than model resolution')
            else:
                yield ax.wait(step_task := model.step(MxModel.StepRequest(image_np=[image_np], pred_mask=True)))

                if step_task.succeeded:
                    pred_mask_np = step_task.result.pred_mask_np[0]
                else:
                    err = step_task.error

        if err is None:
            try:
                H, W, _ = image_np.shape
                mask_np = pred_mask_np.resize(W, H, interp=NPImage.Interp.LANCZOS4)
                mask_np.save(output_path)
            except Exception as e:
                err = e

        if err is not None:
            yield ax.cancel(err)

