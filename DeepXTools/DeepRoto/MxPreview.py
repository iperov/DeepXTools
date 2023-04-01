from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from core import ax, mx
from core.lib import path as lib_path
from core.lib.image import NPImage, Patcher

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxPreview(mx.Disposable):
    """
    Manages preview.
    """
    class SourceType(Enum):
        DataGenerator = auto()
        Directory = auto()

    @dataclass(frozen=True)
    class Sample:
        image_np : NPImage
        target_mask_np : NPImage|None
        pred_mask_np : NPImage
        name : str

        @staticmethod
        def from_state(state : dict) -> MxPreview.Sample|None:
            """Try to construct Sample from state. If state is not valid, returns None."""
            try:
                sample = MxPreview.Sample(
                        image_np = NPImage(state['image_np']),
                        target_mask_np = NPImage(target_mask_np) if (target_mask_np := state.get('target_mask_np', None)) is not None else None,
                        pred_mask_np = NPImage(state['pred_mask_np']),
                        name = state['name']
                    )
                return sample
            except Exception as e:
                return None

        def get_state(self) -> dict:
            return {'image_np' : self.image_np.HWC(),
                    'target_mask_np' : self.target_mask_np.HWC() if self.target_mask_np is not None else None,
                    'pred_mask_np' : self.pred_mask_np.HWC(),
                    'name' : self.name,
                    }


    def __init__(self, data_generator : MxDataGenerator, model : MxModel, state : dict = None):
        super().__init__()
        self._state = state = state or {}
        state['directory_state'] = state.get('directory_state', {})
        
        self._main_thread = ax.get_current_thread()
        self._sub_thread = ax.Thread().dispose_with(self)

        self._data_generator = data_generator
        self._model = model
        self._imagespaths = []

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_sample = mx.Property[MxPreview.Sample|None]( MxPreview.Sample.from_state(sample_state) if (sample_state := state.get('sample_state', None)) is not None else None ).dispose_with(self)

        self._source_type_disp_bag = mx.Disposable().dispose_with(self)
        self._mx_source_type = mx.SingleChoice[MxPreview.SourceType](None,  avail=lambda: [*MxPreview.SourceType],
                                                                            filter=self._on_source_type).dispose_with(self)
        self._mx_source_type.set( MxPreview.SourceType(state.get('source_type', MxPreview.SourceType.DataGenerator.value)) )

    @property
    def mx_data_generator(self) -> MxDataGenerator: return self._data_generator
    @property
    def mx_model(self) -> MxModel: return self._model
    @property
    def mx_error(self) -> mx.ITextEmitter_r:
        return self._mx_error
    @property
    def mx_sample(self) -> mx.IProperty_r[Sample|None]:
        """Current preview sample."""
        return self._mx_sample
    @property
    def mx_source_type(self) -> mx.ISingleChoice[SourceType]: return self._mx_source_type
    @property
    def mx_directory_path(self) -> mx.IPathState:
        """Avail when mx_source_type == Directory"""
        return self._mx_directory_path
    @property
    def mx_directory_image_idx(self) -> mx.INumber:
        """Control current image from directory.

        Avail when `mx_source_type == Directory` and `mx_directory_path.mx_path is not None`
        """
        return self._mx_directory_image_idx
    @property
    def mx_patch_mode(self) -> mx.IFlag:
        """Avail when `mx_source_type == Directory` and `mx_directory_path.mx_path is not None`"""
        return self._mx_patch_mode
    @property
    def mx_sample_count(self) -> mx.INumber:
        """Avail when mx_patch_mode == True"""
        return self._mx_sample_count
    @property
    def mx_fix_borders(self) -> mx.IFlag:
        """Avail when mx_patch_mode == True"""
        return self._mx_fix_borders

    @ax.task
    def generate_one(self):
        """avail when `mx_source_type == DataGenerator`"""
        yield ax.switch_to(self._main_thread)
        if self._data_gen_tg.count != 0:
            yield ax.cancel()
        yield ax.attach_to(self._data_gen_tg)

        model = self._model

        yield ax.wait(data_gen_task := self._data_generator.generate(batch_size=1,
                                                                     W =(res := model.get_input_resolution()),
                                                                     H = res,
                                                                     grayscale=model.get_input_ch()==1))
        if data_gen_task.succeeded:
            gen_result = data_gen_task.result

            yield ax.wait(infer_task := model.infer(gen_result.image_np))

            if infer_task.succeeded:
                self._mx_sample.set(MxPreview.Sample(   image_np       = gen_result.image_np[0],
                                                        target_mask_np = gen_result.target_mask_np[0],
                                                        pred_mask_np   = infer_task.result.pred_mask_np[0],
                                                        name           = gen_result.image_paths[0].name ))
            else:
                yield ax.cancel(infer_task.error)
        else:
            yield ax.cancel(data_gen_task.error)

    def _on_source_type(self, new_source_type, _):
        self.get_state()
        self._source_type_disp_bag = self._source_type_disp_bag.dispose_and_new().dispose_with(self)

        if new_source_type == MxPreview.SourceType.DataGenerator:
            self._data_gen_tg = ax.TaskGroup().dispose_with(self._source_type_disp_bag)

        elif new_source_type == MxPreview.SourceType.Directory:
            self._mx_directory_path = mx.PathState( config=mx.PathStateConfig(dir_only=True),
                                                    on_close=self._on_directory_path_close,
                                                    on_open=self._on_directory_path_open,
                                                    ).dispose_with(self._source_type_disp_bag)

            if (directory_path := self._state['directory_state'].get('directory_path', None)) is not None:
                self._mx_directory_path.open(directory_path)

        return new_source_type

    def _on_directory_path_close(self):
        self.get_state()
        self._directory_tg = self._directory_tg.dispose()
        self._directory_disp_bag.dispose()
        self._imagespaths = []

    def _on_directory_path_open(self, path : Path):
        try:
            imagespaths = lib_path.get_files_paths(path, extensions=NPImage.avail_image_suffixes())
        except Exception as e:
            self._mx_error.emit(str(e))
            return False
        self._imagespaths = imagespaths

        self._directory_disp_bag = mx.Disposable()
        self._directory_tg = ax.TaskGroup()

        self._mx_directory_image_idx = mx.Number(self._state['directory_state'].get('directory_image_idx', 0), mx.NumberConfig(min=0, max=len(self._imagespaths)-1)).dispose_with(self._directory_disp_bag)
        self._mx_directory_image_idx.listen(lambda _: self._infer_directory_sample())

        self._mx_patch_mode = mx.Flag(self._state['directory_state'].get('patch_mode', False)).dispose_with(self._directory_disp_bag)
        self._mx_patch_mode.listen(lambda _: self._infer_directory_sample())

        self._mx_sample_count = mx.Number(self._state['directory_state'].get('sample_count', 2), mx.NumberConfig(min=1, max=4)).dispose_with(self._directory_disp_bag)
        self._mx_sample_count.listen(lambda _: self._infer_directory_sample())
        
        self._mx_fix_borders = mx.Flag(self._state['directory_state'].get('fix_borders', False)).dispose_with(self._directory_disp_bag)
        self._mx_fix_borders.listen(lambda _: self._infer_directory_sample())
        
        self._infer_directory_sample()

        return True

    @ax.task
    def _infer_directory_sample(self):
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._directory_tg, cancel_all=True)

        if len(self._imagespaths) == 0:
            yield ax.cancel()

        idx = self._mx_directory_image_idx.get()
        patch_mode = self._mx_patch_mode.get()
        sample_count = self._mx_sample_count.get()
        fix_borders = self._mx_fix_borders.get()

        model = self._model
        imagepath = self._imagespaths[idx]

        yield ax.switch_to(self._sub_thread)

        err = None
        try:
            image_np = NPImage.from_file(imagepath)
        except Exception as e:
            err = e

        pred_mask_np = None

        if err is None:

            if patch_mode:
                patcher = Patcher(image_np, model.get_input_resolution(), sample_count=sample_count, use_padding=fix_borders)

                for i in range(patcher.patch_count):
                    yield ax.wait(t := model.infer([patcher.get_patch(i)]))

                    if t.succeeded:
                        patcher.merge_patch(i, t.result.pred_mask_np[0] )
                    else:
                        err = t.error
                        break

                if err is None:
                    pred_mask_np = patcher.get_merged_image()

            else:
                yield ax.wait(t := model.infer([image_np]))

                if t.succeeded:
                    pred_mask_np=t.result.pred_mask_np[0]
                else:
                    err = t.error

        yield ax.switch_to(self._main_thread)

        if err is None:
            sample = MxPreview.Sample(  image_np=image_np,
                                        target_mask_np=None,
                                        pred_mask_np=pred_mask_np,
                                        name = imagepath.name,
                                        )

            self._mx_sample.set(sample)
        else:
            self._mx_error.emit(str(err))
            yield ax.cancel(error=err)

    def get_state(self) -> dict:
        # Also updates current state in order to save nested controls
        d = self._state
        d['sample_state'] = sample.get_state() if (sample := self._mx_sample.get()) else None
        d['source_type'] = self._mx_source_type.get().value

        if self._mx_source_type.get() == MxPreview.SourceType.Directory:
            d['directory_state']['directory_path'] = self._mx_directory_path.mx_path.get()

            if self._mx_directory_path.mx_path.get() is not None:
                d['directory_state']['directory_image_idx'] = self._mx_directory_image_idx.get()
                d['directory_state']['patch_mode'] = self._mx_patch_mode.get()
                d['directory_state']['sample_count'] = self._mx_sample_count.get()
                d['directory_state']['fix_borders'] = self._mx_fix_borders.get()
        return d

