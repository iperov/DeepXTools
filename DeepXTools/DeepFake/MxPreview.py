from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import numpy as np

from common.SSI import MxSSI
from core import ax, mx
from core.lib import path as lib_path
from core.lib.image import NPImage

from .MxDataGenerator import MxDataGenerator
from .MxModel import MxModel


class MxPreview(mx.Disposable):
    """
    Manages preview.
    """
    class SourceType(Enum):
        DataGenerator = auto()
        Directory = auto()

    def __init__(self, data_generator_src : MxDataGenerator,
                       data_generator_dst : MxDataGenerator,
                       model : MxModel, state : dict = None):
        super().__init__()
        self._state = state = state or {}
        state['directory_state'] = state.get('directory_state', {})

        self._main_thread = ax.get_current_thread()
        self._sub_thread = ax.Thread().dispose_with(self)

        self._gen_src = data_generator_src
        self._gen_dst = data_generator_dst

        self._model = model
        self._imagespaths = []

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_ssi_sheet = mx.Property[MxSSI.Sheet]( MxSSI.Sheet.from_state(ssi_sheet_state) if (ssi_sheet_state := state.get('ssi_sheet_state', None)) is not None else MxSSI.Sheet() ).dispose_with(self)

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
    def mx_ssi_sheet(self) -> mx.IProperty_r[MxSSI.Sheet]:
        """Current SSI sheet"""
        return self._mx_ssi_sheet
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

        gen_src_task = self._gen_src.generate(  batch_size=1,
                                                W =(res := model.get_input_resolution()),
                                                H = res,
                                                grayscale=model.get_input_ch()==1)

        gen_dst_task = self._gen_dst.generate(  batch_size=1,
                                                W =(res := model.get_input_resolution()),
                                                H = res,
                                                grayscale=model.get_input_ch()==1)

        yield ax.wait([gen_src_task, gen_dst_task])

        if not gen_src_task.succeeded:
            yield ax.cancel(gen_src_task.error)
        if not gen_dst_task.succeeded:
            yield ax.cancel(gen_dst_task.error)

        gen_src_result = gen_src_task.result
        gen_dst_result = gen_dst_task.result

        yield ax.wait(step_task := model.step(MxModel.StepRequest(
                                                src_image_np=gen_src_result.image_np,
                                                dst_image_np=gen_dst_result.image_np,
                                                pred_src_image=True,
                                                pred_src_mask=True,
                                                pred_dst_image=True,
                                                pred_dst_mask=True,
                                                pred_swap_image=True,
                                                pred_swap_mask=True,
                                                pred_src_enhance=True,
                                                pred_swap_enhance=True,
                                                )))

        if step_task.succeeded:
            step_task_result = step_task.result
            sections = {}

            sections['src'] = MxSSI.Grid( { (0,0) : MxSSI.Image(image=gen_src_result.image_np[0], caption=gen_src_result.image_paths[0].name),
                                            (0,1) : MxSSI.Image(image=step_task_result.pred_src_image_np[0] if step_task_result.pred_src_image_np is not None else None, caption='@(QxPreview.Predicted_image)'),
                                            (0,2) : MxSSI.Image(image=step_task_result.pred_src_mask_np[0] if step_task_result.pred_src_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)')} )

            if step_task_result.pred_src_enhance_np is not None:
                sections['src_enhance'] = MxSSI.Grid( { (0,0) : MxSSI.Image(image=gen_src_result.image_np[0], caption=gen_src_result.image_paths[0].name),
                                                        (0,1) : MxSSI.Image(image=step_task_result.pred_src_enhance_np[0], caption='@(QxPreview.Predicted_image)'),
                                                        (0,2) : MxSSI.Image(image=step_task_result.pred_src_mask_np[0] if step_task_result.pred_src_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)') } )

            sections['dst'] = MxSSI.Grid( { (0,0) : MxSSI.Image(image=gen_dst_result.image_np[0], caption=gen_dst_result.image_paths[0].name),
                                            (0,1) : MxSSI.Image(image=step_task_result.pred_dst_image_np[0] if step_task_result.pred_dst_image_np is not None else None, caption='@(QxPreview.Predicted_image)'),
                                            (0,2) : MxSSI.Image(image=step_task_result.pred_dst_mask_np[0] if step_task_result.pred_dst_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)')  } )

            sections['swap'] = MxSSI.Grid( {(0,0) : MxSSI.Image(image=gen_dst_result.image_np[0], caption=gen_dst_result.image_paths[0].name),
                                            (0,1) : MxSSI.Image(image=step_task_result.pred_swap_image_np[0] if step_task_result.pred_swap_image_np is not None else None, caption='@(QxPreview.Predicted_image)'),
                                            (0,2) : MxSSI.Image(image=step_task_result.pred_swap_mask_np[0] if step_task_result.pred_swap_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)') } )

            if step_task_result.pred_swap_enhance_np is not None:
                sections['swap_enhance'] = MxSSI.Grid( {(0,0) : MxSSI.Image(image=gen_dst_result.image_np[0], caption=gen_dst_result.image_paths[0].name),
                                                        (0,1) : MxSSI.Image(image=step_task_result.pred_swap_enhance_np[0], caption='@(QxPreview.Predicted_image)'),
                                                        (0,2) : MxSSI.Image(image=step_task_result.pred_swap_mask_np[0] if step_task_result.pred_swap_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)') } )


            self._mx_ssi_sheet.set(MxSSI.Sheet(sections=sections))

        else:
            yield ax.cancel(step_task.error)

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

        self._mx_directory_image_idx = mx.Number(0, mx.NumberConfig(min=0, max=len(self._imagespaths)-1)).dispose_with(self._directory_disp_bag)
        self._mx_directory_image_idx.set( self._state['directory_state'].get('directory_image_idx', 0) )
        self._mx_directory_image_idx.listen(lambda _: self.update_directory_sample())

        self.update_directory_sample()

        return True

    @ax.task
    def update_directory_sample(self):
        """
        Avail when `mx_source_type == Directory` and `mx_directory_path.mx_path is not None`
        """
        yield ax.switch_to(self._main_thread)
        yield ax.attach_to(self._directory_tg, cancel_all=True)

        if len(self._imagespaths) == 0:
            yield ax.cancel()

        idx = self._mx_directory_image_idx.get()

        model = self._model
        imagepath = self._imagespaths[idx]

        yield ax.switch_to(self._sub_thread)

        err = None
        try:
            image_np = NPImage.from_file(imagepath)
        except Exception as e:
            err = e

        if err is None:
            yield ax.wait(step_task := model.step(MxModel.StepRequest(dst_image_np=[image_np],
                                                                      pred_swap_image=True,
                                                                      pred_swap_enhance=True,
                                                                      pred_swap_mask=True)))
            if step_task.succeeded:
                step_task_result = step_task.result
                pred_swap_image_np = step_task_result.pred_swap_image_np[0]
                pred_swap_mask_np = step_task_result.pred_swap_mask_np[0]
            else:
                err = step_task.error

        yield ax.switch_to(self._main_thread)

        if err is None:
            sections = {'swap' : MxSSI.Grid( {  (0,0) : MxSSI.Image(image=image_np, caption=imagepath.name),
                                                (0,1) : MxSSI.Image(image=pred_swap_image_np, caption='@(QxPreview.Predicted_image)'),
                                                (0,2) : MxSSI.Image(image=pred_swap_mask_np, caption='@(QxPreview.Predicted_mask)') } ), }

            if step_task_result.pred_swap_enhance_np is not None:
                sections['swap_enhance'] = MxSSI.Grid( {(0,0) : MxSSI.Image(image=image_np, caption=imagepath.name),
                                                        (0,1) : MxSSI.Image(image=step_task_result.pred_swap_enhance_np[0], caption='@(QxPreview.Predicted_image)'),
                                                        (0,2) : MxSSI.Image(image=step_task_result.pred_swap_mask_np[0] if step_task_result.pred_swap_mask_np is not None else None, caption='@(QxPreview.Predicted_mask)') } )

            self._mx_ssi_sheet.set(MxSSI.Sheet(sections=sections))

        else:
            self._mx_error.emit(str(err))
            yield ax.cancel(error=err)

    def get_state(self) -> dict:
        # Also updates current state in order to save nested controls
        d = self._state
        d['ssi_sheet_state'] = self._mx_ssi_sheet.get().get_state()
        d['source_type'] = self._mx_source_type.get().value

        if self._mx_source_type.get() == MxPreview.SourceType.Directory:
            d['directory_state']['directory_path'] = self._mx_directory_path.mx_path.get()

            if self._mx_directory_path.mx_path.get() is not None:
                d['directory_state']['directory_image_idx'] = self._mx_directory_image_idx.get()

        return d




    # @dataclass(frozen=True)
    # class Sample:
    #     src_image_np : NPImage|None
    #     src_target_mask_np : NPImage|None
    #     dst_image_np : NPImage|None
    #     dst_target_mask_np : NPImage|None
    #     pred_src_image_np : NPImage|None
    #     pred_src_mask_np : NPImage|None
    #     pred_dst_image_np : NPImage|None
    #     pred_dst_mask_np : NPImage|None
    #     pred_swap_image_np : NPImage|None
    #     pred_swap_mask_np : NPImage|None

    #     src_name : str|None
    #     dst_name : str|None

    #     @staticmethod
    #     def from_state(state : dict) -> MxPreview.Sample|None:
    #         """Try to construct Sample from state. If state is not valid, returns None."""
    #         try:
    #             sample = MxPreview.Sample(
    #                     src_image_np       = NPImage(state['src_image_np']),
    #                     src_target_mask_np = NPImage(src_target_mask_np) if (src_target_mask_np := state.get('src_target_mask_np', None)) is not None else None,
    #                     dst_image_np       = NPImage(state['dst_image_np']),
    #                     dst_target_mask_np = NPImage(dst_target_mask_np) if (dst_target_mask_np := state.get('dst_target_mask_np', None)) is not None else None,

    #                     pred_src_image_np  = NPImage(state['pred_src_image_np']),
    #                     pred_src_mask_np   = NPImage(state['pred_src_mask_np']),
    #                     pred_dst_image_np  = NPImage(state['pred_dst_image_np']),
    #                     pred_dst_mask_np   = NPImage(state['pred_dst_mask_np']),
    #                     pred_swap_image_np = NPImage(state['pred_swap_image_np']),
    #                     pred_swap_mask_np  = NPImage(state['pred_swap_mask_np']),

    #                     src_name = state['src_name'],
    #                     dst_name = state['dst_name'],
    #                 )
    #             return sample
    #         except Exception as e:
    #             return None

    #     def get_state(self) -> dict:
    #         return {'src_image_np'      : self.src_image_np.HWC(),
    #                 'src_target_mask_np' : self.src_target_mask_np.HWC() if self.src_target_mask_np is not None else None,
    #                 'dst_image_np'       : self.dst_image_np.HWC(),
    #                 'dst_target_mask_np' : self.dst_target_mask_np.HWC() if self.dst_target_mask_np is not None else None,

    #                 'pred_src_image_np'  : self.pred_src_image_np.HWC(),
    #                 'pred_src_mask_np'   : self.pred_src_mask_np.HWC(),
    #                 'pred_dst_image_np'  : self.pred_dst_image_np.HWC(),
    #                 'pred_dst_mask_np'   : self.pred_dst_mask_np.HWC(),
    #                 'pred_swap_image_np' : self.pred_swap_image_np.HWC(),
    #                 'pred_swap_mask_np'  : self.pred_swap_mask_np.HWC(),

    #                 'src_name' : self.src_name,
    #                 'dst_name' : self.dst_name,
    #                 }