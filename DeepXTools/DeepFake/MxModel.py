import math
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F

from core import ax, mx
from core.lib import math as lib_math
from core.lib import time as lib_time
from core.lib import torch as lib_torch
from core.lib.image import NPImage
from core.lib.python import cache
from core.lib.torch import functional as xF
from core.lib.torch.init import xavier_uniform
from core.lib.torch.modules import PatchDiscriminator
from core.lib.torch.optim import AdaBelief, Optimizer


class MxModel(mx.Disposable):

    class Stage(Enum):
        AutoEncoder = auto()
        Enhancer = auto()


    @dataclass
    class StepRequest:
        src_image_np : List[NPImage]|None = None
        src_target_image_np : List[NPImage]|None = None
        src_target_mask_np : List[NPImage]|None = None

        dst_image_np : List[NPImage]|None = None
        dst_target_image_np : List[NPImage]|None = None
        dst_target_mask_np : List[NPImage]|None = None

        pred_src_image : bool = False
        pred_src_mask : bool = False
        pred_dst_image : bool = False
        pred_dst_mask : bool = False
        pred_swap_image : bool = False
        pred_swap_mask : bool = False

        pred_src_enhance : bool = False
        pred_swap_enhance : bool = False


        mse_power : float = 1.0
        dssim_x4_power : float = 0.0
        dssim_x8_power : float = 0.0
        dssim_x16_power : float = 1.0
        dssim_x32_power : float = 1.0

        enhancer_gan_power : float = 0.0

        masked_training : bool = True
        edges_priority : bool = True

        batch_acc : int = 1
        lr : float = 5e-5
        lr_dropout : float = 0.3



    @dataclass
    class StepResult:
        src_image_np : List[NPImage]|None = None
        src_target_image_np : List[NPImage]|None = None
        src_target_mask_np : List[NPImage]|None = None
        dst_image_np : List[NPImage]|None = None
        dst_target_image_np : List[NPImage]|None = None
        dst_target_mask_np : List[NPImage]|None = None

        pred_src_image_np : List[NPImage]|None = None
        pred_src_mask_np : List[NPImage]|None = None
        pred_dst_image_np : List[NPImage]|None = None
        pred_dst_mask_np : List[NPImage]|None = None
        pred_swap_image_np : List[NPImage]|None = None
        pred_swap_mask_np : List[NPImage]|None = None

        pred_src_enhance_np : List[NPImage]|None = None
        pred_swap_enhance_np : List[NPImage]|None = None

        time : float = 0.0
        iteration : int = 0
        error_src : float = 0.0
        error_dst : float = 0.0


    def __init__(self, state : dict = None):
        super().__init__()
        state = state or {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._io_thread = ax.Thread('io_thread').dispose_with(self)
        self._prepare_thread_pool = ax.ThreadPool(name='prepare_thread_pool').dispose_with(self)
        self._model_thread = ax.Thread('model_thread').dispose_with(self)

        # Model variables. Changed in _model_thread.
        self._device     = lib_torch.DeviceRef.from_state(state.get('device_state', None))
        self._iteration  = state.get('iteration', 0)
        self._opt_class  = AdaBelief
        self._resolution       = state.get('resolution', 224)
        self._encoder_dim      = state.get('encoder_dim', 64)
        self._ae_dim           = state.get('ae_dim', 384)
        self._decoder_dim      = state.get('decoder_dim', 64)
        self._decoder_mask_dim = state.get('decoder_mask_dim', 24)
        self._enhancer_depth   = state.get('enhancer_depth', 4)
        self._enhancer_dim     = state.get('enhancer_dim', 32)
        self._enhancer_upscale_factor = state.get('enhancer_upscale_factor', 1)
        self._enhancer_gan_dim = state.get('enhancer_gan_dim', 32)
        self._enhancer_gan_patch_size = state.get('enhancer_gan_patch_size', 0)

        self._stage            = MxModel.Stage( state.get('stage', MxModel.Stage.AutoEncoder.value) )
        self._mixed_precision = state.get('mixed_precision', False)

        self._mx_info = mx.TextEmitter().dispose_with(self)
        self._mx_device = mx.SingleChoice[lib_torch.DeviceRef](self._device,
                                                               avail=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)
        self._mx_resolution       = mx.Number(self._resolution, config=mx.NumberConfig(min=64, max=1024, step=32)).dispose_with(self)
        self._mx_encoder_dim      = mx.Number(self._encoder_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_ae_dim           = mx.Number(self._ae_dim, config=mx.NumberConfig(min=32, max=1024, step=8)).dispose_with(self)
        self._mx_decoder_dim      = mx.Number(self._decoder_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_decoder_mask_dim = mx.Number(self._decoder_mask_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_stage = mx.SingleChoice[MxModel.Stage](self._stage, avail=lambda: [x for x in MxModel.Stage]).dispose_with(self)
        self._mx_enhancer_depth    = mx.Number(self._enhancer_depth, config=mx.NumberConfig(min=3, max=5, step=1)).dispose_with(self)
        self._mx_enhancer_dim      = mx.Number(self._enhancer_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_enhancer_upscale_factor = mx.Number(self._enhancer_upscale_factor, config=mx.NumberConfig(min=1, max=4, step=1)).dispose_with(self)
        self._mx_enhancer_gan_dim        = mx.Number(self._enhancer_gan_dim, config=mx.NumberConfig(min=8, max=256, step=8)).dispose_with(self)
        self._mx_enhancer_gan_patch_size = mx.Number(self._enhancer_gan_patch_size, config=mx.NumberConfig(min=0, max=PatchDiscriminator.get_max_patch_size(max_downs=5), step=1, zero_is_auto=True)).dispose_with(self)

        self._mx_mixed_precision = mx.Flag(self._mixed_precision).dispose_with(self)




        self._mod = lib_torch.ModulesOnDemand(  {'encoder': lambda mod: Encoder(resolution=self._resolution, in_ch=3, base_dim=self._encoder_dim),
                                                 'encoder_opt': lambda mod: self._opt_class(mod.get_module('encoder').parameters()),

                                                 'inter_src': lambda mod: Inter(in_ch=(in_ch := mod.get_module('encoder').get_out_ch()), in_res=(in_res := mod.get_module('encoder').get_out_res()), bottleneck_ch=self._ae_dim,
                                                                                out_res=in_res, out_ch=in_ch),
                                                 'inter_src_opt': lambda mod: self._opt_class(mod.get_module('inter_src').parameters()),

                                                 'inter_dst': lambda mod: Inter(in_ch=(in_ch := mod.get_module('encoder').get_out_ch()), in_res=(in_res := mod.get_module('encoder').get_out_res()), bottleneck_ch=self._ae_dim,
                                                                                out_res=in_res, out_ch=in_ch),
                                                 'inter_dst_opt': lambda mod: self._opt_class(mod.get_module('inter_dst').parameters()),

                                                 'decoder': lambda mod: Decoder(in_ch=mod.get_module('inter_src').get_out_ch()*2, base_dim=self._decoder_dim, out_ch=3),
                                                 'decoder_opt': lambda mod: self._opt_class(mod.get_module('decoder').parameters()),

                                                 'decoder_mask': lambda mod: Decoder(in_ch=mod.get_module('inter_src').get_out_ch()*2, base_dim=self._decoder_mask_dim, out_ch=1, use_residuals=False),
                                                 'decoder_mask_opt': lambda mod: self._opt_class(mod.get_module('decoder_mask').parameters()),

                                                 'enhancer': lambda mod: Enhancer(in_ch=3, out_ch=3, base_dim=self._enhancer_dim, n_downs=self._enhancer_depth),
                                                 'enhancer_opt': lambda mod: self._opt_class(mod.get_module('enhancer').parameters()),

                                                 'enhancer_dis': lambda mod: PatchDiscriminator(in_ch=3,
                                                                                                patch_size=((self._resolution*self._enhancer_upscale_factor) // 8) if self._enhancer_gan_patch_size == 0 else self._enhancer_gan_patch_size,
                                                                                                base_dim=self._enhancer_gan_dim,
                                                                                                max_downs=5),
                                                 'enhancer_dis_opt': lambda mod: self._opt_class(mod.get_module('enhancer_dis').parameters()),
                                                },
                                                state=state.get('mm_state', None) ).dispose_with(self)
    @property
    def mx_info(self) -> mx.ITextEmitter_r: return self._mx_info
    @property
    def mx_device(self) -> mx.ISingleChoice[lib_torch.DeviceRef]:
        return self._mx_device
    @property
    def mx_resolution(self) -> mx.INumber:
        return self._mx_resolution
    @property
    def mx_encoder_dim(self) -> mx.INumber:
        return self._mx_encoder_dim
    @property
    def mx_ae_dim(self) -> mx.INumber:
        return self._mx_ae_dim
    @property
    def mx_decoder_dim(self) -> mx.INumber:
        return self._mx_decoder_dim
    @property
    def mx_decoder_mask_dim(self) -> mx.INumber:
        return self._mx_decoder_mask_dim
    @property
    def mx_enhancer_depth(self) -> mx.INumber:
        return self._mx_enhancer_depth
    @property
    def mx_enhancer_dim(self) -> mx.INumber:
        return self._mx_enhancer_dim
    @property
    def mx_enhancer_upscale_factor(self) -> mx.INumber:
        return self._mx_enhancer_upscale_factor
    @property
    def mx_enhancer_gan_dim(self) -> mx.INumber:
        return self._mx_enhancer_gan_dim
    @property
    def mx_enhancer_gan_patch_size(self) -> mx.INumber:
        return self._mx_enhancer_gan_patch_size

    @property
    def mx_stage(self) -> mx.ISingleChoice[Stage]:
        return self._mx_stage
    @property
    def mx_mixed_precision(self) -> mx.IFlag:
        return self._mx_mixed_precision

    @ax.task
    def get_state(self) -> dict:
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        return {'device_state' : self._device.get_state(),
                'iteration'    : self._iteration } | self._get_model_state()

    def _get_model_state(self) -> dict:
        self._model_thread.assert_current_thread()

        return {'resolution'       : self._resolution,
                'encoder_dim'      : self._encoder_dim,
                'ae_dim'           : self._ae_dim,
                'decoder_dim'      : self._decoder_dim,
                'decoder_mask_dim' : self._decoder_mask_dim,
                'enhancer_depth'   : self._enhancer_depth,
                'enhancer_dim'     : self._enhancer_dim,
                'enhancer_upscale_factor' : self._enhancer_upscale_factor,
                'enhancer_gan_dim' : self._enhancer_gan_dim,
                'enhancer_gan_patch_size' : self._enhancer_gan_patch_size,
                'stage'            : self._stage.value,
                'mixed_precision'  : self._mixed_precision,
                'mm_state'         : self._mod.get_state(), }

    def get_input_resolution(self) -> int:
        if self._stage == MxModel.Stage.Enhancer:
            return self._resolution * self._enhancer_upscale_factor

        return self._resolution

    def get_input_ch(self) -> int: return 3

    @ax.task
    def apply_model_settings(self):
        """Apply mx model settings to actual model."""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        new_device           = self._mx_device.get()
        new_resolution       = self._mx_resolution.get()
        new_encoder_dim      = self._mx_encoder_dim.get()
        new_ae_dim           = self._mx_ae_dim.get()
        new_decoder_dim      = self._mx_decoder_dim.get()
        new_decoder_mask_dim = self._mx_decoder_mask_dim.get()
        new_enhancer_depth   = self._mx_enhancer_depth.get()
        new_enhancer_dim     = self._mx_enhancer_dim.get()
        new_enhancer_upscale_factor = self._mx_enhancer_upscale_factor.get()
        new_enhancer_gan_dim = self._mx_enhancer_gan_dim.get()
        new_enhancer_gan_patch_size = self._mx_enhancer_gan_patch_size.get()

        new_stage            = self._mx_stage.get()
        new_mixed_precision  = self._mx_mixed_precision.get()

        yield ax.switch_to(self._model_thread)

        mod = self._mod

        device_info, self._device = self._device, new_device
        resolution, self._resolution = self._resolution, new_resolution
        encoder_dim, self._encoder_dim = self._encoder_dim, new_encoder_dim
        ae_dim, self._ae_dim = self._ae_dim, new_ae_dim
        decoder_dim, self._decoder_dim = self._decoder_dim, new_decoder_dim
        decoder_mask_dim, self._decoder_mask_dim = self._decoder_mask_dim, new_decoder_mask_dim

        enhancer_depth, self._enhancer_depth = self._enhancer_depth, new_enhancer_depth
        enhancer_dim, self._enhancer_dim = self._enhancer_dim, new_enhancer_dim
        enhancer_upscale_factor, self._enhancer_upscale_factor = self._enhancer_upscale_factor, new_enhancer_upscale_factor
        enhancer_gan_patch_size, self._enhancer_gan_patch_size = self._enhancer_gan_patch_size, new_enhancer_gan_patch_size
        enhancer_gan_dim, self._enhancer_gan_dim = self._enhancer_gan_dim, new_enhancer_gan_dim

        stage, self._stage = self._stage, new_stage
        mixed_precision, self._mixed_precision = self._mixed_precision, new_mixed_precision

        reset_encoder      = (resolution != new_resolution or encoder_dim != new_encoder_dim)
        reset_inter        = (reset_encoder or ae_dim != new_ae_dim)
        reset_decoder      = (reset_inter or decoder_dim != new_decoder_dim)
        reset_decoder_mask = (reset_inter or decoder_mask_dim != new_decoder_mask_dim)
        reset_enhancer     = (enhancer_depth != new_enhancer_depth) or (enhancer_dim != new_enhancer_dim)
        reset_enhancer_dis = reset_enhancer or (enhancer_upscale_factor != new_enhancer_upscale_factor) \
                                            or (enhancer_gan_dim != new_enhancer_gan_dim) \
                                            or (enhancer_gan_patch_size != new_enhancer_gan_patch_size)

        if reset_encoder:
            mod.reset_module('encoder')
            mod.reset_module('encoder_opt')
        if reset_inter:
            mod.reset_module('inter_src')
            mod.reset_module('inter_src_opt')
            mod.reset_module('inter_dst')
            mod.reset_module('inter_dst_opt')
        if reset_decoder:
            mod.reset_module('decoder')
            mod.reset_module('decoder_opt')
        if reset_decoder_mask:
            mod.reset_module('decoder_mask')
            mod.reset_module('decoder_mask_opt')
        if reset_enhancer:
            mod.reset_module('enhancer')
            mod.reset_module('enhancer_opt')
        if reset_enhancer_dis:
            mod.reset_module('enhancer_dis')
            mod.reset_module('enhancer_dis_opt')

        if new_stage == MxModel.Stage.AutoEncoder:
            ...
        elif new_stage == MxModel.Stage.Enhancer:
            ...

        if reset_encoder or reset_inter or reset_decoder or reset_decoder_mask or reset_enhancer or reset_enhancer_dis:
            torch.cuda.empty_cache()

    @ax.task
    def revert_model_settings(self):
        """Revert mx model settings to actual from the model."""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        device_info = self._device
        resolution  = self._resolution
        encoder_dim      = self._encoder_dim
        ae_dim           = self._ae_dim
        decoder_dim      = self._decoder_dim
        decoder_mask_dim = self._decoder_mask_dim
        enhancer_dim     = self._enhancer_dim
        enhancer_depth   = self._enhancer_depth
        enhancer_upscale_factor = self._enhancer_upscale_factor
        enhancer_gan_dim = self._enhancer_gan_dim
        enhancer_gan_patch_size = self._enhancer_gan_patch_size

        stage            = self._stage
        mixed_precision  = self._mixed_precision

        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device_info)
        self._mx_resolution.set(resolution)
        self._mx_encoder_dim.set(encoder_dim)
        self._mx_ae_dim.set(ae_dim)
        self._mx_decoder_dim.set(decoder_dim)
        self._mx_decoder_mask_dim.set(decoder_mask_dim)
        self._mx_enhancer_dim.set(enhancer_dim)
        self._mx_enhancer_depth.set(enhancer_depth)
        self._mx_enhancer_upscale_factor.set(enhancer_upscale_factor)
        self._mx_enhancer_gan_dim.set(enhancer_gan_dim)
        self._mx_enhancer_gan_patch_size.set(enhancer_gan_patch_size)

        self._mx_stage.set(stage)
        self._mx_mixed_precision.set(mixed_precision)

    ######################################
    ### RESET / IMPORT / EXPORT / DOWNLOAD
    ######################################

    @ax.task
    def reset_encoder(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('encoder')
        self._mod.reset_module('encoder_opt')

    @ax.task
    def reset_inter_src(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('inter_src')
        self._mod.reset_module('inter_src_opt')

    @ax.task
    def reset_inter_dst(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('inter_dst')
        self._mod.reset_module('inter_dst_opt')

    @ax.task
    def reset_decoder(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('decoder')
        self._mod.reset_module('decoder_opt')

    @ax.task
    def reset_decoder_mask(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('decoder_mask')
        self._mod.reset_module('decoder_mask_opt')

    @ax.task
    def reset_enhancer(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('enhancer')
        self._mod.reset_module('enhancer_opt')
        self._mod.reset_module('enhancer_dis')
        self._mod.reset_module('enhancer_dis_opt')

    # @ax.task
    # def _import_model_state(self, model_state : dict):
    #     yield ax.attach_to(self._tg)
    #     yield ax.switch_to(self._model_thread)

    #     err = None
    #     try:
    #         input_type = MxModel.InputType(model_state['input_type'])
    #         resolution = model_state['resolution']
    #         base_dim   = model_state['base_dim']
    #         generalization_level = model_state['generalization_level']
    #         mm_state   = model_state['mm_state']
    #     except Exception as e:
    #         err = e

    #     if err is None:
    #         self._input_type = input_type
    #         self._resolution = resolution
    #         self._base_dim = base_dim
    #         self._generalization_level = generalization_level
    #         self._mod.set_state(mm_state)
    #         yield ax.wait(self.revert_model_settings())

    #     if err is not None:
    #         yield ax.cancel(err)

    #################
    ### STEP INFER / TRAIN
    #################

    @ax.task
    def step(self, req : StepRequest) -> StepResult:
        yield ax.attach_to(self._tg, detach_parent=False)


        @cache
        def get_encoder_opt() -> Optimizer: return self._mod.get_module('encoder_opt', device=device)
        @cache
        def get_inter_src_opt() -> Optimizer: return self._mod.get_module('inter_src_opt', device=device)
        @cache
        def get_inter_dst_opt() -> Optimizer: return self._mod.get_module('inter_dst_opt', device=device)
        @cache
        def get_decoder_opt() -> Optimizer: return self._mod.get_module('decoder_opt', device=device)
        @cache
        def get_decoder_mask_opt() -> Optimizer: return self._mod.get_module('decoder_mask_opt', device=device)
        @cache
        def get_enhancer_opt() -> Optimizer: return self._mod.get_module('enhancer_opt', device=device)
        @cache
        def get_enhancer_dis_opt() -> Optimizer: return self._mod.get_module('enhancer_dis_opt', device=device)

        @cache
        def get_encoder() -> Encoder:
            encoder = self._mod.get_module('encoder', device=device, train=train_encoder)
            if train_encoder and (iteration % batch_acc) == 0:
                get_encoder_opt().zero_grad()
            return encoder
        @cache
        def get_inter_src() -> Inter:
            inter_src = self._mod.get_module('inter_src', device=device, train=train_inter)
            if train_inter and (iteration % batch_acc) == 0:
                get_inter_src_opt().zero_grad()
            return inter_src
        @cache
        def get_inter_dst() -> Inter:
            inter_dst = self._mod.get_module('inter_dst', device=device, train=train_inter)
            if train_inter and (iteration % batch_acc) == 0:
                get_inter_dst_opt().zero_grad()
            return inter_dst
        @cache
        def get_decoder() -> Decoder:
            decoder = self._mod.get_module('decoder', device=device, train=train_decoder)
            if train_decoder and (iteration % batch_acc) == 0:
                get_decoder_opt().zero_grad()
            return decoder
        @cache
        def get_decoder_mask() -> Decoder:
            decoder_mask = self._mod.get_module('decoder_mask', device=device, train=train_decoder_mask)
            if train_decoder_mask and (iteration % batch_acc) == 0:
                get_decoder_mask_opt().zero_grad()
            return decoder_mask
        @cache
        def get_enhancer() -> Enhancer:
            enhancer = self._mod.get_module('enhancer', device=device, train=train_enhancer)
            if train_enhancer and (iteration % batch_acc) == 0:
                get_enhancer_opt().zero_grad()
            return enhancer
        @cache
        def get_enhancer_dis() -> PatchDiscriminator:
            enhancer_dis = self._mod.get_module('enhancer_dis', device=device, train=train_enhancer_dis)
            if train_enhancer_dis and (iteration % batch_acc) == 0:
                get_enhancer_dis_opt().zero_grad()
            return enhancer_dis

        def encoder_forward(x : torch.Tensor) -> torch.Tensor:
            with torch.set_grad_enabled(train_encoder):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_encoder()(x)

        def inter_src_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_inter):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_inter_src()(x)

        def inter_dst_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_inter):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_inter_dst()(x)

        def decoder_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_decoder):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_decoder()(x)

        def decoder_mask_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_decoder_mask):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_decoder_mask()(x)

        def enhancer_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_enhancer):
                with torch.autocast(device_type=device.backend, enabled=mixed_precision):
                    return get_enhancer()(x)

        def enhancer_dis_forward(x, no_network_grad=False) -> torch.Tensor:
            with torch.set_grad_enabled(train_enhancer_dis):
                enhancer_dis = get_enhancer_dis()
                if no_network_grad:
                    enhancer_dis.set_requires_grad(False)
                result = enhancer_dis(x)
                if no_network_grad:
                    enhancer_dis.set_requires_grad(True)
                return result

        @cache
        def get_src_image_t() -> torch.Tensor|None: return torch.tensor(src_image_nd, device=device.device) if src_image_nd is not None else None
        @cache
        def get_src_target_mask_t_0() -> torch.Tensor|None: return torch.tensor(src_target_mask_nd, device=device.device) if src_target_mask_nd is not None else None
        @cache
        def get_src_target_mask_t(res : int = None) -> torch.Tensor|None:
            if (src_target_mask_t := get_src_target_mask_t_0()) is not None and res is not None and res != src_target_mask_t.shape[2]:
                src_target_mask_t = F.interpolate(src_target_mask_t, size=res, mode='bilinear')
            return src_target_mask_t
        @cache
        def get_src_target_mask_blur_t(res : int) -> torch.Tensor|None:
            if (src_target_mask_t := get_src_target_mask_t(res=res)) is not None:
                src_target_mask_t = torch.clamp(xF.gaussian_blur(src_target_mask_t, sigma=max(1, src_target_mask_t.shape[2] // 32) ), 0.0, 0.5) * 2.0
            return src_target_mask_t
        @cache
        def get_src_target_image_t_0() -> torch.Tensor|None: return torch.tensor(src_target_image_nd, device=device.device) if src_target_image_nd is not None else None
        @cache
        def get_src_target_image_t_1(sobel : bool = False) -> torch.Tensor|None:
            if (src_target_image_t := get_src_target_image_t_0()) is not None and sobel:
                src_target_image_t = xF.sobel_edges_2d(src_target_image_t)
            return src_target_image_t
        @cache
        def get_src_target_image_t(masked : bool, sobel : bool = False) -> torch.Tensor|None:
            if (src_target_image_t := get_src_target_image_t_1(sobel=sobel)) is not None and masked:
                if (src_target_mask_blur_t := get_src_target_mask_blur_t(res=src_target_image_t.shape[2])) is not None:
                    src_target_image_t = src_target_image_t * src_target_mask_blur_t
            return src_target_image_t
        @cache
        def get_dst_image_t() -> torch.Tensor|None: return torch.tensor(dst_image_nd, device=device.device) if dst_image_nd is not None else None

        @cache
        def get_dst_target_mask_t_0() -> torch.Tensor|None: return torch.tensor(dst_target_mask_nd, device=device.device) if dst_target_mask_nd is not None else None
        @cache
        def get_dst_target_mask_t(res : int = None) -> torch.Tensor|None:
            if (dst_target_mask_t := get_dst_target_mask_t_0()) is not None and res is not None and res != dst_target_mask_t.shape[2]:
                dst_target_mask_t = F.interpolate(dst_target_mask_t, size=res, mode='bilinear')
            return dst_target_mask_t
        @cache
        def get_dst_target_mask_blur_t(res : int = None) -> torch.Tensor|None:
            if (dst_target_mask_t := get_dst_target_mask_t(res=res)) is not None:
                dst_target_mask_t = torch.clamp(xF.gaussian_blur(dst_target_mask_t, sigma=max(1, dst_target_mask_t.shape[2] // 32) ), 0.0, 0.5) * 2.0
            return dst_target_mask_t
        @cache
        def get_dst_target_image_t_0() -> torch.Tensor|None: return torch.tensor(dst_target_image_nd, device=device.device) if dst_target_image_nd is not None else None
        @cache
        def get_dst_target_image_t_1(sobel : bool = False) -> torch.Tensor|None:
            if (dst_target_image_t := get_dst_target_image_t_0()) is not None and sobel:
                dst_target_image_t = xF.sobel_edges_2d(dst_target_image_t)
            return dst_target_image_t
        @cache
        def get_dst_target_image_t(masked : bool, sobel : bool = False) -> torch.Tensor|None:
            if (dst_target_image_t := get_dst_target_image_t_1(sobel=sobel)) is not None and masked:
                if (dst_target_mask_blur_t := get_dst_target_mask_blur_t(res=dst_target_image_t.shape[2])) is not None:
                    dst_target_image_t = dst_target_image_t * dst_target_mask_blur_t
            return dst_target_image_t
        @cache
        def get_src_enc_t() -> torch.Tensor|None: return encoder_forward(src_image_t) if (src_image_t := get_src_image_t()) is not None else None
        @cache
        def get_dst_enc_t() -> torch.Tensor|None: return encoder_forward(dst_image_t) if (dst_image_t := get_dst_image_t()) is not None else None
        @cache
        def get_src_src_code_t() -> torch.Tensor|None: return inter_src_forward(src_enc_t) if (src_enc_t := get_src_enc_t()) is not None else None
        @cache
        def get_src_dst_code_t() -> torch.Tensor|None: return inter_src_forward(dst_enc_t) if (dst_enc_t := get_dst_enc_t()) is not None else None
        @cache
        def get_dst_dst_code_t() -> torch.Tensor|None: return inter_dst_forward(dst_enc_t) if (dst_enc_t := get_dst_enc_t()) is not None else None
        @cache
        def get_src_code_t() -> torch.Tensor|None: return torch.cat([src_src_code_t, src_src_code_t], 1) if (src_src_code_t := get_src_src_code_t()) is not None else None
        @cache
        def get_dst_code_t() -> torch.Tensor|None: return torch.cat([src_dst_code_t, dst_dst_code_t], 1) if (src_dst_code_t := get_src_dst_code_t()) is not None and (dst_dst_code_t := get_dst_dst_code_t()) is not None else None
        @cache
        def get_swap_code_t() -> torch.Tensor|None: return torch.cat([src_dst_code_t, src_dst_code_t], 1) if (src_dst_code_t := get_src_dst_code_t()) is not None else None
        @cache
        def get_pred_src_image_t_0() -> torch.Tensor|None: return decoder_forward(src_code_t) if (src_code_t := get_src_code_t()) is not None else None
        @cache
        def get_pred_src_image_t_1(detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := get_pred_src_image_t_0()) is not None and detach:
                pred_src_image_t = pred_src_image_t.detach()
            return pred_src_image_t
        @cache
        def get_pred_src_image_t_2(sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := get_pred_src_image_t_1(detach=detach)) is not None and sobel:
                pred_src_image_t = xF.sobel_edges_2d(pred_src_image_t)
            return pred_src_image_t
        @cache
        def get_pred_src_image_t_3(masked : bool, sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := get_pred_src_image_t_2(sobel=sobel, detach=detach)) is not None and masked:
                if (src_target_mask_blur_t := get_src_target_mask_blur_t(res=pred_src_image_t.shape[2])) is not None:
                    pred_src_image_t = pred_src_image_t * src_target_mask_blur_t
            return pred_src_image_t
        @cache
        def get_pred_src_image_t(masked : bool, res : int = None, sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_src_image_t := get_pred_src_image_t_3(masked=masked, sobel=sobel, detach=detach)) is not None and res is not None and res != pred_src_image_t.shape[2]:
                pred_src_image_t = F.interpolate(pred_src_image_t, size=res, mode='bilinear')
            return pred_src_image_t
        @cache
        def get_pred_src_enhanced_image_t() -> torch.Tensor|None:
            return enhancer_forward(pred_src_image_t) if (pred_src_image_t := get_pred_src_image_t(masked=False, res=enhancer_resolution)) is not None else None
        @cache
        def get_pred_src_mask_t() -> torch.Tensor|None: return decoder_mask_forward(src_code_t) if (src_code_t := get_src_code_t()) is not None else None


        @cache
        def get_pred_dst_image_t_0() -> torch.Tensor|None: return decoder_forward(dst_code_t) if (dst_code_t := get_dst_code_t()) is not None else None
        @cache
        def get_pred_dst_image_t_1(detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := get_pred_dst_image_t_0()) is not None and detach:
                pred_dst_image_t = pred_dst_image_t.detach()
            return pred_dst_image_t
        @cache
        def get_pred_dst_image_t_2(sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := get_pred_dst_image_t_1(detach=detach)) is not None and sobel:
                pred_dst_image_t = xF.sobel_edges_2d(pred_dst_image_t)
            return pred_dst_image_t
        @cache
        def get_pred_dst_image_t_3(masked : bool, sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := get_pred_dst_image_t_2(sobel=sobel, detach=detach)) is not None and masked:
                if (dst_target_mask_blur_t := get_dst_target_mask_blur_t(res=pred_dst_image_t.shape[2])) is not None:
                    pred_dst_image_t = pred_dst_image_t * dst_target_mask_blur_t
            return pred_dst_image_t
        @cache
        def get_pred_dst_image_t(masked : bool, res : int = None, sobel : bool = False, detach=False) -> torch.Tensor|None:
            if (pred_dst_image_t := get_pred_dst_image_t_3(masked=masked, sobel=sobel, detach=detach)) is not None and res is not None and res != pred_dst_image_t.shape[2]:
                pred_dst_image_t = F.interpolate(pred_dst_image_t, size=res, mode='bilinear')
            return pred_dst_image_t
        @cache
        def get_pred_dst_mask_t() -> torch.Tensor|None: return decoder_mask_forward(dst_code_t) if (dst_code_t := get_dst_code_t()) is not None else None


        @cache
        def get_pred_swap_image_t_0() -> torch.Tensor|None:
            return decoder_forward(swap_code_t) if (swap_code_t := get_swap_code_t()) is not None else None
        @cache
        def get_pred_swap_image_t_1(detach=False) -> torch.Tensor|None:
            if (pred_swap_image_t := get_pred_swap_image_t_0()) is not None and detach:
                pred_swap_image_t = pred_swap_image_t.detach()
            return pred_swap_image_t
        @cache
        def get_pred_swap_image_t(res : int = None, detach=False) -> torch.Tensor|None:
            if (pred_swap_image_t := get_pred_swap_image_t_1(detach=detach)) is not None and res is not None and res != pred_swap_image_t.shape[2]:
                pred_swap_image_t = F.interpolate(pred_swap_image_t, size=res, mode='bilinear')
            return pred_swap_image_t

        @cache
        def get_pred_swap_enhanced_image_t() -> torch.Tensor|None:
            return enhancer_forward(pred_swap_image_t) if (pred_swap_image_t := get_pred_swap_image_t(res=enhancer_resolution)) is not None else None
        @cache
        def get_pred_swap_mask_t() -> torch.Tensor|None:
            return decoder_mask_forward(swap_code_t) if (swap_code_t := get_swap_code_t()) is not None else None

        result = MxModel.StepResult()

        src_image_nd = None
        src_target_image_nd = None
        src_target_mask_nd = None
        dst_image_nd = None
        dst_target_image_nd = None
        dst_target_mask_nd = None
        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            stage = self._stage
            resolution = self._resolution
            enhancer_upscale_factor = self._enhancer_upscale_factor
            enhancer_resolution = resolution * enhancer_upscale_factor

            target_resolution = enhancer_resolution if stage == MxModel.Stage.Enhancer else resolution

            if req.src_image_np is not None:
                src_image_np = result.src_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.src_image_np]
                src_image_nd = np.stack([x.CHW() for x in src_image_np])

            if req.src_target_image_np is not None:
                src_target_image_np = result.src_target_image_np = [ x.bgr().resize(target_resolution, target_resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.src_target_image_np]
                src_target_image_nd = np.stack([x.CHW() for x in src_target_image_np])

            if req.src_target_mask_np is not None:
                src_target_mask_np = result.src_target_mask_np = [x.grayscale().resize(target_resolution, target_resolution, interp=NPImage.Interp.LINEAR).f32() for x in req.src_target_mask_np]
                src_target_mask_nd = np.stack([x.CHW() for x in src_target_mask_np])

            if req.dst_image_np is not None:
                dst_image_np = result.dst_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.dst_image_np]
                dst_image_nd = np.stack([x.CHW() for x in dst_image_np])

            if req.dst_target_image_np is not None:
                dst_target_image_np = result.dst_target_image_np = [ x.bgr().resize(target_resolution, target_resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.dst_target_image_np]
                dst_target_image_nd = np.stack([x.CHW() for x in dst_target_image_np])

            if req.dst_target_mask_np is not None:
                dst_target_mask_np = result.dst_target_mask_np = [x.grayscale().resize(target_resolution, target_resolution, interp=NPImage.Interp.LINEAR).f32() if x is not None else None for x in req.dst_target_mask_np]
                dst_target_mask_nd = np.stack([x.CHW() for x in dst_target_mask_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution and stage == self._stage and enhancer_upscale_factor == self._enhancer_upscale_factor:
                # Prepared data matches model parameters.
                break


        batch_acc  = req.batch_acc
        lr         = req.lr
        lr_dropout = req.lr_dropout

        masked_training = result.masked_training = req.masked_training
        edges_priority  = req.edges_priority

        train_src_image    = stage == MxModel.Stage.AutoEncoder and req.src_target_image_np is not None
        train_src_mask     = stage == MxModel.Stage.AutoEncoder and req.src_target_mask_np is not None
        train_src_enhance  = stage == MxModel.Stage.Enhancer    and req.src_target_image_np is not None
        train_dst_image    = stage == MxModel.Stage.AutoEncoder and req.dst_target_image_np is not None
        train_dst_mask     = stage == MxModel.Stage.AutoEncoder and req.dst_target_mask_np is not None

        train_encoder      = train_src_image or train_dst_image or train_src_mask or train_dst_mask
        train_inter        = train_src_image or train_dst_image or train_src_mask or train_dst_mask
        train_decoder      = train_src_image or train_dst_image
        train_decoder_mask = train_src_mask  or train_dst_mask
        train_enhancer     = train_src_enhance
        train_enhancer_dis = train_enhancer

        iteration = self._iteration
        device = self._device
        mixed_precision = self._mixed_precision

        step_time = lib_time.measure()

        try:

            # Collect losses
            src_losses = []
            dst_losses = []
            dis_losses = []
            if stage == MxModel.Stage.AutoEncoder:

                if (src_target_masked_image_t := get_src_target_image_t(masked=masked_training)) is not None:
                  if (src_target_image_t := get_src_target_image_t(masked=False)) is not None:
                    if (pred_src_masked_image_t := get_pred_src_image_t(masked=masked_training)) is not None:

                        if (dssim_x4_power := req.dssim_x4_power) != 0.0:
                            src_losses.append( dssim_x4_power*xF.dssim(pred_src_masked_image_t, src_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//4), use_padding=False).mean([-1]) )

                        if (dssim_x8_power := req.dssim_x8_power) != 0.0:
                            src_losses.append( dssim_x8_power*xF.dssim(pred_src_masked_image_t, src_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//8), use_padding=False).mean([-1]) )

                        if (dssim_x16_power := req.dssim_x16_power) != 0.0:
                            src_losses.append( dssim_x16_power*xF.dssim(pred_src_masked_image_t, src_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//16), use_padding=False).mean([-1]) )

                        if (dssim_x32_power := req.dssim_x32_power) != 0.0:
                            src_losses.append( dssim_x32_power*xF.dssim(pred_src_masked_image_t, src_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//32), use_padding=False).mean([-1]) )

                        if (mse_power := req.mse_power) != 0.0:

                            src_losses.append( torch.mean(mse_power*10*torch.square(pred_src_masked_image_t-src_target_masked_image_t), (1,2,3)) )

                            if edges_priority and \
                              (src_target_sobel_image_t := get_src_target_image_t(masked=masked_training, sobel=True)) is not None and \
                              (pred_src_sobel_image_t := get_pred_src_image_t(masked=masked_training, sobel=True)) is not None:
                                src_losses.append( torch.mean(mse_power*10*torch.abs(pred_src_sobel_image_t-src_target_sobel_image_t), (1,2,3)) )



                if (pred_src_mask_t := get_pred_src_mask_t()) is not None:
                    if (src_target_mask_t := get_src_target_mask_t(res=pred_src_mask_t.shape[2])) is not None:
                        if (mse_power := req.mse_power) != 0.0:
                            src_losses.append( torch.mean(mse_power*10*torch.square(pred_src_mask_t-src_target_mask_t), (1,2,3)) )

                if (dst_target_masked_image_t := get_dst_target_image_t(masked=masked_training)) is not None:
                  if (dst_target_image_t := get_dst_target_image_t(masked=False)) is not None:
                    if (pred_dst_masked_image_t := get_pred_dst_image_t(masked=masked_training)) is not None:

                        if (dssim_x4_power := req.dssim_x4_power) != 0.0:
                            dst_losses.append( dssim_x4_power*xF.dssim(pred_dst_masked_image_t, dst_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//4), use_padding=False).mean([-1]) )

                        if (dssim_x8_power := req.dssim_x8_power) != 0.0:
                            dst_losses.append( dssim_x8_power*xF.dssim(pred_dst_masked_image_t, dst_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//8), use_padding=False).mean([-1]) )

                        if (dssim_x16_power := req.dssim_x16_power) != 0.0:
                            dst_losses.append( dssim_x16_power*xF.dssim(pred_dst_masked_image_t, dst_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//16), use_padding=False).mean([-1]) )

                        if (dssim_x32_power := req.dssim_x32_power) != 0.0:
                            dst_losses.append( dssim_x32_power*xF.dssim(pred_dst_masked_image_t, dst_target_masked_image_t, kernel_size=lib_math.next_odd(resolution//32), use_padding=False).mean([-1]) )

                        if (mse_power := req.mse_power) != 0.0:
                            dst_losses.append( torch.mean(mse_power*10*torch.square(pred_dst_masked_image_t-dst_target_masked_image_t), (1,2,3)) )

                            if edges_priority and \
                              (dst_target_sobel_image_t := get_dst_target_image_t(masked=masked_training, sobel=True)) is not None and \
                              (pred_dst_sobel_image_t := get_pred_dst_image_t(masked=masked_training, sobel=True)) is not None:
                                dst_losses.append( torch.mean(mse_power*10*torch.abs(pred_dst_sobel_image_t-dst_target_sobel_image_t), (1,2,3)) )



                if (pred_dst_mask_t := get_pred_dst_mask_t()) is not None:
                    if (dst_target_mask_t := get_dst_target_mask_t(res=pred_dst_mask_t.shape[2])) is not None:
                        if (mse_power := req.mse_power) != 0.0:
                            dst_losses.append( torch.mean(mse_power*10*torch.square(pred_dst_mask_t-dst_target_mask_t), (1,2,3)) )

            elif stage == MxModel.Stage.Enhancer:
                if  (src_target_image_t := get_src_target_image_t(masked=False)) is not None and \
                    (pred_src_image_t   := get_pred_src_image_t(masked=False, res=src_target_image_t.shape[2], detach=True)) is not None:

                        if masked_training and (src_target_mask_blur_t := get_src_target_mask_blur_t(res=src_target_image_t.shape[2])) is not None:
                            src_target_image_t = src_target_image_t*src_target_mask_blur_t + pred_src_image_t*(1-src_target_mask_blur_t)

                        pred_src_enhanced_image_t = enhancer_forward(pred_src_image_t)

                        src_losses.append( torch.mean( 5*torch.abs(pred_src_enhanced_image_t-src_target_image_t), (1,2,3)) )

                        if (enhancer_gan_power := req.enhancer_gan_power) != 0:
                            for logit in (logits := enhancer_dis_forward(pred_src_enhanced_image_t, no_network_grad=True)):
                                src_losses.append( enhancer_gan_power*0.1*F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := enhancer_dis_forward(src_target_image_t)):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none').mean((1,2,3))/len(logits) )

                            for logit in (logits := enhancer_dis_forward(pred_src_enhanced_image_t.detach())):
                                dis_losses.append( F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit), reduction='none').mean((1,2,3))/len(logits) )

            if len(dis_losses) != 0:
                sum(dis_losses).mean().backward()

            src_loss_t = None
            if len(src_losses) != 0:
                src_loss_t = sum(src_losses).mean()
                src_loss_t.backward()

            dst_loss_t = None
            if len(dst_losses) != 0:
                dst_loss_t = sum(dst_losses).mean()
                dst_loss_t.backward()

            # Optimization
            if (iteration % batch_acc) == (batch_acc-1):
                grad_mult = 1.0 / batch_acc

                opts = [get_encoder_opt.get_cached() if get_encoder_opt.is_cached() else None,
                        get_inter_src_opt.get_cached() if get_inter_src_opt.is_cached() else None,
                        get_inter_dst_opt.get_cached() if get_inter_dst_opt.is_cached() else None,
                        get_decoder_opt.get_cached() if get_decoder_opt.is_cached() else None,
                        get_decoder_mask_opt.get_cached() if get_decoder_mask_opt.is_cached() else None,
                        get_enhancer_opt.get_cached() if get_enhancer_opt.is_cached() else None,
                        get_enhancer_dis_opt.get_cached() if get_enhancer_dis_opt.is_cached() else None,
                        ]

                for opt in opts:
                    if opt is not None:
                        opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                if any(opt is not None for opt in opts):
                    self._iteration = iteration + 1


            # Collect predicts
            if req.pred_src_image:
                if (pred_src_image_t := get_pred_src_image_t(masked=False)) is not None:
                    result.pred_src_image_np = [ NPImage(x, channels_last=False) for x in pred_src_image_t.detach().cpu().numpy().clip(0, 1) ]

            if req.pred_src_mask:
                if (pred_src_mask_t := get_pred_src_mask_t()) is not None:
                    result.pred_src_mask_np = [ NPImage(x, channels_last=False) for x in pred_src_mask_t.detach().cpu().numpy().clip(0, 1) ]

            if req.pred_dst_image:
                if (pred_dst_image_t := get_pred_dst_image_t(masked=False)) is not None:
                    result.pred_dst_image_np = [ NPImage(x, channels_last=False) for x in pred_dst_image_t.detach().cpu().numpy().clip(0, 1) ]

            if req.pred_dst_mask:
                if (pred_dst_mask_t := get_pred_dst_mask_t()) is not None:
                    result.pred_dst_mask_np = [ NPImage(x, channels_last=False) for x in pred_dst_mask_t.detach().cpu().numpy().clip(0, 1) ]

            if req.pred_swap_image:
                if (pred_swap_image_t := get_pred_swap_image_t()) is not None:
                    result.pred_swap_image_np = [ NPImage(x, channels_last=False) for x in pred_swap_image_t.detach().cpu().numpy().clip(0, 1) ]

            if req.pred_swap_mask:
                if (pred_swap_mask_t := get_pred_swap_mask_t()) is not None:
                    result.pred_swap_mask_np = [ NPImage(x, channels_last=False) for x in pred_swap_mask_t.detach().cpu().numpy().clip(0, 1) ]

            if stage == MxModel.Stage.Enhancer:
                if req.pred_src_enhance:
                    if (pred_src_enhanced_image_t := get_pred_src_enhanced_image_t()) is not None:
                        result.pred_src_enhance_np = [ NPImage(x, channels_last=False) for x in pred_src_enhanced_image_t.detach().cpu().numpy().clip(0, 1) ]

                if req.pred_swap_enhance:
                    if (pred_swap_enhanced_image_t := get_pred_swap_enhanced_image_t()) is not None:
                        result.pred_swap_enhance_np = [ NPImage(x, channels_last=False) for x in pred_swap_enhanced_image_t.detach().cpu().numpy().clip(0, 1) ]


            # Metrics
            result.error_src = float( src_loss_t.detach().cpu().numpy() ) if src_loss_t is not None else 0
            result.error_dst = float( dst_loss_t.detach().cpu().numpy() ) if dst_loss_t is not None else 0
            result.time = step_time.elapsed()
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield ax.cancel(e)


    ##########################
    ##########################
    ##########################





# Network blocks

class SimpleAtten(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self._c0 = nn.Conv2d(ch, ch, 3, 1, 1)
        self._c1 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, inp):
        a = F.leaky_relu(self._c0(inp), 0.1)
        a = F.leaky_relu(self._c1(a), 0.1)

        _, _, H, W = a.size()
        d = (a - a.mean(dim=[2,3], keepdim=True)).pow(2)
        a = d / (4 * (d.sum(dim=[2,3], keepdim=True) / (W * H - 1) + 1e-4)) + 0.5

        return inp*torch.sigmoid(a)

class ResidualBlock(nn.Module):
    def __init__(self, ch, is_decoder=False, atten=False):
        super().__init__()

        self._atten = atten

        self._c0 = nn.ConvTranspose2d(ch, ch, 3, 1, 1) if is_decoder else nn.Conv2d(ch, ch, 3, 1, 1)
        self._c1 = nn.ConvTranspose2d(ch, ch, 3, 1, 1) if is_decoder else nn.Conv2d(ch, ch, 3, 1, 1)

        if atten:
            self._atten = SimpleAtten(ch)

    def forward(self, inp, emb=None):
        x = inp
        x = self._c0(x)
        if emb is not None:
            x = x + emb
        x = F.leaky_relu(x, 0.2)
        x = self._c1(x)
        if self._atten:
            x = self._atten(x)
        x = F.leaky_relu(x + inp, 0.2)
        return x

class Encoder(nn.Module):
    def __init__(self, resolution, in_ch, base_dim, n_downs=5):
        super().__init__()

        dim_mults=[ min(2**i, 8) for i in range(n_downs+1) ]
        dims = [ base_dim * mult for mult in dim_mults[1:] ]

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch*4, dims[0], 1, 1, 0)

        down_c_list = self._down_c_list = nn.ModuleList()
        down_r_list = self._down_r_list = nn.ModuleList()

        for (up_ch, down_ch) in list(zip(dims[:-1], dims[1:])):
            down_c_list.append( nn.Conv2d(up_ch, down_ch, 5, 2, 2) )
            down_r_list.append( ResidualBlock(down_ch) )

        self._out_ch = dims[-1]
        self._out_res = resolution // (2**n_downs)

        xavier_uniform(self)

    def get_out_ch(self): return self._out_ch
    def get_out_res(self): return self._out_res

    def forward(self, inp : torch.Tensor):
        x = inp
        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]

        x = self._in(F.pixel_unshuffle(x, 2))

        for c, r in zip(self._down_c_list, self._down_r_list):
            x = F.leaky_relu(c(x), 0.1)
            x = r(x)

        x = x * (x.square().mean(dim=[1,2,3], keepdim=True) + 1e-06).rsqrt()
        return x

class Inter(nn.Module):
    def __init__(self, in_ch, in_res, bottleneck_ch, out_res, out_ch):
        super().__init__()
        self._in_ch = in_ch
        self._in_res = in_res
        self._out_res = out_res
        self._out_ch = out_ch

        self._fc1 = nn.Linear(in_ch*in_res*in_res, bottleneck_ch)
        self._fc2 = nn.Linear(bottleneck_ch, out_ch*out_res*out_res)
        xavier_uniform(self)

    def get_in_ch(self): return self._in_ch
    def get_out_res(self): return self._out_res
    def get_out_ch(self): return self._out_ch

    def forward(self, inp):
        x = inp
        x = x.reshape(-1, self._in_ch*self._in_res*self._in_res)
        x = self._fc1(x)
        x = self._fc2(x)
        x = x.reshape(-1, self._out_ch, self._out_res, self._out_res)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, base_dim, out_ch, n_ups=5, use_residuals=True):
        super().__init__()
        self._in_ch = in_ch
        self._decoder_ch = base_dim
        self._out_ch = out_ch
        self._use_residuals = use_residuals

        dim_mults=[ min(2**i, 8) for i in range(n_ups+1) ]
        dims = [ base_dim * mult for mult in dim_mults ][1:]

        self._c_in = nn.Conv2d(in_ch, dims[-1], 1, 1, 0)

        up_c_list = self._up_c_list = nn.ModuleList()
        up_r_list = self._up_r_list = nn.ModuleList()

        for (up_ch, down_ch) in list(zip(dims[:-1], dims[1:])):
            up_c_list.insert(0, nn.Conv2d(down_ch, up_ch*4, 3, 1, 1) )
            up_r_list.insert(0, ResidualBlock(up_ch, is_decoder=True, atten=True) if use_residuals else nn.Identity())

        self._up0_0 = nn.Conv2d(dims[0], out_ch, 1, 1, 0)
        self._up0_1 = nn.Conv2d(dims[0], out_ch, 3, 1, 1)
        self._up0_2 = nn.Conv2d(dims[0], out_ch, 3, 1, 1)
        self._up0_3 = nn.Conv2d(dims[0], out_ch, 3, 1, 1)

        self._out_gamma = nn.parameter.Parameter( torch.ones(out_ch,), requires_grad=True)
        self._out_beta = nn.parameter.Parameter( torch.zeros(out_ch,), requires_grad=True)

        xavier_uniform(self)



    def forward(self, inp):
        x = inp

        x = self._c_in(x)

        for c, r in zip(self._up_c_list, self._up_r_list):
            x = F.pixel_shuffle(F.leaky_relu(c(x), 0.1), 2)
            x = r(x)

        x = F.pixel_shuffle( torch.cat([self._up0_0(x),
                                        self._up0_1(x),
                                        self._up0_2(x),
                                        self._up0_3(x) ], 1 ), 2 )

        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x

####

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self._c0 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self._c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, inp):
        x = inp
        x = F.leaky_relu(self._c0(x), 0.1)
        x = F.leaky_relu(self._c1(x), 0.1)
        return x



class Enhancer(nn.Module):

    def __init__(self, in_ch, out_ch, base_dim=32, n_downs=6, dim_mults=(1, 2, 4, 8, 8, 8, 8, 8, 8) ):
        super().__init__()
        self._base_dim = base_dim
        self._n_downs = n_downs

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0, bias=False)

        down_c1_list = self._down_c1_list = nn.ModuleList()
        down_c2_list = self._down_c2_list = nn.ModuleList()
        down_p_list = self._down_p_list = nn.ModuleList()

        up_c1_list = self._up_c1_list = nn.ModuleList()
        up_s_list = self._up_s_list = nn.ModuleList()
        up_r_list = self._up_r_list = nn.ModuleList()
        up_c2_list = self._up_c2_list = nn.ModuleList()

        dims = [ base_dim * mult for mult in dim_mults[:n_downs+1] ]
        for level, (up_ch, down_ch) in enumerate(list(zip(dims[:-1], dims[1:]))):
            down_c1_list.append( ConvBlock(up_ch, up_ch) )
            down_c2_list.append( ConvBlock(up_ch, down_ch) )
            down_p_list.append( nn.MaxPool2d(2) )

            up_c1_list.insert(0, ConvBlock(down_ch, up_ch*4) )

            up_s_list.insert(0, nn.Sequential(  nn.Conv2d(down_ch, up_ch*4, 3, 1, 1),
                                                nn.LeakyReLU(0.2),
                                                nn.PixelShuffle(2),
                                                nn.Conv2d(up_ch, up_ch, 3, 1, 1)))

            up_r_list.insert(0, ResidualBlock(up_ch, is_decoder=True, atten=True) )

            up_c2_list.insert(0, ConvBlock(up_ch, up_ch) )


        self._out = nn.Conv2d(base_dim, out_ch, 1, 1, 0, bias=False)
        self._out_gamma = nn.parameter.Parameter( torch.ones(out_ch,), requires_grad=True)
        self._out_beta = nn.parameter.Parameter( torch.zeros(out_ch,), requires_grad=True)

        xavier_uniform(self)

    def forward(self, inp):
        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        shortcuts = []
        for down_c1, down_c2, down_p in zip(self._down_c1_list, self._down_c2_list, self._down_p_list):
            x = down_c1(x)
            x = down_c2(x)
            x = down_p(x)
            shortcuts.insert(0, x)

        x = x * (x.square().mean(dim=[1,2,3], keepdim=True) + 1e-06).rsqrt()

        for shortcut_x, up_c1, up_s, up_r, up_c2 in zip(shortcuts, self._up_c1_list, self._up_s_list, self._up_r_list, self._up_c2_list,  ):
            x = F.pixel_shuffle(up_c1(x), 2)
            x = up_r(x, emb=up_s(shortcut_x))
            x = up_c2(x)

        x = self._out(x)
        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x


