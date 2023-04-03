from dataclasses import dataclass
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
from core.lib.torch import functional as xF
from core.lib.torch.init import xavier_uniform
from core.lib.torch.optim import AdaBelief, Optimizer


class MxModel(mx.Disposable):

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

        dssim_x4_power : float = 1.0
        dssim_x8_power : float = 1.0
        dssim_x16_power : float = 1.0
        dssim_x32_power : float = 1.0
        mse_power : float = 1.0
        gan_power : float = 0.0
        masked_training : bool = True

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

        time : float = 0.0
        iteration : int = 0
        error_src : float = 0.0
        error_dst : float = 0.0
        accuracy_src : float = 0.0
        accuracy_dst : float = 0.0


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
        self._resolution       = state.get('resolution', 256)
        self._encoder_dim      = state.get('encoder_dim', 64)
        self._ae_dim           = state.get('ae_dim', 256)
        self._decoder_dim      = state.get('decoder_dim', 64)
        self._decoder_mask_dim = state.get('decoder_mask_dim', 24)

        self._mx_info = mx.TextEmitter().dispose_with(self)
        self._mx_device = mx.SingleChoice[lib_torch.DeviceRef](self._device,
                                                               avail=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)
        self._mx_resolution       = mx.Number(self._resolution, config=mx.NumberConfig(min=64, max=1024, step=32)).dispose_with(self)
        self._mx_encoder_dim      = mx.Number(self._encoder_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_ae_dim           = mx.Number(self._ae_dim, config=mx.NumberConfig(min=32, max=1024, step=8)).dispose_with(self)
        self._mx_decoder_dim      = mx.Number(self._decoder_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_decoder_mask_dim = mx.Number(self._decoder_mask_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)


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
                'mm_state'         : self._mod.get_state(), }

    def get_input_resolution(self) -> int: return self._resolution
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

        yield ax.switch_to(self._model_thread)

        mod = self._mod

        device_info, self._device = self._device, new_device
        resolution, self._resolution = self._resolution, new_resolution
        encoder_dim, self._encoder_dim = self._encoder_dim, new_encoder_dim
        ae_dim, self._ae_dim = self._ae_dim, new_ae_dim
        decoder_dim, self._decoder_dim = self._decoder_dim, new_decoder_dim
        decoder_mask_dim, self._decoder_mask_dim = self._decoder_mask_dim, new_decoder_mask_dim

        reset_encoder      = (resolution != new_resolution or encoder_dim != new_encoder_dim)
        reset_inter        = (reset_encoder or ae_dim != new_ae_dim)
        reset_decoder      = (reset_inter or decoder_dim != new_decoder_dim)
        reset_decoder_mask = (reset_inter or decoder_mask_dim != new_decoder_mask_dim)

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

        if reset_encoder or reset_inter or reset_decoder or reset_decoder_mask:
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

        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device_info)
        self._mx_resolution.set(resolution)
        self._mx_encoder_dim.set(encoder_dim)
        self._mx_ae_dim.set(ae_dim)
        self._mx_decoder_dim.set(decoder_dim)
        self._mx_decoder_mask_dim.set(decoder_mask_dim)

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

    #################
    ### STEP INFER / TRAIN
    #################
    @ax.task
    def step(self, req : StepRequest) -> StepResult:
        yield ax.attach_to(self._tg, detach_parent=False)

        result = MxModel.StepResult()

        batch_acc  = req.batch_acc
        lr         = req.lr
        lr_dropout = req.lr_dropout

        masked_training = result.masked_training = req.masked_training

        train_src_image = req.src_image_np is not None and req.src_target_image_np is not None
        train_src_mask  = req.src_image_np is not None and req.src_target_mask_np is not None
        train_dst_image = req.dst_image_np is not None and req.dst_target_image_np is not None
        train_dst_mask  = req.dst_image_np is not None and req.dst_target_mask_np is not None

        pred_src_image  = req.src_image_np is not None and (train_src_image or req.pred_src_image)
        pred_src_mask   = req.src_image_np is not None and (train_src_mask or (train_src_image and masked_training) or req.pred_src_mask)
        pred_dst_image  = req.dst_image_np is not None and (train_dst_image or req.pred_dst_image)
        pred_dst_mask   = req.dst_image_np is not None and (train_dst_mask or (train_dst_image and masked_training) or req.pred_dst_mask)
        pred_swap_image = req.pred_swap_image
        pred_swap_mask  = req.pred_swap_mask

        train = train_src_image or train_dst_image or train_src_mask or train_dst_mask
        train_encoder = train
        train_inter = train
        train_decoder = train_src_image or train_dst_image
        train_decoder_mask = train_src_mask or train_dst_mask

        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            resolution = self._resolution

            if req.src_image_np is not None:
                src_image_np = result.src_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.src_image_np]
                src_image_nd = np.stack([x.CHW() for x in src_image_np])

            if req.src_target_image_np is not None:
                src_target_image_np = result.src_target_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.src_target_image_np]
                src_target_image_nd = np.stack([x.CHW() for x in src_target_image_np])

            if req.src_target_mask_np is not None:
                src_target_mask_np = result.src_target_mask_np = [x.grayscale().resize(resolution, resolution, interp=NPImage.Interp.LINEAR).f32() for x in req.src_target_mask_np]
                src_target_mask_nd = np.stack([x.CHW() for x in src_target_mask_np])

            if req.dst_image_np is not None:
                dst_image_np = result.dst_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.dst_image_np]
                dst_image_nd = np.stack([x.CHW() for x in dst_image_np])

            if req.dst_target_image_np is not None:
                dst_target_image_np = result.dst_target_image_np = [ x.bgr().resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.dst_target_image_np]
                dst_target_image_nd = np.stack([x.CHW() for x in dst_target_image_np])

            if req.dst_target_mask_np is not None:
                dst_target_mask_np = result.dst_target_mask_np = [x.grayscale().resize(resolution, resolution, interp=NPImage.Interp.LINEAR).f32() if x is not None else None for x in req.dst_target_mask_np]
                dst_target_mask_nd = np.stack([x.CHW() for x in dst_target_mask_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution:
                # Prepared data matches model parameters.
                break

        try:
            step_time = lib_time.measure()
            iteration = self._iteration
            device = self._device

            encoder : Encoder = self._mod.get_module('encoder', device=device, train=train)
            inter_src : Inter = self._mod.get_module('inter_src', device=device, train=train)
            inter_dst : Inter = self._mod.get_module('inter_dst', device=device, train=train)
            decoder : Decoder = self._mod.get_module('decoder', device=device, train=train)
            decoder_mask : Decoder = self._mod.get_module('decoder_mask', device=device, train=train)


            if pred_src_image or pred_src_mask:
                src_image_t = torch.tensor(src_image_nd, device=device.device)
            if pred_dst_image or pred_dst_mask or pred_swap_image or pred_swap_mask:
                dst_image_t = torch.tensor(dst_image_nd, device=device.device)
            if train_src_image:
                src_target_image_t = torch.tensor(src_target_image_nd, device=device.device)
            if train_dst_image:
                dst_target_image_t = torch.tensor(dst_target_image_nd, device=device.device)
            if train_src_mask:
                src_target_mask_t  = torch.tensor(src_target_mask_nd, device=device.device)
            if train_dst_mask:
                dst_target_mask_t  = torch.tensor(dst_target_mask_nd, device=device.device)

            if pred_src_image or pred_src_mask:
                src_enc_t = encoder(src_image_t)
                src_src_code_t = inter_src(src_enc_t)
                src_code_t = torch.cat([src_src_code_t, src_src_code_t], 1)

            if pred_dst_image or pred_dst_mask or pred_swap_image or pred_swap_mask:
                dst_enc_t = encoder(dst_image_t)
                src_dst_code_t = inter_src(dst_enc_t)

            if pred_dst_image or pred_dst_mask:
                dst_dst_code_t = inter_dst(dst_enc_t)
                dst_code_t = torch.cat([src_dst_code_t, dst_dst_code_t], 1)

            if pred_swap_image or pred_swap_mask:
                swap_code_t = torch.cat([src_dst_code_t, src_dst_code_t], 1)

            if pred_src_image:
                pred_src_image_t = decoder(src_code_t)
                if req.pred_src_image:
                    result.pred_src_image_np = [ NPImage(x, channels_last=False) for x in pred_src_image_t.detach().cpu().numpy().clip(0, 1) ]

            if pred_src_mask:
                pred_src_mask_t = decoder_mask(src_code_t)
                if req.pred_src_mask:
                    result.pred_src_mask_np = [ NPImage(x, channels_last=False) for x in pred_src_mask_t.detach().cpu().numpy().clip(0, 1) ]

            if pred_dst_image:
                pred_dst_image_t = decoder(dst_code_t)
                if req.pred_dst_image:
                    result.pred_dst_image_np = [ NPImage(x, channels_last=False) for x in pred_dst_image_t.detach().cpu().numpy().clip(0, 1) ]

            if pred_dst_mask:
                pred_dst_mask_t = decoder_mask(dst_code_t)
                if req.pred_dst_mask:
                    result.pred_dst_mask_np = [ NPImage(x, channels_last=False) for x in pred_dst_mask_t.detach().cpu().numpy().clip(0, 1) ]

            if pred_swap_image:
                pred_swap_image_t = decoder(swap_code_t)
                if req.pred_swap_image:
                    result.pred_swap_image_np = [ NPImage(x, channels_last=False) for x in pred_swap_image_t.detach().cpu().numpy().clip(0, 1) ]

            if pred_swap_mask:
                pred_swap_mask_t = decoder_mask(swap_code_t)
                if req.pred_swap_mask:
                    result.pred_swap_mask_np = [ NPImage(x, channels_last=False) for x in pred_swap_mask_t.detach().cpu().numpy().clip(0, 1) ]


            if train_src_image and masked_training:
                src_target_mask_blur_t = xF.gaussian_blur(src_target_mask_t, sigma=max(1, resolution // 32) )
                src_target_mask_blur_t = torch.clamp(src_target_mask_blur_t, 0.0, 0.5) * 2.0

                src_target_image_t = src_target_image_t * src_target_mask_blur_t
                pred_src_image_t   = pred_src_image_t   * src_target_mask_blur_t

            if train_dst_image and masked_training:
                dst_target_mask_blur_t = xF.gaussian_blur(dst_target_mask_t, sigma=max(1, resolution // 32) )
                dst_target_mask_blur_t = torch.clamp(dst_target_mask_blur_t, 0.0, 0.5) * 2.0

                dst_target_image_t = dst_target_image_t * dst_target_mask_blur_t
                pred_dst_image_t   = pred_dst_image_t   * dst_target_mask_blur_t



            if train_encoder:
                encoder_opt : Optimizer = self._mod.get_module('encoder_opt', device=device)
                if (iteration % batch_acc) == 0:
                    encoder_opt.zero_grad()

            if train_inter:
                inter_src_opt : Optimizer = self._mod.get_module('inter_src_opt', device=device)
                if (iteration % batch_acc) == 0:
                    inter_src_opt.zero_grad()

                inter_dst_opt : Optimizer = self._mod.get_module('inter_dst_opt', device=device)
                if (iteration % batch_acc) == 0:
                    inter_dst_opt.zero_grad()

            if train_decoder:
                decoder_opt : Optimizer = self._mod.get_module('decoder_opt', device=device)
                if (iteration % batch_acc) == 0:
                    decoder_opt.zero_grad()

            if train_decoder_mask:
                decoder_mask_opt : Optimizer = self._mod.get_module('decoder_mask_opt', device=device)
                if (iteration % batch_acc) == 0:
                    decoder_mask_opt.zero_grad()

            # Collect losses
            src_losses = []
            dst_losses = []


            if (dssim_x4_power := req.dssim_x4_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//4)
                if train_src_image:
                    src_losses.append( dssim_x4_power*xF.dssim(pred_src_image_t, src_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )
                if train_dst_image:
                    dst_losses.append( dssim_x4_power*xF.dssim(pred_dst_image_t, dst_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x8_power := req.dssim_x8_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//8)
                if train_src_image:
                    src_losses.append( dssim_x8_power*xF.dssim(pred_src_image_t, src_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )
                if train_dst_image:
                    dst_losses.append( dssim_x8_power*xF.dssim(pred_dst_image_t, dst_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x16_power := req.dssim_x16_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//16)
                if train_src_image:
                    src_losses.append( dssim_x16_power*xF.dssim(pred_src_image_t, src_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )
                if train_dst_image:
                    dst_losses.append( dssim_x16_power*xF.dssim(pred_dst_image_t, dst_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x32_power := req.dssim_x32_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//32)
                if train_src_image:
                    src_losses.append( dssim_x32_power*xF.dssim(pred_src_image_t, src_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )
                if train_dst_image:
                    dst_losses.append( dssim_x32_power*xF.dssim(pred_dst_image_t, dst_target_image_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (mse_power := req.mse_power) != 0.0:
                if train_src_image:
                    src_losses.append( torch.mean(mse_power*10*torch.square(pred_src_image_t-src_target_image_t), (1,2,3)) )
                if train_src_mask:
                    src_losses.append( torch.mean(mse_power*10*torch.square(pred_src_mask_t-src_target_mask_t), (1,2,3)) )
                if train_dst_image:
                    dst_losses.append( torch.mean(mse_power*10*torch.square(pred_dst_image_t-dst_target_image_t), (1,2,3)) )
                if train_dst_mask:
                    dst_losses.append( torch.mean(mse_power*10*torch.square(pred_dst_mask_t-dst_target_mask_t), (1,2,3)) )


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

                if train_encoder:
                    encoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                if train_inter:
                    inter_src_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)
                    inter_dst_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                if train_decoder:
                    decoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                if train_decoder_mask:
                    decoder_mask_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

            # # Metrics
            result.error_src = float( src_loss_t.detach().cpu().numpy() ) if src_loss_t is not None else 0
            result.error_dst = float( dst_loss_t.detach().cpu().numpy() ) if dst_loss_t is not None else 0
            result.accuracy_src = float( (1 - torch.abs(pred_src_image_t-src_target_image_t).sum((-1,-2)).mean() / (resolution*resolution)).detach().cpu().numpy() ) if train_src_image else 0
            result.accuracy_dst = float( (1 - torch.abs(pred_dst_image_t-dst_target_image_t).sum((-1,-2)).mean() / (resolution*resolution)).detach().cpu().numpy() ) if train_dst_image else 0

            # Update vars
            if train:
                self._iteration = iteration + 1

            result.time = step_time.elapsed()
            return result

        except Exception as e:
            yield ax.cancel(e)

    ##########################
    ##########################
    ##########################

# Network blocks

class ResidualBlock(nn.Module):
    def __init__(self, ch, last_act=True):
        super().__init__()
        self._last_act = last_act
        self.c0 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, inp):
        x = F.leaky_relu(self.c0(inp), 0.2)
        x = self.c1(x)
        if self._last_act:
            x = F.leaky_relu(inp+x, 0.2)
        return x

class Encoder(nn.Module):
    def __init__(self, resolution, in_ch, base_dim):
        super().__init__()
        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, kernel_size=1, stride=1, padding=0)

        self._c0 = nn.Conv2d(base_dim*1, base_dim*2, kernel_size=5, stride=2, padding=2)
        self._c1 = nn.Conv2d(base_dim*2, base_dim*4, kernel_size=5, stride=2, padding=2)
        self._c2 = nn.Conv2d(base_dim*4, base_dim*8, kernel_size=5, stride=2, padding=2)
        self._c3 = nn.Conv2d(base_dim*8, base_dim*8, kernel_size=5, stride=2, padding=2)
        self._c4 = nn.Conv2d(base_dim*8, base_dim*8, kernel_size=5, stride=2, padding=2)

        self._out_ch = base_dim*8
        self._out_res = resolution // (2**5)

        xavier_uniform(self)

    def get_out_ch(self): return self._out_ch
    def get_out_res(self): return self._out_res

    def forward(self, inp : torch.Tensor):
        x = inp
        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(inp)
        x = F.leaky_relu(self._c0(x), 0.1)
        x = F.leaky_relu(self._c1(x), 0.1)
        x = F.leaky_relu(self._c2(x), 0.1)
        x = F.leaky_relu(self._c3(x), 0.1)
        x = F.leaky_relu(self._c4(x), 0.1)
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
    def __init__(self, in_ch, base_dim, out_ch, use_residuals=True):
        super().__init__()
        self._in_ch = in_ch
        self._decoder_ch = base_dim
        self._out_ch = out_ch
        self._use_residuals = use_residuals

        self._c_in = nn.Conv2d(in_ch, base_dim*8, kernel_size=1, stride=1, padding=0)
        self._c3 = nn.Conv2d(base_dim*8, base_dim*8 *4, kernel_size=3, stride=1, padding=1)
        self._c2 = nn.Conv2d(base_dim*8, base_dim*8 *4, kernel_size=3, stride=1, padding=1)
        self._c1 = nn.Conv2d(base_dim*8, base_dim*4 *4, kernel_size=3, stride=1, padding=1)
        self._c0 = nn.Conv2d(base_dim*4, base_dim*2 *4, kernel_size=3, stride=1, padding=1)
        if use_residuals:
            self._r3 = ResidualBlock(base_dim*8)
            self._r2 = ResidualBlock(base_dim*8)
            self._r1 = ResidualBlock(base_dim*4)
            self._r0 = ResidualBlock(base_dim*2)

        self._out0 = nn.Conv2d(base_dim*2, out_ch, kernel_size=1, stride=1, padding=0)
        self._out1 = nn.Conv2d(base_dim*2, out_ch, kernel_size=3, stride=1, padding=1)
        self._out2 = nn.Conv2d(base_dim*2, out_ch, kernel_size=3, stride=1, padding=1)
        self._out3 = nn.Conv2d(base_dim*2, out_ch, kernel_size=3, stride=1, padding=1)

        self._out_gamma = nn.parameter.Parameter( torch.ones(out_ch,), requires_grad=True)
        self._out_beta = nn.parameter.Parameter( torch.zeros(out_ch,), requires_grad=True)

        xavier_uniform(self)

    def forward(self, inp):
        x = inp

        x = self._c_in(x)
        x = F.pixel_shuffle(F.leaky_relu(self._c3(x), 0.1), 2)
        if self._use_residuals:
            x = self._r3(x)

        x = F.pixel_shuffle(F.leaky_relu(self._c2(x), 0.1), 2)
        if self._use_residuals:
            x = self._r2(x)

        x = F.pixel_shuffle(F.leaky_relu(self._c1(x), 0.1), 2)
        if self._use_residuals:
            x = self._r1(x)

        x = F.pixel_shuffle(F.leaky_relu(self._c0(x), 0.1), 2)
        if self._use_residuals:
            x = self._r0(x)

        x = F.pixel_shuffle( torch.cat([    self._out0(x),
                                            self._out1(x),
                                            self._out2(x),
                                            self._out3(x) ], 1 ), 2 )

        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x