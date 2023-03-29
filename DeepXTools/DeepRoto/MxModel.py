from dataclasses import dataclass
from enum import Enum, auto
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
from core.lib.torch.modules import BlurPool
from core.lib.torch.optim import AdaBelief, Optimizer


class MxModel(mx.Disposable):
    @dataclass
    class InferResult:
        image_np : List[NPImage]
        pred_mask_np : List[NPImage]

    @dataclass
    class TrainStepResult:
        image_np : List[NPImage]
        target_mask_np : List[NPImage]
        step_time : float
        iteration : int
        error : float
        accuracy : float

    class InputType(Enum):
        Color = auto()
        Luminance = auto()

    def __init__(self, state : dict = None):
        super().__init__()
        state = state or {}

        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._prepare_thread_pool = ax.ThreadPool(name='prepare_thread_pool').dispose_with(self)
        self._model_thread = ax.Thread('model_thread').dispose_with(self)


        # Model variables. Changed in _model_thread.
        self._device     = lib_torch.DeviceRef.from_state(state.get('device_state', None))
        self._input_type = MxModel.InputType(state.get('input_type', MxModel.InputType.Luminance.value))
        self._resolution = state.get('resolution', 256)
        self._base_dim   = state.get('base_dim', 32)
        self._iteration  = state.get('iteration', 0)
        self._opt_class = AdaBelief


        self._mx_device = mx.SingleChoice[lib_torch.DeviceRef](self._device,
                                                               avail=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)
        self._mx_input_type = mx.SingleChoice[MxModel.InputType](self._input_type, avail=lambda: [*MxModel.InputType]).dispose_with(self)
        self._mx_resolution = mx.Number(self._resolution, config=mx.NumberConfig(min=64, max=1024, step=64)).dispose_with(self)
        self._mx_base_dim   = mx.Number(self._base_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)

        self._mod = lib_torch.ModulesOnDemand(  {'encoder': lambda mod: Encoder(resolution=self._resolution, in_ch=1 if self._input_type == MxModel.InputType.Luminance else 3, base_dim=self._base_dim),
                                                 'encoder_opt': lambda mod: self._opt_class(mod.get_module('encoder').parameters()),

                                                 'decoder': lambda mod: Decoder(out_ch=1, base_dim=self._base_dim),
                                                 'decoder_opt': lambda mod: self._opt_class(mod.get_module('decoder').parameters()),
                                                },
                                                state=state.get('mm_state', None) ).dispose_with(self)

    @property
    def mx_device(self) -> mx.ISingleChoice[lib_torch.DeviceRef]:
        return self._mx_device
    @property
    def mx_input_type(self) ->mx.SingleChoice[InputType]:
        return self._mx_input_type
    @property
    def mx_resolution(self) -> mx.INumber:
        return self._mx_resolution
    @property
    def mx_base_dim(self) -> mx.INumber:
        return self._mx_base_dim

    def get_input_resolution(self) -> int: return self._resolution
    def get_input_ch(self) -> int: return 1 if self._input_type == MxModel.InputType.Luminance else 3

    @ax.task
    def get_state(self) -> dict:
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        return {'device_state' : self._device.get_state(),
                'input_type' : self._input_type.value,
                'resolution' : self._resolution,
                'base_dim' : self._base_dim,
                'iteration' : self._iteration,
                'mm_state' : self._mod.get_state(), }

    @ax.task
    def reset_encoder(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('encoder')
        self._mod.reset_module('encoder_opt')

    @ax.task
    def reset_decoder(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('decoder')
        self._mod.reset_module('decoder_opt')

    @ax.task
    def apply_model_settings(self):
        """Apply mx model settings to actual model."""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        new_device         = self._mx_device.get()
        new_input_type     = self._mx_input_type.get()
        new_resolution     = self._mx_resolution.get()
        new_base_dim       = self._mx_base_dim.get()

        yield ax.switch_to(self._model_thread)

        mod = self._mod

        device_info, self._device = self._device, new_device
        input_type, self._input_type = self._input_type, new_input_type
        resolution, self._resolution = self._resolution, new_resolution
        base_dim, self._base_dim = self._base_dim, new_base_dim

        reset_encoder    = (resolution != new_resolution or base_dim != new_base_dim) or (input_type != new_input_type)
        reset_decoder    = (resolution != new_resolution or base_dim != new_base_dim)

        if reset_encoder:
            mod.reset_module('encoder')
            mod.reset_module('encoder_opt')
        if reset_decoder:
            mod.reset_module('decoder')
            mod.reset_module('decoder_opt')

        if reset_encoder or reset_decoder:
            torch.cuda.empty_cache()

    @ax.task
    def revert_model_settings(self):
        """Revert mx model settings to actual from the model."""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        device_info = self._device
        input_type  = self._input_type
        resolution  = self._resolution
        base_dim    = self._base_dim

        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device_info)
        self._mx_input_type.set(input_type)
        self._mx_resolution.set(resolution)
        self._mx_base_dim.set(base_dim)

    @ax.task
    def infer(self, image_np : List[NPImage]) -> InferResult:
        """
            image_np    List[NPImage]

        Images will be transformed to match current model `.get_input_resolution()` and `.get_input_ch()`.
        To remove performance impact prepare images before infer or use multiple tasks in queue.
        """
        yield ax.attach_to(self._tg, detach_parent=False)

        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            resolution = self._resolution
            input_type = self._input_type

            p_image_np = [ (x.grayscale() if input_type == MxModel.InputType.Luminance else x.bgr()).resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4)
                            .f32() for x in image_np]
            p_image_nd = np.stack([x.CHW() for x in p_image_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution and input_type == self._input_type:
                # Prepared data matches model parameters.
                break

        try:
            device = self._device

            input_t = torch.tensor(p_image_nd, device=device.device)

            # Inference
            encoder : Encoder = self._mod.get_module('encoder', device=device, train=False)
            decoder : Decoder = self._mod.get_module('decoder', device=device, train=False)

            with torch.no_grad():
                skips, x = encoder(input_t)
                pred_mask_t = decoder(skips, x) / 2.0 + 0.5

            pred_mask_nd = pred_mask_t.detach().cpu().numpy()

            yield ax.switch_to(self._prepare_thread_pool)

            # Make result
            return MxModel.InferResult( image_np = p_image_np,
                                        pred_mask_np = [ NPImage(x, channels_last=False) for x in pred_mask_nd.clip(0, 1) ])

        except Exception as e:
            yield ax.cancel(e)

    @ax.task
    def train_step(self,    image_np : List[NPImage],
                            target_mask_np : List[NPImage],
                            mse_power : float = 1.0,
                            dssim_x4_power : float = 0.0,
                            dssim_x8_power : float = 0.0,
                            dssim_x16_power : float = 0.0,
                            dssim_x32_power : float = 0.0,
                            batch_acc : int = 1,
                            lr=5e-5,
                            lr_dropout=0.3,
                            train_encoder=True,
                            train_decoder=True,
                ) -> TrainStepResult:
        """
            image_np    List[NPImage]

            target_mask_np    List[NPImage]

        Images will be transformed to match current model `.get_input_resolution()` and `.get_input_ch()`.
        To remove performance impact prepare images before or use multiple tasks in queue.
        """
        yield ax.attach_to(self._tg, detach_parent=False)

        if len(image_np) != len(target_mask_np):
            raise ValueError('len(image_np) != len(target_mask_np)')

        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            resolution = self._resolution
            input_type = self._input_type

            p_image_np = [ (x.grayscale() if input_type == MxModel.InputType.Luminance else x.bgr()).resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4)
                            .f32() for x in image_np]
            p_image_nd = np.stack([x.CHW() for x in p_image_np])

            p_target_mask_np = [x.grayscale().resize(resolution, resolution, interp=NPImage.Interp.CUBIC).f32() for x in target_mask_np]
            p_target_mask_nd = np.stack([x.CHW() for x in p_target_mask_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution and input_type == self._input_type:
                # Prepared data matches model parameters.
                break

        try:
            step_time = lib_time.measure()
            iteration = self._iteration
            device = self._device

            input_t       = torch.tensor(p_image_nd, device=device.device)
            target_mask_t = torch.tensor(p_target_mask_nd, device=device.device) * 2.0 - 1.0

            # Inference
            encoder : Encoder = self._mod.get_module('encoder', device=device, train=True)
            decoder : Decoder = self._mod.get_module('decoder', device=device, train=True)

            if train_encoder:
                encoder_opt : Optimizer = self._mod.get_module('encoder_opt', device=device)
                if (iteration % batch_acc) == 0:
                    encoder_opt.zero_grad()

            if train_decoder:
                decoder_opt : Optimizer = self._mod.get_module('decoder_opt', device=device)
                if (iteration % batch_acc) == 0:
                    decoder_opt.zero_grad()

            #with torch.autocast(device.backend):
            skips, x = encoder(input_t)
            pred_mask_t = decoder(skips, x)
            _, _, H, W = pred_mask_t.shape

            # Collect losses
            losses = []

            if mse_power != 0.0:
                losses.append( torch.mean(mse_power*10*torch.square(pred_mask_t-target_mask_t), (1,2,3)) )

            if (dssim_x4_power + dssim_x8_power + dssim_x16_power + dssim_x32_power) != 0.0:
                pred_mask_u_t = pred_mask_t / 2 + 0.5
                target_mask_u_t = target_mask_t / 2 + 0.5

            if dssim_x4_power != 0.0:
                kernel_size = lib_math.next_odd(resolution//4)
                losses.append( dssim_x4_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if dssim_x8_power != 0.0:
                kernel_size = lib_math.next_odd(resolution//8)
                losses.append( dssim_x8_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if dssim_x16_power != 0.0:
                kernel_size = lib_math.next_odd(resolution//16)
                losses.append( dssim_x16_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if dssim_x32_power != 0.0:
                kernel_size = lib_math.next_odd(resolution//32)
                losses.append( dssim_x32_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            loss_t = None
            if len(losses) != 0:
                loss_t = sum(losses).mean()
                loss_t.backward()

                # Optimization
                if (iteration % batch_acc) == (batch_acc-1):
                    grad_mult = 1.0 / batch_acc

                    if train_encoder:
                        encoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                    if train_decoder:
                        decoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

            # Metrics
            error    = float( loss_t.detach().cpu().numpy() ) if loss_t is not None else 0
            accuracy = float( (1 - torch.abs(pred_mask_t-target_mask_t).sum((-1,-2)).mean() / (H*W*2)).detach().cpu().numpy() )

            # Update vars
            self._iteration = iteration + 1

            # Make result
            return MxModel.TrainStepResult( image_np = p_image_np,
                                            target_mask_np = p_target_mask_np,
                                            step_time = step_time.elapsed(),
                                            iteration = iteration,
                                            error = error,
                                            accuracy = accuracy )

        except Exception as e:
            yield ax.cancel(e)


# Network blocks

class Encoder(nn.Module):
    def __init__(self, resolution, in_ch, base_dim=32, n_downs=6):
        super().__init__()
        self._resolution = resolution
        self._base_dim = base_dim
        self._n_downs = n_downs

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, kernel_size=1, stride=1, padding=0, bias=False)

        conv_list = []
        bp_list = []
        for i_down in range(n_downs):
            i_ch = base_dim * min(2**(i_down)  , 8)
            o_ch = base_dim * min(2**(i_down+1), 8)

            conv_list.append( nn.Conv2d (i_ch, o_ch, kernel_size=5, padding=2) )
            bp_list.append( BlurPool (o_ch, kernel_size=max(2, 4-i_down)) )

        self._conv_list = nn.ModuleList(conv_list)
        self._bp_list = nn.ModuleList(bp_list)


    def get_out_resolution(self) -> int:
        return self._resolution // (2**self._n_downs)

    def forward(self, inp):
        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        skips = []
        for conv, bp in zip(self._conv_list, self._bp_list):
            x = F.leaky_relu(conv(x), 0.1)
            x = bp(x)
            skips.insert(0, x)

        return skips, x



class Decoder(nn.Module):
    def __init__(self, out_ch, base_dim=32, n_downs=6):
        super().__init__()
        self._n_downs = n_downs

        conv_s1_list = []
        conv_s2_list = []
        conv_r_list = []
        conv_c_list = []

        for i_down in range(n_downs-1, -1, -1):
            i_ch = base_dim * min(2**(i_down+1), 8)
            o_ch = base_dim * min(2**(i_down)  , 8)

            conv_s1_list.append( nn.Conv2d(i_ch, i_ch, kernel_size=3, padding=1) )
            conv_s2_list.append( nn.Conv2d(i_ch, i_ch, kernel_size=3, padding=1) )
            conv_r_list.append( nn.Conv2d(i_ch, i_ch, kernel_size=3, padding=1) )
            conv_c_list.append( nn.Conv2d(i_ch, o_ch*4, kernel_size=3, padding=1) )

        self._conv_s1_list = nn.ModuleList(conv_s1_list)
        self._conv_s2_list = nn.ModuleList(conv_s2_list)
        self._conv_r_list = nn.ModuleList(conv_r_list)
        self._conv_c_list = nn.ModuleList(conv_c_list)

        self._out_conv = nn.Conv2d (base_dim, base_dim, kernel_size=3, padding=1)
        self._out = nn.Conv2d(base_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self._out_gamma = nn.parameter.Parameter( torch.ones(out_ch,), requires_grad=True)
        self._out_beta = nn.parameter.Parameter( torch.zeros(out_ch,), requires_grad=True)


    def forward(self, skips : List[torch.Tensor], x):

        for skip_x, s1, s2, r, c, in zip(skips, self._conv_s1_list, self._conv_s2_list, self._conv_r_list, self._conv_c_list):
            x = F.leaky_relu(x + r(x) + s2(F.leaky_relu(s1(skip_x), 0.2)) , 0.2)
            x = F.leaky_relu(c(x), 0.1)
            x = F.pixel_shuffle(x, 2)

        x = F.leaky_relu(self._out_conv(x), 0.1)
        x = self._out(x)
        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x
