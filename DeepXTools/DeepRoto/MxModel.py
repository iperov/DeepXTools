import io
import itertools
import pickle
import ssl
import urllib.request
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
from core.lib.torch import functional as xF
from core.lib.torch.init import xavier_uniform
from core.lib.torch.modules import BlurPool
from core.lib.torch.optim import AdaBelief, Optimizer


class MxModel(mx.Disposable):

    @dataclass
    class StepRequest:
        image_np : List[NPImage]|None = None
        target_mask_np : List[NPImage]|None = None

        pred_mask : bool = False

        dssim_x4_power : float = 1.0
        dssim_x8_power : float = 1.0
        dssim_x16_power : float = 1.0
        dssim_x32_power : float = 1.0
        mse_power : float = 1.0

        batch_acc : int = 1
        lr : float = 5e-5
        lr_dropout : float = 0.3

    @dataclass
    class StepResult:
        image_np : List[NPImage]|None = None
        target_mask_np : List[NPImage]|None = None
        pred_mask_np : List[NPImage]|None = None

        time : float = 0.0
        iteration : int = 0
        error : float = 0.0
        accuracy : float = 0.0

    class InputType(Enum):
        Color = auto()
        Luminance = auto()

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
        self._input_type = MxModel.InputType(state.get('input_type', MxModel.InputType.Luminance.value))
        self._resolution = state.get('resolution', 256)
        self._base_dim   = state.get('base_dim', 32)
        self._generalization_level = state.get('generalization_level', 0)
        self._iteration  = state.get('iteration', 0)
        self._opt_class = AdaBelief
        
        n_downs = 6

        self._mx_info = mx.TextEmitter().dispose_with(self)
        self._mx_device = mx.SingleChoice[lib_torch.DeviceRef](self._device,
                                                               avail=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)
        self._mx_input_type = mx.SingleChoice[MxModel.InputType](self._input_type, avail=lambda: [*MxModel.InputType]).dispose_with(self)
        self._mx_resolution = mx.Number(self._resolution, config=mx.NumberConfig(min=64, max=1024, step=64)).dispose_with(self)
        self._mx_base_dim   = mx.Number(self._base_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_generalization_level = mx.Number(self._generalization_level, config=mx.NumberConfig(min=0, max=n_downs, step=1)).dispose_with(self)
        self._mx_url_download_menu = mx.Menu[str](avail_choices=lambda: ['https://github.com/iperov/DeepXTools/releases/download/DXRM_0/Luminance_256_32_UNet_from_ImageNet.dxrm'],
                                         on_choose=lambda x: self._download_model_state(x)
                                         ).dispose_with(self)

        self._mod = lib_torch.ModulesOnDemand(  {'model': lambda mod: XSegModel(in_ch=1 if self._input_type == MxModel.InputType.Luminance else 3,
                                                                                out_ch=1,
                                                                                generalization_level=self._generalization_level,
                                                                                base_dim=self._base_dim, n_downs=n_downs),
                                                 'model_opt': lambda mod: self._opt_class(mod.get_module('model').parameters()),

                                                },
                                                state=state.get('mm_state', None) ).dispose_with(self)
    @property
    def mx_info(self) -> mx.ITextEmitter_r: return self._mx_info
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
    @property
    def mx_generalization_level(self) -> mx.INumber:
        return self._mx_generalization_level
    @property
    def mx_url_download_menu(self) -> mx.IMenu[str]:
        return self._mx_url_download_menu

    @ax.task
    def get_state(self) -> dict:
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        return {'device_state' : self._device.get_state(),
                'iteration'    : self._iteration } | self._get_model_state()

    def _get_model_state(self) -> dict:
        self._model_thread.assert_current_thread()

        return {'input_type'   : self._input_type.value,
                'resolution'   : self._resolution,
                'base_dim'     : self._base_dim,
                'generalization_level' : self._generalization_level,
                'mm_state'     : self._mod.get_state(), }

    def get_input_resolution(self) -> int: return self._resolution
    def get_input_ch(self) -> int: return 1 if self._input_type == MxModel.InputType.Luminance else 3

    @ax.task
    def apply_model_settings(self):
        """Apply mx model settings to actual model."""
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        new_device         = self._mx_device.get()
        new_input_type     = self._mx_input_type.get()
        new_resolution     = self._mx_resolution.get()
        new_base_dim       = self._mx_base_dim.get()
        new_generalization_level = self._mx_generalization_level.get()

        yield ax.switch_to(self._model_thread)

        mod = self._mod

        device_info, self._device = self._device, new_device
        input_type, self._input_type = self._input_type, new_input_type
        resolution, self._resolution = self._resolution, new_resolution
        base_dim, self._base_dim = self._base_dim, new_base_dim
        generalization_level, self._generalization_level = self._generalization_level, new_generalization_level

        reset_model = (resolution != new_resolution or base_dim != new_base_dim) or (input_type != new_input_type)

        if reset_model:
            mod.reset_module('model')
            mod.reset_module('model_opt')

        if reset_model or generalization_level != new_generalization_level:
            model : XSegModel = self._mod.get_module('model')
            model.set_generalization_level(new_generalization_level)

        if reset_model:
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
        generalization_level = self._generalization_level

        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device_info)
        self._mx_input_type.set(input_type)
        self._mx_resolution.set(resolution)
        self._mx_base_dim.set(base_dim)
        self._mx_generalization_level.set(generalization_level)

    ######################################
    ### RESET / IMPORT / EXPORT / DOWNLOAD
    ######################################
    @ax.task
    def reset_model(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('model')
        self._mod.reset_module('model_opt')


    @ax.task
    def import_model(self, filepath : Path):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        self._mx_info.emit(f'@(MxModel.Importing_model_from) {filepath} ...')

        yield ax.switch_to(self._model_thread)

        err = None
        try:
            model_state = pickle.loads(filepath.read_bytes())
        except Exception as e:
            err = e

        if err is None:
            yield ax.wait(t := self._import_model_state(model_state))
            if not t.succeeded:
                err = t.error
                if err is None:
                    err = Exception('Unknown')

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_info.emit(f'@(Error): {err}')
            yield ax.cancel(err)
        else:
            self._mx_info.emit(f'@(Success).')

    @ax.task
    def _import_model_state(self, model_state : dict):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)

        err = None
        try:
            input_type = MxModel.InputType(model_state['input_type'])
            resolution = model_state['resolution']
            base_dim   = model_state['base_dim']
            generalization_level = model_state['generalization_level']
            mm_state   = model_state['mm_state']
        except Exception as e:
            err = e

        if err is None:
            self._input_type = input_type
            self._resolution = resolution
            self._base_dim = base_dim
            self._generalization_level = generalization_level
            self._mod.set_state(mm_state)
            yield ax.wait(self.revert_model_settings())

        if err is not None:
            yield ax.cancel(err)

    @ax.task
    def _download_model_state(self, url : str):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        self._mx_info.emit(f'@(MxModel.Downloading_model_from) {url} ...')

        yield ax.switch_to(self._io_thread)

        err = None
        try:
            url_request = urllib.request.urlopen(url, context=ssl._create_unverified_context())
            url_size = int( url_request.getheader('content-length') )

            bytes_io = io.BytesIO()

            file_size_dl = 0
            for i in itertools.count():
                buffer = url_request.read(8192)

                if (i % 1000) == 0 or not buffer:
                    yield ax.switch_to(self._main_thread)
                    self._mx_info.emit(f'{file_size_dl} / {url_size}')
                    yield ax.switch_to(self._io_thread)

                if not buffer:
                    break

                bytes_io.write(buffer)
                file_size_dl += len(buffer)

            bytes_io.seek(0)
            model_state = pickle.load(bytes_io)
        except Exception as e:
            err = e

        if err is None:
            yield ax.wait(t := self._import_model_state(model_state))
            if not t.succeeded:
                err = t.error
                if err is None:
                    err = Exception('Unknown')

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_info.emit(f'@(Error): {err}')
            yield ax.cancel(err)
        else:
            self._mx_info.emit(f'@(Success).')

    @ax.task
    def export_model(self, filepath : Path):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        self._mx_info.emit(f'@(MxModel.Exporting_model_to) {filepath} ...')

        yield ax.switch_to(self._model_thread)

        err = None
        try:
            filepath.write_bytes( pickle.dumps(self._get_model_state()) )
        except Exception as e:
            err = e

        yield ax.switch_to(self._main_thread)

        if err is not None:
            self._mx_info.emit(f'@(Error): {err}')
            yield ax.cancel(err)
        else:
            self._mx_info.emit(f'@(Success).')

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

        train_mask  = req.target_mask_np is not None

        train_model = train_mask

        model : XSegModel = ...
        model_opt : Optimizer = ...

        image_t : torch.Tensor = ...
        target_mask_u_t : torch.Tensor = ...
        target_mask_t : torch.Tensor = ...
        pred_mask_t : torch.Tensor = ...
        pred_mask_u_t : torch.Tensor = ...

        def get_model() -> XSegModel:
            nonlocal model
            nonlocal model_opt
            if model is Ellipsis:
                model = self._mod.get_module('model', device=device, train=train_model)
                if train_model:
                    model_opt = self._mod.get_module('model_opt', device=device)
                    if (iteration % batch_acc) == 0:
                        model_opt.zero_grad()
            return model

        def model_forward(x) -> torch.Tensor:
            with torch.set_grad_enabled(train_model):
                return get_model()(x)

        def get_image_t() -> torch.Tensor|None:
            nonlocal image_t
            if image_t is Ellipsis:
                image_t = torch.tensor(image_nd, device=device.device) if image_nd is not None else None
            return image_t

        def get_target_mask_u_t() -> torch.Tensor|None:
            nonlocal target_mask_u_t
            if target_mask_u_t is Ellipsis:
                target_mask_u_t = torch.tensor(target_mask_nd, device=device.device) if target_mask_nd is not None else None
            return target_mask_u_t

        def get_target_mask_t() -> torch.Tensor|None:
            nonlocal target_mask_t
            if target_mask_t is Ellipsis:
                target_mask_t = None
                if (target_mask_u_t := get_target_mask_u_t()) is not None:
                    target_mask_t = target_mask_u_t * 2.0 - 1.0
            return target_mask_t

        def get_pred_mask_t() -> torch.Tensor|None:
            nonlocal pred_mask_t
            if pred_mask_t is Ellipsis:
                pred_mask_t = None
                if (image_t := get_image_t()) is not None:
                    pred_mask_t = model_forward(image_t)

            return pred_mask_t

        def get_pred_mask_u_t() -> torch.Tensor|None:
            nonlocal pred_mask_u_t
            if pred_mask_u_t is Ellipsis:
                pred_mask_u_t = None
                if (pred_mask_t := get_pred_mask_t()) is not None:
                    pred_mask_u_t = pred_mask_t / 2.0 + 0.5

            return pred_mask_u_t

        image_nd = None
        target_mask_nd = None
        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            resolution = self._resolution
            input_type = self._input_type

            if req.image_np is not None:
                image_np = result.image_np = [ (x.grayscale() if input_type == MxModel.InputType.Luminance else x.bgr()).resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.image_np]
                image_nd = np.stack([x.CHW() for x in image_np])

            if req.target_mask_np is not None:
                target_mask_np = result.target_mask_np = [x.grayscale().resize(resolution, resolution, interp=NPImage.Interp.LINEAR).f32() for x in req.target_mask_np]
                target_mask_nd = np.stack([x.CHW() for x in target_mask_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution and input_type == self._input_type:
                # Prepared data matches model parameters.
                break

        iteration = self._iteration
        device = self._device
        step_time = lib_time.measure()
        try:
            # Collect losses
            losses = []

            if  (target_mask_t := get_target_mask_t()) is not None and \
                (target_mask_u_t := get_target_mask_u_t()) is not None and \
                (pred_mask_t := get_pred_mask_t()) is not None and \
                (pred_mask_u_t := get_pred_mask_u_t()) is not None:

                    if (mse_power := req.mse_power) != 0.0:
                        losses.append( torch.mean(mse_power*10*torch.square(pred_mask_t-target_mask_t), (1,2,3)) )

                    if (dssim_x4_power := req.dssim_x4_power) != 0.0:
                        kernel_size = lib_math.next_odd(resolution//4)
                        losses.append( dssim_x4_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

                    if (dssim_x8_power := req.dssim_x8_power) != 0.0:
                        kernel_size = lib_math.next_odd(resolution//8)
                        losses.append( dssim_x8_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

                    if (dssim_x16_power := req.dssim_x16_power) != 0.0:
                        kernel_size = lib_math.next_odd(resolution//16)
                        losses.append( dssim_x16_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

                    if (dssim_x32_power := req.dssim_x32_power) != 0.0:
                        kernel_size = lib_math.next_odd(resolution//32)
                        losses.append( dssim_x32_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            loss_t = None
            if len(losses) != 0:
                loss_t = sum(losses).mean()
                loss_t.backward()

            # Collect predicts
            if req.pred_mask:
                if (pred_mask_u_t := get_pred_mask_u_t()) is not None:
                    result.pred_mask_np = [ NPImage(x, channels_last=False) for x in pred_mask_u_t.detach().cpu().numpy().clip(0, 1) ]

            # Optimization
            if (iteration % batch_acc) == (batch_acc-1):
                grad_mult = 1.0 / batch_acc

                opts = [model_opt]
                for opt in opts:
                    if opt is not Ellipsis:
                        opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

                if any(opt is not Ellipsis for opt in opts):
                    self._iteration = iteration + 1

            # Metrics
            result.error    = float( loss_t.detach().cpu().numpy() ) if loss_t is not None else 0

            result.accuracy = 0
            if  (target_mask_t := get_target_mask_t()) is not None and \
                (pred_mask_t := get_pred_mask_t()) is not None:
                result.accuracy = float( (1 - torch.abs(pred_mask_t-target_mask_t).sum((-1,-2)).mean() / (resolution*resolution*2)).detach().cpu().numpy() )

            result.time = step_time.elapsed()
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield ax.cancel(e)

    ##########################
    ##########################
    ##########################

class XSegModel(nn.Module):

    def __init__(self, in_ch, out_ch, generalization_level=6, base_dim=32, n_downs=6, dim_mults=(1, 2, 4, 8, 8, 8, 8, 8, 8)):
        super().__init__()
        self._base_dim = base_dim
        self._n_downs = n_downs
        self._generalization_level = min(n_downs, generalization_level)

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0, bias=False)

        down_list = self._down_list = nn.ModuleList()

        up_s_list = self._up_s_list = nn.ModuleList()
        up_r_list = self._up_r_list = nn.ModuleList()
        up_c_list = self._up_c_list = nn.ModuleList()

        dims = [ base_dim * mult for mult in dim_mults[:n_downs+1] ]
        for level, (up_ch, down_ch) in enumerate(list(zip(dims[:-1], dims[1:]))):

            down_list.append( nn.Sequential(nn.Conv2d (up_ch, down_ch, 3, 1, 1), nn.LeakyReLU(0.1),
                                            nn.Conv2d (down_ch, down_ch, 3, 1, 1), nn.LeakyReLU(0.1),
                                            BlurPool (down_ch, kernel_size=max(2, 4-level)),
                                        ))

            up_s_list.insert(0, nn.Sequential(  nn.Conv2d(down_ch, down_ch, 3, 1, 1),
                                                nn.LeakyReLU(0.2),
                                                nn.Conv2d(down_ch, down_ch, 3, 1, 1)))
            up_r_list.insert(0, nn.Conv2d(down_ch, down_ch, 3, 1, 1) )
            up_c_list.insert(0, nn.Conv2d(down_ch, up_ch*4, 3, 1, 1) )


        self._out = nn.Conv2d(base_dim, out_ch, 1, 1, 0, bias=False)
        self._out_gamma = nn.parameter.Parameter( torch.ones(out_ch,), requires_grad=True)
        self._out_beta = nn.parameter.Parameter( torch.zeros(out_ch,), requires_grad=True)

        xavier_uniform(self)

    def set_generalization_level(self, generalization_level : int):
        generalization_level = min(self._n_downs, generalization_level)

        if self._generalization_level != generalization_level:
            if generalization_level < self._generalization_level:
                # Reset shortcuts
                for i in range(generalization_level, self._generalization_level):
                    xavier_uniform(self._up_s_list[i])

            self._generalization_level = generalization_level
        return self._generalization_level

    def forward(self, inp):
        generalization_level = self._generalization_level

        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        shortcuts = []
        for down in self._down_list:
            x = down(x)
            shortcuts.insert(0, x)

        x = x * (x.square().mean(dim=[1,2,3], keepdim=True) + 1e-06).rsqrt()

        for i, (shortcut_x, up_s, up_r, up_c) in enumerate(zip(shortcuts, self._up_s_list, self._up_r_list, self._up_c_list)):
            level = self._n_downs-i-1

            x = x + up_r(x)
            if level >= generalization_level:
                x = x + up_s(shortcut_x)
            x = F.leaky_relu(x, 0.2)

            x = F.leaky_relu(up_c(x), 0.1)
            x = F.pixel_shuffle(x, 2)

        x = self._out(x)
        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x

