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
        self._unet_mode  = state.get('unet_mode', True)
        self._iteration  = state.get('iteration', 0)
        self._opt_class = AdaBelief
        
        self._mx_info = mx.TextEmitter().dispose_with(self)
        self._mx_device = mx.SingleChoice[lib_torch.DeviceRef](self._device,
                                                               avail=lambda: [lib_torch.get_cpu_device()]+lib_torch.get_avail_gpu_devices()).dispose_with(self)
        self._mx_input_type = mx.SingleChoice[MxModel.InputType](self._input_type, avail=lambda: [*MxModel.InputType]).dispose_with(self)
        self._mx_resolution = mx.Number(self._resolution, config=mx.NumberConfig(min=64, max=1024, step=64)).dispose_with(self)
        self._mx_base_dim   = mx.Number(self._base_dim, config=mx.NumberConfig(min=16, max=256, step=8)).dispose_with(self)
        self._mx_unet_mode = mx.Flag(self._unet_mode).dispose_with(self)
        self._mx_url_download_menu = mx.Menu[str](avail_choices=lambda: ['https://github.com/iperov/DeepXTools/releases/download/DXRM_0/Luminance_256_32_UNet_from_ImageNet.dxrm'], 
                                         on_choose=lambda x: self._download_model_state(x)
                                         ).dispose_with(self)

        self._mod = lib_torch.ModulesOnDemand(  {'encoder': lambda mod: Encoder(resolution=self._resolution, in_ch=1 if self._input_type == MxModel.InputType.Luminance else 3, base_dim=self._base_dim),
                                                 'encoder_opt': lambda mod: self._opt_class(mod.get_module('encoder').parameters()),

                                                 'decoder': lambda mod: Decoder(out_ch=1, base_dim=self._base_dim, unet_mode=self._unet_mode),
                                                 'decoder_opt': lambda mod: self._opt_class(mod.get_module('decoder').parameters()),
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
    def mx_unet_mode(self) -> mx.IFlag:
        return self._mx_unet_mode
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
                'unet_mode'    : self._unet_mode,
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
        new_unet_mode      = self._mx_unet_mode.get()

        yield ax.switch_to(self._model_thread)

        mod = self._mod

        device_info, self._device = self._device, new_device
        input_type, self._input_type = self._input_type, new_input_type
        resolution, self._resolution = self._resolution, new_resolution
        base_dim, self._base_dim = self._base_dim, new_base_dim
        unet_mode, self._unet_mode = self._unet_mode, new_unet_mode

        reset_encoder    = (resolution != new_resolution or base_dim != new_base_dim) or (input_type != new_input_type)
        reset_decoder    = (resolution != new_resolution or base_dim != new_base_dim)

        if reset_encoder:
            mod.reset_module('encoder')
            mod.reset_module('encoder_opt')
        if reset_decoder:
            mod.reset_module('decoder')
            mod.reset_module('decoder_opt')

        if unet_mode != new_unet_mode:
            decoder : Decoder = self._mod.get_module('decoder')
            decoder.set_unet_mode(new_unet_mode)

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
        unet_mode   = self._unet_mode
        
        yield ax.switch_to(self._main_thread)

        self._mx_device.set(device_info)
        self._mx_input_type.set(input_type)
        self._mx_resolution.set(resolution)
        self._mx_base_dim.set(base_dim)
        self._mx_unet_mode.set(unet_mode)
        
    ######################################
    ### RESET / IMPORT / EXPORT / DOWNLOAD
    ######################################
    @ax.task
    def reset_model(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._model_thread)
        self._mod.reset_module('encoder')
        self._mod.reset_module('encoder_opt')
        self._mod.reset_module('decoder')
        self._mod.reset_module('decoder_opt')
    
    
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
            unet_mode  = model_state['unet_mode']
            mm_state   = model_state['mm_state']
        except Exception as e:
            err = e
            
        if err is None:
            self._input_type = input_type
            self._resolution = resolution
            self._base_dim = base_dim
            self._unet_mode = unet_mode
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
        
        train_mask  = req.image_np is not None and req.target_mask_np is not None
        pred_mask   = req.image_np is not None and (train_mask or req.pred_mask)
        
        train = train_mask
        
        while True:
            # Prepare data in pool.
            yield ax.switch_to(self._prepare_thread_pool)

            resolution = self._resolution
            input_type = self._input_type
            
            if req.image_np is not None:
                p_image_np = result.image_np = [ (x.grayscale() if input_type == MxModel.InputType.Luminance else x.bgr()).resize(resolution, resolution, interp=NPImage.Interp.LANCZOS4).f32() for x in req.image_np]
                p_image_nd = np.stack([x.CHW() for x in p_image_np])
            
            if req.target_mask_np is not None:
                p_target_mask_np = result.target_mask_np = [x.grayscale().resize(resolution, resolution, interp=NPImage.Interp.LINEAR).f32() for x in req.target_mask_np]
                p_target_mask_nd = np.stack([x.CHW() for x in p_target_mask_np])

            yield ax.switch_to(self._model_thread)

            if resolution == self._resolution and input_type == self._input_type:
                # Prepared data matches model parameters.
                break
        
        try:
            step_time = lib_time.measure()
            iteration = self._iteration
            device = self._device

            encoder : Encoder = self._mod.get_module('encoder', device=device, train=train)
            decoder : Decoder = self._mod.get_module('decoder', device=device, train=train)
            
            if pred_mask:            
                input_t = torch.tensor(p_image_nd, device=device.device)
            if train_mask:                
                target_mask_t = torch.tensor(p_target_mask_nd, device=device.device) * 2.0 - 1.0
            
            if pred_mask:                
                shortcuts, x = encoder(input_t)
                pred_mask_t = decoder(shortcuts, x)
                
                if req.pred_mask:
                    result.pred_mask_np = [ NPImage(x, channels_last=False) for x in pred_mask_t.detach().cpu().numpy().clip(0, 1) ]
                
            if train:
                encoder_opt : Optimizer = self._mod.get_module('encoder_opt', device=device)
                decoder_opt : Optimizer = self._mod.get_module('decoder_opt', device=device)
                if (iteration % batch_acc) == 0:
                    encoder_opt.zero_grad()
                    decoder_opt.zero_grad()
            
            # Collect losses
            losses = []

            if (mse_power := req.mse_power) != 0.0:
                if train_mask:
                    losses.append( torch.mean(mse_power*10*torch.square(pred_mask_t-target_mask_t), (1,2,3)) )

            if (req.dssim_x4_power + req.dssim_x8_power + req.dssim_x16_power + req.dssim_x32_power) != 0.0:
                if train_mask:
                    pred_mask_u_t = pred_mask_t / 2 + 0.5
                    target_mask_u_t = target_mask_t / 2 + 0.5

            if (dssim_x4_power := req.dssim_x4_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//4)
                if train_mask:
                    losses.append( dssim_x4_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x8_power := req.dssim_x8_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//8)
                if train_mask:
                    losses.append( dssim_x8_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x16_power := req.dssim_x16_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//16)
                if train_mask:
                    losses.append( dssim_x16_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            if (dssim_x32_power := req.dssim_x32_power) != 0.0:
                kernel_size = lib_math.next_odd(resolution//32)
                if train_mask:
                    losses.append( dssim_x32_power*xF.dssim(pred_mask_u_t, target_mask_u_t, kernel_size=kernel_size, use_padding=False).mean([-1]) )

            loss_t = None
            if len(losses) != 0:
                loss_t = sum(losses).mean()
                loss_t.backward()

                # Optimization
                if (iteration % batch_acc) == (batch_acc-1):
                    grad_mult = 1.0 / batch_acc

                    if train:
                        encoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)
                        decoder_opt.step(iteration=iteration, grad_mult=grad_mult, lr=lr, lr_dropout=lr_dropout)

            # Metrics
            result.error    = float( loss_t.detach().cpu().numpy() ) if loss_t is not None else 0
            result.accuracy = float( (1 - torch.abs(pred_mask_t-target_mask_t).sum((-1,-2)).mean() / (resolution*resolution*2)).detach().cpu().numpy() ) if train else 0

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

        xavier_uniform(self)

    def get_out_resolution(self) -> int:
        return self._resolution // (2**self._n_downs)

    def forward(self, inp):
        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        shortcuts = []
        for conv, bp in zip(self._conv_list, self._bp_list):
            x = F.leaky_relu(conv(x), 0.1)
            x = bp(x)
            shortcuts.insert(0, x)

        return shortcuts, x


class Decoder(nn.Module):
    def __init__(self, out_ch, base_dim=32, n_downs=6, unet_mode=True):
        super().__init__()
        self._n_downs = n_downs
        self._unet_mode = unet_mode

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

        xavier_uniform(self)

    def set_unet_mode(self, unet_mode : bool):
        if self._unet_mode != unet_mode:
            self._unet_mode = unet_mode
            if unet_mode:
                # Reset shortcuts
                xavier_uniform(self._conv_s1_list)
                xavier_uniform(self._conv_s2_list)

    def forward(self, shortcuts : List[torch.Tensor], x):

        for shortcut_x, s1, s2, r, c, in zip(shortcuts, self._conv_s1_list, self._conv_s2_list, self._conv_r_list, self._conv_c_list):
            x = x + r(x)
            if self._unet_mode:
                x = x + s2(F.leaky_relu(s1(shortcut_x), 0.2))
            x = F.leaky_relu(x, 0.2)
            x = F.leaky_relu(c(x), 0.1)
            x = F.pixel_shuffle(x, 2)

        x = F.leaky_relu(self._out_conv(x), 0.1)
        x = self._out(x)
        x = x * self._out_gamma[None,:,None,None]
        x = x + self._out_beta[None,:,None,None]

        return x
