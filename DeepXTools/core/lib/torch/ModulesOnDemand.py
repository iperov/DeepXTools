from __future__ import annotations

import io
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.optim as opt

from ... import mx
from .device import DeviceRef, get_cpu_device


class ModulesOnDemand(mx.Disposable):
    """
    Convenient class to operate multiple torch Module/Optimizer on demand by key.

    Non thread-safe.
    """

    def __init__(self, module_dict : Dict[Any, Callable[[ModulesOnDemand], nn.Module]],
                       state : dict = None):
        """
        ```
            module_dict     dict of callable by key which instantiates module on demand

            state           keeps state until first module request
        ```

        raise no errors.
        """
        super().__init__()
        
        self._module_dict = module_dict
        self._initial_state = state = state or {}

        self._modules = {}
        self._modules_device : Dict [Any, DeviceRef|None] = {}
        self._modules_train = {}

        # Remove unused keys from state
        dict_keys = set(f'{key}_state_bytes' for key in module_dict.keys())

        for state_key in set(state.keys()):
            if state_key not in dict_keys:
                state.pop(state_key)

    def __dispose__(self):
        self.reset_modules()
        super().__dispose__()

    def get_state(self) -> dict:
        d = self._initial_state.copy()

        for key, module in self._modules.items():
            if module is not None:
                model_state_bytes = io.BytesIO()
                torch.save(module.state_dict(), model_state_bytes)
                model_state_bytes = model_state_bytes.getbuffer().tobytes()
                d[f'{key}_state_bytes'] = model_state_bytes
        return d

    def reset_modules(self):
        """reset all modules. Modules will be reinstantiated on next request."""
        for key in list(self._modules.keys()):
            self.reset_module(key)
        
    def reset_module(self, key):
        """Reset specific module. Module will be reinstantiated on next request."""
        self._initial_state[f'{key}_state_bytes'] = None
        if key in self._modules:
            self._modules.pop(key)
            self._modules_train.pop(key)
            device = self._modules_device.pop(key)

            if device is not None and device.backend == 'cuda':
                torch.cuda.empty_cache()

    def get_module(self, key : Any, device : DeviceRef = None, train : bool = None) -> nn.Module:
        """
        Get/create module or optimizer by key.

        module's `device` and `train` will be updated

        raises on error. If error: Module is unusable and should be resetted or request with CPU device.
        """
        module = self._modules.get(key, None)
        if module is None:
            if key not in self._module_dict.keys():
                raise ValueError(f'Key {key} not in avail {self._module_dict.keys()}')

            # Module does not exists. Instantiate new.
            # Load from initial state
            state_key = f'{key}_state_bytes'
            if (model_state_bytes := self._initial_state.get(state_key, None)) is not None:
                model_state = torch.load(io.BytesIO(model_state_bytes), map_location='cpu')
            else:
                model_state = None

            # Try to instantiate module on CPU
            module = self._module_dict[key](self)
            if not isinstance(module, (nn.Module, opt.Optimizer)):
                raise ValueError(f'Instantiation func must return nn.Module or Optimizer')

            if model_state is not None:
                try:
                    module.load_state_dict(model_state)#, strict=False
                except:
                    # Something goes wrong during loading.
                    # We need to instantiate again, because weights may be loaded partially that is unacceptable.
                    module = None # Delete reference in order to free RAM before reinstantiation
                    module = self._module_dict[key](self)

            if isinstance(module, nn.Module):
                module.eval()

            # No exception at this point, now we can update vars.

            # Consume initial state
            self._initial_state[state_key] = None
            # Update vars
            self._modules[key] = module
            self._modules_device[key] = get_cpu_device()
            self._modules_train[key] = False

        if device is not None and (prev_device := self._modules_device[key]) != device:
            # Update device to the requested
            try:
                if isinstance(module, nn.Module):
                    module.to(device.device)
                else:
                    module.load_state_dict(module.state_dict())
            except Exception as e:
                # Exception during full or partially changed device
                # reset device in order to set again in next time.
                self._modules_device[key] = None
                # raise error assuming User will choose other device
                raise e
            
            if prev_device is not None and prev_device.backend == 'cuda':
                torch.cuda.empty_cache()

            # No exception at this point, success
            self._modules_device[key] = device

        if isinstance(module, nn.Module):
            if train is not None and self._modules_train[key] != train:
                # Update train/eval to requested
                # No exception at this point, success
                module.train() if train else module.eval()
                self._modules_train[key] = train

        return module
