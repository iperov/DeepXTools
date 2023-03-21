from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List

import torch

#torch.backends.cudnn.benchmark = True

try:
    import torch_directml
except:
    torch_directml = None
    
@dataclass
class DeviceInfo:
    backend : str
    index : int
    name : str
    total_memory : int

class DeviceRef:
    """
    """
    @staticmethod
    def from_state(state : dict = None) -> DeviceRef:
        state = state or {}
        return DeviceRef(backend = state.get('backend', 'cpu'), index = state.get('index', 0) )

    def __init__(self, backend : str, index : int):
        self._backend = backend
        self._index = index

    @property
    def backend(self) -> str: return self._backend
    @property
    def index(self) -> int: return self._index
    @property
    def info(self) -> DeviceInfo:
        if not self.is_cpu:
            for info in get_avail_gpu_devices_info():
                if info.backend == self._backend and info.index == self._index:
                    return info
        # Fallback to CPU
        return get_cpu_device_info()

    @property
    def device(self) -> torch.device:
        """get torch.device. If invalid, returns CPU device."""
        info = self.info
        backend = info.backend
        
        if backend == 'dml':
            if torch_directml is not None:
                return torch_directml.device(info.index)
        elif info.backend == 'cpu':
            return torch.device(f'{info.backend}')        
        
        return torch.device(f'{info.backend}:{info.index}')

    @property
    def is_cpu(self) -> bool: return self._backend == 'cpu'

    def get_state(self) -> dict: return {'backend' : self._backend, 'index': self._index}

    def __hash__(self): return (self._backend, self._index).__hash__()
    def __eq__(self, other):
        if self is not None and other is not None and isinstance(self, DeviceRef) and isinstance(other, DeviceRef):
            return self._backend == other._backend and self._index == other._index
        return False

    def __str__(self):
        info = self.info
        if info.backend == 'cpu':
            s = f'[{info.backend.upper()}]'
        else:
            s = f'[{info.backend.upper()}:{info.index}] {info.name}'
            if (total_memory := info.total_memory) != 0:
                s += f' [{(total_memory / 1024**3) :.3}Gb]'
        return s

    def __repr__(self):
        return f'{self.__class__.__name__} object: ' + self.__str__()

_gpu_devices_info_lock = threading.Lock()
_gpu_devices_info = None

def get_cpu_device_info() -> DeviceInfo: return DeviceInfo(backend='cpu', index=0, name='CPU', total_memory=0)

def get_avail_gpu_devices_info() -> List[DeviceInfo]:
    global _gpu_devices_info_lock
    global _gpu_devices_info
    with _gpu_devices_info_lock:
        if _gpu_devices_info is None:
            _gpu_devices_info = []

            for i in range (torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                _gpu_devices_info.append( DeviceInfo(backend='cuda', index=i, name=device_props.name, total_memory=device_props.total_memory) )

            if torch_directml is not None:
                for i in range (torch_directml.device_count()):
                    _gpu_devices_info.append( DeviceInfo(backend='dml', index=i, name=torch_directml.device_name(i), total_memory=0) )


    return _gpu_devices_info


def get_cpu_device() -> DeviceRef: return DeviceRef(backend='cpu', index=0)

def get_avail_gpu_devices() -> List[DeviceRef]:
    return [ DeviceRef(backend=dev.backend, index=dev.index) for dev in get_avail_gpu_devices_info() ]


