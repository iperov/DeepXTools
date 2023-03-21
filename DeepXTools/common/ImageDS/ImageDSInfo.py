from pathlib import Path
from typing import Set

from core.lib import path as lib_path


class ImageDSInfo:
    def __init__(self, path : Path):
        self._path = path
        
    def get_mask_dir_path(self, mask_type : str) -> Path:
        return self._path.parent / (self._path.name + f'_{mask_type}')
        
    def load_mask_types(self) -> Set[str]:
        result = set()
        
        mask_dir_prefix = f'{self._path.name}_'
        for dirpath in lib_path.get_dir_paths(self._path.parent):
            if (dirpath_name := dirpath.name).startswith(mask_dir_prefix):
                result.add(dirpath_name[len(mask_dir_prefix):])
                
        return result