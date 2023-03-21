from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

from core.lib import path as lib_path
from core.lib.image import NPImage

from .ImageDSInfo import ImageDSInfo


class ImageDS:
    """
    a class to read/write images and masks from directories
    
    example
    
    \\foodir
    
    \\foodir_mask1
    
    \\foodir_mask2    
    """

    @staticmethod
    def open(path : Path) -> ImageDS:
        """```
            path    directory Path.

        raise errors.
        ```"""
        if not path.is_dir():
            raise Exception(f'{path} must be a dir.')

        if len(path.parents) == 0:
            raise Exception(f'Path cannot be a root drive dir.')

        return ImageDS(path)


    def __init__(self, path : Path):
        """```
        Don't use constructor directly.
        Use ImageDS.open(path)
        ```"""
        super().__init__()
        self._path = path

        self._info = ImageDSInfo(path)

        # Loading images paths
        self._images_paths : List[Path] = lib_path.get_files_paths(self._path, extensions=NPImage.avail_image_suffixes())

        # Loading mask types
        self._mask_types : Set[str] = self._info.load_mask_types()

        self._image_count = len(self._images_paths)

    @property
    def image_count(self) -> int:
        return self._image_count

    def get_path(self) -> Path: return self._path
    def get_image_count(self) -> int: return len(self._images_paths)
    def get_image_name(self, id : int) -> str: self._assert_id(id); return self._images_paths[id].name
    def get_mask_types(self) -> List[str]: return sorted(self._mask_types)
    def get_mask_dir_path(self, mask_type : str) -> Path:  return self._info.get_mask_dir_path(mask_type)
    def get_mask_path(self, id : int, mask_type : str) -> Path: return self.get_mask_dir_path(mask_type) / f'{self._images_paths[id].stem}.png'

    def add_mask_type(self, mask_type : str):
        """```
            mask_type   str     will be automatically filtered
                                allowed chars: letter(unicode),number,_
                                regex filter: re.sub('\W', '', s)

        returns filtered mask_type
        ```"""
        mask_type = re.findall('\w+', mask_type)[0]

        if mask_type not in self._mask_types:
            self._mask_types.add(mask_type)
            try:
                self.get_mask_dir_path(mask_type).mkdir(parents=True, exist_ok=True)
            except: ...

    def load_image(self, id : int) -> NPImage:
        """
        raises on error
        """
        self._assert_id(id)
        return NPImage.from_file(self._images_paths[id])

    def load_mask(self, id : int, mask_type : str) -> NPImage:
        """
        returns mask HW1 float32

        raises on error
        """
        self._assert_id(id)
        self._assert_mask_type(mask_type)
        return NPImage.from_file(self.get_mask_path(id, mask_type))

    def save_mask(self, id : int, mask_type : str, mask : NPImage):
        """
        raises on error
        """
        self._assert_id(id)
        self._assert_mask_type(mask_type)

        mask.grayscale().save(self.get_mask_path(id, mask_type))

    def delete_mask(self, id : int, mask_type : str):
        """
        raises on error
        """
        self._assert_id(id)
        mask_path = self.get_mask_path(id, mask_type)
        if mask_path.exists():
            mask_path.unlink()

    def has_mask(self, id : int, mask_type : str) -> bool:
        """raises on error"""
        self._assert_id(id)
        self._assert_mask_type(mask_type)

        return self.get_mask_path(id, mask_type).exists()


    def get_next_image_id_with_mask(self, id : int, mask_type : str, forward : bool = True) -> int|None:
        """raises on error"""
        self._assert_id(id)
        self._assert_mask_type(mask_type)

        mask_dir_path = self.get_mask_dir_path(mask_type)

        for i in (range(id+1, len(self._images_paths)) if forward else \
                    range(id-1, -1, -1)):
            if (mask_dir_path / f'{self._images_paths[i].stem}.png').exists():
                return i

        return None

    def _assert_id(self, id : int):
        if id < 0 or id >= len(self._images_paths):
            raise ValueError('id must be in range [0..image_count-1]')

    def _assert_mask_type(self, mask_type : str):
        if mask_type not in self._mask_types:
            raise ValueError('mask_type not in defined')


    def __repr__(self): return self.__str__()
    def __str__(self): return f"{super().__str__()}[ImageDS]"
