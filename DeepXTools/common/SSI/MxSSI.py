from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from core.lib.image import NPImage


class MxSSI:
    """Sheet-Section-Item"""

    @dataclass(frozen=True)
    class Section: ...

    @dataclass(frozen=True)
    class Item: ...

    @dataclass(frozen=True)
    class Image(Item):
        image   : NPImage|None = None
        caption : str|None     = None

        @staticmethod
        def from_state(state : dict) -> MxSSI.Image|None:
            try:
                return MxSSI.Image( image   = NPImage(image) if (image := state.get('image', None)) is not None else None,
                                    caption = caption        if (caption := state.get('caption', None)) is not None else None, )
            except Exception as e:
                return None

        def get_state(self) -> dict:
            return {'caption' : self.caption if self.caption is not None else None,
                    'image'   : self.image.HWC() if self.image is not None else None}

    @dataclass(frozen=True)
    class Grid(Section):
        """ [ (row,col) ] = item """
        
        items : Dict[ Tuple[int,int], MxSSI.Item ] = field(default_factory=dict)

        @staticmethod
        def from_state(state : dict) -> MxSSI.Grid:
            try:
                items = {}
                for type_name, key, item_state in state.get('items', []):
                    if (type_cls := getattr(MxSSI, type_name, None)) is not None:
                        if issubclass(type_cls, MxSSI.Item):

                            if (item := type_cls.from_state(item_state) ) is not None:
                                items[key] = item

                return MxSSI.Grid(items=items)
            except Exception as e:
                return MxSSI.Grid()

        def get_state(self) -> dict:
            return {'items' : [ (type(item).__name__, key, item.get_state()) for key, item in self.items.items() ] }

    @dataclass(frozen=True)
    class Sheet:
        sections : Dict[str, MxSSI.Section] = field(default_factory=dict)

        @staticmethod
        def from_state(state : dict) -> MxSSI.Sheet:
            
            try:
                sections = {}
                for type_name, key, section_state in state.get('sections', []):

                    if (type_cls := getattr(MxSSI, type_name, None)) is not None:
                        if issubclass(type_cls, MxSSI.Section):

                            if (section := type_cls.from_state(section_state) ) is not None:
                                sections[key] = section

                return MxSSI.Sheet(sections=sections)
            except Exception as e:
                return MxSSI.Sheet()

        def get_state(self) -> dict:
            return {'sections' : [ (type(section).__name__, key, section.get_state()) for key, section in self.sections.items() ] }
