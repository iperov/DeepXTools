from pathlib import Path

from core import mx

from .ImageDSInfo import ImageDSInfo


class MxImageDSRefListItem(mx.Disposable):
    def __init__(self, state : dict = None):
        super().__init__()
        self._state = state = state or {}

        self._mx_image_ds_path = mx.PathState(  config=mx.PathStateConfig(dir_only=True),
                                                on_close=self._on_close,
                                                on_open=self._on_open).dispose_with(self)
        self._mx_mask_type = None

        if (path := state.get('image_ds_path', None)) is not None:
            self._mx_image_ds_path.open(path)

        self._state = {}

    @property
    def mx_image_ds_path(self) -> mx.IPathState:
        return self._mx_image_ds_path

    @property
    def mx_mask_type(self) -> mx.ISingleChoice[str|None]|None:
        """avail when mx_image_ds_path.mx_path is not None"""
        return self._mx_mask_type

    def get_state(self) -> dict:
        d = {}

        if (path := self._mx_image_ds_path.mx_path.get()) is not None:
            d['image_ds_path'] = path
            d['mask_type'] = self._mx_mask_type.get()

        return d

    def _on_close(self):
        self._mx_mask_type = self._mx_mask_type.dispose()

    def _on_open(self, path : Path):
        mx_mask_type = self._mx_mask_type = mx.SingleChoice[str|None](None, avail=lambda: [None]+sorted(ImageDSInfo(path).load_mask_types()))
        mx_mask_type.set(self._state.get('mask_type', None))

        return True

class MxImageDSRefList(mx.List[MxImageDSRefListItem]):
    """list of references to ImageDS with mask type"""
    
    def __init__(self, state : dict = None):
        super().__init__()
        state = state or {}

        for value_state in state.get('values_states', []):
            super().append( MxImageDSRefListItem(state=value_state) )


    def __dispose__(self):
        for item in self.values():
            item.dispose()
        super().__dispose__()

    def get_state(self) -> dict:
        return {'values_states' : [item.get_state() for item in self.values()] }

    def new(self) -> MxImageDSRefListItem:
        super().append( MxImageDSRefListItem() )

    def remove(self, item : MxImageDSRefListItem):
        super().remove(item)
        item.dispose()