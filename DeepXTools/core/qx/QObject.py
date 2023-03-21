from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, Sequence, Type, TypeVar

from .. import mx, qt
from ._helpers import q_init
from .QFuncWrap import QFuncWrap
from .QSettings import QSettings

if TYPE_CHECKING:
    from .QApplication import QApplication

class QObject(mx.Disposable):
    TObject = TypeVar('TObject', bound='QObject')

    def __init__(self, **kwargs):
        """
        kwargs:
        
            wrap_mode(False)    if True we don't touch wrapped qt object on dispose,
                                no hide, no delete later, etc
        
        """
        if QObject._QApplication is None:
            raise Exception('QApplication must be instantiated first.')

        super().__init__()
        self.__q_object = q_object = q_init('q_object', qt.QObject, **kwargs)
        self.__wrap_mode = kwargs.get('wrap_mode', False)

        # save wrapper reference in QObject
        setattr(self.__q_object, '__owner', self)

        self.__object_name = self.__class__.__qualname__
        self.__object_name_id = 0
        self.__parent : QObject = None
        self.__childs : Deque[QObject] = deque()
        self.__obj_name_counter : Dict[str, int] = {}
        self.__obj_name_count : Dict[str, int] = {}

        self.__vtp_listeners = set() # Visible to parent listeners
        if self._visible_to_parent_event.__func__ != QObject._visible_to_parent_event or \
           self._invisible_to_parent_event.__func__ != QObject._invisible_to_parent_event or \
           self._settings_event.__func__ != QObject._settings_event:
            # _visible_to_parent_event-related methods are overriden, subscribe self
            self.__vtp_listeners.add(self)
        
        self.__mx_visible_to_parent = mx.ReplayEvent1[QObject](replayer=self._mx_visible_to_parent_replayer).dispose_with(self)
        self.__mx_invisible_to_parent = mx.Event1[QObject]().dispose_with(self)
        self.__mx_child_added = mx.Event1[QObject]().dispose_with(self)
        self.__mx_child_remove = mx.Event1[QObject]().dispose_with(self)

        self.__eventFilter_wrap = QFuncWrap(q_object, 'eventFilter', lambda *args, **kwargs: self._event_filter(*args, **kwargs)).dispose_with(self)
        self.__event_wrap = QFuncWrap(q_object, 'event', lambda *args, **kwargs: self._event(*args, **kwargs)).dispose_with(self)

    @property
    def mx_visible_to_parent(self) -> mx.IReplayEvent1_r[QObject]:
        self.__vtp_listeners.add(self)
        return self.__mx_visible_to_parent
    @property
    def mx_invisible_to_parent(self) -> mx.IEvent1_r[QObject]:
        self.__vtp_listeners.add(self)
        return self.__mx_invisible_to_parent
    @property
    def mx_child_added(self) -> mx.IEvent1_r[QObject]: return self.__mx_child_added
    @property
    def mx_child_remove(self) -> mx.IEvent1_r[QObject]: return self.__mx_child_remove

    def _is_wrap_mode(self) -> bool:
        return self.__wrap_mode

    def _mx_visible_to_parent_replayer(self, conn : mx.EventConnection):
        parent = self.get_parent()
        while parent is not None:
            conn.emit(parent)
            parent = parent.get_parent()

    def __dispose__(self):
        """Dispose this object. All childs will be disposed. This object will be removed from parent."""
        self.dispose_childs()
        
        if not self._is_wrap_mode():
            self.set_parent(None)
            self.__q_object.deleteLater()
            
        self.__q_object.__owner = None
        self.__q_object = None

        super().__dispose__()

    def dispose_childs(self):
        """Dispose all child objects."""
        while len(self.__childs) != 0:
            child = self.__childs[0]
            child.dispose()
        return self

    def dispose_with_childs(self, object : QObject):
        self.dispose_with( QObject().set_parent(object) )
        return self

    def get_q_object(self) -> qt.QObject: return self.__q_object

    def get_childs(self) -> Sequence[QObject]: return reversed(tuple(self.__childs))
    def get_parent(self) -> QObject|None: return self.__parent

    # def create_null_child(self) -> QObject:
    #     """same as QObject().set_parent(obj)"""
    #     return QObject().set_parent(self)

    def get_object_name(self) -> str:
        """object name. Not unique relative to parent. Example: XLabel"""
        return self.__object_name
    def get_object_name_id(self) -> str:
        """object name id relative to parent. Example: 0"""
        return self.__object_name_id
    def get_name(self) -> str:
        """unique name relative to parent (if exists). Example: XLabel:0"""
        return f'{self.__object_name}:{self.__object_name_id}'
    def get_tree_name(self,) -> str:
        """Unique tree name up to current top parent (if exists). Example: QWindow:0/XLabel:0"""
        s = deque()
        parent = self
        while parent is not None:
            s.appendleft(parent.get_name() )
            parent = parent.__parent
        return '/'.join(s)

    def get_top_parent(self) -> QObject|None:
        parent = self
        while parent is not None:
            top_parent = parent
            parent = parent.__parent
        return top_parent

    def get_top_parent_by_class(self, cls_ : Type[TObject]) -> TObject|None:
        parent = self
        top_parent = None
        while parent is not None:
            if isinstance(parent, cls_):
                top_parent = parent
            parent = parent.__parent
        return top_parent

    def get_first_parent_by_class(self, cls_ : Type[TObject]) -> TObject|None:
        parent = self
        while parent is not None:
            if isinstance(parent, cls_):
                return parent
            parent = parent.__parent
        return None

    def set_object_name(self, name : str|None):
        if self.__parent is not None:
            raise Exception('object_name must be set before parent')
        self.__object_name = f"{self.__class__.__qualname__}{ ('_'+name) if name is not None else ''}"
        return self

    def set_parent(self, new_parent : QObject|None):
        if self._is_wrap_mode():
            raise Exception('set_parent is not allowed in wrap_mode')
        
        if self.__parent != new_parent:
            if self.__parent is not None:
                self.__parent._child_remove_event(self)
                self.__q_object.setParent(None)

            self.__parent = new_parent

            if new_parent is not None:
                new_parent._child_added_event(self)

        return self

    def _visible_to_parent_event(self, parent : QObject):
        """inheritable at first. Event appears when this object became visible to parent or far parent via parent-child traverse."""
        self.__mx_visible_to_parent.emit(parent)
        if isinstance(parent, self._QApplication):
            self._settings_event(parent.get_settings(self.get_tree_name()))

    def _invisible_to_parent_event(self, parent : QObject):
        """inheritable at last. Event appears when this object became invisible to parent or far parent via parent-child traverse."""
        self.__mx_invisible_to_parent.emit(parent, reverse=True)

    def _settings_event(self, settings : QSettings):
        """```
        inheritable
        Settings event appears when QObject is visible to QApplication via parent-child traverse.

            settings    QSettings
                            lifetime mutable dict of local settings of this QObject
                            which are saved/loaded with QApplication
        ```"""

    def _event_filter(self, object : qt.QObject, ev : qt.QEvent) -> bool:
        """inheritable. Return True to eat the event"""
        return self.__eventFilter_wrap.get_super()(object, ev)

    def _event(self, ev : qt.QEvent) -> bool:
        """inheritable"""
        return self.__event_wrap.get_super()(ev)

    def _child_added_event(self, child : QObject):
        """inheritable at first. Called when child QObject has been added"""
        obj_name = child.get_object_name()

        obj_name_count = self.__obj_name_count
        obj_name_count[obj_name] = obj_name_count.get(obj_name, 0) + 1

        obj_name_counter = self.__obj_name_counter
        count = child.__object_name_id = obj_name_counter.get(obj_name, 0)
        obj_name_counter[obj_name] = count + 1

        self.__childs.appendleft(child)
        parent = self
        while parent is not None:
            parent.__vtp_listeners.update(child.__vtp_listeners)
            for obj in child.__vtp_listeners:
                obj._visible_to_parent_event(parent)

            parent = parent.__parent

        self.__mx_child_added.emit(child)

    def _child_remove_event(self, child : QObject):
        """inheritable at last. Called when child QObject will be removed from childs."""
        self.__mx_child_remove.emit(child, reverse=True)

        parent = self
        while parent is not None:
            for obj in child.__vtp_listeners:
                obj._invisible_to_parent_event(parent)
            parent.__vtp_listeners.difference_update(child.__vtp_listeners)
            parent = parent.__parent
        self.__childs.remove(child)

        obj_name = child.get_object_name()
        obj_name_count = self.__obj_name_count
        new_count = obj_name_count[obj_name] = obj_name_count[obj_name] - 1
        if new_count == 0:
            obj_name_counter = self.__obj_name_counter
            obj_name_counter[obj_name] = 0
        child.__object_name_id = 0

    def __repr__(self): return self.__str__()
    def __str__(self):
        return f'{super().__str__()}[{self.get_tree_name()}][Childs:{len(self.__childs)}]'#

    _QApplication : Type[QApplication] = None
