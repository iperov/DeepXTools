from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from core import ax, mx


class MxGraph(mx.Disposable):
    @dataclass(frozen=True)
    class Data:
        array : np.ndarray    # shape (N, C)
        names : Sequence[str] # name for every N
                
    def __init__(self,  state : dict = None):
        super().__init__()
        state = state or {}
        
        self._tg = ax.TaskGroup().dispose_with(self)
        self._main_thread = ax.get_current_thread()
        self._graph_thread = ax.Thread().dispose_with(self)
        
        ###
        self._length = state.get('length', 0)
        self._all_names = state.get('all_names', ())
        self._array = state.get('array', np.empty((len(self._all_names),self._length), np.float32) )
        #self._array = state.get('array', np.random.rand( len(self._all_names),self._length )).astype(np.float32)
        
        ### ^ operated only in _graph_thread
        
        self._mx_data = mx.Property[MxGraph.Data](MxGraph.Data(names=(), array=np.empty( (0,0), np.float32 ))).dispose_with(self)
        
        self._mx_names = mx.MultiChoice[str](avail=lambda: self._all_names).dispose_with(self)
        self._mx_names.listen(lambda _: self._update_mx_data(instant=True))
        self._mx_names.set( state.get('names', self._all_names) )
        
        
    @property
    def mx_data(self) -> mx.IProperty_r[Data]:
        """current snapshot of graph data. Updated not often than 1 per sec"""
        return self._mx_data

    @property
    def mx_names(self) -> mx.MultiChoice[str]:
        """Controls names presented in mx_data."""
        return self._mx_names

    # @property
    # def length(self) -> int:
    #     return self._length
                
    @ax.task
    def add(self, values : Dict[str, float]): 
        """
        Task to add values at the end of graph.
        
        if graph of name does not exist it will be created with zero values.
        
        if name exists and value not specified in argument, it will not be set.
        """
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._graph_thread)
        
        length = self._length
        array = self._array 
        all_names = self._all_names 
        
        N, C = array.shape
        
        if length >= C:
            # Requested length more than avail capacity
            # Expand capacity
            NEW_C = length+256*1024 
            
            new_array = np.empty( (N, NEW_C), np.float32 )
            new_array[:, :C] = array
            new_array[:, C:] = 0.0
            
            array = new_array
            C = NEW_C
            
        for name in values.keys():
            if name not in all_names:
                all_names = all_names + (name,)
                
        if (N_diff := (len(all_names) - N)) > 0:
            # Expand length
            array = np.concatenate([array,
                                    np.zeros( (N_diff, C), np.float32) ], 0)
            N += N_diff
            
        for name, value in values.items():
            array[all_names.index(name), length] = value
            
        self._length = length + 1
        self._array = array
        
        if self._all_names != all_names:
            old_all_names, self._all_names = self._all_names, all_names
            
            yield ax.switch_to(self._main_thread)
            
            self._mx_names.set(self._mx_names.get() + tuple(x for x in all_names if x not in old_all_names))
            
        
        
        self._update_mx_data()
        
        
    @ax.task
    def trim(self, f_start : float, f_end : float):
        """
        Task to trim the graph
        
            f_start [0..1] inclusive
            
            f_end   [0..1] inclusive        
        """
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._graph_thread)
        
        f_end   = max(0.0, min(f_end, 1.0))
        f_start = max(0.0, min(f_start, f_end, 1.0))
        
        start_length = int(f_start * self._length)
        end_length   = int(f_end * self._length)
        
        self._length = end_length - start_length 
        array = self._array[:, start_length:end_length]
        valid_n = [ n for n in range(array.shape[0]) if np.any(array[n] != 0) ]
        self._array = array[valid_n, :]
        
        self._all_names = tuple( x for n, x in enumerate(self._all_names) if n in valid_n )
        
        yield ax.switch_to(self._main_thread)
        
        self._mx_names.update()
        self._update_mx_data(instant=True)
    
    @ax.task
    def get_state(self) -> dict:
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)
        names = self._mx_names.get()
        yield ax.switch_to(self._graph_thread)
        
        return {'length' : self._length,
                'all_names' : self._all_names,
                'array' : self._array,
                'names' : names,
                }
        
    @ax.task
    def _update_mx_data(self, instant = False):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)
                
        if not instant:
            yield ax.time_barrier(interval=1.0, max_task=1)

        names = self._mx_names.get()
        
        yield ax.switch_to(self._graph_thread)
        
        data = MxGraph.Data(names=[ name for name in self._all_names if name in names ],
                            array=self._array[ [ i for i, name in enumerate(self._all_names) if name in names ], :self._length],)
        
        yield ax.switch_to(self._main_thread)
            
        self._mx_data.set(data)
        
        
    