from typing import Sequence

import numpy as np


class ProbIndexer1D:
    """Probabilistic 1D index generator."""
    def __init__(self, size : int):
        self._size = size
        self._rnd_state = np.random.RandomState()

        self.reset_probs()

    @property
    def size(self) -> int:
        return self._size

    def reset_probs(self):
        self._probs = [1]*self._size
        self._probs_counters = [0]*self._size

    def set_probs(self, probs : Sequence[int]):
        """
        update probs of idxs

            probs     sequence of int >= 1
        """
        if len(probs) != self._size:
            raise ValueError(f'probs must have len of {self._size}')
        self._probs = probs

    def generate(self, count : int) -> Sequence[int]:
        """Generate indexes of `count`"""
        out = []
        if self._size != 0:
            while len(out) < count:
                idx = self._rnd_state.randint(self._size)

                prob = self._probs_counters[idx] = self._probs_counters[idx] + 1
                if prob % self._probs[idx] == 0:
                    out.append(idx)
        return out