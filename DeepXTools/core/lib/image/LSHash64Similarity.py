import numpy as np

from ..math import bit_count
from .LSHash64 import LSHash64


class LSHash64Similarity:
    def __init__(self, count : int, similarity_factor : int = 8):
        """
        a class for continuous computation every-with-every similarity of LSHash64'es 
            
            count       number of indexes to be computed
            
            similarity_factor(8)  0..63
        """
        if not (count > 0):
            raise ValueError(f'count must be > 0')
        if not (similarity_factor >= 0 and similarity_factor <= 63):
            raise ValueError(f'similarity_factor must be in range [0..63]')
            
        self._count = count
        self._similarity_factor = similarity_factor

        self._hashed_map = np.zeros( (count,), dtype=np.uint8 )
        
        self._hashed_count = 0
        self._hashed_idxs  = np.zeros( (count,), dtype=np.uint32 )
        self._hashes       = np.zeros( (count,), dtype=np.uint64 )

        self._similarities = np.ones( (count,), dtype=np.uint32 )


    def add(self, idx : int, hash : LSHash64):
        """
        add hash with idx, and update similarities with already added hashes
        
        if idx already hashed, nothing will happen
        """
        if self._hashed_map[idx] == 0:
            self._hashed_map[idx] = 1
            hashed_count = self._hashed_count

            hashes_diffs = bit_count(self._hashes[:hashed_count] ^ hash)
            idxs = self._hashed_idxs[ np.argwhere( hashes_diffs <= self._similarity_factor ) ]

            self._similarities[idxs] += 1
            self._similarities[idx] += len(idxs)

            self._hashed_idxs[hashed_count] = idx
            self._hashes[hashed_count] = hash
            self._hashed_count += 1

    def hashed(self, idx : int) -> bool: 
        return self._hashed_map[idx] != 0
    
    def get_similarities(self) -> np.ndarray:
        return self._similarities
