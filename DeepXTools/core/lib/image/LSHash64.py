from __future__ import annotations

from typing import Sequence

import numpy as np

from ..math import bit_count


class LSHash64(np.uint64):
    """Local-Sensitive 64-bit image hash"""

    @staticmethod
    def sorted_by_dissim(hashes : Sequence[LSHash64]) -> Sequence[int]:
        """
        returns Sequence of idx of hashes with most dissimilarities in descending order.
        """
        hashes = np.array(hashes)

        x = []
        for i, hash in enumerate(hashes):

            dissim_count = bit_count(hashes ^ hash).sum()

            x.append( (i, dissim_count) )

        x = sorted(x, key=lambda v: v[1], reverse=True)
        x = [v[0] for v in x]
        
        return x