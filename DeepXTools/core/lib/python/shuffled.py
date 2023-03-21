from typing import Sequence

import numpy


def shuffled(seq : Sequence):

    idxs = [*range(len(seq))]
    numpy.random.shuffle(idxs)

    for idx in idxs:
        yield seq[idx]
