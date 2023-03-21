import math

import numba as nb
import numpy as np

from ..NPImage import NPImage


def noise(W : int, H : int) -> NPImage:
    return NPImage(_noise(W, H, seed=np.random.uniform()))


def circle_faded(W : int, H : int, cx : float, cy : float, fs : float, fe : float) -> NPImage:
    """H,W,1"""
    return NPImage(_circle_faded(W, H, cx, cy, fs, fe))


@nb.njit(nogil=True)
def _circle_faded(W : int, H : int, cx : float, cy : float, fs : float, fe : float):
    img = np.empty( (H,W,1), np.float32 )

    d = max(fs, fe) - fs
    if d == 0:
        d = 1

    for h in range(H):
        for w in range(W):
            dist = math.sqrt((cx-w)**2 + (cy-h)**2)
            img[h,w,0] = min(1.0, max(0, 1.0 - (dist - fs) / d ))

    return img

@nb.njit(nogil=True)
def _noise(W : int, H : int, seed ):
    img = np.empty( (H,W,1), np.float32 )

    for h in range(H):
        for w in range(W):
            img[h, w] = _hash(w+seed, h+seed)
    return img

@nb.njit
def _hash3(x, y):
    return ((math.sin(x*127.1 + y*311.7)*43758.5453) % 1,
            (math.sin(x*269.5 + y*183.3)*43758.5453) % 1,
            (math.sin(x*419.2 + y*371.9)*43758.5453) % 1)
@nb.njit
def _hash(x, y):
    return (math.sin(x*127.1 + y*311.7)*43758.5453) % 1

@nb.njit
def _smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)
