import functools

import numpy as np


@functools.cache
def get_gaussian_kernel(ch, kernel_size : int, sigma : float, dtype=np.float32) -> np.ndarray:
    """returns (ksize,ksize) np gauss kernel"""
    x = np.arange(0, kernel_size, dtype=dtype)
    x -= (kernel_size - 1 ) / 2.0
    x = x**2
    x *= ( -0.5 / (sigma**2) )
    x = np.reshape (x, (1,-1)) + np.reshape(x, (-1,1) )
    kernel_exp = np.exp(x)
    x = kernel_exp / kernel_exp.sum()
    return np.tile (x[None,None,...], (ch,1,1,1))
