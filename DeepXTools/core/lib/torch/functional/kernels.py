import functools

import numpy as np


@functools.cache
def get_gaussian_kernel(ch, kernel_size : int, sigma : float, dtype=np.float32) -> np.ndarray:
    """returns (ksize,ksize) np gauss kernel"""
    x = np.arange(0, kernel_size, dtype=np.float64)
    x -= (kernel_size - 1 ) / 2.0
    x = x**2
    x *= ( -0.5 / (sigma**2) )
    x = np.reshape (x, (1,-1)) + np.reshape(x, (-1,1) )
    kernel_exp = np.exp(x)
    x = kernel_exp / kernel_exp.sum()
    return np.tile (x[None,None,...], (ch,1,1,1)).astype(dtype)




@functools.cache
def get_sobel_kernel(ch, kernel_size=9, dtype=np.float32) -> np.ndarray:
    """returns (ksize,ksize) np gauss kernel"""
    
    range = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    #sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, kernel_size // 2] = 1  # avoid division by zero
    kx = x / sobel_2D_denominator
    ky = y / sobel_2D_denominator
    
    k = np.concatenate([kx[None,...],ky[None,...]], 0)[:,None,:,:]
    k = np.tile(k, (ch,1,1,1))
    
    return k.astype(np.float32)
    
    return k[None,None,:,:].astype(np.float32)

    import code
    code.interact(local=dict(globals(), **locals()))

    
    kx = np.array([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]], dtype=dtype)
    ky = np.array([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]], dtype=dtype)
    
    k = np.concatenate([kx[None,...],ky[None,...]], 0)[:,None,:,:]
    return k