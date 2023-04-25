import functools

import numpy as np
import torch
import torch.nn.functional as F

from .gaussian import gaussian_blur


@functools.cache
def get_sobel_kernel(ch, kernel_size=9, dtype=np.float32) -> np.ndarray:
    """returns (ksize,ksize) np sobel kernel"""
    
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
    
    
def sobel_edges_2d(x : torch.Tensor, kernel_size=5):
    """
    some hand-crafted func for sobel edges. 
    
    Returns [0..1] N,1,H,W
    """
    _, C, H, W = x.shape
        
    kernel_np = get_sobel_kernel(C, kernel_size=kernel_size)
    kernel_t = torch.tensor(kernel_np, device=x.device)
    
    x_sobel = F.conv2d(F.pad(x, (kernel_size//2,)*4, mode='reflect'), kernel_t, stride=1, padding=0, groups=C)
    x_sobel = x_sobel.pow(2).sum(1, keepdim=True).sqrt()
    x_sobel = gaussian_blur(x_sobel, sigma=max(H,W)/64.0)
    x_sobel /= x_sobel.max()
    return x_sobel