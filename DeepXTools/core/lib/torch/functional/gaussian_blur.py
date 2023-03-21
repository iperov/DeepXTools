import numpy as np
import torch
import torch.nn.functional as F

from .gaussian import get_gaussian_kernel


def gaussian_kernel(kernel_size : int, sigma : float, dtype=np.float32):
    """returns (ksize,ksize) np gauss kernel"""
    x = np.arange(0, kernel_size, dtype=dtype)
    x -= (kernel_size - 1 ) / 2.0
    x = x**2
    x *= ( -0.5 / (sigma**2) )
    x = np.reshape (x, (1,-1)) + np.reshape(x, (-1,1) )        
    kernel_exp = np.exp(x)
    x = kernel_exp / kernel_exp.sum()
    return x
    
def gaussian_blur(img_t : torch.Tensor, sigma : float = 2.0):
    _,C,_,_ = img_t.shape
    kernel_size = max(3, int(2 * 2 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_t = torch.tensor(get_gaussian_kernel(C, kernel_size, sigma), device=img_t.device)
    
    out_t = F.conv2d(img_t, kernel_t, stride=1, padding=kernel_size // 2, groups=C)
    return out_t
    
