import cv2
import numpy as np
import numpy.random as nprnd

from .. import ps as lib_ps
from ..NPImage import NPImage


def circle_faded_mask(W, H=None, cx_range=[-1.0, 2.0], cy_range=[-1.0, 2.0], f_range=[0.0, 1.5] ) -> NPImage:
    if H is None:
        H = W
    HW_max = max(H,W)
    
    return lib_ps.circle_faded(W, H,    np.random.uniform(cx_range[0]*W, cx_range[1]*W), 
                                        np.random.uniform(cy_range[0]*H, cy_range[1]*H), 
                                        fs := np.random.uniform(f_range[0]*HW_max, f_range[1]*HW_max), 
                                        fe = np.random.uniform(fs, f_range[1]*HW_max) )

def noise_clouds_mask(W, H=None) -> NPImage:
    if H is None:
        H = W
    
    img = lib_ps.noise(1,1).resize(W, H, interp=NPImage.Interp.LANCZOS4)
        
    d = 2
    while True:
        dW = W // d
        dH = W // d
        if dW <= 1 or dH <= 1:
            break
        d *= 2
        if np.random.randint(2) == 0:
            continue
        
        x = lib_ps.noise(W // d, H // d).resize(W, H, interp=NPImage.Interp.CUBIC  if np.random.randint(2) == 0 else
                                                            NPImage.Interp.LANCZOS4)
        if np.random.randint(2) == 0:
            img *= x
        else:
            img += x
    img = img.apply(lambda x: x / x.max())
    return img

def levels(img : NPImage, in_b_range=[0.0, 0.25], in_w_range=[0.75, 1.0], in_g_range=[0.5, 1.5],
                          out_b_range=[0.0, 0.25], out_w_range=[0.75, 1.0]):
    C = img.shape[-1]
    
    return img.levels(  in_b = np.float32([ nprnd.uniform(in_b_range[0], in_b_range[1]) for _ in range(C) ]),
                        in_w = np.float32([ nprnd.uniform(in_w_range[0], in_w_range[1]) for _ in range(C) ]),
                        in_g = np.float32([ nprnd.uniform(in_g_range[0], in_g_range[1]) for _ in range(C) ]),
                        
                        out_b = np.float32([ nprnd.uniform(out_b_range[0], out_b_range[1]) for _ in range(C) ]),
                        out_w = np.float32([ nprnd.uniform(out_w_range[0], out_w_range[1]) for _ in range(C) ]))


def hsv_shift(img : NPImage, h_offset_range=[0.0,1.0], s_offset_range=[-0.5,0.5], v_offset_range=[-0.5,0.5]) -> NPImage:
    return img.hsv_shift(nprnd.uniform(*h_offset_range),nprnd.uniform(*s_offset_range),nprnd.uniform(*v_offset_range))


def box_sharpen(img : NPImage, kernel_range=None, power_range=[0.0, 2.0]) -> NPImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 64) ]
    return img.box_sharpen(kernel_size=nprnd.randint(kernel_range[0], kernel_range[1]+1), power=nprnd.uniform(*power_range))

def gaussian_sharpen(img : NPImage, sigma_range=None, power_range=[0.0, 2.0]) -> NPImage:
    if sigma_range is None:
        sigma_range = [0, max(img.shape[0:2]) / 64.0 ]
    
    return img.gaussian_sharpen(sigma=nprnd.uniform(*sigma_range), power=nprnd.uniform(*power_range))

def gaussian_blur(img : NPImage, sigma_range=None) -> NPImage:
    if sigma_range is None:
        sigma_range = [0, max(img.shape[0:2]) / 64.0 ]
        
    return img.gaussian_blur(sigma=nprnd.uniform(*sigma_range))

def median_blur(img : NPImage, kernel_range=None) -> NPImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 64) ]
    return img.median_blur(kernel_size=nprnd.randint(kernel_range[0], kernel_range[1]+1))

def motion_blur(img : NPImage, kernel_range=None, angle_range=[0,360]) -> NPImage:
    if kernel_range is None:
        kernel_range = [1, max(1, max(img.shape[0:2]) // 16) ]
    return img.motion_blur(kernel_size=nprnd.randint(kernel_range[0], kernel_range[1]+1), angle=nprnd.randint(*angle_range))

def glow_shade(img : NPImage, mask : NPImage, inner=True, glow=False) -> NPImage:
    """"""
    H,W,_ = img.shape
    HW_max = max(H,W)
    
    img = img.f32()
    mask = mask.f32()
    
    if inner:
        halo = img*mask
    else:
        halo = img*(1-mask)    
    halo = halo.gaussian_blur(sigma=np.random.uniform(HW_max/16, HW_max/4))
    
    if glow:
        img = img + halo
    else:
        img = img - halo
    img.clip(inplace=True)
        
    return img

def resize(img : NPImage, size_range=[0.25, 1.0], interp=NPImage.Interp.LINEAR) -> NPImage:
    H,W,C = img.shape
    
    s = nprnd.uniform(*size_range)
    
    img = img.resize (int(s*W), int(s*H), interp=interp )
    img = img.resize (W, H, interp=interp )

    return img


def jpeg_artifacts(img : NPImage, quality_range = [10,100] ) -> NPImage:
    """
     quality    0-100
    """
    quality = nprnd.randint(quality_range[0],quality_range[1]+1)
    
    dtype = img.dtype
    img = img.u8().HWC()

    ret, result = cv2.imencode('.jpg', img, params=[cv2.IMWRITE_JPEG_QUALITY, quality] )
    if not ret:
        raise Exception('unable to compress jpeg')
    img = cv2.imdecode(result, flags=cv2.IMREAD_UNCHANGED)
        
    return NPImage(img).to_dtype(dtype)