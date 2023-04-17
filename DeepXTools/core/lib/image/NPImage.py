"""
Checklist:

@nb.njit(nogil=True)
"""
from __future__ import annotations

import functools
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cv2
import numba as nb
import numpy as np
from .LSHash64 import LSHash64

class NPImage:
    """
    Wrapper of numpy ndarray HWC images (u8/f32).
    Provides image processing methods.

    Saves lines of code, because you don't need to check image channels, dtype or format.
    You just get NPImage as arg, and then transform it to your needs.

    Example
    ```
    npi.grayscale().u8().CHW()
    ```
    """
    __slots__ = ['_img']

    class Interp(Enum):
        NEAREST = auto()
        LINEAR = auto()
        CUBIC = auto()
        LANCZOS4 = auto()

    class Border(Enum):
        CONSTANT = auto()
        REFLECT = auto()
        REPLICATE = auto()

    @staticmethod
    def _border_to_cv(border : Border):
        return _cv_border[border]


    @staticmethod
    def avail_image_suffixes() -> List[str]:
        return ('.jpeg','.jpg','.jpe','.jp2','.png','.webp','.tiff','.tif')

    @staticmethod
    def from_file(path : Path) -> NPImage:
        with open(path, "rb") as stream:
            b = bytearray(stream.read())
        numpyarray = np.asarray(b, dtype=np.uint8)

        return NPImage(cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED))

    def __init__(self, img : np.ndarray, channels_last = True):
        """```
            img     np.ndarray

            dtype must be uint8 or float32 in range [0..1]

            acceptable format:
            HW[C] channels_last(True)
            [C]HW channels_last(False)

            C must be
                0: assume 1 ch
                1: assume grayscale
                3: assume BGR   (opencv & qt compliant)
                4: assume BGRA  (opencv & qt compliant)
        ```"""
        if channels_last:
            if (shape_len := len(img.shape)) != 3:
                if shape_len == 2:
                    img = img[:,:,None]
                else:
                    raise ValueError(f'Wrong shape len {shape_len}')
        else:
            if (shape_len := len(img.shape)) != 3:
                if shape_len == 2:
                    img = img[None,:,:]
                else:
                    raise ValueError(f'Wrong shape len {shape_len}')
            img = img.transpose(1,2,0)

        if (C := img.shape[-1]) not in [1,3,4]:
            raise ValueError(f'img.C must be 0,1,3,4 but {C} provided.')

        if np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32, copy=False)
        elif img.dtype != np.uint8:
            raise ValueError(f'img.dtype must be np.uint8 or np.floating, but passed {img.dtype}.')

        self._img : np.ndarray = img

    @property
    def shape(self) -> Tuple[int, int, int]:
        """get (H,W,C) dims"""
        return self._img.shape

    @property
    def dtype(self) -> np.dtype:
        return self._img.dtype

    def copy(self) -> NPImage:
        return NPImage(img=self._img.copy())

    def HWC(self) -> np.ndarray:
        return self._img

    def CHW(self) -> np.ndarray:
        return self._img.transpose(2,0,1)

    def HW(self) -> np.ndarray:
        return self.HWC()[:,:,0]

    def grayscale(self) -> NPImage:
        """"""
        C = (img := self._img).shape[-1]
        if C == 1:
            return self
        return NPImage(np.dot(img[...,0:3], np.array([0.1140, 0.5870, 0.299], np.float32))[...,None].astype(img.dtype, copy=False))

    def bgr(self) -> NPImage:
        """"""
        C = (img := self._img).shape[-1]
        if C == 3:
            return self
        if C == 1:
            return NPImage(np.repeat(img, 3, -1))
        if C == 4:
            return NPImage(img[...,:3])
        return NPImage(img)

    def bgra(self) -> NPImage:
        """"""
        C = (img := self._img).shape[-1]
        if C == 4:
            return self
        if C == 1:
            img = np.repeat(img, 3, -1)
            C = 3
        if C == 3:
            img = np.pad(img, ( (0,0), (0,0), (0,1) ), mode='constant', constant_values=255 if img.dtype == np.uint8 else 1.0 )
        return NPImage(img)

    def to_dtype(self, dtype) -> NPImage:
        """allowed dtypes: np.uint8, np.float32"""
        if dtype == np.uint8:     return self.u8()
        elif dtype == np.float32: return self.f32()
        else: raise ValueError('unsupported dtype')

    def f32(self, **kwargs) -> NPImage:
        """convert to uniform float32. if current image dtype uint8, then image will be divided by / 255.0"""
        dtype = (img := self._img).dtype
        if dtype == np.uint8:
            return NPImage(np.divide(img, 255.0, dtype=np.float32))
        return NPImage(img.astype(np.float32, copy=kwargs.get('copy', False)))

    def u8(self, **kwargs) -> NPImage:
        """
        convert to uint8

        if current image dtype is f32, then image will be multiplied by *255
        """
        img = self._img
        if img.dtype == np.float32:
            img = img * 255.0
            img = np.clip(img, 0, 255, out=img).astype(np.uint8)
            return NPImage(img)
        return NPImage(img.astype(np.uint8, copy=kwargs.get('copy', False)))

    def apply(self, func : Callable[ [np.ndarray], np.ndarray]) -> NPImage:
        """
        apply your own function on image.

        image has NHWC format. Do not change format, keep dtype either u8 or float, dims can be changed.

        ```
        example:
        .apply( lambda img: img-[102,127,63] )
        ```
        """
        img = func(self._img)
        if img.dtype not in [np.uint8, np.float32]:
            raise Exception('dtype result of apply() must be u8/f32')
        return NPImage(img)

    def clip(self, min=None, max=None, inplace=False) -> NPImage:
        """clip to min,max.

        if min/max not specified and dtype is f32, clips to 0..1
        """
        img = self._img
        if min is None and max is None:
            if img.dtype == np.float32:
                min = 0.0
                max = 1.0
        if min is not None and max is not None:
            return NPImage(np.clip(img, min, max, out=img if inplace else None))
        return self


    # def erode_blur(self, erode : int, blur : int, fade_to_border : bool = False) -> NPImageProcessor:
    #     """
    #     apply erode and blur to the mask image

    #      erode  int     != 0
    #      blur   int     > 0
    #      fade_to_border(False)  clip the image in order
    #                             to fade smoothly to the border with specified blur amount


    #     """
    #     erode, blur = int(erode), int(blur)

    #     N,H,W,C = (img := self.f32()._img).shape
    #     img = img.transpose( (1,2,0,3) ).reshape( (H,W,N*C) )
    #     img = np.pad (img, ( (H,H), (W,W), (0,0) ) )

    #     if erode > 0:
    #         el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    #         iterations = max(1,erode//2)
    #         img = cv2.erode(img, el, iterations = iterations )

    #     elif erode < 0:
    #         el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    #         iterations = max(1,-erode//2)
    #         img = cv2.dilate(img, el, iterations = iterations )

    #     if fade_to_border:
    #         h_clip_size = H + blur // 2
    #         w_clip_size = W + blur // 2
    #         img[:h_clip_size,:] = 0
    #         img[-h_clip_size:,:] = 0
    #         img[:,:w_clip_size] = 0
    #         img[:,-w_clip_size:] = 0

    #     if blur > 0:
    #         sigma = blur * 0.125 * 2
    #         img = cv2.GaussianBlur(img, (0, 0), sigma)

    #     img = img[H:-H,W:-W]
    #     img = img.reshape( (H,W,N,C) ).transpose( (2,0,1,3) )

    #     self._npi = img
    #     return self

    def gaussian_blur(self, sigma : float) -> NPImage:
        """
        Spatial gaussian blur.

            sigma  float
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self._img).shape
        return NPImage(cv2.GaussianBlur(img, (0,0), sigma).reshape(H,W,C))

    def blend(self, other : NPImage, mask : NPImage, alpha = 1.0) -> NPImage:
        """
        Pixel-wise blending `self*(1-mask*alpha) + other*mask*alpha`

        Image will be forced to f32.

            alpha  [0.0 ... 1.0]
        """
        if self.dtype == np.uint8:
            self.f32()._img
        else:
            self._img.copy()

        img = self.f32(copy=True)._img
        other = other.f32()._img
        mask = mask.f32()._img

        return NPImage( _nb_blend_f32(img, other, mask, alpha) )

    def box_sharpen(self, kernel_size : int, power : float) -> NPImage:
        """
         kernel_size   int     kernel size

         power  float   0 .. 1.0 (or higher)

        Image will be forced to f32.
        """
        power = max(0, power)
        if power == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.filter2D(img, -1, _box_sharpen_kernel(kernel_size, power))
        img = np.clip(img, 0, 1, out=img)
        img = img.reshape(H,W,C)

        return NPImage(img)

    def gaussian_sharpen(self, sigma : float, power : float) -> NPImage:
        """
         sigma  float

         power  float   0 .. 1.0 and higher

        Image will be forced to f32.
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.addWeighted(img, 1.0 + power,
                              cv2.GaussianBlur(img, (0, 0), sigma), -power, 0)
        img = np.clip(img, 0, 1, out=img)
        img = img.reshape(H,W,C)

        return NPImage(img)

    def gaussian_blur(self, sigma : float) -> NPImage:
        """
         sigma  float

        Image will be forced to f32.
        """
        sigma = max(0, sigma)
        if sigma == 0:
            return self

        H,W,C = (img := self.f32()._img).shape

        img = cv2.GaussianBlur(img, (0,0), sigma)
        img = img.reshape(H,W,C)

        return NPImage(img)

    def h_flip(self) -> NPImage:
        return NPImage(self._img[:,::-1,:])
    def v_flip(self) -> NPImage:
        return NPImage(self._img[::-1,:,:])

    def levels(self, in_b, in_w, in_g, out_b, out_w) -> NPImage:
        """```
            in_b
            in_w
            in_g
            out_b
            out_w       (C,) float32

        Image will be forced to f32.
        ```"""
        H,W,C = (img := self._img).shape

        if C != in_b.shape[0] != in_w.shape[0] != in_g.shape[0] != out_b.shape[0] != out_w.shape[0]:
            raise Exception('in_b, in_w, in_g, out_b, out_w dim must match C dims')

        return NPImage( _nb_levels_f32(self.f32(copy=True)._img, in_b, in_w, in_g, out_b, out_w) )

    def bilateral_filter(self, sigma : float) -> NPImage:
        H,W,C = (img := self._img).shape
        return NPImage(cv2.bilateralFilter(img, 0, sigma, sigma).reshape(H,W,C))

    # def median_blur(self, kernel_size : int) -> NPImage:
    #     """
    #      kernel_size   int     median kernel size

    #     Image will be forced to f32.
    #     """
    #     if kernel_size % 2 == 0:
    #         kernel_size += 1
    #     kernel_size = max(1, kernel_size)

    #     H,W,C = (img := self.f32()._img).shape

    #     img = cv2.medianBlur(img, kernel_size)
    #     img = img.reshape(H,W,C)
    #     return NPImage(img)

    def motion_blur( self, kernel_size : int, angle : int):
        """
            kernel_size    >= 1

            angle   degrees

        Image will be forced to f32.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        H,W,C = (img := self.f32()._img).shape
        return NPImage(cv2.filter2D(img, -1, _motion_blur_kernel(kernel_size, angle)).reshape(H,W,C))

    def hsv_shift(self, h_offset : float, s_offset : float, v_offset : float) -> NPImage:
        """```
            H,S,V in [-1.0..1.0]
        ```"""
        H,W,C = (img := self._img).shape
        if C != 3:
            raise Exception('C must be == 3')

        return NPImage( _nb_hsv_shift(img.copy(), h_offset, s_offset, v_offset, u8=img.dtype==np.uint8) )

    def pad(self, pads : Sequence[int]) -> NPImage:
        """```
        pads per HWC axes

        ( (PADT,PADB), (PADL, PADR), (PADC_before, PADC_after) )
        ```"""
        return NPImage(np.pad(self._img, pads))

    def resize(self, OW : int, OH : int, interp : Interp = Interp.LINEAR) -> NPImage:
        """resize to (OW,OH)"""
        H,W,C = (img := self._img).shape
        if OW != W or OH != H:
            img = cv2.resize (img, (OW, OH), interpolation=_cv_inter[interp])
            img = img.reshape(OH,OW,C)
            return NPImage(img)
        return self

    def remap(self, grid : np.ndarray, interp : Interp = Interp.LINEAR, border : Border = Border.CONSTANT) -> NPImage:
        """```
            grid    HW2
        ```"""
        OH,OW,_ = grid.shape

        H,W,C = (img := self._img).shape
        return NPImage(cv2.remap(img, grid, None, interpolation=_cv_inter[interp], borderMode=_cv_border[border] ).reshape(OH,OW,C))

    def warp_affine(self, mat, OW, OH, interp : Interp = Interp.LINEAR ) -> NPImage:
        """
        """
        H,W,C = (img := self._img).shape

        img = cv2.warpAffine(img, mat, (OW, OH), flags=_cv_inter[interp] )
        img = img.reshape(OH,OW,C)
        return NPImage(img)

    def save(self, path : Path, quality : int = 100):
        """raise on error"""

        suffix = path.suffix
        if suffix in ['.png', '.webp', '.jpeg','.jpg','.jpe','.jp2']:
            img = self.u8().HWC()
        elif suffix in ['.tiff', '.tif']:
            img = self.f32().HWC()
        else:
            raise ValueError(f'Unsupported format {suffix}')

        if format == 'webp':
            imencode_args = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        elif format == 'jpg':
            imencode_args = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif format == 'jp2':
            imencode_args = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality*10]
        else:
            imencode_args = []

        ret, buf = cv2.imencode(suffix, img, imencode_args)
        if not ret:
            raise Exception(f'Unable to encode image to {suffix}')

        with open(path, "wb") as stream:
            stream.write( buf )


    def get_ls_hash64(self) -> LSHash64:
        """
        Calculates perceptual local-sensitive 64-bit hash of image

        based on http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

        returns LSHash64
        """
        hash_size, highfreq_factor = 8, 4

        img = self.grayscale().resize(hash_size * highfreq_factor, hash_size * highfreq_factor).f32().HW()

        dct = cv2.dct(img)

        dct_low_freq = dct[:hash_size, :hash_size]
        bits = ( dct_low_freq > np.median(dct_low_freq) ).reshape( (hash_size*hash_size,)).astype(np.uint64)
        bits = bits << np.arange(len(bits), dtype=np.uint64)

        return LSHash64(bits.sum())


    def __add__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(self._img + value)

    def __radd__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(value + self._img)

    def __iadd__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        self._img += value
        return self

    def __sub__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(self._img - value)

    def __rsub__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(value - self._img)

    def __isub__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        self._img -= value
        return self

    def __mul__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(self._img * value)

    def __rmul__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(value * self._img)

    def __imul__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        self._img *= value
        return self

    def __truediv__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(self._img / value)

    def __rtruediv__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        return NPImage(value / self._img)

    def __itruediv__(self, value) -> NPImage:
        if isinstance(value, NPImage):
            value = value._img
        self._img /= value
        return self

    def __repr__(self): return self._img.__repr__()
    def __str__(self): return self._img.__str__()



_cv_inter = { NPImage.Interp.NEAREST : cv2.INTER_NEAREST,
              NPImage.Interp.LINEAR : cv2.INTER_LINEAR,
              NPImage.Interp.CUBIC : cv2.INTER_CUBIC,
              NPImage.Interp.LANCZOS4 : cv2.INTER_LANCZOS4,
               }

_cv_border = {NPImage.Border.CONSTANT : cv2.BORDER_CONSTANT,
              NPImage.Border.REFLECT : cv2.BORDER_REFLECT,
              NPImage.Border.REPLICATE : cv2.BORDER_REPLICATE,
            }

@functools.cache
def _box_sharpen_kernel(kernel_size, power) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
    k[ kernel_size//2, kernel_size//2] = 1.0
    b = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    k = k + (k - b) * power
    return k


@functools.cache
def _motion_blur_kernel(kernel_size, angle) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[ (kernel_size-1)// 2 , :] = np.ones(kernel_size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (kernel_size / 2 -0.5 , kernel_size / 2 -0.5 ) , angle, 1.0), (kernel_size, kernel_size) )
    k = k * ( 1.0 / np.sum(k) )
    return k

@nb.njit(nogil=True)
def _nb_levels_f32(in_out, in_b, in_w, in_g, out_b, out_w):
    H,W,C = in_out.shape

    for h in range(H):
        for w in range(W):
            v = in_out[h,w]
            v = np.clip( (v - in_b) / (in_w - in_b), 0, 1 )
            v = ( v ** (1/in_g) ) *  (out_w - out_b) + out_b
            np.clip(v, 0.0, 1.0, out=v)
            in_out[h,w] = v

    return in_out

@nb.njit(nogil=True)
def _nb_blend_f32(in_out, other, mask, alpha : float):
    H,W,C = in_out.shape
    bH,bW,bC = other.shape
    mH,mW,mC = mask.shape

    for h in range(H):
        for w in range(W):
            for c in range(C):
                a = in_out[h,w,c]
                b = other[h % bH, w % bW, c % bC]
                m = mask[h % mH, w % mW, c % mC]

                in_out[h,w,c] = max(0.0, min(1.0, a*(1-(m*alpha) ) + b*(m*alpha) ))

    return in_out


@nb.njit(nogil=True)
def _bgr_to_hsv(b, g, r):
    maxc = max(b, g, r)
    minc = min(b, g, r)
    v = maxc
    if minc == maxc:
        h=0.0
        s=0.0
    else:
        s = (maxc-minc) / maxc
        rc = (maxc-r) / (maxc-minc)
        gc = (maxc-g) / (maxc-minc)
        bc = (maxc-b) / (maxc-minc)
        if r == maxc:
            h = bc-gc
        elif g == maxc:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0) % 1.0
    return np.float32(h), np.float32(s), np.float32(v)

@nb.njit(nogil=True)
def _hsv_to_bgr(h, s, v):
    if s == 0.0:
        b=g=r=v
    else:
        i = int(h*6.0) # XXX assume int() truncates!
        f = (h*6.0) - i
        p = v*(1.0 - s)
        q = v*(1.0 - s*f)
        t = v*(1.0 - s*(1.0-f))
        i = i%6
        if i == 0:
            b=t; g=p; r=v
        if i == 1:
            b=v; g=p; r=q
        if i == 2:
            b=v; g=t; r=p
        if i == 3:
            b=q; g=v; r=p
        if i == 4:
            b=p; g=v; r=t
        if i == 5:
            b=p; g=q; r=v
    return np.float32(b), np.float32(g), np.float32(r)


@nb.njit(nogil=True)
def _nb_hsv_shift(in_out, h_offset, s_offset, v_offset, u8 = False):
    H,W,C = in_out.shape

    for h in range(H):
        for w in range(W):
            b,g,r = in_out[h,w]

            if u8:
                b = np.float32(b) / 255.0
                g = np.float32(g) / 255.0
                r = np.float32(r) / 255.0

            h_,s_,v_ = _bgr_to_hsv(b,g,r)

            h_ = (h_ + h_offset) % 1.0
            s_ = max(0, min(1, (s_ + s_offset)))
            v_ = max(0, min(1, (v_ + v_offset)))

            b,g,r = _hsv_to_bgr(h_,s_,v_)

            if u8:
                b = np.uint8(max(0, min(255, np.float32(b) * 255.0)))
                g = np.uint8(max(0, min(255, np.float32(g) * 255.0)))
                r = np.uint8(max(0, min(255, np.float32(r) * 255.0)))

            in_out[h,w] = b,g,r

    return in_out
