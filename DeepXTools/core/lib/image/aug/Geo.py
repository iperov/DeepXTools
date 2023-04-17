from __future__ import annotations

import cv2
import numba as nb
import numpy as np

from ...math import Affine2DMat
from ..NPImage import NPImage


class TransformParams:
    def __init__(self,  tx : float = 0.0,
                        ty : float = 0.0,
                        scale : float = 0,
                        rot_deg : float = 0.0,
                        ):
        """"""
        self._tx = tx
        self._ty = ty
        self._scale = scale
        self._rot_deg = rot_deg

    @property
    def _affine_scale(self) -> float:
        return (1 / (1 - scale)) if (scale := self._scale) < 0 else 1 + scale


    def copy(self) -> TransformParams:
        return TransformParams( tx = self._tx,
                                ty = self._ty,
                                scale = self._scale,
                                rot_deg = self._rot_deg)

    @staticmethod
    def generate(   tx_var : float = 0.3,
                    ty_var : float = 0.3,
                    scale_var : float = 0.4,
                    rot_deg_var : float = 15.0,
                    rnd_state : np.random.RandomState|None = None,) -> TransformParams:
        if rnd_state is None:
            rnd_state = np.random.RandomState()

        return TransformParams( tx = rnd_state.uniform(-tx_var, tx_var),
                                ty = rnd_state.uniform(-ty_var, ty_var),
                                scale = rnd_state.uniform(-scale_var, scale_var),
                                rot_deg = rnd_state.uniform(-rot_deg_var, rot_deg_var) )

    def added(self, tx : float = 0.0,
                    ty : float = 0.0,
                    scale : float = 0,
                    rot_deg : float = 0.0,) -> TransformParams:
        return TransformParams( tx = self._tx + tx,
                                ty = self._ty + ty,
                                scale = self._scale + scale,
                                rot_deg = self._rot_deg + rot_deg)

    def scaled(self, value : float) -> TransformParams:
        return TransformParams( tx = self._tx*value,
                                ty = self._ty*value,
                                scale = self._scale*value,
                                rot_deg = self._rot_deg*value)

    def __add__(self, value : TransformParams) -> TransformParams:
        return  self.added( tx=value._tx,
                            ty=value._ty,
                            scale=value._scale,
                            rot_deg=value._rot_deg )


class Geo:
    """Max quality one-pass image augmentation using geometric transformations."""

    def __init__(self,  offset_transform_params : TransformParams,
                        transform_params : TransformParams,
                        deform_transform_params : TransformParams = None,
                ):
        self._rnd_state = np.random.RandomState()

        if deform_transform_params is None:
            deform_transform_params = TransformParams.generate( tx_var = 0.0,
                                                                ty_var = 0.0,
                                                                scale_var = 0.2,
                                                                rot_deg_var = 180.0,
                                                                rnd_state=self._rnd_state)


        self._offset_transform_params = offset_transform_params
        self._transform_params = transform_params
        self._deform_transform_params = deform_transform_params
        self._deform_grid_cell_count = self._rnd_state.randint(1,6)


    def transform(self, img : NPImage,
                        OW : int,
                        OH : int,
                        center_fit = True,
                        transform_intensity : float = 1.0,
                        deform_intensity : float = 1.0,
                        interp : NPImage.Interp = NPImage.Interp.LANCZOS4,
                        border : NPImage.Border = NPImage.Border.CONSTANT,
                        ) -> NPImage:
        """
        transform an image.

        Subsequent calls will output the same result for any img shape and out_res.

        """
        H,W,_ = img.shape

        offset_tr_params = self._offset_transform_params

        if center_fit:
            offset_tr_params = offset_tr_params.copy()
            scale = offset_tr_params._affine_scale * min(W,H)/ ( OW if W < H else OH )
            scale = 1 - (1 / scale) if scale < 1.0 else scale - 1
            offset_tr_params._scale = scale

        tr_params = self._transform_params.scaled(transform_intensity)

        rnd_state = np.random.RandomState()
        rnd_state.set_state(self._rnd_state.get_state())

        remap_grid, mask = _gen_remap_grid(W, H, OW, OH,
                                            tr_params=offset_tr_params + tr_params,
                                            deform_tr_params=self._deform_transform_params,
                                            deform_cell_count=self._deform_grid_cell_count,
                                            deform_intensity=deform_intensity,
                                            border=border,
                                            rnd_state=rnd_state,
                                            )

        img = img.remap(remap_grid, interp=interp, border=border).HWC()

        if border == NPImage.Border.CONSTANT:
            img *= mask

        return NPImage(img)

def _gen_remap_grid(SW, SH, TW, TH,
                    tr_params : TransformParams,
                    deform_tr_params : TransformParams,
                    deform_cell_count : int,
                    deform_intensity : float,
                    border,
                    rnd_state,
                    ):
    """Generate remap grid and mask"""

    # Make identity remap grid in source space
    s_remap_grid = np.stack(np.meshgrid(np.linspace(0., SW-1, SW, dtype=np.float32),
                                        np.linspace(0., SH-1, SH, dtype=np.float32), ), -1)

    # Make mat for transform target space to source space
    t2s_mat = (Affine2DMat().translated( SW*(0.5+tr_params._tx), SH*(0.5+tr_params._ty) )
                            .scaled(tr_params._affine_scale)
                            .rotated(tr_params._rot_deg)
                            .translated(-0.5*TW, -0.5*TH) )

    if deform_intensity != 0.0:
        # Apply random deformations of target space in s_remap_grid

        # Make transform mat for deform_grid
        t2s_deform_mat = (Affine2DMat() .translated( TW*(0.5+deform_tr_params._tx), TH*(0.5+deform_tr_params._ty) )
                                        .scaled(deform_tr_params._affine_scale)
                                        .rotated(deform_tr_params._rot_deg)
                                        .translated(-0.5*TW,-0.5*TH) )

        # Warp border-reflected s_remap_grid to target space with deform_mat
        t_remap_grid = cv2.warpAffine(s_remap_grid, t2s_mat*t2s_deform_mat, (TW,TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Make random coord shifts in target space
        t_deform_grid = _gen_rw_coord_diff_grid(TW, TH, deform_cell_count, rnd_state)
        # Scale with intensity
        t_deform_grid *= deform_intensity
        # Merge with identity
        t_deform_grid += np.stack(np.meshgrid(np.linspace(0., TW-1, TW, dtype=np.float32),
                                              np.linspace(0., TH-1, TH, dtype=np.float32), ), -1)

        # Remap t_remap_grid with t_deform_grid and get diffs
        t_deform_grid = t_remap_grid - cv2.remap(t_remap_grid, t_deform_grid, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Fade to zero at borders
        w_border_size = TW // deform_cell_count
        w_dumper = np.linspace(0, 1, w_border_size, dtype=np.float32)
        t_deform_grid[:,:w_border_size ,:] *= w_dumper[None,:,None]
        t_deform_grid[:,-w_border_size:,:] *= w_dumper[None,::-1,None]

        h_border_size = TH // deform_cell_count
        h_dumper = np.linspace(0, 1, h_border_size, dtype=np.float32)
        t_deform_grid[:h_border_size, :,:] *= h_dumper[:,None,None]
        t_deform_grid[-h_border_size:,:,:] *= h_dumper[::-1,None,None]

        # Warp t_deform_grid to source space. BORDER_CONSTANT ensures zero coord diff outside.
        s_deform_grid = cv2.warpAffine(t_deform_grid, t2s_mat*t2s_deform_mat, (SW,SH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Merge s_deform_grid with s_remap_grid
        s_remap_grid += s_deform_grid


    # Warp s_remap_grid to target space
    t_remap_grid = cv2.warpAffine(s_remap_grid, t2s_mat, (TW,TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, 
                                  borderMode=NPImage._border_to_cv(NPImage.Border.REPLICATE if border == NPImage.Border.CONSTANT else border))

    # make binary mask to refine image-boundary
    mask = cv2.warpAffine( np.ones( (SH,SW), dtype=np.uint8), t2s_mat, (TW,TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)[...,None]

    return t_remap_grid, mask


def _gen_rw_coord_diff_grid(W, H, cell_count, rnd_state) -> np.ndarray:
    """
    generates square random deform coordinate differences

    grid is faded to 0 at borders

    grid of shape (H, W, 2)  (x,y)

    cell_count        1+
    """
    cell_count = max(1, cell_count)

    cell_var = ( 1.0 / (cell_count+1) ) * 0.4

    grid = np.zeros( (cell_count+3,cell_count+3, 2), dtype=np.float32 )

    grid[1:1+cell_count,1:1+cell_count, 0:2] = rnd_state.uniform (low=-cell_var, high=cell_var, size=(cell_count, cell_count, 2) )
    grid = grid_interp(W, H, grid)
    grid *= (W, H)

    return grid


@nb.njit(nogil=True)
def grid_interp(W, H, grid : np.ndarray):
    """Special bilinear interpolation which respects the borders."""
    GH, GW, N = grid.shape

    xs = float(W-1) / (GW-2)
    ys = float(H-1) / (GH-2)

    out = np.zeros( (H,W,N), np.float32)

    for h in range(H):
        gy  = float(h) / ys
        gyi = int(gy)
        gym = gy-gyi

        for w in range(W):
            gx  = float(w) / xs
            gxi = int(gx)
            gxm = gx-gxi

            v00 = grid[gyi,   gxi]
            v01 = grid[gyi,   gxi+1]
            v10 = grid[gyi+1, gxi]
            v11 = grid[gyi+1, gxi+1]

            a00 = (1.0-gym) * (1.0-gxm)
            a01 = (1.0-gym) * gxm
            a10 = gym       * (1.0-gxm)
            a11 = gym       * gxm

            out[h,w] = v00*a00 + v01*a01 + v10*a10 + v11*a11

    return out

