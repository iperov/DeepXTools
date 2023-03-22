from __future__ import annotations

import cv2
import numba as nb
import numpy as np
import numpy.random as nprnd

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
            deform_transform_params = TransformParams.generate( tx_var = 0.2,
                                                                ty_var = 0.2,
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
                                            rnd_state=rnd_state,
                                            )

        img = img.remap(remap_grid, interp=interp).HWC()
        img *= mask

        return NPImage(img)

def _gen_remap_grid(W, H, OW, OH,
                    tr_params : TransformParams,
                    deform_tr_params : TransformParams,
                    deform_cell_count : int,
                    deform_intensity : float,
                    rnd_state,
                    ):
    """Generate remap grid and mask"""

    # Make identity uniform remap grid in source space
    remap_grid = np.stack(np.meshgrid(np.linspace(0., W-1, W, dtype=np.float32),
                                      np.linspace(0., H-1, H, dtype=np.float32), ), -1)

    # Make transform mat
    mat = (Affine2DMat()    .translated( W*(0.5+tr_params._tx), H*(0.5+tr_params._ty) )
                            .scaled(tr_params._affine_scale)
                            .rotated(tr_params._rot_deg)
                            .translated(-0.5*OW, -0.5*OH) )
    mat_inv = mat.inverted()
    

    if deform_intensity != 0.0:
        # Apply random deform to remap grid

        # Make transform mat of deform grid
        deform_mat = (Affine2DMat() .translated( OW*(0.5+deform_tr_params._tx), OH*(0.5+deform_tr_params._ty) )
                                    .scaled(deform_tr_params._affine_scale)
                                    .rotated(deform_tr_params._rot_deg)
                                    .translated(-0.5*OW,-0.5*OH) )

        # Make random deform diff_grid
        diff_grid = _gen_rw_coord_uni_diff_grid(OW, OH, deform_cell_count, rnd_state)
        diff_grid *= tr_params._affine_scale * deform_intensity 
        diff_grid *= (OW-1,OH-1)
        
        # Transform diff_grid to source space using mat*deform_mat 
        diff_grid = cv2.warpAffine(diff_grid, mat*deform_mat, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Merge diff_grid with remap_grid
        remap_grid += diff_grid

    # Warp remap_grid to target space
    remap_grid = cv2.warpAffine(remap_grid, mat_inv, (OW,OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE )

    # make binary mask to refine image-boundary
    mask = cv2.warpAffine( np.ones( (H,W), dtype=np.uint8), mat_inv, (OW,OH), flags=cv2.INTER_NEAREST)[...,None]

    return remap_grid, mask

            

def _gen_rw_coord_uni_diff_grid(W, H, cell_count,  rnd_state) -> np.ndarray:
    """
    generates square uniform random deform coordinate differences

    grid of shape (H, W, 2)  (x,y)

    cell_count        1+
    """
    cell_count = max(1, cell_count)
    
    cell_var = ( 1.0 / (cell_count+1) ) * 0.4
    
    grid = np.zeros( (cell_count+3,cell_count+3, 2), dtype=np.float32 )

    grid[1:1+cell_count,1:1+cell_count, 0:2] = rnd_state.uniform (low=-cell_var, high=cell_var, size=(cell_count, cell_count, 2) )
    grid = grid_interp(W, H, grid)
    
   
    return grid


@nb.njit(nogil=True)
def grid_interp(W, H, grid : np.ndarray):
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
    
    
     
    
        # align_uni_mat = align_uni_mat.inverted() if align_uni_mat is not None else Affine2DUniMat()

        # align_uni_mat = self._align_uni_mat = (align_uni_mat * Affine2DUniMat()
        #                                                 .translated(-align_tx+0.5, -align_ty+0.5)
        #                                                 .scaled(align_scale)
        #                                                 .rotated(align_rot_deg)
        #                                                 #.translated(-0.5,-0.5)
        #                                                 )

        # self._rw_rnd_mat = align_uni_mat * Affine2DUniMat() .translated(0.5, 0.5) \
        #                                                     .rotated(rw_rot_deg) \
        #                                                     .translated(-rw_tx, -rw_ty) \
        #                                                     .scaled(rw_scale) \
        #                                                     .translated(-0.5,-0.5)

        # self._cached = {}

    # def _get_tr_rnd_mat(self, power : float = 1.0)-> Affine2DUniMat:
    #     return self._align_uni_mat
            # * Affine2DUniMat()   .translated(0.5, 0.5) \
            #                                             .rotated( self._tr_rot_deg*power ) \
            #                                             .translated(-self._tr_tx*power, -self._tr_ty*power) \
            #                                             .scaled( 1.0 + (self._tr_scale - 1.0) * power ) \
            #                                             .translated(-0.5,-0.5)


    # def get_aligned_random_transform_mat(self, aug_transform_power=1.0) -> Affine2DUniMat:
    #     """
    #     returns Affine2DUniMat that represents transformation from aligned image to randomly transformed aligned image
    #     """
    #     pts = [ [0,0], [1,0], [1,1]]
    #     src_pts = self._align_uni_mat.inverted().transform_points(pts)
    #     dst_pts = self._get_tr_rnd_mat(power=aug_transform_power).inverted().transform_points(pts)

    #     return Affine2DUniMat.from_3_pairs(src_pts, dst_pts)
      # key = (H,W, aug_warp_power, aug_transform_power)
        # data = self._cached.get(key, None)
        # if data is None:
        #     rnd_state = np.random.RandomState()
        #     rnd_state.set_state( self._rnd_state )
        #     self._cached[key] = data = self._gen_remap_grid(H,W, aug_warp_power, aug_transform_power, out_res, rnd_state=rnd_state )

        # remap_grid, mask = data