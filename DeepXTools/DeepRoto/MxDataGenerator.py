from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy.random as nprnd

from common.ImageDS import ImageDSInfo, MxImageDSRefList
from core import ax, mx
from core.lib import index as lib_index
from core.lib import path as lib_path
from core.lib.image import NPImage
from core.lib.image import aug as lib_aug
from core.lib.python import shuffled


class MxDataGenerator(mx.Disposable):

    class Mode(Enum):
        Fit = auto()
        Patch = auto()

    # class OutputType(Enum):
    #     Image_n_Mask = auto()
    #     Image_n_ImageGrayscaled = auto()
    
    @dataclass
    class GenResult:
        batch_size : int
        W : int
        H : int
        image_paths : List[Path]
        target_mask_paths : List[Path]
        image_np : List[NPImage]
        target_mask_np : List[NPImage]
        
    def __init__(self,  default_rnd_flip : bool = True,
                        state : dict = None):
        super().__init__()
        
        self._main_thread = ax.get_current_thread()
        self._tg = ax.TaskGroup().dispose_with(self)
        self._ds_tg = ax.TaskGroup().dispose_with(self)
        self._ds_thread = ax.Thread('ds_thread').dispose_with(self)
        self._job_thread = ax.Thread('job_thread').dispose_with(self)
        self._job_thread_pool = ax.ThreadPool(name='DataGeneratorThreadPool').dispose_with(self)

        #self._image_paths : List[Path] = []
        self._image_mask_paths : List[Tuple[Path,Path]] = []

        #self._image_indexer : lib_index.ProbIndexer1D = None
        self._image_mask_indexer : lib_index.ProbIndexer1D = None

        self._reloading = False

        state = state or {}
        
        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_image_ds_ref_list = MxImageDSRefList(state=state.get('image_ds_ref_list', None)).dispose_with(self)

        self._mx_mode = mx.SingleChoice[MxDataGenerator.Mode](MxDataGenerator.Mode(state.get('mode', MxDataGenerator.Mode.Fit.value)), avail=lambda: [*MxDataGenerator.Mode]).dispose_with(self)

        #self._mx_output_type = mx.SingleChoice[MxDataGenerator.OutputType](MxDataGenerator.OutputType(state.get('output_type', MxDataGenerator.OutputType.Image_n_Mask.value)), avail=lambda: [*MxDataGenerator.OutputType]).dispose_with(self)

        self._mx_offset_tx = mx.Number(state.get('offset_tx', 0.0), config=mx.NumberConfig(min=-2.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_ty = mx.Number(state.get('offset_ty', 0.0), config=mx.NumberConfig(min=-2.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_scale = mx.Number(state.get('offset_scale', 0.0), config=mx.NumberConfig(min=-4.0, max=4.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_rot_deg = mx.Number(state.get('offset_rot_deg', 0.0), config=mx.NumberConfig(min=-180, max=180)).dispose_with(self)

        self._mx_rnd_tx_var = mx.Number(state.get('rnd_tx_var', 0.25), config=mx.NumberConfig(min=0.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_ty_var = mx.Number(state.get('rnd_ty_var', 0.25), config=mx.NumberConfig(min=0.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_scale_var = mx.Number(state.get('rnd_scale_var', 0.30), config=mx.NumberConfig(min=0.0, max=4.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_rot_deg_var = mx.Number(state.get('rnd_rot_deg_var', 15), config=mx.NumberConfig(min=0, max=180)).dispose_with(self)

        self._mx_rnd_flip = mx.Flag( state.get('rnd_flip', default_rnd_flip) ).dispose_with(self)

        self._mx_transform_intensity = mx.Number(state.get('transform_intensity', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_image_deform_intensity = mx.Number(state.get('image_deform_intensity', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_image_deform_intensity.listen(lambda v: self._mx_mask_deform_intensity.set(v) if self._mx_mask_deform_intensity.get() > v else ... )
        self._mx_mask_deform_intensity = mx.Number(state.get('mask_deform_intensity', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.01, decimals=2), 
                                                    filter=lambda n,o: min(n, self._mx_image_deform_intensity.get())  ).dispose_with(self)
        
        self._mx_rnd_levels_shift   = mx.Flag( state.get('rnd_levels_shift', False) ).dispose_with(self)
        self._mx_rnd_sharpen_blur   = mx.Flag( state.get('rnd_sharpen_blur', False) ).dispose_with(self)
        self._mx_rnd_glow_shade     = mx.Flag( state.get('rnd_glow_shade', False) ).dispose_with(self)
        self._mx_rnd_resize         = mx.Flag( state.get('rnd_resize', False) ).dispose_with(self)
        self._mx_rnd_jpeg_artifacts = mx.Flag( state.get('rnd_jpeg_artifacts', False) ).dispose_with(self)
        
        self.reload()

    @property
    def mx_error(self) -> mx.ITextEmitter_r:
        return self._mx_error
    @property
    def mx_image_ds_ref_list(self) -> MxImageDSRefList:
        return self._mx_image_ds_ref_list
    @property
    def mx_mode(self) -> mx.ISingleChoice[MxDataGenerator.Mode]:
        return self._mx_mode
    # @property
    # def mx_output_type(self) -> mx.ISingleChoice[MxDataGenerator.OutputType]:
    #     return self._mx_output_type
    @property
    def mx_offset_tx(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_offset_tx
    @property
    def mx_offset_ty(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_offset_ty
    @property
    def mx_offset_scale(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_offset_scale
    @property
    def mx_offset_rot_deg(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_offset_rot_deg
    @property
    def mx_rnd_tx_var(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_rnd_tx_var
    @property
    def mx_rnd_ty_var(self) -> mx.INumber:
        """Avail when mx_mode == Fit"""
        return self._mx_rnd_ty_var
    @property
    def mx_rnd_scale_var(self) -> mx.INumber:
        return self._mx_rnd_scale_var
    @property
    def mx_rnd_rot_deg_var(self) -> mx.INumber:
        return self._mx_rnd_rot_deg_var
    @property
    def mx_rnd_flip(self) -> mx.IFlag:
        return self._mx_rnd_flip
    @property
    def mx_transform_intensity(self) -> mx.INumber:
        return self._mx_transform_intensity
    @property
    def mx_image_deform_intensity(self) -> mx.INumber:
        return self._mx_image_deform_intensity
    @property
    def mx_mask_deform_intensity(self) -> mx.INumber:
        return self._mx_mask_deform_intensity
    @property
    def mx_rnd_levels_shift(self) -> mx.IFlag:
        return self._mx_rnd_levels_shift
    @property
    def mx_rnd_sharpen_blur(self) -> mx.IFlag:
        return self._mx_rnd_sharpen_blur
    @property
    def mx_rnd_glow_shade(self) -> mx.IFlag:
        return self._mx_rnd_glow_shade
    @property
    def mx_rnd_resize(self) -> mx.IFlag:
        return self._mx_rnd_resize
    @property
    def mx_rnd_jpeg_artifacts(self) -> mx.IFlag:
        return self._mx_rnd_jpeg_artifacts
    @property
    def workers_count(self) -> int: return self._job_thread_pool.count

    def get_state(self) -> dict:
        return {'image_ds_ref_list' : self._mx_image_ds_ref_list.get_state(),
                'mode' : self._mx_mode.get().value,
                # 'output_type' : self._mx_output_type.get().value,

                'offset_tx' : self._mx_offset_tx.get(),
                'offset_ty' : self._mx_offset_ty.get(),
                'offset_scale' : self._mx_offset_scale.get(),
                'offset_rot_deg' : self._mx_offset_rot_deg.get(),

                'rnd_tx_var' : self._mx_rnd_tx_var.get(),
                'rnd_ty_var' : self._mx_rnd_ty_var.get(),
                'rnd_scale_var' : self._mx_rnd_scale_var.get(),
                'rnd_rot_deg_var' : self.mx_rnd_rot_deg_var.get(),
                'rnd_flip' : self._mx_rnd_flip.get(),

                'transform_intensity' : self._mx_transform_intensity.get(),
                'image_deform_intensity' : self._mx_image_deform_intensity.get(),
                'mask_deform_intensity' : self._mx_mask_deform_intensity.get(),

                'rnd_levels_shift'   : self._mx_rnd_levels_shift.get(),
                'rnd_sharpen_blur'   : self._mx_rnd_sharpen_blur.get(),
                'rnd_glow_shade'     : self._mx_rnd_glow_shade.get(),
                'rnd_resize'         : self._mx_rnd_resize.get(),
                'rnd_jpeg_artifacts' : self._mx_rnd_jpeg_artifacts.get(),
                }

    @ax.protected_task
    def reload(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        self._reloading = True

        yield ax.attach_to(self._ds_tg, cancel_all=True)

        # Collect (path, mask_type) from mx controls.
        ar = []
        for v in self._mx_image_ds_ref_list.values():
            if (path := v.mx_image_ds_path.mx_path.get()) is not None:
                ar.append( (path, v.mx_mask_type.get()) )

        yield ax.switch_to(self._ds_thread)

        err = None
        try:
            # Collect file paths from disk.
            #all_image_paths = []
            all_image_mask_paths = []
            for path, mask_type in ar:
                #all_image_paths += (
                image_paths = lib_path.get_files_paths(path, extensions=NPImage.avail_image_suffixes())

                imagepath_by_stem = {imagepath.stem : imagepath for imagepath in image_paths}

                if mask_type is not None:
                    # Collect valid image+mask paths
                    mask_paths = lib_path.get_files_paths(ImageDSInfo(path).get_mask_dir_path(mask_type), extensions=NPImage.avail_image_suffixes())
                    maskpath_by_stem = { maskpath.stem : maskpath for maskpath in mask_paths}
                    common_stems = set(imagepath_by_stem.keys()).intersection( set(maskpath_by_stem.keys()) )
                    all_image_mask_paths += [ (imagepath_by_stem[x], maskpath_by_stem[x]) for x in common_stems ]
        except Exception as e:
            err = e

        yield ax.switch_to(self._main_thread)

        self._reloading = False

        if err is None:
            #self._image_paths = all_image_paths
            self._image_mask_paths = all_image_mask_paths

            #self._image_indexer = lib_index.ProbIndexer1D(len(all_image_paths))
            self._image_mask_indexer = lib_index.ProbIndexer1D(len(all_image_mask_paths))
        else:
            self._mx_error.emit(str(err))
            yield ax.cancel(err)


    @ax.task
    def generate(self,  batch_size : int,
                        W : int, H : int,
                        grayscale : bool = False,
                    ) -> GenResult:

        if batch_size < 1: raise ValueError('batch_size must be >= 1')
        if W < 4: raise ValueError('W must be >= 4')
        if H < 4: raise ValueError('H must be >= 4')

        yield ax.attach_to(self._tg, detach_parent=False)
        yield ax.switch_to(self._main_thread)

        while self._reloading:
            yield ax.sleep(0.1)

        #image_paths = self._image_paths
        #image_indexer = self._image_indexer

        image_mask_paths = self._image_mask_paths
        image_mask_indexer = self._image_mask_indexer

        mode = self._mx_mode.get()
        #output_type = self._mx_output_type.get()

        #if output_type == MxDataGenerator.OutputType.Image_n_Mask:
        if len(image_mask_paths) == 0:
            yield ax.cancel(error=Exception('No training data.'))
        # elif output_type == MxDataGenerator.OutputType.Image_n_ImageGrayscaled:
        #     if len(image_paths) == 0:
        #         yield ax.cancel(error=Exception('No training data.'))

        offset_tx      = self._mx_offset_tx.get()
        offset_ty      = self._mx_offset_ty.get()
        offset_scale   = self._mx_offset_scale.get()
        offset_rot_deg = self._mx_offset_rot_deg.get()

        rnd_tx_var      = self._mx_rnd_tx_var.get()
        rnd_ty_var      = self._mx_rnd_ty_var.get()
        rnd_scale_var   = self._mx_rnd_scale_var.get()
        rnd_rot_deg_var = self._mx_rnd_rot_deg_var.get()
        rnd_flip        = self._mx_rnd_flip.get()

        transform_intensity  = self._mx_transform_intensity.get()
        image_deform_intensity = self._mx_image_deform_intensity.get()
        mask_deform_intensity  = self._mx_mask_deform_intensity.get()

        rnd_levels_shift    = self._mx_rnd_levels_shift.get()
        rnd_sharpen_blur    = self._mx_rnd_sharpen_blur.get()
        rnd_glow_shade      = self._mx_rnd_glow_shade.get()
        rnd_resize          = self._mx_rnd_resize.get()
        rnd_jpeg_artifacts  = self._mx_rnd_jpeg_artifacts.get()

        yield ax.switch_to(self._job_thread_pool)

        out_image_paths = []
        out_target_mask_paths = []
        out_image_np : List[NPImage] = []
        out_target_mask : List[NPImage] = []

        aug_list = [
            (lambda img: (lib_aug.hsv_shift(img) if (nprnd.randint(2) == 0 and not grayscale) else
                          lib_aug.levels(img))) if rnd_levels_shift else None,

            [   lambda img: lib_aug.box_sharpen(img),
                lambda img: lib_aug.gaussian_sharpen(img),
                lambda img: lib_aug.motion_blur(img),
                lambda img: lib_aug.gaussian_blur(img),
            ] if rnd_sharpen_blur else None,

            (lambda img: lib_aug.glow_shade(img, mask, inner=True, glow=nprnd.randint(2)==0)) if rnd_glow_shade else None,
            (lambda img: lib_aug.glow_shade(img, mask, inner=False, glow=nprnd.randint(2)==0)) if rnd_glow_shade else None,

            (lambda img: (lib_aug.resize(img, interp=NPImage.Interp.NEAREST) if (rnd := nprnd.randint(4)) == 0 else
                          lib_aug.resize(img, interp=NPImage.Interp.LINEAR) if rnd == 1 else
                          lib_aug.resize(img, interp=NPImage.Interp.CUBIC) if rnd == 2 else
                          lib_aug.resize(img, interp=NPImage.Interp.LANCZOS4))
            ) if rnd_resize else None,

            (lambda img: lib_aug.jpeg_artifacts(img)) if rnd_jpeg_artifacts else None,
        ]


        for idx in (image_mask_indexer.generate(batch_size)):# if output_type == MxDataGenerator.OutputType.Image_n_Mask else
                    #image_indexer.generate(batch_size) if output_type == MxDataGenerator.OutputType.Image_n_ImageGrayscaled else ...):

            #if output_type == MxDataGenerator.OutputType.Image_n_Mask:
            imagepath, maskpath = image_mask_paths[idx]
            #elif output_type == MxDataGenerator.OutputType.Image_n_ImageGrayscaled:
            #    imagepath = maskpath = image_paths[idx]

            out_image_paths.append(imagepath)
            out_target_mask_paths.append(maskpath)

            try:
                #if output_type == MxDataGenerator.OutputType.Image_n_Mask:
                img = NPImage.from_file(imagepath)
                mask = NPImage.from_file(maskpath)
                #elif output_type == MxDataGenerator.OutputType.Image_n_ImageGrayscaled:
                #    img = mask = NPImage.from_file(imagepath)
            except Exception as e:
                yield ax.cancel(error=e)

            img = img.grayscale() if grayscale else img.bgr()
            mask = mask.grayscale()

            if mode == MxDataGenerator.Mode.Fit:
                offset_transform_params = lib_aug.TransformParams(  tx=offset_tx,
                                                                    ty=offset_ty,
                                                                    scale=offset_scale,
                                                                    rot_deg=offset_rot_deg)
                
                transform_params=lib_aug.TransformParams(tx=nprnd.uniform(-rnd_tx_var, rnd_tx_var),
                                                         ty=nprnd.uniform(-rnd_ty_var, rnd_ty_var),
                                                         scale=nprnd.uniform(-rnd_scale_var, rnd_scale_var),
                                                         rot_deg=nprnd.uniform(-rnd_rot_deg_var, rnd_rot_deg_var))

            elif mode == MxDataGenerator.Mode.Patch:
                offset_transform_params = lib_aug.TransformParams()
                transform_params=lib_aug.TransformParams(
                                                    tx=nprnd.uniform(-0.50, 0.50),
                                                    ty=nprnd.uniform(-0.50, 0.50),
                                                    scale=nprnd.uniform(-rnd_scale_var, rnd_scale_var),
                                                    rot_deg=nprnd.uniform(-rnd_rot_deg_var, rnd_rot_deg_var))
            geo_aug = lib_aug.Geo(offset_transform_params=offset_transform_params, transform_params=transform_params)

            img  = geo_aug.transform(img, W, H, center_fit=mode == MxDataGenerator.Mode.Fit, transform_intensity=transform_intensity, deform_intensity=image_deform_intensity, )
            mask = geo_aug.transform(mask, W, H, center_fit=mode == MxDataGenerator.Mode.Fit, transform_intensity=transform_intensity, deform_intensity=mask_deform_intensity, )

            if rnd_flip and nprnd.randint(2) == 0:
                img = img.h_flip()
                mask = mask.h_flip()


            for aug in shuffled(aug_list):
                if aug is None:
                    continue
                if isinstance(aug, Sequence):
                    aug = random.choice(aug)

                aug_mask = lib_aug.noise_clouds_mask(W, H)
                aug_mask = aug_mask.motion_blur( nprnd.randint(1, max(1, max(aug_mask.shape[0:2]) // 8)), nprnd.randint(0, 360)  )
                aug_mask *= lib_aug.circle_faded_mask(W, H)

                img = img.blend(aug(img), aug_mask)

            img = img.clip(inplace=True)

            out_image_np.append(img)
            out_target_mask.append(mask)

            yield ax.sleep(0) # Let other tasks to do the work

        return MxDataGenerator.GenResult(batch_size = batch_size,
                                         W = W,
                                         H = H,
                                         image_paths = out_image_paths,
                                         target_mask_paths = out_target_mask_paths,
                                         image_np = out_image_np,
                                         target_mask_np = out_target_mask)

    

