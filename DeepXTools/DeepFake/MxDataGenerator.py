from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.random as nprnd

from common.ImageDS import ImageDSInfo, MxImageDSRefList
from core import ax, mx
from core.lib import index as lib_index
from core.lib import path as lib_path
from core.lib.image import LSHash64, LSHash64Similarity, NPImage
from core.lib.image import aug as lib_aug


class MxDataGenerator(mx.Disposable):

    @dataclass
    class GenResult:
        batch_size : int
        W : int
        H : int
        image_paths : List[Path]
        target_mask_paths : List[Path]
        image_deformed_np : List[NPImage]
        image_np : List[NPImage]
        target_mask_np : List[NPImage]

    BorderType = NPImage.Border

    def __init__(self,  default_rnd_flip : bool = True,
                        state : dict = None):
        super().__init__()

        self._main_thread = ax.get_current_thread()
        self._tg = ax.TaskGroup().dispose_with(self)
        self._ds_tg = ax.TaskGroup().dispose_with(self)
        self._ds_thread = ax.Thread('ds_thread').dispose_with(self)
        self._dcs_pool = ax.ThreadPool().dispose_with(self)
        self._job_thread = ax.Thread('job_thread').dispose_with(self)
        self._job_thread_pool = ax.ThreadPool(name='DataGeneratorThreadPool').dispose_with(self)

        self._image_mask_paths : List[Tuple[Path,Path]] = []
        self._image_mask_indexer : lib_index.ProbIndexer1D = None

        self._reloading = False

        state = state or {}

        self._mx_error = mx.TextEmitter().dispose_with(self)
        self._mx_image_ds_ref_list = MxImageDSRefList(state=state.get('image_ds_ref_list', None)).dispose_with(self)

        self._mx_offset_tx = mx.Number(state.get('offset_tx', 0.0), config=mx.NumberConfig(min=-2.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_ty = mx.Number(state.get('offset_ty', 0.0), config=mx.NumberConfig(min=-2.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_scale = mx.Number(state.get('offset_scale', 0.0), config=mx.NumberConfig(min=-4.0, max=4.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_offset_rot_deg = mx.Number(state.get('offset_rot_deg', 0.0), config=mx.NumberConfig(min=-180, max=180)).dispose_with(self)

        self._mx_rnd_tx_var = mx.Number(state.get('rnd_tx_var', 0.30), config=mx.NumberConfig(min=0.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_ty_var = mx.Number(state.get('rnd_ty_var', 0.30), config=mx.NumberConfig(min=0.0, max=2.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_scale_var = mx.Number(state.get('rnd_scale_var', 0.30), config=mx.NumberConfig(min=0.0, max=4.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_rnd_rot_deg_var = mx.Number(state.get('rnd_rot_deg_var', 20), config=mx.NumberConfig(min=0, max=180)).dispose_with(self)
        self._mx_rnd_flip = mx.Flag( state.get('rnd_flip', default_rnd_flip) ).dispose_with(self)

        self._mx_transform_intensity = mx.Number(state.get('transform_intensity', 1.0), config=mx.NumberConfig(min=0.0, max=1.0, step=0.01, decimals=2)).dispose_with(self)
        self._mx_image_deform_intensity = mx.Number(state.get('image_deform_intensity', 0.5), config=mx.NumberConfig(min=0.0, max=1.0, step=0.01, decimals=2)).dispose_with(self)

        self._mx_border_type = mx.SingleChoice[MxDataGenerator.BorderType]( MxDataGenerator.BorderType(state.get('border_type', MxDataGenerator.BorderType.REPLICATE.value)),
                                                                            avail=lambda: [x for x in [ MxDataGenerator.BorderType.CONSTANT,
                                                                                                        MxDataGenerator.BorderType.REFLECT,
                                                                                                        MxDataGenerator.BorderType.REPLICATE]]).dispose_with(self)

        self._mx_dcs = mx.Flag( state.get('dcs', False) ).dispose_with(self)
        self._mx_dcs.listen(lambda b: self.reload())

        self._mx_dcs_computing = mx.Flag(False).dispose_with(self)

        self.reload()

    @property
    def mx_error(self) -> mx.ITextEmitter_r:
        return self._mx_error
    @property
    def mx_image_ds_ref_list(self) -> MxImageDSRefList:
        return self._mx_image_ds_ref_list
    @property
    def mx_offset_tx(self) -> mx.INumber:
        return self._mx_offset_tx
    @property
    def mx_offset_ty(self) -> mx.INumber:
        return self._mx_offset_ty
    @property
    def mx_offset_scale(self) -> mx.INumber:
        return self._mx_offset_scale
    @property
    def mx_offset_rot_deg(self) -> mx.INumber:
        return self._mx_offset_rot_deg
    @property
    def mx_rnd_tx_var(self) -> mx.INumber:
        return self._mx_rnd_tx_var
    @property
    def mx_rnd_ty_var(self) -> mx.INumber:
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
    def mx_border_type(self) -> mx.ISingleChoice[BorderType]:
        return self._mx_border_type
    @property
    def mx_dcs(self) -> mx.IFlag:
        return self._mx_dcs
    @property
    def mx_dcs_computing(self) -> mx.IFlag_r:
        """Indicates that DCS currently computing"""
        return self._mx_dcs_computing
    @property
    def workers_count(self) -> int: return self._job_thread_pool.count

    def get_state(self) -> dict:
        return {'image_ds_ref_list' : self._mx_image_ds_ref_list.get_state(),

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

                'border_type' : self._mx_border_type.get().value,
                'dcs' : self._mx_dcs.get(),
                }

    @ax.protected_task
    def reload(self):
        yield ax.attach_to(self._tg)
        yield ax.switch_to(self._main_thread)

        self._reloading = True
        self._mx_dcs_computing.set(False)

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
            all_image_mask_paths = []
            for path, mask_type in ar:
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
            all_image_mask_paths_len = len(all_image_mask_paths)
            self._image_mask_paths = all_image_mask_paths
            self._image_mask_indexer = indexer = lib_index.ProbIndexer1D(all_image_mask_paths_len)
        else:
            self._mx_error.emit(str(err))
            yield ax.cancel(err)

        if self._mx_dcs.get() and all_image_mask_paths_len != 0:
            self._mx_dcs_computing.set(True)

            yield ax.switch_to(self._ds_thread)

            hash_sim = LSHash64Similarity(all_image_mask_paths_len, similarity_factor=8)

            dcs_tasks = ax.TaskSet()
            dcs_pool = self._dcs_pool
            n_hashed = 0

            while err is None:
                while dcs_tasks.count < dcs_pool.count*10 and n_hashed < all_image_mask_paths_len:
                    dcs_tasks.add( self._compute_hash(dcs_pool, idx=n_hashed, path=all_image_mask_paths[n_hashed][0]) )

                    n_hashed += 1

                if dcs_tasks.empty and n_hashed == all_image_mask_paths_len:
                    # Done
                    yield ax.switch_to(self._main_thread)

                    sim = hash_sim.get_similarities()
                    sim = (sim / sim.min()).astype(np.int32)

                    # Update probabilities in Indexer
                    indexer.set_probs(sim)
                    break

                for t in dcs_tasks.fetch(finished=True):

                    if t.succeeded:
                        idx, hash = t.result
                        hash_sim.add(idx, LSHash64(hash) )
                    else:
                        err = t.error or Exception('Unknown')
                        break

                yield ax.sleep(0.005)

            yield ax.switch_to(self._main_thread)

        self._mx_dcs_computing.set(False)

        if err is not None:
            self._mx_error.emit(str(err))
            yield ax.cancel(err)

    @ax.task
    def _compute_hash(self, pool : ax.ThreadPool, idx : int, path : Path):
        yield ax.switch_to(pool)

        return idx, NPImage.from_file(path).get_ls_hash64()



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

        image_mask_paths = self._image_mask_paths
        image_mask_indexer = self._image_mask_indexer

        if len(image_mask_paths) == 0:
            yield ax.cancel(error=Exception('No training data.'))

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

        border_type = self._mx_border_type.get()

        yield ax.switch_to(self._job_thread_pool)

        out_image_paths = []
        out_target_mask_paths = []
        out_image_deformed_np : List[NPImage] = []
        out_image_np : List[NPImage] = []
        out_target_mask : List[NPImage] = []

        for idx in (image_mask_indexer.generate(batch_size)):
            imagepath, maskpath = image_mask_paths[idx]

            out_image_paths.append(imagepath)
            out_target_mask_paths.append(maskpath)

            try:
                img = NPImage.from_file(imagepath)
                mask = NPImage.from_file(maskpath)
            except Exception as e:
                yield ax.cancel(error=e)

            img = img.grayscale() if grayscale else img.bgr()
            mask = mask.grayscale()

            # Mode FIT
            geo_aug = lib_aug.Geo(offset_transform_params = lib_aug.TransformParams(tx=offset_tx,
                                                                                    ty=offset_ty,
                                                                                    scale=offset_scale,
                                                                                    rot_deg=offset_rot_deg),
                                  transform_params = lib_aug.TransformParams(tx=nprnd.uniform(-rnd_tx_var, rnd_tx_var),
                                                                             ty=nprnd.uniform(-rnd_ty_var, rnd_ty_var),
                                                                             scale=nprnd.uniform(-rnd_scale_var, rnd_scale_var),
                                                                             rot_deg=nprnd.uniform(-rnd_rot_deg_var, rnd_rot_deg_var)) )

            img_deformed = geo_aug.transform(img, W, H, center_fit=True, transform_intensity=transform_intensity, deform_intensity=image_deform_intensity, border=border_type)
            img          = geo_aug.transform(img, W, H, center_fit=True, transform_intensity=transform_intensity, deform_intensity=0.0, border=border_type)
            mask         = geo_aug.transform(mask, W, H, center_fit=True, transform_intensity=transform_intensity, deform_intensity=0.0, )

            if rnd_flip and nprnd.randint(2) == 0:
                img_deformed = img_deformed.h_flip()
                img          = img.h_flip()
                mask         = mask.h_flip()

            img_deformed = img_deformed.clip(inplace=True)
            img          = img.clip(inplace=True)
            mask         = mask.clip(inplace=True)

            out_image_deformed_np.append(img_deformed)
            out_image_np.append(img)
            out_target_mask.append(mask)

            yield ax.sleep(0) # Let other tasks to do the work

        return MxDataGenerator.GenResult(batch_size = batch_size,
                                         W = W,
                                         H = H,
                                         image_paths = out_image_paths,
                                         target_mask_paths = out_target_mask_paths,
                                         image_deformed_np = out_image_deformed_np,
                                         image_np          = out_image_np,
                                         target_mask_np = out_target_mask)



