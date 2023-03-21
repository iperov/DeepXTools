from __future__ import annotations

import math

import cv2
import numpy as np


class Affine2DMat(np.ndarray):
    """
    affine transformation matrix for 2D
    shape is (2,3)
    """
    @staticmethod
    def from_3_pairs(src_pts, dst_pts) -> Affine2DMat:
        """
        calculates Affine2DMat from three pairs of the corresponding points.
        """
        return Affine2DMat(cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts)))


    @staticmethod
    def from_umeyama(src, dst, estimate_scale=True):
        """
        Estimate N-D similarity transformation with or without scaling.
        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.

        Returns
        -------
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.

        Reference
        Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
        """
        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = np.dot(dst_demean.T, src_demean) / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = np.dot(U, V)
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
                d[dim - 1] = s
        else:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
        T[:dim, :dim] *= scale

        return Affine2DMat(T[:2])

    def __new__(cls,*args,**kwargs):
        return super().__new__(cls, shape=(2,3), dtype=np.float32, buffer=None, offset=0, strides=None, order=None)

    def __init__(self, values = None):
        """Creates identity mat"""
        super().__init__()
        self[:] = values if values is not None else [ [1,0,0], [0,1,0] ]

    def translated(self, tx : float, ty : float) -> Affine2DMat:
        return self*Affine2DMat( ((1,0,tx), (0,1,ty)))

    def scaled(self, sx : float, sy : float = None) -> Affine2DMat:
        return self*Affine2DMat( ((sx,0,0), (0,sy if sy is not None else sx,0)))

    def rotated(self, rot_deg : float) -> Affine2DMat:
        rot_rad = rot_deg * math.pi / 180.0
        alpha, beta = math.cos(rot_rad), math.sin(rot_rad)
        return self*Affine2DMat( ((alpha, beta,  0), (-beta, alpha, 0)) )

    def inverted(self) -> Affine2DMat:
        """returns inverted Affine2DMat"""
        ((a, b, c),
         (d, e, f)) = self
        D = a*e - b*d
        D = 1.0 / D if D != 0.0 else 0.0
        a, b, c, d, e, f = ( e*D, -b*D, (b*f-e*c)*D ,
                            -d*D,  a*D, (d*c-a*f)*D )

        return Affine2DMat( ((a, b, c),
                             (d, e, f)) )

    def map(self, points : np.ndarray) -> np.ndarray:
        """maps (N,2) points"""
        if not isinstance(points, np.ndarray):
            points = np.float32(points)

        dtype = points.dtype

        points = np.pad(points, ((0,0), (0,1) ), constant_values=(1,), mode='constant')

        return np.matmul( np.concatenate( [ self, [[0,0,1]] ], 0), points.T).T[:,:2].astype(dtype)

    def __rmul__(self, other) -> Affine2DMat:
        if isinstance(other, Affine2DMat):
            return Affine2DMat( np.matmul( np.concatenate( [ other, [[0,0,1]] ], 0),
                                           np.concatenate( [ self,  [[0,0,1]] ], 0) )[:2] )
        raise ValueError('You can multiplicate Affine2DMat only with Affine2DMat')

    def __mul__(self, other) -> Affine2DMat:
        if isinstance(other, Affine2DMat):
            return Affine2DMat( np.matmul( np.concatenate( [ self,  [[0,0,1]] ], 0),
                                           np.concatenate( [ other, [[0,0,1]] ], 0) )[:2] )
        raise ValueError('You can multiplicate Affine2DMat only with Affine2DMat')


# def as_uni_mat(self) -> 'Affine2DUniMat':
#     """
#     represent this mat as Affine2DUniMat
#     """
#     return Affine2DUniMat(self)


# class Affine2DUniMat(Affine2DMat):
#     """
#     same as Affine2DMat but for transformation of uniform coordinates
#     """
#     def __rmul__(self, other) -> 'Affine2DUniMat':
#         return super().__rmul__(other).as_uni_mat()

#     def __mul__(self, other) -> 'Affine2DUniMat':
#         return super().__mul__(other).as_uni_mat()

#     @staticmethod
#     def identity(): return Affine2DMat.identity().as_uni_mat()

#     @staticmethod
#     def umeyama(src, dst, estimate_scale=True): return Affine2DMat.umeyama(src, dst, estimate_scale=estimate_scale).as_uni_mat()

#     def translated(self, tx : float, ty : float) -> Affine2DMat: return super().translated(tx, ty).as_uni_mat()
#     def scaled(self, sx : float, sy : float = None) -> Affine2DMat: return super().scaled(sx, sy).as_uni_mat()
#     def rotated(self, rot_deg : float) -> Affine2DMat: return super().rotated(rot_deg).as_uni_mat()

#     @staticmethod
#     def from_3_pairs(src_pts, dst_pts) -> 'Affine2DUniMat': return Affine2DMat.from_3_pairs(src_pts, dst_pts).as_uni_mat()

#     def inverted(self) -> 'Affine2DUniMat': return super().inverted().as_uni_mat()

#     def source_scaled_around_center(self, sw : float, sh: float) -> 'Affine2DUniMat':
#         """
#         produces scaled UniMat around center in source space

#             sw, sh      source width/height scale
#         """
#         src_pts = np.float32([(0,0),(1,0),(0,1)])

#         dst_pts = self.map(src_pts)

#         src_pts = (src_pts-0.5)/(sw,sh)+0.5

#         return Affine2DUniMat.from_3_pairs(src_pts, dst_pts)

#     def source_translated(self, utw : float, uth: float) -> 'Affine2DUniMat':
#         """
#         produces translated UniMat in source space

#             utw, uth      uniform translate values
#         """
#         src_pts = np.float32([(0,0),(1,0),(0,1)])
#         dst_pts = self.map(src_pts)
#         src_pts += (utw, uth)
#         return Affine2DUniMat.from_3_pairs(src_pts, dst_pts)

#     def to_exact_mat(self, sw : float, sh: float, tw: float, th: float) -> Affine2DMat:
#         """
#         calculate exact Affine2DMat using provided source and target sizes

#             sw, sh      source width/height
#             tw, th      target width/height
#         """
#         return Affine2DMat.from_3_pairs([[0,0],[sw,0],[0,sh]],
#                                         self.map( [(0,0),(1,0),(0,1)] ) * (tw,th) )



# def scaled(self, sw : float, sh: float, tw: float, th: float) -> Affine2DMat:
#     """

#         sw, sh      source width/height scale
#         tw, th      target width/height scale
#     """
#     src_pts = np.float32([(0,0),(1,0),(0,1),(0.5,0.5)])
#     src_pts -= 0.5

#     dst_pts = self.map(src_pts)

#     print(src_pts, dst_pts)

#     src_pts = src_pts*(sw,sh)

#     dst_cpt = dst_pts[-1]

#     dst_pts = (dst_pts-dst_cpt)*(tw,th) + dst_cpt*(tw,th)



#     return Affine2DUniMat.from_3_pairs(src_pts[:3], dst_pts[:3] )
