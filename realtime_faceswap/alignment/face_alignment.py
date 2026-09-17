"""Facial Landmark-based Affine Alignment (5-Point ArcFace/InSwapper Standard).

Requirements (Sections 12, 57):
- Standard 5-point facial landmark reference for 128x128 crop (InSwapper-128 requirement).
- Similarity transform estimation (Umeyama algorithm).
- Computes forward transformation matrix M (frame -> 128x128 aligned face)
- Computes inverse transformation matrix M_inv (128x128 -> frame coordinates) for ROI compositing.
"""
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Standard 112x112 ArcFace reference landmarks scaled to 128x128 for InSwapper-128
ARCFACE_REF_128 = np.array(
    [
        [38.2946 * (128.0 / 112.0), 51.6963 * (128.0 / 112.0)],  # Left eye
        [73.5318 * (128.0 / 112.0), 51.6963 * (128.0 / 112.0)],  # Right eye
        [56.0252 * (128.0 / 112.0), 71.7366 * (128.0 / 112.0)],  # Nose tip
        [41.5493 * (128.0 / 112.0), 92.3655 * (128.0 / 112.0)],  # Left mouth corner
        [70.7299 * (128.0 / 112.0), 92.3655 * (128.0 / 112.0)],  # Right mouth corner
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Estimates 2x3 affine similarity transform matrix via Umeyama algorithm."""
    num = src_pts.shape[0]
    dim = src_pts.shape[1]

    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_demean = src_pts - src_mean
    dst_demean = dst_pts - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
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

    src_var = np.var(src_pts, axis=0).sum()
    scale = 1.0 / src_var * np.dot(S, d)

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T[:2, :]


class FaceAligner:
    """Handles face cropping, 128x128 alignment, and inverse ROI transformation."""

    def __init__(self, crop_size: int = 128) -> None:
        self.crop_size = crop_size
        self.ref_pts = ARCFACE_REF_128.copy()

    def get_alignment_matrix(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates forward M and inverse M_inv affine transform matrices."""
        M = estimate_similarity_transform(landmarks, self.ref_pts)
        
        # Calculate inverse transform matrix
        if HAS_CV2:
            M_inv = cv2.invertAffineTransform(M)
        else:
            # Mathematical inverse of 2x3 affine matrix: [A | b] -> [A^-1 | -A^-1 * b]
            A = M[:2, :2]
            b = M[:2, 2:]
            A_inv = np.linalg.inv(A)
            b_inv = -np.dot(A_inv, b)
            M_inv = np.hstack([A_inv, b_inv])

        return M.astype(np.float32), M_inv.astype(np.float32)

    def crop_face(self, frame: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Warps frame into aligned 128x128 face crop using matrix M."""
        if HAS_CV2:
            return cv2.warpAffine(
                frame,
                M,
                (self.crop_size, self.crop_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        # Fallback dummy crop
        return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

    def inverse_warp(
        self, face_128: np.ndarray, M_inv: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Warps aligned 128x128 face or mask back to original frame dimensions."""
        h, w = target_shape[:2]
        if HAS_CV2:
            return cv2.warpAffine(
                face_128,
                M_inv,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        return np.zeros((h, w, 3), dtype=np.uint8)
