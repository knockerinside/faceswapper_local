"""Real-Time ROI Face Compositor and Inverse Warper.

Requirements (Sections 18, 57):
- Inverse-warps 128x128 swapped face and soft mask directly into frame ROI.
- Fast alpha blending without processing full frame needlessly.
- Blends seamlessly with background video feed.
"""
from typing import Tuple, Optional
import numpy as np
import logging
from .color import ColorMatcher

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FaceBlender:
    """Combines original video frame with swapped face crop and feathered mask."""

    def __init__(self) -> None:
        self.color_matcher = ColorMatcher()

    def blend(
        self,
        original_frame: np.ndarray,
        aligned_target_face: np.ndarray,
        swapped_face_128: np.ndarray,
        mask_128: np.ndarray,
        M_inv: np.ndarray,
        blend_strength: float = 1.0,
        enable_color_correction: bool = True,
    ) -> np.ndarray:
        """Inverse-warps swapped 128x128 face and mask to frame and composites."""
        if not HAS_CV2:
            return original_frame

        h, w = original_frame.shape[:2]

        # 1. Optional statistical color matching in 128x128 crop space (extremely fast)
        if enable_color_correction:
            swapped_face_128 = self.color_matcher.transfer_color_lab(
                source_bgr=swapped_face_128, target_bgr=aligned_target_face, strength=0.85
            )

        # 2. Warp swapped face (128x128 -> frame) using inverse affine matrix M_inv
        warped_face = cv2.warpAffine(
            swapped_face_128,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # 3. Warp mask (128x128 -> frame) using inverse affine matrix M_inv
        warped_mask = cv2.warpAffine(
            mask_128,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Apply global blend strength
        effective_mask = warped_mask * blend_strength
        mask_3d = effective_mask[..., np.newaxis]

        # 4. Fast in-place alpha composite: output = frame * (1 - mask) + warped_face * mask
        orig_f = original_frame.astype(np.float32)
        warp_f = warped_face.astype(np.float32)

        composited = orig_f * (1.0 - mask_3d) + warp_f * mask_3d
        return np.clip(composited, 0, 255).astype(np.uint8)
