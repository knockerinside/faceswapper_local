"""Facial Region Mask Generation and Temporal Smoothing.

Requirements (Sections 13, 17):
- Generates soft-edged boundary masks around 128x128 face crop.
- Supports erosion, dilation, Gaussian feathering, and temporal smoothing.
- Prevents hard box boundary artifacts or jittering mask edges.
"""
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MaskGenerator:
    """Creates softened, eroded, and temporally smoothed face masks."""

    def __init__(self, crop_size: int = 128) -> None:
        self.crop_size = crop_size
        self._prev_mask: Optional[np.ndarray] = None
        self._base_template = self._create_base_ellipse_mask(crop_size)

    @staticmethod
    def _create_base_ellipse_mask(size: int) -> np.ndarray:
        """Creates a smooth default face oval for 128x128 aligned crops."""
        mask = np.zeros((size, size), dtype=np.uint8)
        if HAS_CV2:
            center = (size // 2, int(size * 0.52))
            axes = (int(size * 0.38), int(size * 0.44))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            y, x = np.ogrid[:size, :size]
            dist = ((x - size // 2) ** 2) / (0.38 * size) ** 2 + ((y - size * 0.52) ** 2) / (0.44 * size) ** 2
            mask[dist <= 1.0] = 255
        return mask

    def generate_mask(
        self,
        custom_mask: Optional[np.ndarray] = None,
        feather_px: int = 15,
        erosion_px: int = 4,
        temporal_alpha: float = 0.4,
    ) -> np.ndarray:
        """Generates a refined 128x128 float32 mask [0.0, 1.0] with feathering."""
        raw_mask = custom_mask if custom_mask is not None else self._base_template.copy()

        if HAS_CV2:
            # 1. Erosion to pull mask inside face boundary
            if erosion_px > 0:
                k_size = erosion_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                raw_mask = cv2.erode(raw_mask, kernel)

            # 2. Gaussian feathering for seamless edge transition
            if feather_px > 0:
                f_size = feather_px * 2 + 1
                raw_mask = cv2.GaussianBlur(raw_mask, (f_size, f_size), 0)

        mask_float = raw_mask.astype(np.float32) / 255.0

        # 3. Temporal smoothing to eliminate inter-frame flicker
        if self._prev_mask is None or self._prev_mask.shape != mask_float.shape:
            self._prev_mask = mask_float
        else:
            mask_float = temporal_alpha * self._prev_mask + (1.0 - temporal_alpha) * mask_float
            self._prev_mask = mask_float

        return np.clip(mask_float, 0.0, 1.0)

    def reset(self) -> None:
        self._prev_mask = None
