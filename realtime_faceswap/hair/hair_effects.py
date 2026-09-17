"""Real-time Hair Boundary Protection and Selective Recoloring.

Requirements (Sections 15, 16):
- Preserves natural hair boundary by carving hair region out of the face swap mask.
- Real-time hair recoloring: uses HSV / soft-light color blending restricted strictly
  to the hair mask area (never bleeding onto skin or background).
- Temporal smoothing for hair mask stability.
"""
from typing import Tuple, Optional
import numpy as np
import logging
from .base import HairBackend

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class HairProcessor(HairBackend):
    """Handles hair boundary masking and HSV-based real-time hair tinting."""

    def __init__(self) -> None:
        self._previous_hair_mask: Optional[np.ndarray] = None

    def protect_hair_boundary(
        self, face_mask: np.ndarray, hair_mask: np.ndarray, dilation_px: int = 3
    ) -> np.ndarray:
        """Removes hair pixels from face swap mask so swapped face doesn't clobber bangs/forehead hair."""
        if hair_mask is None or np.sum(hair_mask) == 0:
            return face_mask

        if HAS_CV2 and dilation_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
            dilated_hair = cv2.dilate(hair_mask, kernel)
        else:
            dilated_hair = hair_mask

        # Clean subtraction: face_mask & ~dilated_hair
        protected_face_mask = np.clip(face_mask.astype(np.int16) - dilated_hair.astype(np.int16), 0, 255).astype(np.uint8)
        return protected_face_mask

    def smooth_hair_mask(self, current_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Applies temporal EMA to hair mask to suppress frame-to-frame boundary shimmer."""
        if self._previous_hair_mask is None or self._previous_hair_mask.shape != current_mask.shape:
            self._previous_hair_mask = current_mask.astype(np.float32)
            return current_mask

        smoothed = alpha * self._previous_hair_mask + (1.0 - alpha) * current_mask.astype(np.float32)
        self._previous_hair_mask = smoothed
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def apply_hair_effects(
        self,
        frame: np.ndarray,
        hair_mask: np.ndarray,
        target_color_rgb: Tuple[int, int, int],
        strength: float = 0.5,
    ) -> np.ndarray:
        """Applies selective hair recoloring in HSV space."""
        if not HAS_CV2 or hair_mask is None or strength <= 0.001:
            return frame

        # Convert target RGB to BGR for OpenCV
        r, g, b = target_color_rgb
        target_bgr = np.array([b, g, r], dtype=np.uint8).reshape(1, 1, 3)
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hair mask alpha [0, 1]
        norm_mask = (hair_mask.astype(np.float32) / 255.0) * strength

        # Blend hue and saturation toward target color while preserving original luminance/value (V channel)
        # to retain natural hair highlights, shadows, and strand texture
        hsv[..., 0] = (1.0 - norm_mask) * hsv[..., 0] + norm_mask * target_hsv[0]
        hsv[..., 1] = np.clip((1.0 - norm_mask) * hsv[..., 1] + norm_mask * max(target_hsv[1], 80), 0, 255)

        recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Composite recolored hair back using 3-channel alpha
        alpha_3d = norm_mask[..., np.newaxis]
        result = (frame.astype(np.float32) * (1.0 - alpha_3d) + recolored_bgr.astype(np.float32) * alpha_3d).astype(np.uint8)
        return result
