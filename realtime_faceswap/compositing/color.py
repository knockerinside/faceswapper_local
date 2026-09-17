"""Lightweight Real-time Color Correction (Reinhard Statistical Transfer).

Requirements (Section 19):
- Matches target video lighting, exposure, and skin tone.
- Computationally lightweight (statistical transfer in LAB color space).
- Zero neural network overhead.
"""
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ColorMatcher:
    """Matches lighting and color tone between swapped face and target frame."""

    @staticmethod
    def transfer_color_lab(
        source_bgr: np.ndarray, target_bgr: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """Transfers mean and standard deviation in LAB color space from target to source."""
        if not HAS_CV2 or strength <= 0.001:
            return source_bgr

        # Convert to float LAB space
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = cv2.meanStdDev(src_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

        src_mean = src_mean.reshape(1, 1, 3)
        src_std = np.maximum(src_std.reshape(1, 1, 3), 1e-4)
        tgt_mean = tgt_mean.reshape(1, 1, 3)
        tgt_std = np.maximum(tgt_std.reshape(1, 1, 3), 1e-4)

        # Scale and shift channels
        scaled = ((src_lab - src_mean) * (tgt_std / src_std)) + tgt_mean
        matched_lab = np.clip(scaled, 0, 255).astype(np.uint8)
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

        if strength < 0.999:
            return cv2.addWeighted(source_bgr, 1.0 - strength, matched_bgr, strength, 0)

        return matched_bgr
