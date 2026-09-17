"""Video File Capture Backend for Development and Headless Testing.

Requirements (Section 69):
- Loop prerecorded video files (MP4, MOV, AVI) through the full AI pipeline.
- Enables reproducible benchmarking and development without requiring physical Android hardware.
"""
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
from .base import CameraBackend

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoFileCameraBackend(CameraBackend):
    """Feeds video frames from a local video file in a real-time paced loop."""

    def __init__(self, file_path: str = "") -> None:
        self.file_path = file_path
        self._cap = None
        self._width: int = 1280
        self._height: int = 720
        self._fps: float = 30.0
        self._last_frame_time: float = 0.0

    def open(self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        if not self.file_path or not Path(self.file_path).exists():
            logger.info("Using synthetic video backend generator.")
            self._width = width
            self._height = height
            self._fps = float(fps)
            return True

        if not HAS_CV2:
            logger.error("OpenCV (cv2) is not installed.")
            return False

        try:
            self._cap = cv2.VideoCapture(self.file_path)
            if not self._cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False

            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or float(fps)
            logger.info(f"Loaded test video file: {self.file_path} ({self._width}x{self._height} @ {self._fps:.1f}fps)")
            return True
        except Exception as e:
            logger.error(f"Error opening test video: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        # Frame rate pacing to simulate real camera delivery
        target_interval = 1.0 / max(1.0, self._fps)
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        self._last_frame_time = time.perf_counter()

        if self._cap is not None and self._cap.isOpened() and HAS_CV2:
            ret, frame = self._cap.read()
            if not ret:
                # Loop back to beginning of video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            return ret, frame

        # Fallback synthetic numpy frame if no video file
        import numpy as np
        synthetic_frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return True, synthetic_frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return True

    def get_actual_resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def get_actual_fps(self) -> float:
        return self._fps
