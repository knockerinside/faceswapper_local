"""OpenCV DirectShow Camera Backend for Windows."""
import sys
import time
import logging
from typing import Optional, Tuple, Any
from .base import CameraBackend

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class OpenCVCameraBackend(CameraBackend):
    """Low-latency capture backend utilizing cv2.CAP_DSHOW on Windows."""

    def __init__(self) -> None:
        self._cap = None
        self._device_index: int = -1
        self._width: int = 1280
        self._height: int = 720
        self._fps: int = 30
        self._actual_width: int = 1280
        self._actual_height: int = 720
        self._actual_fps: float = 30.0

    def open(self, device_index: int, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        if not HAS_CV2:
            logger.error("OpenCV (cv2) is not installed.")
            return False

        self.release()
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps

        backend_flag = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        logger.info(f"Opening camera index {device_index} with backend {backend_flag} ({width}x{height} @ {fps}fps)")

        try:
            self._cap = cv2.VideoCapture(device_index, backend_flag)
            if not self._cap.isOpened():
                logger.error(f"Failed to open video capture device {device_index}")
                return False

            # Crucial: Request driver-level single-frame buffer to eliminate camera lag
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Request MJPG or native stream format
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)

            # Read back actual negotiated parameters from driver
            self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            self._actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or float(fps)

            logger.info(
                f"Camera opened successfully. Negotiated: {self._actual_width}x{self._actual_height} @ {self._actual_fps:.1f} FPS"
            )
            return True
        except Exception as e:
            logger.error(f"Exception opening camera {device_index}: {e}", exc_info=True)
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Camera read returned empty frame or disconnected.")
            return False, None

        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.debug(f"Error releasing camera handle: {e}")
            self._cap = None
            logger.info("Camera handle released cleanly.")

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_actual_resolution(self) -> Tuple[int, int]:
        return self._actual_width, self._actual_height

    def get_actual_fps(self) -> float:
        return self._actual_fps
