"""pyvirtualcam Virtual Camera Backend.

Requirements (Sections 20, 73):
- Exposes output as a standard Windows DirectShow webcam for OBS Studio or Discord.
- Converts OpenCV BGR frames to pyvirtualcam RGB.
- Reports clear AVAILABLE / UNAVAILABLE status without crashing.
"""
from typing import Optional
import logging
import numpy as np
from .base import OutputBackend

logger = logging.getLogger(__name__)

try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except ImportError:
    HAS_PYVIRTUALCAM = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VirtualCameraBackend(OutputBackend):
    """Feeds processed frames into OBS Virtual Camera or Unity Capture via pyvirtualcam."""

    def __init__(self) -> None:
        self._cam = None
        self._is_active: bool = False
        self._width: int = 1280
        self._height: int = 720
        self._fps: int = 30

    def is_supported(self) -> bool:
        return HAS_PYVIRTUALCAM

    def start(self, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        if not HAS_PYVIRTUALCAM:
            logger.warning("pyvirtualcam is not installed. Virtual camera unavailable.")
            return False

        self.stop()
        self._width = width
        self._height = height
        self._fps = fps

        try:
            # Attempts opening native Windows virtual camera (OBS Virtual Camera or Unity)
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB
            )
            self._is_active = True
            logger.info(f"Virtual camera started successfully: {self._cam.device} ({width}x{height} @ {fps}fps)")
            return True
        except Exception as e:
            logger.error(f"Failed to start virtual camera: {e}")
            self._is_active = False
            return False

    def send_frame(self, frame_bgr: np.ndarray) -> bool:
        if not self._is_active or self._cam is None:
            return False

        try:
            # Resize if necessary to match virtual camera dimensions
            h, w = frame_bgr.shape[:2]
            if w != self._width or h != self._height:
                if HAS_CV2:
                    frame_bgr = cv2.resize(frame_bgr, (self._width, self._height))

            # Convert BGR to RGB for pyvirtualcam
            if HAS_CV2:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_bgr[..., ::-1]

            self._cam.send(frame_rgb)
            self._cam.sleep_until_next_frame()
            return True
        except Exception as e:
            logger.debug(f"Error sending frame to virtual camera: {e}")
            return False

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception as e:
                logger.debug(f"Error closing virtual camera: {e}")
            self._cam = None
        self._is_active = False
        logger.info("Virtual camera stopped.")

    def is_active(self) -> bool:
        return self._is_active
