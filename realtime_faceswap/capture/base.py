"""Abstract Base Class for Camera Capture Devices."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List


@dataclass
class CameraDeviceInfo:
    index: int
    name: str
    is_phone_camera: bool = False
    supported_resolutions: List[Tuple[int, int]] = None
    default_fps: int = 30


class CameraBackend(ABC):
    """Abstract Camera Interface supporting DirectShow, Media Foundation, or Synthetic Feeds."""

    @abstractmethod
    def open(self, device_index: int, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Opens camera connection and negotiates requested resolution/FPS."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[Any]]:
        """Reads a single frame. Returns (success, frame_bgr)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Closes and releases hardware camera handles cleanly."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Returns True if the camera handle is active and healthy."""
        pass

    @abstractmethod
    def get_actual_resolution(self) -> Tuple[int, int]:
        """Returns negotiated (width, height)."""
        pass

    @abstractmethod
    def get_actual_fps(self) -> float:
        """Returns negotiated/reported FPS from the device driver."""
        pass
