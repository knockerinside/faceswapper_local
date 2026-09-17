"""Abstract Base Class for Output Backends."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class OutputBackend(ABC):
    @abstractmethod
    def start(self, width: int = 1280, height: int = 720, fps: int = 30) -> bool:
        """Starts virtual camera stream."""
        pass

    @abstractmethod
    def send_frame(self, frame_bgr: np.ndarray) -> bool:
        """Sends frame to virtual camera buffer."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops virtual camera and releases driver resources."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass

    @abstractmethod
    def is_supported(self) -> bool:
        """Returns True if pyvirtualcam and a virtual camera device driver exist."""
        pass
