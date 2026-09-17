"""Virtual Camera Output subsystem."""
from .base import OutputBackend
from .virtual_camera import VirtualCameraBackend

__all__ = ["OutputBackend", "VirtualCameraBackend"]
