"""Camera capture subsystem."""
from .base import CameraBackend, CameraDeviceInfo
from .device_manager import DeviceManager
from .opencv_dshow import OpenCVCameraBackend
from .video_backend import VideoFileCameraBackend

__all__ = [
    "CameraBackend",
    "CameraDeviceInfo",
    "DeviceManager",
    "OpenCVCameraBackend",
    "VideoFileCameraBackend",
]
