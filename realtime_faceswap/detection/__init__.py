"""Face Detection subsystem."""
from .base import DetectorBackend, DetectedFace
from .scrfd import SCRFDDetector

__all__ = ["DetectorBackend", "DetectedFace", "SCRFDDetector"]
