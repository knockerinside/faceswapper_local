"""Abstract Base Class for Face Detectors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import numpy as np


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box, score, and 5-point facial landmarks."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    landmarks: np.ndarray                   # Shape (5, 2): [left_eye, right_eye, nose, left_mouth, right_mouth]
    embedding: Optional[np.ndarray] = None   # ArcFace 512-d embedding if extracted
    track_id: int = -1


class DetectorBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, provider: str = "CUDAExecutionProvider") -> bool:
        """Loads model weights once. Never inside frame loop."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, max_num: int = 1) -> List[DetectedFace]:
        """Runs face detection returning list of detected faces sorted by score/area."""
        pass

    @abstractmethod
    def warmup(self, num_runs: int = 3) -> None:
        """Runs warmup inferences to compile CUDA kernels and stabilize timing."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Returns model name, provider, input shape, loaded status."""
        pass
