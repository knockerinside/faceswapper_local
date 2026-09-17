"""Abstract Base Class for Face Parsing Backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ParsedFaceMasks:
    """Segmented region masks for a face."""
    face_mask: np.ndarray      # Skin, nose, brows, lips (uint8 0-255)
    hair_mask: np.ndarray      # Hair region (uint8 0-255)
    occlusion_mask: np.ndarray # Eyeglasses, hands, microphone (uint8 0-255)
    raw_class_map: Optional[np.ndarray] = None


class ParserBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, provider: str = "CUDAExecutionProvider") -> bool:
        pass

    @abstractmethod
    def parse(self, face_image: np.ndarray) -> ParsedFaceMasks:
        """Runs face and hair segmentation returning binary masks."""
        pass

    @abstractmethod
    def warmup(self, num_runs: int = 3) -> None:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass
