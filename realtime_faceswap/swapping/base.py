"""Abstract Base Class for Face Swapping Backends."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class SwapperBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, provider: str = "CUDAExecutionProvider") -> bool:
        """Loads model weights once into GPU/VRAM. Never inside frame loop."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Releases session and frees GPU VRAM."""
        pass

    @abstractmethod
    def set_source(self, source_image: np.ndarray, source_embedding: Optional[np.ndarray] = None) -> bool:
        """Extracts identity embedding from source image ONCE and caches it."""
        pass

    @abstractmethod
    def process(self, aligned_face_128: np.ndarray) -> np.ndarray:
        """Runs inference swapping source identity onto aligned target face (128x128)."""
        pass

    @abstractmethod
    def warmup(self, num_runs: int = 3) -> None:
        """Runs warmup inferences on GPU before timer starts."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model metadata, status, provider, and VRAM estimate."""
        pass
