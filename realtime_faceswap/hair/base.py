"""Abstract Base Class for Hair Processing and Stylization."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class HairBackend(ABC):
    @abstractmethod
    def apply_hair_effects(
        self,
        frame: np.ndarray,
        hair_mask: np.ndarray,
        target_color_rgb: Tuple[int, int, int],
        strength: float,
    ) -> np.ndarray:
        """Applies recoloring to hair region without bleeding onto face or background."""
        pass

    @abstractmethod
    def protect_hair_boundary(
        self, face_mask: np.ndarray, hair_mask: np.ndarray
    ) -> np.ndarray:
        """Subtracts hair region from face swap mask so swapped face doesn't clobber hair."""
        pass
