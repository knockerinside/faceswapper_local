"""BiSeNet Face and Hair Parsing implementation via ONNX Runtime.

Requirements (Section 14):
- Lightweight ResNet18 backbone trained on CelebAMask-HQ.
- Segments: skin, hair, eyes, eyebrows, mouth, nose, background.
- Fully optional: disabled by default for low latency (Section 82).
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import numpy as np
from .base import ParserBackend, ParsedFaceMasks

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# CelebAMask-HQ Class Indices
# 0: bg, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r
# 10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]
HAIR_CLASSES = [17]


class BiSeNetParser(ParserBackend):
    """BiSeNet ResNet-18 Face Parser."""

    def __init__(self, input_size: Tuple[int, int] = (512, 512)) -> None:
        self.input_size = input_size
        self._session = None
        self._model_path: str = ""
        self._provider: str = "CPUExecutionProvider"
        self._is_loaded: bool = False
        self._input_name: str = ""
        self._output_name: str = ""

    def load(self, model_path: str, provider: str = "CUDAExecutionProvider") -> bool:
        if not HAS_ORT:
            return False

        path = Path(model_path)
        if not path.exists():
            logger.info(f"BiSeNet model not found at {model_path} (parsing remains optional).")
            self._model_path = model_path
            return False

        available = ort.get_available_providers()
        chosen = [provider] if provider in available else ["CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(str(path), providers=chosen)
            self._provider = self._session.get_providers()[0]
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._model_path = str(path)
            self._is_loaded = True
            logger.info(f"BiSeNet face parser loaded. Active Provider: {self._provider}")
            return True
        except Exception as e:
            logger.error(f"Failed loading BiSeNet: {e}")
            return False

    def warmup(self, num_runs: int = 2) -> None:
        if not self._is_loaded or self._session is None:
            return
        dummy = np.zeros((1, 3, self.input_size[1], self.input_size[0]), dtype=np.float32)
        for _ in range(num_runs):
            self._session.run([self._output_name], {self._input_name: dummy})

    def parse(self, face_image: np.ndarray) -> ParsedFaceMasks:
        h, w = face_image.shape[:2]
        if not self._is_loaded or self._session is None or not HAS_CV2:
            # Elliptical heuristic mask fallback when parser model is not loaded
            y, x = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            a, b = w * 0.42, h * 0.46
            dist_from_center = ((x - center_x) ** 2) / (a ** 2) + ((y - center_y) ** 2) / (b ** 2)
            face_mask = (dist_from_center <= 1.0).astype(np.uint8) * 255

            # Hair top arc heuristic
            hair_mask = ((dist_from_center > 0.8) & (dist_from_center <= 1.4) & (y < center_y)).astype(np.uint8) * 255
            empty = np.zeros((h, w), dtype=np.uint8)
            return ParsedFaceMasks(face_mask=face_mask, hair_mask=hair_mask, occlusion_mask=empty)

        # Real BiSeNet inference
        resized = cv2.resize(face_image, self.input_size)
        # ImageNet normalization
        normalized = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        out = self._session.run([self._output_name], {self._input_name: tensor})[0]
        class_map = np.argmax(out[0], axis=0).astype(np.uint8)

        # Scale class map back to input face size
        class_map_orig = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)

        face_mask = np.isin(class_map_orig, FACE_CLASSES).astype(np.uint8) * 255
        hair_mask = np.isin(class_map_orig, HAIR_CLASSES).astype(np.uint8) * 255
        occ_mask = (class_map_orig == 6).astype(np.uint8) * 255  # eyeglasses

        return ParsedFaceMasks(
            face_mask=face_mask,
            hair_mask=hair_mask,
            occlusion_mask=occ_mask,
            raw_class_map=class_map_orig,
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "BiSeNet ResNet-18 Face Parser",
            "path": self._model_path,
            "provider": self._provider,
            "loaded": self._is_loaded,
            "vram_estimate_mb": 180 if "CUDA" in self._provider else 0,
        }
