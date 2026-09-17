"""InSwapper-128 ONNX Face Swap Backend.

Requirements (Sections 9, 10, 11, 53, 54, 56):
- InSwapper-128 ONNX model loaded ONCE. Never in frame loop.
- Keeps ONNX Runtime session alive in GPU VRAM (RTX 4050 6GB target).
- Source identity embedding calculated ONCE and cached.
- Operates on aligned 128x128 face crop (ROI).
- Warm-up inference on startup.
- Graceful handling of CUDA out-of-memory.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import logging
import numpy as np
from .base import SwapperBackend
from .arcface import ArcFaceExtractor

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


class InSwapperBackend(SwapperBackend):
    """Production InSwapper-128 ONNX inference engine."""

    def __init__(self) -> None:
        self._session = None
        self._model_path: str = ""
        self._provider: str = "CPUExecutionProvider"
        self._is_loaded: bool = False
        self._source_embedding: Optional[np.ndarray] = None
        self._source_face_crop: Optional[np.ndarray] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._vram_estimate_mb: int = 600
        self.arcface = ArcFaceExtractor()

    def load(self, model_path: str, provider: str = "CUDAExecutionProvider") -> bool:
        if not HAS_ORT:
            logger.error("ONNX Runtime is not installed.")
            return False

        path = Path(model_path)
        # Attempt to load ArcFace embedder if available
        arcface_candidates = [
            path.parent / "w600k_r50.onnx",
            Path(__file__).resolve().parents[2] / "models" / "w600k_r50.onnx",
            Path("models/w600k_r50.onnx"),
        ]
        for arc_cand in arcface_candidates:
            if arc_cand.exists() and not self.arcface.is_loaded:
                self.arcface.load(str(arc_cand), provider)
                break

        # Check if specified model exists
        if not path.exists():
            # Check alternative fp16 or fp32 in same directory
            fp16_path = path.parent / "inswapper_128_fp16.onnx"
            fp32_path = path.parent / "inswapper_128.onnx"
            if fp16_path.exists():
                path = fp16_path
                logger.info(f"Using high-performance FP16 InSwapper model: {path}")
            elif fp32_path.exists():
                path = fp32_path
                logger.info(f"Using FP32 InSwapper model: {path}")
            else:
                logger.warning(
                    f"InSwapper model file not found at {model_path}. "
                    "The engine will run in preview simulation mode until models are downloaded."
                )
                self._model_path = model_path
                return False

        available_providers = ort.get_available_providers()
        chosen = [provider] if provider in available_providers else ["CPUExecutionProvider"]
        if provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in available_providers:
            logger.warning("CUDA acceleration unavailable for InSwapper — running on CPU (will be slow).")

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(path), sess_options=sess_options, providers=chosen
            )
            self._provider = self._session.get_providers()[0]
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            self._model_path = str(path)
            self._is_loaded = True
            is_fp16 = "fp16" in path.name.lower()
            self._vram_estimate_mb = 580 if is_fp16 else 1100
            logger.info(f"InSwapper loaded ({path.name}). Active Provider: {self._provider} (VRAM ~{self._vram_estimate_mb} MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to load InSwapper model: {e}", exc_info=True)
            return False

    def unload(self) -> None:
        self._session = None
        self._is_loaded = False
        logger.info("InSwapper-128 session unloaded from memory.")

    def set_source(self, source_image: np.ndarray, source_embedding: Optional[np.ndarray] = None) -> bool:
        """Stores source face crop and ArcFace 512-d identity embedding."""
        self._source_face_crop = source_image

        if source_embedding is not None:
            emb = source_embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            self._source_embedding = emb[np.newaxis, :]
            logger.info("Source identity embedding set successfully (512-D).")
            return True

        # Extract using ArcFace if available
        if self.arcface.is_loaded:
            emb = self.arcface.extract_embedding(source_image)
            self._source_embedding = emb[np.newaxis, :]
            logger.info("Extracted high-fidelity identity embedding using ArcFace (w600k_r50).")
            return True

        # If embedding not supplied, synthesize normalized vector
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)
        self._source_embedding = dummy_emb[np.newaxis, :]
        logger.info("Synthesized normalized identity embedding (ArcFace model not yet loaded).")
        return True

    def warmup(self, num_runs: int = 3) -> None:
        if not self._is_loaded or self._session is None:
            return
        logger.info(f"Warming up InSwapper GPU inference ({num_runs} runs)...")
        dummy_target = np.zeros((1, 3, 128, 128), dtype=np.float32)
        dummy_emb = np.zeros((1, 512), dtype=np.float32)
        dummy_emb[0, 0] = 1.0

        for _ in range(num_runs):
            try:
                inputs = {
                    self._input_names[0]: dummy_target,
                    self._input_names[1]: dummy_emb,
                }
                self._session.run(self._output_names, inputs)
            except Exception as e:
                logger.warning(f"InSwapper warmup iteration failed: {e}")
                break
        logger.info("InSwapper GPU warmup complete.")

    def process(self, aligned_face_128: np.ndarray) -> np.ndarray:
        """Swaps source identity into aligned 128x128 face crop with natural color matching."""
        if not self._is_loaded or self._session is None or self._source_embedding is None:
            # Simulation / fallback: slight color blend indicating transformation
            swapped = aligned_face_128.copy()
            if self._source_face_crop is not None:
                if HAS_CV2:
                    resized_src = cv2.resize(self._source_face_crop, (128, 128))
                    swapped = cv2.addWeighted(aligned_face_128, 0.4, resized_src, 0.6, 0)
            return swapped

        try:
            # InSwapper expects RGB in float32 [0.0, 1.0] or normalized
            if HAS_CV2 and len(aligned_face_128.shape) == 3 and aligned_face_128.shape[2] == 3:
                face_rgb = cv2.cvtColor(aligned_face_128, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = aligned_face_128

            face_float = face_rgb.astype(np.float32) / 255.0
            face_nchw = np.transpose(face_float, (2, 0, 1))[np.newaxis, ...]

            inputs = {
                self._input_names[0]: face_nchw,
                self._input_names[1]: self._source_embedding,
            }
            output_tensors = self._session.run(self._output_names, inputs)
            output_tensor = output_tensors[0][0]  # Shape (3, 128, 128)

            # Transpose back to (128, 128, 3) RGB and scale to uint8 [0, 255]
            swapped_face_rgb = np.transpose(output_tensor, (1, 2, 0)) * 255.0
            swapped_face_rgb = np.clip(swapped_face_rgb, 0, 255).astype(np.uint8)

            # Convert RGB back to BGR for OpenCV pipeline
            if HAS_CV2:
                swapped_face_bgr = cv2.cvtColor(swapped_face_rgb, cv2.COLOR_RGB2BGR)
            else:
                swapped_face_bgr = swapped_face_rgb

            return swapped_face_bgr

        except Exception as e:
            logger.error(f"Inference error in InSwapper: {e}")
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.critical("CUDA OUT OF MEMORY during InSwapper inference!")
            return aligned_face_128

    def get_model_info(self) -> Dict[str, Any]:
        precision = "FP16 (Ultra-Fast)" if "fp16" in self._model_path.lower() else "FP32"
        return {
            "name": f"InSwapper-128 {precision}",
            "path": self._model_path,
            "provider": self._provider,
            "loaded": self._is_loaded,
            "has_source": self._source_embedding is not None,
            "has_arcface": self.arcface.is_loaded,
            "vram_estimate_mb": self._vram_estimate_mb if "CUDA" in self._provider else 0,
        }
