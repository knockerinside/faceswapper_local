"""Configuration management and presets for Real-Time Face-Swap."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

# Optional PyYAML import with fallback
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ProfileType(Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass
class CameraConfig:
    device_index: int = -1
    device_name: str = ""
    backend: str = "dshow"
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1


@dataclass
class DetectionConfig:
    model_path: str = "models/scrfd_10g_bnkps.onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    interval: int = 4
    execution_provider: str = "CUDAExecutionProvider"


@dataclass
class TrackingConfig:
    enabled: bool = True
    bbox_smoothing: float = 0.5
    landmark_smoothing: float = 0.5
    lost_threshold_frames: int = 5


@dataclass
class SwapConfig:
    enabled: bool = True
    model_path: str = "models/inswapper_128.onnx"
    source_image_path: str = "models/source/source.jpg"
    blend_strength: float = 1.0
    mask_feather: int = 15
    mask_erosion: int = 4
    mask_smoothing: float = 0.4
    color_correction: bool = True
    execution_provider: str = "CUDAExecutionProvider"


@dataclass
class ParsingConfig:
    enabled: bool = False
    model_path: str = "models/bisenet_face.onnx"
    input_size: Tuple[int, int] = (512, 512)
    interval: int = 3
    execution_provider: str = "CUDAExecutionProvider"


@dataclass
class HairConfig:
    enabled: bool = False
    mask_visualization: bool = False
    recolor_enabled: bool = False
    color_rgb: Tuple[int, int, int] = (180, 50, 50)
    strength: float = 0.5
    boundary_protection: bool = True


@dataclass
class AdaptiveConfig:
    enabled: bool = False
    target_fps: float = 30.0
    min_fps_threshold: float = 24.0
    cooldown_frames: int = 60


@dataclass
class OutputConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    virtual_camera_enabled: bool = False
    virtual_camera_device: str = "OBS Virtual Camera"


@dataclass
class DebugConfig:
    enabled: bool = False
    show_fps: bool = True
    show_bboxes: bool = True
    show_landmarks: bool = True
    show_masks: bool = False
    show_latency: bool = True
    send_overlays_to_vcam: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    swap: SwapConfig = field(default_factory=SwapConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    hair: HairConfig = field(default_factory=HairConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def apply_profile(self, profile: ProfileType) -> None:
        """Apply pre-configured optimization profiles without altering device choices."""
        if profile == ProfileType.PERFORMANCE:
            self.camera.width = 1280
            self.camera.height = 720
            self.camera.fps = 30
            self.detection.interval = 6
            self.detection.confidence_threshold = 0.55
            self.parsing.enabled = False
            self.hair.enabled = False
            self.swap.mask_feather = 10
            self.swap.mask_smoothing = 0.3
            self.swap.color_correction = True
            self.output.fps = 30
            logger.info("Applied Profile: PERFORMANCE (Detection interval: 6, Parsing: OFF)")

        elif profile == ProfileType.BALANCED:
            self.camera.width = 1280
            self.camera.height = 720
            self.camera.fps = 30
            self.detection.interval = 4
            self.detection.confidence_threshold = 0.5
            self.parsing.enabled = False
            self.hair.enabled = False
            self.swap.mask_feather = 15
            self.swap.mask_smoothing = 0.4
            self.swap.color_correction = True
            self.output.fps = 30
            logger.info("Applied Profile: BALANCED (Default safe 720p30)")

        elif profile == ProfileType.QUALITY:
            self.camera.width = 1280
            self.camera.height = 720
            self.camera.fps = 30
            self.detection.interval = 2
            self.detection.confidence_threshold = 0.45
            self.parsing.enabled = True
            self.hair.enabled = True
            self.swap.mask_feather = 20
            self.swap.mask_smoothing = 0.5
            self.swap.color_correction = True
            self.output.fps = 30
            logger.info("Applied Profile: QUALITY (Detection interval: 2, Parsing: ON, Hair: ON)")


class ConfigManager:
    @staticmethod
    def load(file_path: str = "config/default.yaml") -> AppConfig:
        candidates = [
            Path(file_path),
            Path(__file__).resolve().parents[1] / file_path,
            Path(__file__).resolve().parents[2] / file_path,
            Path("realtime_faceswap") / file_path,
        ]
        path = None
        for cand in candidates:
            if cand.exists():
                path = cand
                break

        config = AppConfig()

        if path is None:
            logger.warning(f"Config file {file_path} not found in search paths. Returning safe default config.")
            return config

        try:
            content = path.read_text(encoding="utf-8")
            data: Dict[str, Any] = {}
            if HAS_YAML:
                data = yaml.safe_load(content) or {}
            else:
                # Basic JSON fallback if content is valid JSON
                try:
                    data = json.loads(content)
                except Exception:
                    pass

            if "camera" in data:
                for k, v in data["camera"].items():
                    if hasattr(config.camera, k):
                        setattr(config.camera, k, v)
            if "detection" in data:
                for k, v in data["detection"].items():
                    if hasattr(config.detection, k):
                        setattr(config.detection, k, v)
            if "tracking" in data:
                for k, v in data["tracking"].items():
                    if hasattr(config.tracking, k):
                        setattr(config.tracking, k, v)
            if "swap" in data:
                for k, v in data["swap"].items():
                    if hasattr(config.swap, k):
                        setattr(config.swap, k, v)
            if "parsing" in data:
                for k, v in data["parsing"].items():
                    if hasattr(config.parsing, k):
                        setattr(config.parsing, k, v)
            if "hair" in data:
                for k, v in data["hair"].items():
                    if hasattr(config.hair, k):
                        setattr(config.hair, k, v)
            if "adaptive" in data:
                for k, v in data["adaptive"].items():
                    if hasattr(config.adaptive, k):
                        setattr(config.adaptive, k, v)
            if "output" in data:
                for k, v in data["output"].items():
                    if hasattr(config.output, k):
                        setattr(config.output, k, v)
            if "debug" in data:
                for k, v in data["debug"].items():
                    if hasattr(config.debug, k):
                        setattr(config.debug, k, v)

            logger.info(f"Loaded config from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load config {file_path}: {e}. Using defaults.")

        return config

    @staticmethod
    def save(config: AppConfig, file_path: str = "config/default.yaml") -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(config)
        if HAS_YAML:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        logger.info(f"Saved configuration to {file_path}")
