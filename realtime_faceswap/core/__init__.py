"""Core pipeline components for real-time face-swap."""
from .events import EventBus, EventType, PipelineEvent
from .config import AppConfig, ConfigManager, ProfileType
from .frame_buffer import LatestFrameBuffer, FrameMetadata
from .profiler import PerformanceProfiler
from .pipeline import FaceSwapPipeline

__all__ = [
    "EventBus",
    "EventType",
    "PipelineEvent",
    "AppConfig",
    "ConfigManager",
    "ProfileType",
    "LatestFrameBuffer",
    "FrameMetadata",
    "PerformanceProfiler",
    "FaceSwapPipeline",
]
