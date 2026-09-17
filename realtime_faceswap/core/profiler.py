"""High-resolution performance profiler and latency telemetry.

Requirements (Sections 23, 48, 58, 59, 77):
- Explicitly separates camera capture latency from AI latency.
- Measures individual pipeline stages:
  1. Capture
  2. Preprocessing
  3. Face Detection (SCRFD)
  4. Face Tracking
  5. Face Alignment
  6. InSwapper Face Swap
  7. BiSeNet Parsing
  8. Compositing / Blending
  9. Output / Virtual Camera
- Computes metrics: Mean, P50 (Median), P95, FPS, and dropped frames.
- Reusable rolling window to avoid memory growth.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    last_ms: float = 0.0


@dataclass
class ProfilerSummary:
    fps_capture: float = 0.0
    fps_processing: float = 0.0
    fps_output: float = 0.0
    total_latency_ms: float = 0.0
    ai_latency_ms: float = 0.0
    camera_latency_ms: float = 0.0
    output_latency_ms: float = 0.0
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    dropped_frames: int = 0
    drop_rate_pct: float = 0.0


class PerformanceProfiler:
    """Zero-allocation rolling-window timer for latency benchmarking."""

    STAGE_NAMES = [
        "capture",
        "preprocessing",
        "detection",
        "tracking",
        "alignment",
        "face_swap",
        "parsing",
        "compositing",
        "output",
        "gpu_transfer",
    ]

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self._history: Dict[str, deque] = {
            stage: deque(maxlen=window_size) for stage in self.STAGE_NAMES
        }
        self._stage_start_times: Dict[str, float] = {}

        # Frame counters for FPS calculations
        self._capture_timestamps = deque(maxlen=window_size)
        self._process_timestamps = deque(maxlen=window_size)
        self._output_timestamps = deque(maxlen=window_size)

        self.dropped_frames: int = 0
        self.total_captured: int = 0

    def start_stage(self, stage_name: str) -> None:
        self._stage_start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        start = self._stage_start_times.pop(stage_name, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if stage_name in self._history:
            self._history[stage_name].append(elapsed_ms)
        return elapsed_ms

    def record_stage_time(self, stage_name: str, duration_ms: float) -> None:
        if stage_name in self._history:
            self._history[stage_name].append(duration_ms)

    def record_capture_event(self) -> None:
        now = time.perf_counter()
        self._capture_timestamps.append(now)
        self.total_captured += 1

    def record_process_event(self) -> None:
        now = time.perf_counter()
        self._process_timestamps.append(now)

    def record_output_event(self) -> None:
        now = time.perf_counter()
        self._output_timestamps.append(now)

    def record_dropped_frame(self) -> None:
        self.dropped_frames += 1

    @staticmethod
    def _calculate_fps(timestamps: deque) -> float:
        if len(timestamps) < 2:
            return 0.0
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(timestamps) - 1) / duration

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_v = sorted(values)
        idx = (len(sorted_v) - 1) * p
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return sorted_v[int(idx)]
        return sorted_v[lower] * (upper - idx) + sorted_v[upper] * (idx - lower)

    def get_summary(self) -> ProfilerSummary:
        stages_metrics: Dict[str, StageMetrics] = {}
        for stage, history in self._history.items():
            if not history:
                stages_metrics[stage] = StageMetrics()
                continue
            vals = list(history)
            mean_v = sum(vals) / len(vals)
            p50_v = self._percentile(vals, 0.50)
            p95_v = self._percentile(vals, 0.95)
            last_v = vals[-1]
            stages_metrics[stage] = StageMetrics(
                mean_ms=round(mean_v, 2),
                p50_ms=round(p50_v, 2),
                p95_ms=round(p95_v, 2),
                last_ms=round(last_v, 2),
            )

        fps_cap = round(self._calculate_fps(self._capture_timestamps), 1)
        fps_proc = round(self._calculate_fps(self._process_timestamps), 1)
        fps_out = round(self._calculate_fps(self._output_timestamps), 1)

        cap_lat = stages_metrics.get("capture", StageMetrics()).mean_ms
        out_lat = stages_metrics.get("output", StageMetrics()).mean_ms

        ai_stages = [
            "preprocessing",
            "detection",
            "tracking",
            "alignment",
            "face_swap",
            "parsing",
            "compositing",
        ]
        ai_lat = sum(stages_metrics.get(st, StageMetrics()).mean_ms for st in ai_stages)
        total_lat = cap_lat + ai_lat + out_lat

        drop_rate = 0.0
        if self.total_captured > 0:
            drop_rate = round((self.dropped_frames / self.total_captured) * 100.0, 1)

        return ProfilerSummary(
            fps_capture=fps_cap,
            fps_processing=fps_proc,
            fps_output=fps_out,
            total_latency_ms=round(total_lat, 2),
            ai_latency_ms=round(ai_lat, 2),
            camera_latency_ms=round(cap_lat, 2),
            output_latency_ms=round(out_lat, 2),
            stages=stages_metrics,
            dropped_frames=self.dropped_frames,
            drop_rate_pct=drop_rate,
        )

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()
        self._capture_timestamps.clear()
        self._process_timestamps.clear()
        self._output_timestamps.clear()
        self.dropped_frames = 0
        self.total_captured = 0
