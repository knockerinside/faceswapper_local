"""Performance Telemetry Table Widget.

Requirements (Sections 22, 23):
- Displays granular stage latencies: Capture, Detection, Tracking, Face Swap, Parsing, Compositing, Total.
- Displays Average, P50, P95, and dropped frame counts.
"""
from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QGridLayout, QLabel, QGroupBox
    )
    from PySide6.QtCore import Qt
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    class QWidget: pass


class ProfilerWidget(QWidget):
    """Real-time UI metrics table."""

    def __init__(self, parent=None) -> None:
        if HAS_PYSIDE:
            super().__init__(parent)
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)

            self.group = QGroupBox("Performance & Latency (RTX 4050 CUDA)")
            grid = QGridLayout(self.group)

            self.lbl_cap_fps = QLabel("0.0")
            self.lbl_proc_fps = QLabel("0.0")
            self.lbl_out_fps = QLabel("0.0")
            self.lbl_dropped = QLabel("0 (0.0%)")

            self.lbl_det_ms = QLabel("0.0 ms")
            self.lbl_track_ms = QLabel("0.0 ms")
            self.lbl_swap_ms = QLabel("0.0 ms")
            self.lbl_parse_ms = QLabel("0.0 ms")
            self.lbl_comp_ms = QLabel("0.0 ms")
            self.lbl_total_ms = QLabel("0.0 ms")

            # Layout metrics
            grid.addWidget(QLabel("Capture FPS:"), 0, 0)
            grid.addWidget(self.lbl_cap_fps, 0, 1)
            grid.addWidget(QLabel("AI FPS:"), 0, 2)
            grid.addWidget(self.lbl_proc_fps, 0, 3)

            grid.addWidget(QLabel("Output FPS:"), 1, 0)
            grid.addWidget(self.lbl_out_fps, 1, 1)
            grid.addWidget(QLabel("Dropped Frames:"), 1, 2)
            grid.addWidget(self.lbl_dropped, 1, 3)

            grid.addWidget(QLabel("Face Detection:"), 2, 0)
            grid.addWidget(self.lbl_det_ms, 2, 1)
            grid.addWidget(QLabel("Face Tracking:"), 2, 2)
            grid.addWidget(self.lbl_track_ms, 2, 3)

            grid.addWidget(QLabel("Face Swap (128x128):"), 3, 0)
            grid.addWidget(self.lbl_swap_ms, 3, 1)
            grid.addWidget(QLabel("Face Parsing:"), 3, 2)
            grid.addWidget(self.lbl_parse_ms, 3, 3)

            grid.addWidget(QLabel("Compositing:"), 4, 0)
            grid.addWidget(self.lbl_comp_ms, 4, 1)
            grid.addWidget(QLabel("Total Latency:"), 4, 2)
            grid.addWidget(self.lbl_total_ms, 4, 3)

            self._layout.addWidget(self.group)

    def update_metrics(self, summary: Any) -> None:
        if not HAS_PYSIDE or summary is None:
            return

        self.lbl_cap_fps.setText(f"{summary.fps_capture}")
        self.lbl_proc_fps.setText(f"{summary.fps_processing}")
        self.lbl_out_fps.setText(f"{summary.fps_output}")
        self.lbl_dropped.setText(f"{summary.dropped_frames} ({summary.drop_rate_pct}%)")

        stages = summary.stages
        det = stages.get("detection")
        if det: self.lbl_det_ms.setText(f"{det.mean_ms} ms (P95: {det.p95_ms})")
        track = stages.get("tracking")
        if track: self.lbl_track_ms.setText(f"{track.mean_ms} ms")
        swap = stages.get("face_swap")
        if swap: self.lbl_swap_ms.setText(f"{swap.mean_ms} ms (P95: {swap.p95_ms})")
        parse = stages.get("parsing")
        if parse: self.lbl_parse_ms.setText(f"{parse.mean_ms} ms")
        comp = stages.get("compositing")
        if comp: self.lbl_comp_ms.setText(f"{comp.mean_ms} ms")

        self.lbl_total_ms.setText(f"{summary.total_latency_ms} ms (AI: {summary.ai_latency_ms}ms)")
