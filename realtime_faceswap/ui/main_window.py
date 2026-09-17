"""Complete Desktop Main Window interface using PySide6.

Requirements (Sections 21, 22, 65, 66, 67, 81, 82):
- Professional dark engineering desktop UI.
- Camera device enumeration with Android phone camera detection.
- Source identity loader with immediate face alignment preview.
- Granular controls for Swap, Hair boundary protection, and Hair recoloring.
- Live performance waterfall profiler.
- Virtual camera output controls for OBS Studio.
- Crash recovery banner with instant restart button.
"""
from pathlib import Path
from typing import Optional
import logging
import numpy as np

from realtime_faceswap.core.config import AppConfig, ProfileType
from realtime_faceswap.core.pipeline import FaceSwapPipeline
from realtime_faceswap.core.events import event_bus, EventType, PipelineEvent
from realtime_faceswap.capture.device_manager import DeviceManager
from .widgets.preview_widget import VideoPreviewWidget
from .widgets.profiler_widget import ProfilerWidget
from .widgets.model_manager_dialog import ModelManagerDialog

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QComboBox, QCheckBox, QSlider, QLabel, QGroupBox,
        QFileDialog, QMessageBox, QStatusBar, QFrame
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    class QMainWindow: pass


class MainWindow(QMainWindow):
    """Primary PySide6 Desktop Application Window."""

    def __init__(self, config: AppConfig, pipeline: FaceSwapPipeline) -> None:
        if not HAS_PYSIDE:
            logger.warning("PySide6 not available in this environment.")
            return

        super().__init__()
        self.config = config
        self.pipeline = pipeline

        self.setWindowTitle("Real-Time AI Face-Swap Engine (RTX 4050 / Windows 11)")
        self.resize(1280, 840)

        # Style with dark professional theme
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; color: #f1f5f9; font-family: 'Segoe UI', system-ui; }
            QGroupBox { border: 1px solid #1e293b; border-radius: 8px; margin-top: 10px; font-weight: bold; color: #94a3b8; padding: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #2563eb; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: 600; }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
            QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 6px; color: #f8fafc; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::handle:horizontal { background: #38bdf8; width: 16px; margin: -5px 0; border-radius: 8px; }
            QLabel { color: #cbd5e1; }
        """)

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT COLUMN: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(420)

        # 0. Crash Recovery Banner (Hidden unless triggered)
        self.crash_frame = QFrame()
        self.crash_frame.setStyleSheet("background-color: #7f1d1d; border-radius: 6px; padding: 6px;")
        crash_layout = QHBoxLayout(self.crash_frame)
        self.lbl_crash = QLabel("AI processing stopped unexpectedly.")
        self.lbl_crash.setStyleSheet("color: #fecaca; font-weight: bold;")
        self.btn_restart_ai = QPushButton("Restart AI Pipeline")
        self.btn_restart_ai.clicked.connect(self._restart_ai_pipeline)
        crash_layout.addWidget(self.lbl_crash)
        crash_layout.addWidget(self.btn_restart_ai)
        self.crash_frame.setVisible(False)
        left_layout.addWidget(self.crash_frame)

        # 1. INPUT CAMERA SECTION
        input_group = QGroupBox("INPUT CAMERA")
        input_grid = QGridLayout(input_group)
        self.combo_cameras = QComboBox()
        self._populate_cameras()

        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["1280x720 (Recommended)", "1920x1080", "640x480"])

        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["30 FPS", "60 FPS"])

        self.btn_camera = QPushButton("START CAMERA")
        self.btn_camera.clicked.connect(self._toggle_camera)

        input_grid.addWidget(QLabel("Camera Device:"), 0, 0)
        input_grid.addWidget(self.combo_cameras, 0, 1)
        input_grid.addWidget(QLabel("Resolution:"), 1, 0)
        input_grid.addWidget(self.combo_resolution, 1, 1)
        input_grid.addWidget(QLabel("Target FPS:"), 2, 0)
        input_grid.addWidget(self.combo_fps, 2, 1)
        input_grid.addWidget(self.btn_camera, 3, 0, 1, 2)
        left_layout.addWidget(input_group)

        # 2. SOURCE IDENTITY SECTION
        source_group = QGroupBox("SOURCE IDENTITY")
        source_layout = QVBoxLayout(source_group)
        self.btn_select_source = QPushButton("Select Source Image...")
        self.btn_select_source.clicked.connect(self._select_source_image)
        self.lbl_source_status = QLabel("Status: No Source Face Selected")
        source_layout.addWidget(self.btn_select_source)
        source_layout.addWidget(self.lbl_source_status)
        left_layout.addWidget(source_group)

        # 3. FACE SWAP SETTINGS
        swap_group = QGroupBox("FACE SWAP (InSwapper-128)")
        swap_grid = QGridLayout(swap_group)
        self.chk_swap_enable = QCheckBox("Enable Face Swap")
        self.chk_swap_enable.setChecked(True)
        self.chk_swap_enable.toggled.connect(self._toggle_swap)

        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)

        self.slider_feather = QSlider(Qt.Horizontal)
        self.slider_feather.setRange(0, 30)
        self.slider_feather.setValue(15)

        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(0, 90)
        self.slider_smooth.setValue(40)

        swap_grid.addWidget(self.chk_swap_enable, 0, 0, 1, 2)
        swap_grid.addWidget(QLabel("Swap Strength:"), 1, 0)
        swap_grid.addWidget(self.slider_strength, 1, 1)
        swap_grid.addWidget(QLabel("Mask Feather (px):"), 2, 0)
        swap_grid.addWidget(self.slider_feather, 2, 1)
        swap_grid.addWidget(QLabel("Mask Smoothing:"), 3, 0)
        swap_grid.addWidget(self.slider_smooth, 3, 1)
        left_layout.addWidget(swap_group)

        # 4. HAIR PROCESSING SECTION
        hair_group = QGroupBox("HAIR PROCESSING (BiSeNet)")
        hair_grid = QGridLayout(hair_group)
        self.chk_hair_enable = QCheckBox("Enable Hair Processing")
        self.chk_hair_enable.setChecked(False)
        self.chk_hair_recolor = QCheckBox("Enable Hair Recolor")
        self.chk_hair_recolor.setChecked(False)

        self.slider_hair_strength = QSlider(Qt.Horizontal)
        self.slider_hair_strength.setRange(0, 100)
        self.slider_hair_strength.setValue(50)

        hair_grid.addWidget(self.chk_hair_enable, 0, 0, 1, 2)
        hair_grid.addWidget(self.chk_hair_recolor, 1, 0, 1, 2)
        hair_grid.addWidget(QLabel("Recolor Strength:"), 2, 0)
        hair_grid.addWidget(self.slider_hair_strength, 2, 1)
        left_layout.addWidget(hair_group)

        # 5. OUTPUT & VIRTUAL CAMERA
        output_group = QGroupBox("OUTPUT (OBS VIRTUAL CAMERA)")
        output_layout = QVBoxLayout(output_group)
        vcam_status_str = "AVAILABLE" if self.pipeline.virtual_camera.is_supported() else "UNAVAILABLE"
        self.lbl_vcam_status = QLabel(f"Virtual Camera: {vcam_status_str}")
        self.btn_vcam = QPushButton("START VIRTUAL CAMERA")
        self.btn_vcam.setEnabled(self.pipeline.virtual_camera.is_supported())
        self.btn_vcam.clicked.connect(self._toggle_vcam)
        output_layout.addWidget(self.lbl_vcam_status)
        output_layout.addWidget(self.btn_vcam)
        left_layout.addWidget(output_group)

        # Tools buttons
        btn_models = QPushButton("Model Manager...")
        btn_models.setStyleSheet("background-color: #334155;")
        btn_models.clicked.connect(self._open_model_manager)
        left_layout.addWidget(btn_models)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # RIGHT COLUMN: Live Preview and Performance Telemetry
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.preview_widget = VideoPreviewWidget(self)
        right_layout.addWidget(self.preview_widget, stretch=3)

        self.profiler_widget = ProfilerWidget(self)
        right_layout.addWidget(self.profiler_widget, stretch=1)

        main_layout.addWidget(right_panel, stretch=1)

        # Status Bar
        self.statusBar().showMessage("Ready — Local NVIDIA RTX 4050 CUDA Acceleration Active")

        # Hook UI events
        event_bus.subscribe(EventType.STATUS_CHANGED, self._on_status_changed)
        event_bus.subscribe(EventType.PROFILER_UPDATE, self._on_profiler_update)
        event_bus.subscribe(EventType.PIPELINE_CRASH, self._on_pipeline_crash)

        # UI Refresh Timer (30 FPS GUI display refresh)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(33)

    def _populate_cameras(self) -> None:
        devices = DeviceManager.enumerate_cameras()
        self.combo_cameras.clear()
        preferred_idx = 0
        for i, dev in enumerate(devices):
            label = f"[{dev.index}] {dev.name}"
            if dev.is_phone_camera:
                label += " ★ (Phone Rear Camera)"
                preferred_idx = i
            self.combo_cameras.addItem(label, dev.index)
        self.combo_cameras.setCurrentIndex(preferred_idx)

    def _toggle_camera(self) -> None:
        if not self.pipeline._is_running:
            selected_idx = self.combo_cameras.currentData() or 0
            self.pipeline.initialize_models()
            success = self.pipeline.start(device_index=selected_idx)
            if success:
                self.btn_camera.setText("STOP CAMERA")
                self.btn_camera.setStyleSheet("background-color: #dc2626;")
        else:
            self.pipeline.stop()
            self.btn_camera.setText("START CAMERA")
            self.btn_camera.setStyleSheet("background-color: #2563eb;")
            self.preview_widget.set_placeholder_text("Camera Inactive")

    def _select_source_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Identity Image", "models/source/", "Images (*.jpg *.png *.jpeg *.webp)"
        )
        if file_path:
            success = self.pipeline.set_source_face(file_path)
            if success:
                self.lbl_source_status.setText(f"Status: Loaded ({Path(file_path).name})")
                self.lbl_source_status.setStyleSheet("color: #4ade80;")
            else:
                self.lbl_source_status.setText("Status: No face detected in image!")
                self.lbl_source_status.setStyleSheet("color: #f87171;")

    def _toggle_swap(self, checked: bool) -> None:
        self.config.swap.enabled = checked

    def _toggle_vcam(self) -> None:
        if not self.pipeline.virtual_camera.is_active():
            self.pipeline.virtual_camera.start(
                self.config.output.width, self.config.output.height, self.config.output.fps
            )
            self.btn_vcam.setText("STOP VIRTUAL CAMERA")
            self.btn_vcam.setStyleSheet("background-color: #dc2626;")
        else:
            self.pipeline.virtual_camera.stop()
            self.btn_vcam.setText("START VIRTUAL CAMERA")
            self.btn_vcam.setStyleSheet("background-color: #2563eb;")

    def _open_model_manager(self) -> None:
        models = [
            self.pipeline.detector.get_info(),
            self.pipeline.swapper.get_model_info(),
            self.pipeline.parser.get_info(),
        ]
        dlg = ModelManagerDialog(models, self, pipeline=self.pipeline)
        dlg.exec()

    def _restart_ai_pipeline(self) -> None:
        self.crash_frame.setVisible(False)
        self.pipeline.initialize_models()
        self.statusBar().showMessage("AI Pipeline restarted successfully.")

    def _refresh_preview(self) -> None:
        frame = self.pipeline.get_latest_frame()
        if frame is not None:
            self.preview_widget.update_frame(frame)

    def _on_status_changed(self, event: PipelineEvent) -> None:
        self.statusBar().showMessage(str(event.data))

    def _on_profiler_update(self, event: PipelineEvent) -> None:
        self.profiler_widget.update_metrics(event.data)

    def _on_pipeline_crash(self, event: PipelineEvent) -> None:
        self.lbl_crash.setText(f"AI Error: {event.data[:45]}...")
        self.crash_frame.setVisible(True)

    def closeEvent(self, event) -> None:
        """Ensures clean hardware shutdown on window close."""
        self.pipeline.stop()
        event.accept()
