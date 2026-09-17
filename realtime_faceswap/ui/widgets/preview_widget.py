"""Preview canvas widget supporting Processed, Original, Side-by-side, and Mask views."""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    # Mock base class if PySide6 is not in environment
    class QWidget: pass


class VideoPreviewWidget(QWidget):
    """High-performance PySide6 video render canvas."""

    def __init__(self, parent=None) -> None:
        if HAS_PYSIDE:
            super().__init__(parent)
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)
            self.label = QLabel("Camera Preview Inactive")
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setStyleSheet(
                "background-color: #0f172a; color: #94a3b8; font-size: 14px; border: 1px solid #334155; border-radius: 8px;"
            )
            self.label.setMinimumSize(640, 360)
            self._layout.addWidget(self.label)
        self.preview_mode = "Processed"  # Processed, Original, Side-by-side, Mask

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        if not HAS_PYSIDE or frame_bgr is None:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        # Qt requires RGB
        rgb_data = frame_bgr[..., ::-1].copy()
        q_img = QImage(rgb_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled_pixmap)

    def set_placeholder_text(self, text: str) -> None:
        if HAS_PYSIDE:
            self.label.setText(text)
