"""Interactive Model Manager and Downloader Dialog for PySide6.

Hardware Target: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
Features:
- Visual comparison cards showing FP16 vs FP32 swapper, detectors, ArcFace, and GFPGAN
- Detailed breakdown of advantages, latency, FPS, and VRAM footprints
- Asynchronous background model downloads with live progress bar and speed display
- 1-Click "Set as Active Model" to switch active model in the running pipeline
- 1-Click "Download Recommended Suite (RTX 4050)" button
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import threading
import subprocess
import os
import sys

from realtime_faceswap.tools.download_models import (
    MODEL_CATALOG,
    get_models_dir,
    check_model_installed,
    download_model,
)

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
        QProgressBar, QScrollArea, QWidget, QFrame, QTabWidget, QMessageBox,
        QSplitter
    )
    from PySide6.QtCore import Qt, Signal, QObject
    from PySide6.QtGui import QColor, QFont
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    class QDialog: pass
    class QObject: pass


class DownloadSignals(QObject):
    """Safe cross-thread signals for GUI updates during download."""
    progress = Signal(str, int, int, float, str)  # key, downloaded, total, speed, status
    finished = Signal(str, bool, str)              # key, success, message


class ModelCard(QFrame):
    """Individual card displaying model metadata, advantages, and download controls."""

    def __init__(self, key: str, info: Dict[str, Any], parent_dialog, pipeline=None) -> None:
        super().__init__()
        self.key = key
        self.info = info
        self.parent_dialog = parent_dialog
        self.pipeline = pipeline

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ModelCard {
                background-color: #131d2e;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 12px;
            }
            ModelCard:hover {
                border: 1px solid #3b82f6;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header: Name, Category, Status Badge
        hdr_layout = QHBoxLayout()
        self.lbl_title = QLabel(info["name"])
        self.lbl_title.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold;")
        hdr_layout.addWidget(self.lbl_title)

        hdr_layout.addStretch()

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        hdr_layout.addWidget(self.lbl_status)
        layout.addLayout(hdr_layout)

        # Metrics Bar: Size, Precision, Speed on RTX 4050, Est. VRAM
        metrics_layout = QHBoxLayout()
        lbl_specs = QLabel(
            f"File: <b>{info['size_mb']} MB</b> ({info['precision']}) | "
            f"RTX 4050 Latency: <b style='color:#38bdf8;'>{info['speed_rtx4050']}</b> | "
            f"Est. VRAM: <b style='color:#a78bfa;'>{info['vram_rtx4050']}</b>"
        )
        lbl_specs.setStyleSheet("color: #94a3b8; font-size: 12px;")
        metrics_layout.addWidget(lbl_specs)
        layout.addLayout(metrics_layout)

        # Recommended for line
        lbl_rec = QLabel(f"<b>Best For:</b> {info['recommended_for']}")
        lbl_rec.setStyleSheet("color: #e2e8f0; font-size: 12px;")
        layout.addWidget(lbl_rec)

        # Advantages bullet points
        adv_text = "<b>Advantages:</b><ul style='margin: 2px 0 0 0; padding-left: 18px;'>"
        for adv in info["advantages"]:
            adv_text += f"<li style='color: #cbd5e1;'>{adv}</li>"
        adv_text += "</ul>"
        lbl_adv = QLabel(adv_text)
        lbl_adv.setWordWrap(True)
        lbl_adv.setStyleSheet("font-size: 11px;")
        layout.addWidget(lbl_adv)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
                height: 16px;
                text-align: center;
                color: white;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status subtext
        self.lbl_subtext = QLabel("")
        self.lbl_subtext.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.lbl_subtext)

        # Actions
        actions_layout = QHBoxLayout()
        self.btn_download = QPushButton("Download Model")
        self.btn_download.setStyleSheet("background-color: #2563eb; color: white; border-radius: 4px; padding: 6px 12px; font-weight: 600;")
        self.btn_download.clicked.connect(self._on_download_clicked)
        actions_layout.addWidget(self.btn_download)

        self.btn_activate = QPushButton("Set as Active Model")
        self.btn_activate.setStyleSheet("background-color: #059669; color: white; border-radius: 4px; padding: 6px 12px; font-weight: 600;")
        self.btn_activate.clicked.connect(self._on_activate_clicked)
        actions_layout.addWidget(self.btn_activate)

        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        self.refresh_status()

    def refresh_status(self) -> None:
        """Updates UI status according to whether model file is present and active."""
        is_inst, path, sz = check_model_installed(self.key)
        is_active = self._check_if_active()

        if is_active:
            self.lbl_status.setText("ACTIVE IN ENGINE")
            self.lbl_status.setStyleSheet("background-color: #065f46; color: #6ee7b7; border: 1px solid #059669; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            self.btn_activate.setEnabled(False)
            self.btn_activate.setText("Active Model")
            self.btn_download.setText("Re-Download")
            self.btn_download.setStyleSheet("background-color: #334155; color: #cbd5e1; border-radius: 4px; padding: 6px 12px; font-size: 11px;")
        elif is_inst:
            self.lbl_status.setText(f"READY ({sz/(1024*1024):.0f} MB)")
            self.lbl_status.setStyleSheet("background-color: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            self.btn_activate.setEnabled(True)
            self.btn_activate.setText("Use This Model")
            self.btn_activate.setStyleSheet("background-color: #059669; color: white; border-radius: 4px; padding: 6px 12px; font-weight: 600;")
            self.btn_download.setText("Re-Download")
            self.btn_download.setStyleSheet("background-color: #334155; color: #cbd5e1; border-radius: 4px; padding: 6px 12px; font-size: 11px;")
        else:
            req_tag = "REQUIRED" if self.info["required"] else "OPTIONAL"
            self.lbl_status.setText(f"NOT INSTALLED ({req_tag})")
            self.lbl_status.setStyleSheet("background-color: #7f1d1d; color: #fecaca; border: 1px solid #dc2626; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
            self.btn_activate.setEnabled(False)
            self.btn_activate.setText("Use This Model")
            self.btn_activate.setStyleSheet("background-color: #334155; color: #64748b; border-radius: 4px; padding: 6px 12px;")
            self.btn_download.setText("Download Model")
            self.btn_download.setStyleSheet("background-color: #2563eb; color: white; border-radius: 4px; padding: 6px 12px; font-weight: 600;")

    def _check_if_active(self) -> bool:
        if not self.pipeline:
            return False
        fn = self.info["file_name"].lower()
        if "swapper" in self.info["category"].lower():
            active_p = str(getattr(self.pipeline.swapper, "_model_path", "")).lower()
            return fn in active_p
        elif "detector" in self.info["category"].lower():
            active_p = str(getattr(self.pipeline.detector, "_model_path", "")).lower()
            return fn in active_p
        elif "segmentation" in self.info["category"].lower():
            active_p = str(getattr(self.pipeline.parser, "_model_path", "")).lower()
            return fn in active_p
        return False

    def _on_download_clicked(self) -> None:
        self.parent_dialog.start_download(self.key)

    def _on_activate_clicked(self) -> None:
        if not self.pipeline:
            return
        fn = self.info["file_name"]
        models_dir = get_models_dir()
        target_path = str(models_dir / fn)

        if "swapper" in self.info["category"].lower():
            self.pipeline.config.swap.model_path = target_path
            self.pipeline.swapper.load(target_path, self.pipeline.config.swap.execution_provider)
            QMessageBox.information(self, "Swapper Activated", f"Switched active face swapper to:\n{self.info['name']}")
        elif "detector" in self.info["category"].lower():
            self.pipeline.config.detection.model_path = target_path
            self.pipeline.detector.load(target_path, self.pipeline.config.detection.execution_provider)
            QMessageBox.information(self, "Detector Activated", f"Switched active face detector to:\n{self.info['name']}")
        elif "segmentation" in self.info["category"].lower():
            self.pipeline.config.parsing.model_path = target_path
            self.pipeline.parser.load(target_path, self.pipeline.config.parsing.execution_provider)
            QMessageBox.information(self, "Parser Activated", f"Switched active parser to:\n{self.info['name']}")

        self.parent_dialog.refresh_all_cards()


class ModelManagerDialog(QDialog):
    """Complete, interactive Model Manager & Downloader Dialog."""

    def __init__(self, models_info: Optional[List[Dict[str, Any]]] = None, parent=None, pipeline=None) -> None:
        if not HAS_PYSIDE:
            return

        super().__init__(parent)
        self.pipeline = pipeline
        self.signals = DownloadSignals()
        self.signals.progress.connect(self._on_progress_update)
        self.signals.finished.connect(self._on_download_finished)

        self.setWindowTitle("AI Model Selection & Download Manager — NVIDIA RTX 4050")
        self.resize(880, 680)
        self.setStyleSheet("""
            QDialog {
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Segoe UI', system-ui;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QPushButton {
                font-family: 'Segoe UI', system-ui;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header Bar
        hdr_frame = QFrame()
        hdr_frame.setStyleSheet("background-color: #1e293b; border-radius: 8px; padding: 10px;")
        hdr_layout = QHBoxLayout(hdr_frame)

        info_box = QVBoxLayout()
        lbl_title = QLabel("NVIDIA RTX 4050 Model Center & Downloader")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        lbl_sub = QLabel("Select and download precision-optimized ONNX models with real-time performance profiles.")
        lbl_sub.setStyleSheet("font-size: 12px; color: #94a3b8;")
        info_box.addWidget(lbl_title)
        info_box.addWidget(lbl_sub)
        hdr_layout.addLayout(info_box)

        hdr_layout.addStretch()

        # Quick Actions in Header
        self.btn_download_rec = QPushButton("⚡ Download Recommended Suite (~446 MB)")
        self.btn_download_rec.setStyleSheet("background-color: #2563eb; color: white; padding: 8px 14px; border-radius: 6px; font-weight: bold;")
        self.btn_download_rec.clicked.connect(self._download_recommended_suite)
        hdr_layout.addWidget(self.btn_download_rec)

        self.btn_open_folder = QPushButton("📂 Open Folder")
        self.btn_open_folder.setStyleSheet("background-color: #334155; color: white; padding: 8px 12px; border-radius: 6px; font-weight: 600;")
        self.btn_open_folder.clicked.connect(self._open_models_folder)
        hdr_layout.addWidget(self.btn_open_folder)

        layout.addWidget(hdr_frame)

        # Scroll Area for Model Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.cards_layout = QVBoxLayout(scroll_content)
        self.cards_layout.setSpacing(10)

        self.card_widgets: Dict[str, ModelCard] = {}
        for key, info in MODEL_CATALOG.items():
            card = ModelCard(key, info, self, pipeline=self.pipeline)
            self.card_widgets[key] = card
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Footer / Close
        ftr_layout = QHBoxLayout()
        self.lbl_global_status = QLabel("Ready.")
        self.lbl_global_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        ftr_layout.addWidget(self.lbl_global_status)

        ftr_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("background-color: #334155; color: white; padding: 8px 20px; border-radius: 6px; font-weight: 600;")
        btn_close.clicked.connect(self.accept)
        ftr_layout.addWidget(btn_close)

        layout.addLayout(ftr_layout)

    def refresh_all_cards(self) -> None:
        for card in self.card_widgets.values():
            card.refresh_status()

    def start_download(self, key: str) -> None:
        card = self.card_widgets.get(key)
        if card:
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            card.btn_download.setEnabled(False)
            card.lbl_subtext.setText("Connecting to mirrors...")

        self.lbl_global_status.setText(f"Downloading {MODEL_CATALOG[key]['name']}...")

        def _worker():
            def _cb(downloaded, total, speed, status):
                self.signals.progress.emit(key, downloaded, total, speed, status)

            ok = download_model(key, progress_callback=_cb, force=True)
            msg = "Download Complete" if ok else "Download Failed"
            self.signals.finished.emit(key, ok, msg)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _download_recommended_suite(self) -> None:
        self.btn_download_rec.setEnabled(False)
        keys_to_dl = ["scrfd_10g", "inswapper_fp16", "arcface"]

        def _suite_worker():
            for k in keys_to_dl:
                is_inst, _, _ = check_model_installed(k)
                if not is_inst:
                    self.start_download(k)
                    # Simple wait for completion
                    while True:
                        time.sleep(0.5)
                        inst_now, _, _ = check_model_installed(k)
                        if inst_now:
                            break

        threading.Thread(target=_suite_worker, daemon=True).start()

    def _on_progress_update(self, key: str, downloaded: int, total: int, speed: float, status: str) -> None:
        card = self.card_widgets.get(key)
        if not card:
            return

        dl_mb = downloaded / (1024 * 1024)
        tot_mb = total / (1024 * 1024) if total > 0 else 0

        if total > 0:
            pct = int((downloaded / total) * 100)
            card.progress_bar.setValue(pct)
            card.lbl_subtext.setText(f"{dl_mb:.1f} / {tot_mb:.1f} MB ({pct}%) @ {speed:.1f} MB/s")
        else:
            card.lbl_subtext.setText(f"{dl_mb:.1f} MB downloaded @ {speed:.1f} MB/s")

    def _on_download_finished(self, key: str, success: bool, message: str) -> None:
        card = self.card_widgets.get(key)
        if card:
            card.btn_download.setEnabled(True)
            card.lbl_subtext.setText(message)
            card.refresh_status()
            if success:
                card.progress_bar.setValue(100)
                QTimer_single_shot = getattr(card, "timer", None)

        self.btn_download_rec.setEnabled(True)
        self.lbl_global_status.setText(f"{MODEL_CATALOG[key]['name']}: {message}")

    def _open_models_folder(self) -> None:
        models_dir = get_models_dir().resolve()
        try:
            if sys.platform == "win32":
                os.startfile(str(models_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(models_dir)])
            else:
                subprocess.Popen(["xdg-open", str(models_dir)])
        except Exception as e:
            QMessageBox.information(self, "Models Directory", f"Models are located at:\n{models_dir}")
