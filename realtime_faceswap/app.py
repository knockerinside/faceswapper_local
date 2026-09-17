"""Main Application Entry Point for Real-Time Face-Swap.

Usage:
    python app.py [--config config/default.yaml] [--profile performance|balanced|quality]
"""
import sys
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root and module folder to sys.path so it runs seamlessly from anywhere
_current_file = Path(__file__).resolve()
_realtime_faceswap_dir = _current_file.parent
_project_root = _realtime_faceswap_dir.parent
for _p in [str(_project_root), str(_realtime_faceswap_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _check_runtime_dependencies():
    packages = [
        ("numpy", "numpy<2.0.0"),
        ("cv2", "opencv-python"),
        ("onnxruntime", "onnxruntime-gpu"),
        ("yaml", "PyYAML"),
        ("PIL", "pillow"),
    ]
    missing = []
    for mod, pkg in packages:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        import subprocess
        print(f"[*] Auto-installing missing packages {missing} into active Python: {sys.executable}...")
        req_file = _project_root / "requirements.txt"
        if not req_file.exists():
            req_file = _realtime_faceswap_dir / "requirements.txt"
        try:
            try:
                import pip
            except ImportError:
                subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if req_file.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        except Exception as e:
            print(f"[ERROR] Could not install packages: {e}")
            print(f"Please run: {sys.executable} -m pip install -r requirements.txt")

_check_runtime_dependencies()

from realtime_faceswap.core.config import ConfigManager, ProfileType
from realtime_faceswap.core.pipeline import FaceSwapPipeline

# Setup Rotating File Logging (Section 31)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "realtime_faceswap.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("App")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Real-Time Face-Swap Engine (RTX 4050)")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--profile", choices=["performance", "balanced", "quality"], default=None)
    parser.add_argument("--headless", action="store_true", help="Run without PySide6 GUI")
    args = parser.parse_args()

    logger.info("Starting Real-Time AI Face-Swap Engine...")

    # 1. Load Configuration
    config = ConfigManager.load(args.config)
    if args.profile:
        config.apply_profile(ProfileType(args.profile))

    # 2. Instantiate Master Pipeline
    pipeline = FaceSwapPipeline(config)

    # 3. Launch GUI or Headless Mode
    if args.headless:
        logger.info("Running in headless benchmark mode.")
        pipeline.initialize_models()
        pipeline.start(device_index=0)
        try:
            import time
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pipeline.stop()
            return 0

    try:
        from PySide6.QtWidgets import QApplication
        from realtime_faceswap.ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("Real-Time Face-Swap")
        window = MainWindow(config, pipeline)
        window.show()
        return app.exec()
    except ImportError:
        logger.warning("PySide6 is not installed. To run the desktop GUI: pip install PySide6")
        return 1


if __name__ == "__main__":
    sys.exit(main())
