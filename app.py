"""Root Application Entry Point for Real-Time Face-Swap.
Delegates to realtime_faceswap.app.
"""
import sys
import os
import subprocess
import onnxruntime
onnxruntime.preload_dlls()
from pathlib import Path

# Add directories to sys.path
root_dir = Path(__file__).resolve().parent
realtime_faceswap_dir = root_dir / "realtime_faceswap"

for p in [str(root_dir), str(realtime_faceswap_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

def _ensure_dependencies():
    """Verify essential packages and auto-install into the active environment if missing."""
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
        print("=" * 68)
        print(" [!] Missing required dependencies in current Python environment:")
        for m in missing:
            print(f"     - {m}")
        print(f" [*] Active Python: {sys.executable}")
        print(" [*] Auto-installing missing packages via pip...")
        print("=" * 68)

        req_candidates = [
            root_dir / "requirements.txt",
            realtime_faceswap_dir / "requirements.txt",
            Path.cwd() / "requirements.txt",
        ]
        req_file = next((p for p in req_candidates if p.exists()), None)

        try:
            # If pip module is not loaded, try to bootstrap with ensurepip
            try:
                import pip
            except ImportError:
                subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if req_file and req_file.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("\n [OK] All dependencies successfully installed and verified!")
            print("=" * 68 + "\n")
        except Exception as e:
            print(f"\n [WARNING] Automatic dependency installation could not complete: {e}")
            print(f" If runtime errors occur, run: {sys.executable} -m pip install -r requirements.txt\n")

_ensure_dependencies()

from realtime_faceswap.app import main

if __name__ == "__main__":
    sys.exit(main())


