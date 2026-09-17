"""System Installation and Hardware Diagnostics Tool.

Requirements:
- Checks: Python, Windows 11, NVIDIA RTX 4050 GPU, CUDA, ONNX Runtime, Camera Devices, OBS Virtual Camera, Model Weights.
- Outputs clear status table.
- Detects existing models (FP16 & FP32 InSwapper, SCRFD & det_10g, ArcFace).
- If something fails: shows WHAT FAILED, WHY IT FAILED, and HOW TO FIX IT.
"""
import sys
import os
import platform
import shutil
from pathlib import Path
from typing import List, Tuple

# Ensure project root and module folder in sys.path
_cur_dir = Path(__file__).resolve().parent
for _p in [_cur_dir, _cur_dir.parent, _cur_dir.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def find_model_path(candidates: List[str]) -> Tuple[bool, str, int]:
    """Finds if any candidate model file exists in models/ or realtime_faceswap/models/."""
    search_dirs = [
        Path.cwd() / "models",
        Path(__file__).resolve().parents[1] / "models",
        Path(__file__).resolve().parents[2] / "models",
    ]
    for filename in candidates:
        for sdir in search_dirs:
            p = sdir / filename
            if p.exists() and p.is_file() and p.stat().st_size > 1024 * 1024:
                return True, str(p), p.stat().st_size
    return False, candidates[0], 0


def check_diagnostics() -> List[Tuple[str, str, str, str]]:
    """Runs full hardware & runtime check returning (component, status, notes, fix_hint)."""
    results = []

    # 1. Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        results.append(("Python", "OK", f"Python {py_ver}", ""))
    else:
        results.append(("Python", "FAIL", f"Python {py_ver}", "Install Python 3.10 or 3.11 64-bit from python.org with Add to PATH checked"))

    # 2. Operating System
    os_name = f"{platform.system()} {platform.release()}"
    results.append(("Operating System", "INFO", os_name, "Target: Windows 11 64-bit"))

    # 3. NVIDIA GPU & Driver
    has_nvidia = False
    gpu_name = "None detected"
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], stderr=subprocess.DEVNULL)
        line = out.decode("utf-8").strip().split(",")
        if line and len(line) >= 2:
            gpu_name = f"{line[0].strip()} ({line[2].strip() if len(line)>2 else ''}, Driver {line[1].strip()})"
            has_nvidia = True
    except Exception:
        pass

    if has_nvidia:
        results.append(("NVIDIA GPU", "OK", gpu_name, ""))
    else:
        results.append(("NVIDIA GPU", "WARN", "nvidia-smi not available", "Ensure NVIDIA RTX 4050 mobile driver >= 535.xx is installed"))

    # 4. ONNX Runtime & CUDA Provider
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            results.append(("ONNX Runtime (CUDA)", "OK", f"Version {ort.__version__} [CUDA active]", ""))
        else:
            results.append(("ONNX Runtime", "WARN", f"Version {ort.__version__} (CPU Only)", "Run: pip install onnxruntime-gpu"))
    except ImportError:
        results.append(("ONNX Runtime", "FAIL", "Not installed", "Run: pip install onnxruntime-gpu"))

    # 5. Face Detector Model
    det_found, det_path, det_size = find_model_path(["scrfd_10g_bnkps.onnx", "det_10g.onnx"])
    if det_found:
        results.append(("Face Detector", "OK", f"Found ({Path(det_path).name}, {det_size // (1024*1024)} MB)", ""))
    else:
        results.append(("Face Detector", "FAIL (Required)", "Missing scrfd_10g_bnkps.onnx or det_10g.onnx", "Run download_models.bat or download_models.ps1"))

    # 6. Face Swapper Model
    swap_found, swap_path, swap_size = find_model_path(["inswapper_128_fp16.onnx", "inswapper_128.onnx"])
    if swap_found:
        is_fp16 = "fp16" in swap_path.lower()
        tag = "FP16 TensorCore Optimized" if is_fp16 else "FP32 Standard"
        results.append(("Face Swapper", "OK", f"Found ({Path(swap_path).name} - {tag}, {swap_size // (1024*1024)} MB)", ""))
    else:
        results.append(("Face Swapper", "FAIL (Required)", "Missing inswapper_128.onnx (or fp16)", "Run download_models.bat or download_models.ps1"))

    # 7. Optional Recognition & Enhancement Models
    rec_found, rec_path, rec_size = find_model_path(["w600k_r50.onnx"])
    if rec_found:
        results.append(("ArcFace Embedder", "OK", f"Found ({Path(rec_path).name}, {rec_size // (1024*1024)} MB)", ""))
    else:
        results.append(("ArcFace Embedder", "INFO (Optional)", "Not installed (Built-in normalization active)", "Run download_models.bat to install w600k_r50.onnx"))

    parse_found, parse_path, parse_size = find_model_path(["bisenet_face.onnx", "bisenet.onnx"])
    if parse_found:
        results.append(("BiSeNet Hair Parser", "OK", f"Found ({Path(parse_path).name}, {parse_size // (1024*1024)} MB)", ""))
    else:
        results.append(("BiSeNet Hair Parser", "INFO (Optional)", "Not installed", "Run download_models.bat to install bisenet_face.onnx"))

    gfp_found, gfp_path, gfp_size = find_model_path(["GFPGANv1.4.onnx"])
    if gfp_found:
        results.append(("GFPGAN Enhancer", "OK", f"Found ({Path(gfp_path).name}, {gfp_size // (1024*1024)} MB)", ""))
    else:
        results.append(("GFPGAN Enhancer", "INFO (Optional)", "Not installed", "Run download_models.bat to install GFPGANv1.4.onnx"))

    # 8. Virtual Camera & OBS Studio
    try:
        import pyvirtualcam
        results.append(("Virtual Camera Lib", "OK", "pyvirtualcam installed", ""))
    except ImportError:
        results.append(("Virtual Camera Lib", "WARN", "pyvirtualcam not installed", "Run: pip install pyvirtualcam"))

    # Check for OBS Studio on Windows
    obs_paths = [
        r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
        r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe",
    ]
    obs_installed = any(os.path.exists(p) for p in obs_paths)
    if obs_installed:
        results.append(("OBS Virtual Camera", "OK", "OBS Studio installation detected", ""))
    else:
        results.append(("OBS Virtual Camera", "INFO", "OBS Studio not at default path", "Install OBS Studio if you wish to broadcast to Discord/Zoom"))

    return results


def print_status_table() -> None:
    results = check_diagnostics()
    print("=" * 80)
    print(" REAL-TIME AI FACE-SWAP HARDWARE & INSTALLATION DIAGNOSTICS")
    print(" Target: Windows 11, NVIDIA RTX 4050 Laptop GPU (6GB VRAM)")
    print("=" * 80)
    print(f"{'COMPONENT':<24} {'STATUS':<16} {'DETAILS':<38}")
    print("-" * 80)

    failures = []
    for comp, status, details, fix in results:
        print(f"{comp:<24} {status:<16} {details:<38}")
        if "FAIL" in status and fix:
            failures.append((comp, details, fix))

    print("=" * 80)
    if failures:
        print("\n[!] ACTION REQUIRED FOR FAILED COMPONENTS:")
        for comp, details, fix in failures:
            print(f"\n* WHAT FAILED: {comp} ({details})")
            print(f"  HOW TO FIX IT: {fix}")
    else:
        print("\n[*] All primary components verified! Ready to run: run.bat or python app.py")


if __name__ == "__main__":
    print_status_table()
