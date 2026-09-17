# Real-Time AI Face-Swap Engine & Control Studio
**Optimized for Windows 11 & NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**

Low-latency, zero-queue real-time face replacement designed for video calls (OBS Virtual Camera, Discord, Zoom) and live camera feeds with temporal smoothing, bang/hair protection, and Reinhard color matching.

---

## 1. Quick Start on Windows 11

> **Looking for the in-depth technical manual and architectural breakdown?**  
> Check out the **[Full Documentation & Architectural Manual (docs/MANUAL.md)](docs/MANUAL.md)** covering pipeline mathematics, VRAM budget allocation, model comparison matrices, and troubleshooting.

### Option A: Command Prompt (Double-Click Setup -- Recommended)
Simply double-click:
```cmd
setup.bat
```
*(Or open Command Prompt in the folder and type `setup.bat`)*

**What `setup.bat` does automatically:**
1. Detects Microsoft Visual C++ 2015-2022 (skips if already installed, auto-installs if missing).
2. Detects Python 3.10 / 3.11 and NVIDIA GPU (RTX 4050).
3. Creates an isolated virtual environment (`.venv`) inside the folder.
4. Installs CUDA ONNX Runtime, PySide6, OpenCV, and pyvirtualcam.
5. Smartly detects which models you already have in `models/` (SCRFD / det_10g, FP16 / FP32 InSwapper) and downloads only missing files.
6. **Creates a Desktop Shortcut** (`Real-Time Face-Swap Studio.lnk`) on your Windows Desktop!

**To launch:**
Double-click `run.bat` or use the Desktop shortcut!

---

### Option B: PowerShell Setup
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```
And launch via:
```powershell
.\run.ps1
```

---

## 2. In-Depth Features & Enhancements (Inspired by Deep-Live-Cam)

1. **Dual FP16 / FP32 InSwapper Support**:
   - Includes support for `inswapper_128_fp16.onnx` (264 MB), which cuts VRAM footprint in half and runs up to 1.8x faster on the RTX 4050.
2. **Multi-Mirror Automatic Model Downloader**:
   - `.\download_models.ps1` connects to verified mirrors (including HuggingFace Deep-Live-Cam repositories) with automatic fallback and chunked verification.
3. **Zero-Queue Bounded Frame Buffer**:
   - DirectShow camera input never piles up queued frames. The inference pipeline always swaps on the freshest physical camera frame, eliminating the 100ms+ display lag common in other tools.
4. **Hair & Bangs Boundary Protection**:
   - Prevents swapped facial skin from painting over bangs, glasses, or facial hair.
5. **Reinhard Color & Lighting Transfer**:
   - Dynamically adapts the color tone of the source identity to match the live camera ambient lighting.

---

## 3. Zero-Trash Uninstaller (`.\uninstall.ps1`)

If you want to uninstall and delete everything created by the application without leaving any cache, trash, or temp files:

```powershell
.\uninstall.ps1
```

The script will:
- Terminate any running Python engine processes.
- Delete the isolated `.venv` directory and all installed packages.
- Recursively purge all Python bytecode caches (`__pycache__`, `*.pyc`, `*.pyo`).
- Clear benchmark test logs and compiler caches (`.pytest_cache`, `build/`, `dist/`).
- Remove downloaded ONNX model weights (or keep them if you pass `-KeepModels`).
- Clean up local pip build cache in AppData.

To run silently without confirmation prompts:
```powershell
.\uninstall.ps1 -Force
```
To delete packages and cache while keeping the downloaded model files:
```powershell
.\uninstall.ps1 -KeepModels
```

---

## 4. Diagnostics & Troubleshooting

- **Run Diagnostics manually**:
  ```powershell
  .\diagnostics.ps1
  ```
- **Benchmark performance (exports JSON & CSV)**:
  ```powershell
  .\benchmark.ps1 -Frames 200
  ```
- **View files in PowerShell**:
  Remember that PowerShell's `tree` command only shows directories by default. To list all files:
  ```powershell
  tree /F
  # or
  Get-ChildItem -Recurse
  ```
