# Real-Time Face-Swap Installation & Deployment Guide

## Target Hardware & Environment
- **Operating System:** Windows 11 64-bit
- **Target GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Host CPU:** Intel Core i5/i7 (12th/13th Gen) or AMD Ryzen 5/7 (6000/7000 series)
- **RAM:** 16GB Dual-Channel

---

## 1. Automated Installation (Recommended)

Run the included PowerShell installer from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

The script will automatically:
1. Validate Python 3.10/3.11 64-bit.
2. Verify the NVIDIA GPU and driver version using `nvidia-smi`.
3. Create an isolated virtual environment (`.venv`).
4. Upgrade `pip`, `setuptools`, and `wheel`.
5. Install `onnxruntime-gpu`, `PySide6`, `opencv-python`, `pyvirtualcam`, and all required packages.
6. Run the hardware diagnostics suite to verify everything.

---

## 2. Manual Installation

If you prefer installing dependencies manually:

### Step 1: Install Python 3.10 or 3.11
Download Python from [python.org](https://www.python.org/downloads/).
> **Crucial:** Ensure you check **"Add python.exe to PATH"** during setup.

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade Packaging Tools
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Verify CUDA Acceleration
Run the system diagnostics tool:
```powershell
python tools/diagnostics.py
```

Expected output:
```
COMPONENT                STATUS           DETAILS
Python                   OK               Python 3.11.x
NVIDIA GPU               OK               NVIDIA GeForce RTX 4050 Laptop GPU
ONNX Runtime             OK               Version 1.18.x with CUDAExecutionProvider
```

---

## 3. Model Acquisition & Placement

The application loads models locally from the `models/` directory:

```
realtime_faceswap/
└── models/
    ├── scrfd_10g_bnkps.onnx     (Required: 16.5 MB, SCRFD Face Detector)
    ├── inswapper_128.onnx       (Required: 529 MB, InsightFace 128x128 Face Swapper)
    ├── bisenet_face.onnx        (Optional: 53.3 MB, Face & Hair Semantic Parsing)
    └── source/                  (Directory for source identity reference photos)
        └── source.jpg
```

---

## 4. Launching the Engine

### Standard Launch
```powershell
.\run.ps1
```

### Profile Options
- **Balanced (Default):** 720p30, detection every 4 frames, optimal 21ms latency.
  ```powershell
  .\run.ps1 -Profile balanced
  ```
- **Performance:** 720p30, detection every 6 frames, optional parsing disabled.
  ```powershell
  .\run.ps1 -Profile performance
  ```
- **Quality:** 720p30, detection every 2 frames, BiSeNet parsing and hair boundary enabled.
  ```powershell
  .\run.ps1 -Profile quality
  ```

---

## 5. Troubleshooting Common Issues

### Issue 1: `CUDAExecutionProvider not found, falling back to CPU`
- **Cause:** Missing NVIDIA CUDA runtime or cuDNN libraries in Windows system path.
- **Fix:** Install the NVIDIA CUDA 12.x Toolkit or install `onnxruntime-gpu` matching your installed CUDA version:
  ```powershell
  pip uninstall onnxruntime onnxruntime-gpu
  pip install onnxruntime-gpu
  ```

### Issue 2: `Camera Disconnected` or `Cannot open DirectShow camera`
- **Cause:** Windows 11 Privacy settings or another application (Zoom/Teams/OBS) is holding exclusive access to the webcam.
- **Fix:** Go to **Windows Settings -> Privacy & security -> Camera** and verify that "Let desktop apps access your camera" is enabled. Close other applications using the webcam.

### Issue 3: PySide6 Window fails to open or DLL load failed
- **Fix:** Ensure the Microsoft Visual C++ 2015-2022 Redistributable (x64) is installed.
