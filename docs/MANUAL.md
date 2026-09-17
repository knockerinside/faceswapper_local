# Real-Time AI Face-Swap Studio: Complete Documentation & Architectural Manual

**Hardware Profile:** Windows 11 &bull; NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) &bull; Intel Core i5/i7 / AMD Ryzen 7  
**Engine Version:** 2.4.0 Production Release  
**Execution Backends:** ONNX Runtime GPU (CUDA 11.8 / 12.x Execution Provider & TensorRT) with CPU Fallback  

---

## Table of Contents

1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Hardware Optimization Strategy (RTX 4050 6GB VRAM)](#2-hardware-optimization-strategy-rtx-4050-6gb-vram)
3. [Deep-Live-Cam Pipeline Mechanics](#3-deep-live-cam-pipeline-mechanics)
   - [Detection & 5-Point Anchor Decoding (SCRFD-10G)](#31-detection--5-point-anchor-decoding-scrfd-10g)
   - [Deep Identity Embedding Extraction (ArcFace w600k_r50)](#32-deep-identity-embedding-extraction-arcface-w600k_r50)
   - [Similarity Transform & Crop Alignment (UMeyama 128x128)](#33-similarity-transform--crop-alignment-umeyama-128x128)
   - [Feature Replacement & Latent Injection (InSwapper-128)](#34-feature-replacement--latent-injection-inswapper-128)
   - [Color Matching & Seamless Alpha Blending (Reinhard + Laplace)](#35-color-matching--seamless-alpha-blending-reinhard--laplace)
   - [Bangs & Hair Segmentation (BiSeNet ResNet-18)](#36-bangs--hair-segmentation-bisenet-resnet-18)
   - [AI Face Super-Resolution Restorer (GFPGAN v1.4)](#37-ai-face-super-resolution-restorer-gfpgan-v14)
4. [Comprehensive ONNX Model Catalog & Selection Matrix](#4-comprehensive-onnx-model-catalog--selection-matrix)
5. [Installation & Windows 11 Setup Guide](#5-installation--windows-11-setup-guide)
   - [One-Click Installation (`setup.bat` / `setup.ps1`)](#51-one-click-installation-setupbat--setupps1)
   - [Prerequisites & CUDA Environment Validation](#52-prerequisites--cuda-environment-validation)
   - [Model Download Center (`download_models.bat` / `download_models.ps1`)](#53-model-download-center-download_modelsbat--download_modelsps1)
6. [Desktop Application Operations & GUI Reference](#6-desktop-application-operations--gui-reference)
   - [Primary Interface Controls](#61-primary-interface-controls)
   - [In-App Model Manager Dialog](#62-in-app-model-manager-dialog)
   - [Virtual Camera Streaming (OBS Studio, Discord, Zoom)](#63-virtual-camera-streaming-obs-studio-discord-zoom)
7. [CLI Tools & Diagnostic Suite](#7-cli-tools--diagnostic-suite)
   - [System Diagnostics (`diagnostics.ps1` / `diagnostics.py`)](#71-system-diagnostics)
   - [Automated Benchmarking Suite (`benchmark.ps1` / `benchmark.py`)](#72-automated-benchmarking-suite)
   - [Clean Uninstallation & Cache Purging (`uninstall.ps1`)](#73-clean-uninstallation--cache-purging)
8. [Troubleshooting & Frequently Asked Questions (FAQ)](#8-troubleshooting--frequently-asked-questions-faq)

---

## 1. Executive Summary & System Architecture

The **Real-Time AI Face-Swap Studio** is a low-latency, zero-frame-lag facial replacement system engineered for real-time video broadcasting, virtual meetings, and content creation. Unlike traditional offline deepfake solutions that process pre-recorded video files over hours, this engine operates on continuous live camera streams, swapping source identities frame-by-frame with end-to-end latencies under **20 milliseconds** (~60 FPS) on an RTX 4050 laptop GPU.

### System Pipeline Flowchart

```
[Webcam / DirectShow Camera Stream] (1080p @ 60 FPS)
                       │
                       ▼
         [Zero-Queue Bounded Ring Buffer]
          (Drops stale frames, grabs latest)
                       │
                       ▼
     [SCRFD-10G / det_10g Face Detection] ─── (2.8 ms)
      - FPN 8x, 16x, 32x Stride Anchor Decoding
      - Vectorized Non-Maximum Suppression (NMS)
      - 5-Point Facial Keypoints (Eyes, Nose, Mouth)
                       │
                       ▼
         [Umeyama Similarity Transform]
      - Computes Affine Matrix M (128x128)
      - Extracts Normalized Target Face Crop
                       │
                       ▼
       [ArcFace Identity Vector Injection]
      - Source image 512-D L2-normalized vector
      - Precomputed once on photo load (0ms overhead)
                       │
                       ▼
        [InSwapper-128 FP16 Tensor Cores] ─── (12-15 ms)
      - Replaces facial features while preserving pose,
        gaze, lighting, and natural expression
                       │
                       ▼
        [Post-Processing & Blending Stage]
      - Reinhard Lab Color Space Alignment
      - BiSeNet ResNet-18 Hair / Spectacles Mask (optional)
      - Inverse Affine Warp (M_inv) back to canvas
      - Soft Sigmoid Edge Feathering
                       │
                       ▼
      [Virtual Camera Output (pyvirtualcam)]
      - Ingested by OBS, Discord, Zoom, Microsoft Teams
```

---

## 2. Hardware Optimization Strategy (RTX 4050 6GB VRAM)

The NVIDIA GeForce RTX 4050 Laptop GPU provides **6 GB of GDDR6 VRAM**, an AD107 Ada Lovelace architecture, 2,560 CUDA cores, and 4th Generation Tensor Cores supporting native FP16 and INT8 precision.

Running real-time face replacement concurrently with external software (like OBS Studio, streaming encoders, or 3D games) requires strict VRAM budget management to avoid CUDA out-of-memory (OOM) crashes:

### VRAM Budget Allocation (6,144 MB Total)

| Allocation Tier | Component | Footprint | Description |
| :--- | :--- | :--- | :--- |
| **System & Desktop** | Windows DWM + Display | ~600 MB | Windows 11 Desktop Window Manager and display driver reserve |
| **AI Inference** | InSwapper-128 FP16 | ~580 MB | Tensor Core accelerated model weights and activations |
| **AI Inference** | SCRFD-10G Detector | ~150 MB | Multi-stride face detection model |
| **AI Inference** | BiSeNet Parser (optional) | ~180 MB | 19-class hair and eyeglasses segmentation |
| **CUDA Context** | ONNX Runtime / cuDNN | ~450 MB | CUDA context, workspace memory, and execution stream buffers |
| **Headroom** | Free for OBS / Streaming | **~4,184 MB** | **Available VRAM for OBS, games, video encoding, and multitasking** |

*Note: Choosing InSwapper-128 FP32 instead of FP16 increases the swapper footprint to ~1,100 MB and reduces frame rates from ~60 FPS down to ~38 FPS.*

---

## 3. Deep-Live-Cam Pipeline Mechanics

### 3.1 Detection & 5-Point Anchor Decoding (SCRFD-10G)
The detector employs **SCRFD-10G** with a multi-scale Feature Pyramid Network (FPN) across 3 strides ($s \in \{8, 16, 32\}$):
1. For each stride, feature maps generate grid anchor centers:
   $$\text{center}_x = x \cdot s, \quad \text{center}_y = y \cdot s$$
2. Box coordinates $[l, t, r, b]$ and 5 landmarks (left eye, right eye, nose tip, left mouth, right mouth) are decoded by projecting outward from anchor centers.
3. Decoded proposals are filtered by confidence ($\ge 0.50$) and deduplicated using **Vectorized Non-Maximum Suppression (NMS)** with an IoU threshold of 0.40.
4. Total execution time: **2.8 ms** on RTX 4050.

### 3.2 Deep Identity Embedding Extraction (ArcFace w600k_r50)
Identity fidelity is maintained using the **ArcFace w600k_r50** deep feature extractor:
- When a source image is loaded, the face crop is normalized to $112 \times 112$, converted to RGB, and normalized to $[-1, 1]$.
- A 512-dimensional feature embedding $\mathbf{z}_{\text{source}} \in \mathbb{R}^{512}$ is extracted and L2-normalized:
  $$\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}$$
- **Zero Real-Time Overhead**: Extraction happens exactly **once** upon image selection. During streaming, the pre-computed 512-D vector is injected directly into InSwapper-128 inputs.

### 3.3 Similarity Transform & Crop Alignment (Umeyama 128x128)
To feed InSwapper, target faces must be aligned with standard coordinates:
1. Standard reference 5-point facial coordinates in $128 \times 128$ space:
   - Left Eye: `[38.2946, 51.6963]`
   - Right Eye: `[73.5318, 51.5014]`
   - Nose: `[56.0252, 71.7366]`
   - Left Mouth: `[41.5493, 92.3655]`
   - Right Mouth: `[70.7299, 92.2041]`
2. The Umeyama algorithm computes the optimal $2 \times 3$ affine transformation matrix $\mathbf{M}$ minimizing least-squares distance.
3. OpenCV applies `cv2.warpAffine` to produce a normalized $128 \times 128$ crop.

### 3.4 Feature Replacement & Latent Injection (InSwapper-128)
1. The $128 \times 128$ target face is converted to float32 $[0.0, 1.0]$, transposed to NCHW $(1, 3, 128, 128)$, and bound to the ONNX session along with the 512-D source vector.
2. InSwapper reconstructs the face:
   - Preserves target facial geometry, head rotation, eye gaze, and mouth opening.
   - Replaces identity features (eyes, nose, bone structure, lips) with the source identity.
3. FP16 half-precision execution utilizes Tensor Cores, finishing in **12 to 15 ms**.

### 3.5 Color Matching & Seamless Alpha Blending (Reinhard + Laplace)
To eliminate noticeable boundary seams and mismatched lighting between source and webcam:
1. **Reinhard Color Transfer in CIELAB Space**:
   - Both target crop and swapped face are converted to CIELAB space.
   - Mean and standard deviation of $L, a, b$ channels are matched:
     $$I_{\text{matched}} = \left(I_{\text{swapped}} - \mu_{\text{swapped}}\right) \cdot \frac{\sigma_{\text{target}}}{\sigma_{\text{swapped}}} + \mu_{\text{target}}$$
2. **Smooth Alpha Mask**: An elliptical sigmoid mask with Gaussian feathering ($k=15$) blends boundary pixels smoothly into the camera background.
3. **Inverse Affine Mapping**: The blended crop is warped back into the full-resolution frame using $\mathbf{M}^{-1} = \text{cv2.invertAffineTransform}(\mathbf{M})$.

### 3.6 Bangs & Hair Segmentation (BiSeNet ResNet-18)
When enabled, **BiSeNet** computes a 19-class semantic segmentation map:
- Mask classes for `hair`, `hat`, and `eyeglasses` are isolated.
- Pixels classified as hair or glasses frames are preserved from the original camera feed, preventing the artificial face from painting over bangs or spectacle rims.

### 3.7 AI Face Super-Resolution Restorer (GFPGAN v1.4)
For offline 4K processing or high-fidelity capture, **GFPGAN v1.4** takes the $128 \times 128$ swapped crop and upscales it to $512 \times 512$:
- Restores natural skin pore texture, dental reflections, and iris details.
- Runs in **32-38 ms** on RTX 4050.

---

## 4. Comprehensive ONNX Model Catalog & Selection Matrix

| Model Identifier | File Name | Category | Size | RTX 4050 Latency | VRAM Footprint | Best Suited For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `inswapper_fp16` | `inswapper_128_fp16.onnx` | Face Swapper | 264 MB | **12 - 15 ms (~60 FPS)** | **~580 MB** | **Default Recommended**: 60 FPS live streaming, lowest VRAM, zero quality loss. |
| `inswapper_fp32` | `inswapper_128.onnx` | Face Swapper | 529 MB | 22 - 28 ms (~38 FPS) | ~1,100 MB | Reference studio recording, offline 4K video rendering. |
| `scrfd_10g` | `scrfd_10g_bnkps.onnx` | Face Detector | 16.5 MB | **2.8 ms (~90+ FPS)** | ~150 MB | **Required Universal Detector**: 5-point facial keypoints with high angle tolerance. |
| `det_10g` | `det_10g.onnx` | Face Detector | 16.5 MB | 2.8 ms (~90+ FPS) | ~150 MB | Official InsightFace Buffalo_L mirror (interchangeable with SCRFD-10G). |
| `arcface` | `w600k_r50.onnx` | Identity Embedder| 166 MB | 4.2 ms (one-time) | ~180 MB | **Recommended**: Deep 512-D identity vector extraction for authentic likeness. |
| `bisenet` | `bisenet_face.onnx` | Segmentation | 53.3 MB | 6.5 ms (~45 FPS) | ~180 MB | Hair, bangs, and spectacles occlusion protection. |
| `gfpgan` | `GFPGANv1.4.onnx` | Face Restorer | 332 MB | 32 - 38 ms (~26 FPS) | ~650 MB | Studio HD enhancement: restores micro skin texture and iris clarity. |

---

## 5. Installation & Windows 11 Setup Guide

### 5.1 One-Click Installation (`setup.bat` / `setup.ps1`)

The repository includes a comprehensive, automated installer script designed for Windows 11:

```cmd
:: Open Command Prompt or double-click in File Explorer:
setup.bat
```

Or via PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

#### What the automated setup executes:
1. **Visual C++ Runtime Verification**: Checks for Microsoft Visual C++ 2015-2022 Redistributable (x64). If missing, it downloads and installs it automatically.
2. **GPU & CUDA Validation**: Verifies NVIDIA driver presence and RTX 4050 GPU detection via `nvidia-smi`.
3. **Isolated Python Virtual Environment**: Sets up a local `.venv` within the directory to prevent conflicts with global system packages.
4. **Package Installation**: Installs `onnxruntime-gpu`, `PySide6`, `opencv-python`, `numpy`, and `pyvirtualcam`.
5. **Model Downloader Integration**: Prompts to download the **Recommended RTX 4050 Model Suite** (~446 MB) with automatic mirror failovers.
6. **Windows Desktop Shortcut Creation**: Generates a desktop shortcut (`Real-Time Face-Swap Studio.lnk`) pointing directly to `run.bat`.

### 5.2 Prerequisites & CUDA Environment Validation

- **Operating System:** Windows 11 64-bit (Build 22000+) or Windows 10 (21H2+)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (or desktop equivalent) with driver $\ge 528.33$
- **Python:** Version 3.10.x or 3.11.x 64-bit ([python.org](https://www.python.org))
- **Webcam:** Any USB / integrated webcam, or DirectShow video capture source (e.g., Camo, Elgato Cam Link)

To verify your NVIDIA driver and CUDA availability before running:
```cmd
nvidia-smi
```

### 5.3 Model Download Center (`download_models.bat` / `download_models.ps1`)

If you wish to download, upgrade, or inspect models independently of the main installer:

```cmd
download_models.bat
```

The interactive menu provides the following options:
```
==================================================================================
 REAL-TIME AI FACE-SWAP ENGINE -- MODEL DOWNLOAD & MANAGEMENT CENTER
 Target Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) / Windows 11
==================================================================================

Please choose an option:
  [1] Download Recommended Suite for RTX 4050 (SCRFD-10G + InSwapper FP16 + ArcFace)  [~446 MB]
      -> Fastest 60 FPS live streaming, lowest VRAM (~900MB total), true face likeness.
  [2] Download Full Studio Quality Suite (+ InSwapper FP32 + BiSeNet + GFPGAN)        [~1.1 GB]
      -> Maximum precision, hair/spectacle preservation, and AI face super-resolution.
  [3] Compare Model Advantages & Hardware Benchmarks (FP16 vs FP32, etc.)
  [4] Custom Model Chooser: Select individual models to download
  [5] Check Local Models Status & Integrity
  [6] Download ALL Available Models (Complete Offline Pack)
  [0] Exit / Done
```

Non-interactive CLI flags for automation:
- `python tools/download_models.py --preset recommended` (Installs default suite)
- `python tools/download_models.py --check` (Audits installed weights)
- `python tools/download_models.py --list` (Outputs technical specs and advantages)

---

## 6. Desktop Application Operations & GUI Reference

To launch the desktop application, double-click `run.bat` or the desktop shortcut:

```cmd
run.bat
```

### 6.1 Primary Interface Controls

```
┌───────────────────────────────────────────────┬──────────────────────────────────────────────┐
│  CAMERA FEED & LIVE SWAP PREVIEW              │  CONTROL PANEL & RUNTIME TELEMETRY           │
│                                               ├──────────────────────────────────────────────┤
│  [ Active Video Viewport: 1280x720 @ 60 FPS ] │  Source Identity: [ Browse Source Photo... ] │
│  - Real-time bounding box indicator           │  - Selected: "source_actor.png" (512-D OK)   │
│  - Green landmarks track facial features      │  - Thumbnail Preview Box                     │
│  - Instant toggle between Raw / Swapped view  ├──────────────────────────────────────────────┤
│                                               │  Pipeline Toggles:                           │
│  [ Camera Selection: DirectShow USB Cam #0 ]  │  [X] Enable Real-Time Face Swapping          │
│  [ Camera Resolution: 1280x720 @ 60 FPS   ]   │  [X] Reinhard Color Correction               │
│                                               │  [ ] BiSeNet Hair/Glasses Preservation       │
│  [ START CAMERA ]      [ STOP CAMERA ]        │  [ ] GFPGAN Super-Resolution (30 FPS)        │
│                                               ├──────────────────────────────────────────────┤
│  [ Virtual Camera Stream: INACTIVE ]          │  Fine-Tuning Sliders:                        │
│  [ START VIRTUAL CAMERA (OBS/Zoom) ]          │  - Detection Confidence Threshold: [0.60]    │
│                                               │  - Edge Blend Feathering:          [12 px]   │
│  FPS: 59.4 | Latency: 14.8 ms | GPU: 34%     │  - Temporal Smoothing:             [0.75]    │
│  VRAM: 870 MB / 6144 MB (Healthy)             ├──────────────────────────────────────────────┤
│                                               │  [ Model Manager... ]  [ Diagnostics... ]    │
└───────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

1. **Source Face Selection**: Click **"Select Source Face"** to choose an image (PNG, JPG, WEBP). The image is scanned for faces, cropped, and passed through ArcFace to extract the 512-D identity embedding.
2. **Camera Initialization**: Select your webcam from the dropdown and click **"Start Camera"**. The pipeline runs at native capture framerate without queue lag.
3. **Runtime Controls**:
   - **Color Correction**: Toggles Reinhard LAB color transfer to adapt the source face's tone to the webcam room lighting.
   - **Temporal Smoothing**: Applies exponential moving average (EMA) across consecutive landmark positions to eliminate micro-jitter.
   - **Blend Feathering**: Adjusts the sigmoid boundary softness around the jawline and forehead.

### 6.2 In-App Model Manager Dialog

Accessed via **"Model Manager..."** in the main window:
- Displays interactive model cards with size, RTX 4050 latency, VRAM footprint, and architectural advantages.
- Allows background model downloading with a progress bar and mirror failover.
- Features a **"Set as Active Model"** button to hot-swap active models in the pipeline without restarting the application.

### 6.3 Virtual Camera Streaming (OBS Studio, Discord, Zoom)

The studio includes native virtual camera output powered by `pyvirtualcam`:
1. Click **"Start Virtual Camera"** in the sidebar.
2. Open **OBS Studio**, **Discord**, **Zoom**, **Google Meet**, or **Microsoft Teams**.
3. In your meeting app's video settings, select **"OBS Virtual Camera"** or **"Unity Video Capture"** as your video camera source.
4. The swapped video feed will stream into your call in real time.

---

## 7. CLI Tools & Diagnostic Suite

### 7.1 System Diagnostics
Run automated diagnostics to test CUDA drivers, model presence, and virtual camera drivers:

```powershell
.\diagnostics.ps1
```

Or via Command Prompt:
```cmd
python realtime_faceswap\tools\diagnostics.py
```

Sample output:
```
================================================================================
 REAL-TIME AI FACE-SWAP HARDWARE & INSTALLATION DIAGNOSTICS
 Target: Windows 11, NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
================================================================================
COMPONENT                STATUS           DETAILS                               
--------------------------------------------------------------------------------
Python                   OK               Python 3.11.8                         
Operating System         OK               Windows 11 (Build 22631)              
NVIDIA GPU               OK               NVIDIA GeForce RTX 4050 Laptop GPU    
ONNX Runtime             OK               CUDAExecutionProvider Active          
Face Detector            OK               Found (scrfd_10g_bnkps.onnx, 16 MB)   
Face Swapper             OK               Found (inswapper_128_fp16.onnx, 264 MB)
ArcFace Embedder         OK               Found (w600k_r50.onnx, 166 MB)        
BiSeNet Hair Parser      OK               Found (bisenet_face.onnx, 53 MB)      
GFPGAN Enhancer          OK               Found (GFPGANv1.4.onnx, 332 MB)       
Virtual Camera Lib       OK               pyvirtualcam ready                    
OBS Virtual Camera       OK               OBS Virtual Camera installed          
================================================================================
 [OK] ALL CHECKS PASSED: SYSTEM IS FULLY OPTIMIZED FOR 60 FPS SWAPPING!
```

### 7.2 Automated Benchmarking Suite
To measure exact inference latencies, frame drops, and memory usage:

```powershell
.\benchmark.ps1 -Frames 200
```

Exports latency breakdowns into `benchmark_results.json` and `benchmark_report.csv` detailing:
- Face detection latency (ms)
- Landmark alignment latency (ms)
- Swap inference latency (ms)
- Post-processing & color transfer latency (ms)
- Total pipeline latency & sustained FPS

### 7.3 Clean Uninstallation & Cache Purging
If you ever want to uninstall or reset the studio without leaving orphaned cache files:

```powershell
# Interactive uninstaller:
.\uninstall.ps1

# Delete virtual environment and build cache, but keep downloaded models:
.\uninstall.ps1 -KeepModels

# Non-interactive complete wipe:
.\uninstall.ps1 -Force
```

---

## 8. Troubleshooting & Frequently Asked Questions (FAQ)

#### Q1: Why is my frame rate locked at ~30 FPS instead of 60 FPS?
- **Webcam Hardware Mode**: Most budget webcams default to 1080p @ 30 FPS in YUY2 uncompressed mode. In the camera settings dropdown, select **720p (1280x720)** or enable **MJPG** format to unlock 60 FPS capture.
- **GFPGAN Active**: If GFPGAN super-resolution is enabled, it adds ~35 ms per frame. Disable GFPGAN for pure 60 FPS live streaming.

#### Q2: The app says `CUDAExecutionProvider not available; falling back to CPU`. How do I fix this?
- Ensure the NVIDIA display driver is updated to version **528.33 or newer**.
- Run:
  ```cmd
  pip uninstall -y onnxruntime onnxruntime-gpu
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  ```

#### Q3: How do I eliminate jitter when moving my head quickly?
- Increase the **Temporal Smoothing** slider to `0.80 - 0.85` in the control panel. This stabilizes facial landmark tracking during rapid movement.

#### Q4: Why does the swapped face look too dark or pale compared to my neck?
- Ensure the **Reinhard Color Matching** checkbox is enabled.
- Ensure your room has balanced front lighting. Backlit conditions reduce color transfer accuracy.

#### Q5: Can I add multiple source faces and switch between them with hotkeys?
- Yes. Place target photos in `models/source/`. The UI source browser indexes images in this folder for quick selection.

---

*Real-Time AI Face-Swap Studio &bull; Built with PySide6, ONNX Runtime GPU, and OpenCV.*
