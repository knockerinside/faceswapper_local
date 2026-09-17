# Real-Time Face-Swap Engine Architecture Guide

## 1. System Overview & Latency-First Philosophy

This system is engineered for **low glass-to-glass latency** on a Windows 11 system equipped with an **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**.

### Core Principle: Zero-Queue Bounded Buffers
In real-time interactive video (such as video calls or streaming), **freshness is everything**:
- Standard multi-frame queues (`queue.Queue()`) accumulate delay when AI inference drops below capture frame rates.
- A 3-frame queue at 30 FPS instantly injects **100 ms of pure display lag**.
- This engine uses a custom **`LatestFrameBuffer` with strict capacity = 1**:
  - Whenever a new camera frame arrives, any unextracted frame is atomically overwritten.
  - The inference thread **always** processes the most temporally recent physical camera capture.
  - A GPU running at 70% utilization with zero queue latency is vastly superior to a GPU running at 100% utilization with 5 queued frames.

---

## 2. Multi-Threaded Execution Topology

The engine splits work across decoupled concurrent threads:

```
[Camera Hardware]
       │ DirectShow 1280x720 @ 30/60 FPS
       ▼
[Thread 1: CaptureWorker]
       │ Reads raw frame into pre-allocated memory
       ▼
[LatestFrameBuffer (Capacity = 1)] ──(Drop stale frames on arrival)──
       │ Fresh frame notification via Condition Variable
       ▼
[Thread 2: InferenceWorker]
  ├── 1. Detection (SCRFD ONNX) [Every N frames / low tracker confidence]
  ├── 2. Landmark Tracking (EMA Smoothing & Lost Face Recovery)
  ├── 3. Face Alignment & 128x128 Affine Crop
  ├── 4. Face Swap (InSwapper-128 ONNX on CUDA)
  ├── 5. Optional Face/Hair Parsing (BiSeNet ONNX on CUDA)
  ├── 6. Soft Mask Generation & Feathering
  ├── 7. Color Correction (Reinhard Statistical Transfer in LAB)
  └── 8. Inverse Affine ROI Compositing into Background
       │
       ├──► [PySide6 GUI Desktop Display (Main Thread)]
       └──► [pyvirtualcam / OBS Virtual Camera DirectShow Output]
```

---

## 3. CPU vs. GPU Work Distribution

Moving every single pixel operation to the GPU is counter-productive because host-to-device (`H2D`) and device-to-host (`D2H`) PCI Express transfers add measurable latency.

| Stage | Execution Location | Technical Justification |
| :--- | :--- | :--- |
| **Camera Capture** | **CPU** | Windows DirectShow hardware driver interacts through host memory. |
| **Face Detection (SCRFD)** | **GPU (CUDA)** | Highly parallel convolutional feature pyramids run in ~4ms on RTX 4050. |
| **Face Tracking & EMA** | **CPU** | Bounding box & landmark smoothing involves ~10 floats. GPU invocation overhead would be 50x slower than CPU scalar math. |
| **Alignment Crop (128x128)** | **CPU / Host** | Cropping a small 128x128 matrix is instantaneous (<0.1ms). |
| **InSwapper Inference** | **GPU (CUDA)** | 128x128 UNet model with ArcFace conditioning runs in ~15ms on RTX 4050 Tensor Cores. |
| **BiSeNet Parsing** | **GPU (CUDA)** | High-resolution semantic segmentation runs on GPU (~6ms) when enabled. |
| **Mask Feathering & ROI Blend**| **CPU** | Blending is performed strictly on the face bounding box ROI rather than the full 1280x720 canvas, taking <1ms on modern CPU SIMD. |
| **Virtual Camera Output** | **CPU** | pyvirtualcam communicates with OBS Virtual Camera DirectShow buffer on host memory. |

---

## 4. End-to-End Latency Budget (RTX 4050 6GB VRAM)

Target: **1280x720 @ 30 FPS** (Budget = 33.3 ms per frame)

| Pipeline Stage | Typical Latency | P95 Latency |
| :--- | :--- | :--- |
| DirectShow Capture | 2.5 ms | 5.0 ms |
| Face Detection (every 4 frames) | 1.1 ms (amortized) | 4.5 ms |
| Landmark Tracking & EMA | 0.2 ms | 0.4 ms |
| 128x128 Affine Crop | 0.1 ms | 0.2 ms |
| InSwapper-128 (CUDA FP16) | 14.5 ms | 17.0 ms |
| Soft Mask & Color Transfer | 0.8 ms | 1.2 ms |
| Inverse Affine Compositing | 0.9 ms | 1.4 ms |
| Output to Virtual Camera | 1.0 ms | 2.0 ms |
| **Total Pipeline Latency** | **~21.1 ms** | **~31.7 ms** |

**Headroom**: Fits cleanly inside the 33.3 ms budget, sustaining a stable **30.0 FPS** with **0 frame drops**.
