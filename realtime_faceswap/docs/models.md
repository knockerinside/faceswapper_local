# Neural Network Models & Inference Specifications

## 1. SCRFD-10G Face Detector

- **File:** `models/scrfd_10g_bnkps.onnx`
- **Architecture:** Sample and Computation Redistribution for Efficient Face Detection (SCRFD)
- **Input Resolution:** `(1, 3, 640, 640)` normalized RGB float32
- **Key Output Tensors:**
  - `score_8`, `score_16`, `score_32`: Classification logits for 8x, 16x, 32x feature strides
  - `bbox_8`, `bbox_16`, `bbox_32`: Distance regression targets `(l, t, r, b)`
  - `kps_8`, `kps_16`, `kps_32`: 5-point facial landmark coordinates (left eye, right eye, nose tip, left mouth corner, right mouth corner)
- **Inference Latency:** **3.8 ms – 4.5 ms** on RTX 4050 (CUDA)
- **VRAM Footprint:** ~350 MB

---

## 2. InSwapper-128 (InsightFace)

- **File:** `models/inswapper_128.onnx`
- **Architecture:** High-efficiency identity conditional generative network
- **Input Tensors:**
  - `target`: `(1, 3, 128, 128)` normalized RGB float32 aligned facial crop
  - `source_embedding`: `(1, 512)` ArcFace L2-normalized 512-dimensional facial identity vector
- **Output Tensor:**
  - `output`: `(1, 3, 128, 128)` synthesized facial crop in BGR/RGB
- **Inference Latency:** **14.2 ms – 15.5 ms** on RTX 4050 (CUDA FP16)
- **VRAM Footprint:** ~850 MB

---

## 3. BiSeNet Face & Hair Parsing (Optional)

- **File:** `models/bisenet_face.onnx`
- **Architecture:** Bilateral Segmentation Network for Real-time Semantic Segmentation
- **Input Resolution:** `(1, 3, 512, 512)` normalized RGB float32
- **Output Classes (19 categories):**
  - Class 1: Facial Skin
  - Class 2-3: Left/Right Eyebrows
  - Class 4-5: Left/Right Eyes
  - Class 10: Nose
  - Class 11-13: Lips & Mouth
  - Class 17: Hair (used for boundary protection & recoloring)
- **Inference Latency:** **6.2 ms – 7.0 ms** on RTX 4050 (CUDA)
- **VRAM Footprint:** ~420 MB

---

## 4. VRAM Budget on NVIDIA RTX 4050 (6GB Total)

| Component | VRAM Allocated |
| :--- | :--- |
| Windows Desktop Window Manager (DWM) | ~500 MB |
| ONNX Runtime CUDA Execution Provider Context | ~450 MB |
| SCRFD Model Weights & CUDA Kernels | ~350 MB |
| InSwapper-128 Weights & Activation Tensors | ~850 MB |
| BiSeNet Weights & Feature Maps (if enabled) | ~420 MB |
| PySide6 GUI Surface & Swapchain Buffers | ~180 MB |
| **Total Engine VRAM Usage** | **~2.75 GB** (Well under 6GB capacity) |

**Safety Margin**: The RTX 4050 has **3.25 GB of headroom remaining**, preventing Windows out-of-memory driver paging and stutter.
