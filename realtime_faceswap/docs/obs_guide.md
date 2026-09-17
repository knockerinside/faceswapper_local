# OBS Studio & Virtual Camera Integration Guide

## 1. Output Topology Options

### Option A: App -> Virtual Camera -> OBS / Discord (Recommended)
This is the lowest-latency, cleanest topology:

```
[Camera Hardware] ──► [FaceSwap Engine] ──► [OBS Virtual Camera] ──► [OBS Studio / Discord / Zoom]
```
1. Connect physical camera to Real-Time FaceSwap Engine.
2. In the FaceSwap GUI, click **"START VIRTUAL CAMERA"**.
3. In OBS Studio, Discord, or Zoom, select **"OBS Virtual Camera"** as your Video Capture Device.
4. **Latency:** Direct hardware delivery with zero OBS filter overhead (~21ms total latency).

---

### Option B: OBS Studio -> Virtual Camera -> FaceSwap -> Virtual Camera 2 (Advanced)
Only use this if you require OBS scene overlays (such as VTuber avatars or game capture) to be processed before face swap:

```
[OBS Source] ──► [OBS Virtual Camera] ──► [FaceSwap Engine] ──► [Unity/Third-party Virtual Camera]
```

> ⚠️ **CRITICAL WARNING: Avoid Infinite Video Feedback Loops!**
> Never set FaceSwap's output virtual camera as an input source into the same OBS scene that feeds FaceSwap. This causes an exponential latency cascade and GPU hang.

---

## 2. OBS Virtual Camera Setup on Windows 11

1. Download and install [OBS Studio 29.x or 30.x](https://obsproject.com/).
2. During OBS installation, make sure **Virtual Camera** driver is installed (installed by default).
3. The engine uses `pyvirtualcam` to talk directly to OBS's DirectShow filter:
   - Output Resolution: **1280x720**
   - Output Format: **RGB24 / NV12**
   - Frame Rate: **30 FPS / 60 FPS**

---

## 3. Discord & Zoom Setup
1. In Discord: Go to **User Settings -> Voice & Video -> Camera**.
2. Under Camera dropdown, select **OBS Virtual Camera**.
3. Click "Test Video" to verify smooth real-time video streaming with low latency.
