# Performance Tuning & Thermal Optimization Guide

## 1. 30 FPS vs. 60 FPS Trade-offs on RTX 4050 Laptop GPU

The NVIDIA GeForce RTX 4050 Laptop GPU operates within a **35W – 75W TGP (Total Graphics Power)** thermal envelope:

| Metric | 720p @ 30 FPS (Recommended) | 720p @ 60 FPS |
| :--- | :--- | :--- |
| **Frame Budget** | **33.3 ms** | **16.6 ms** |
| **Engine Latency** | ~21.0 ms | ~21.0 ms |
| **Feasibility** | **100% Stable Real-Time** | Requires 1-in-4 detector interval + TensorRT |
| **GPU Utilization**| ~65% – 72% | ~95% – 100% |
| **Thermals** | 62°C – 68°C (Silent fan curve) | 78°C – 84°C (Thermal throttling risk) |
| **Power Draw** | ~40W | ~65W (Max TGP) |

> **Recommendation:** Run at **720p30**. It delivers rock-solid real-time stability, prevents laptop thermal throttling, and preserves GPU headroom for other streaming/gaming software.

---

## 2. Preventing Laptop Thermal Throttling

During extended sessions (>30 minutes):
1. **Windows Power Mode:** Set to **"Best Performance"** in Windows 11 Settings (avoids sudden CPU core parking).
2. **NVIDIA Control Panel:**
   - Power Management Mode: **Prefer Maximum Performance**.
   - Max Frame Rate: **30 FPS** (prevents unneeded render loops).
3. **Cooling:** Ensure laptop exhaust vents have at least 2 inches of elevation.

---

## 3. Adaptive Quality Hysteresis System

The engine features an automatic adaptive performance engine:
- If processing FPS falls below **24 FPS** for more than 5 consecutive frames:
  1. **Tier 1:** Automatically disable BiSeNet hair and face parsing.
  2. **Tier 2:** Increase face detection interval from 4 to 6 frames.
- **Hysteresis Cooldown (60 frames):** The system enforces a 2-second cooldown before attempting to step quality back up, completely eliminating jarring visual oscillations.

---

## 4. Mask Parameter Tuning

- **Mask Feather (10 – 20 px):** Softens the outer boundary between original skin and swapped face.
  - *Too low (<5px):* Hard boundary edge visible around jaw and forehead.
  - *Too high (>25px):* Outer original face bleeds into swap.
  - *Optimal:* **15 px**.
- **Mask Smoothing (0.3 – 0.5):** Temporal Exponential Moving Average (EMA) applied across consecutive frame masks. Eliminates high-frequency mask jitter and flicker during rapid head movements.
- **Erosion (2 – 6 px):** Insets the mask to prevent background artifacts from creeping inside the facial boundary.
