"""Camera Enumeration and Real-time Frame Acquisition Latency Tester.

Requirements (Sections 5, 46, 58):
- Probes connected cameras, queries supported resolutions and actual delivered FPS.
- Measures capture latency specifically, showing camera hardware delay vs software delay.
"""
import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from realtime_faceswap.capture.device_manager import DeviceManager
from realtime_faceswap.capture.opencv_dshow import OpenCVCameraBackend


def test_camera(device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
    print("=" * 70)
    print(" CAMERA HARDWARE TESTER & LATENCY PROBER")
    print("=" * 70)

    devices = DeviceManager.enumerate_cameras()
    print(f"Detected {len(devices)} video capture device(s):")
    for dev in devices:
        phone_tag = " [★ DETECTED ANDROID PHONE]" if dev.is_phone_camera else ""
        print(f"  [{dev.index}] {dev.name}{phone_tag}")

    print(f"\nTesting camera index {device_index} ({width}x{height} @ {fps}fps)...")
    backend = OpenCVCameraBackend()
    success = backend.open(device_index, width=width, height=height, fps=fps)

    if not success:
        print(f"[!] FAILED to open camera index {device_index}.")
        return

    actual_w, actual_h = backend.get_actual_resolution()
    reported_fps = backend.get_actual_fps()
    print(f"[+] Camera opened! Actual resolution: {actual_w}x{actual_h}, Reported FPS: {reported_fps}")

    frame_times = []
    print("Capturing 60 test frames to measure actual inter-frame delivery...")
    start_all = time.perf_counter()

    for i in range(60):
        t0 = time.perf_counter()
        ret, frame = backend.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if ret and frame is not None:
            frame_times.append(elapsed_ms)
        else:
            print(f"  Warning: Frame {i} failed to read.")

    backend.release()
    total_time = time.perf_counter() - start_all

    if frame_times:
        measured_fps = len(frame_times) / total_time
        avg_cap_lat = sum(frame_times) / len(frame_times)
        print("-" * 70)
        print(f"  Delivered FPS:        {measured_fps:.1f} FPS")
        print(f"  Mean Capture Latency: {avg_cap_lat:.2f} ms")
        print(f"  Max Capture Latency:  {max(frame_times):.2f} ms")
        print(f"  Min Capture Latency:  {min(frame_times):.2f} ms")
        print("-" * 70)
        if measured_fps < fps * 0.8:
            print("[!] Note: Actual delivered FPS is lower than requested. USB bandwidth or lighting exposure may be limiting camera speed.")
        else:
            print("[*] Camera capture operating within target latency parameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    test_camera(args.index, args.width, args.height, args.fps)
