"""Optional TensorRT Engine Compilation Tool for NVIDIA RTX 4050.

Requirements (Section 25):
- Documents TensorRT compatibility with CUDA 12.x / 11.8 on RTX 4050 Laptop GPU.
- Builds FP16 optimized plan file from InSwapper ONNX.
- Explicitly documents that TensorRT engines are GPU architecture and driver-specific.
"""
import sys
import argparse
from pathlib import Path


def build_engine(onnx_path: str, output_engine: str, fp16: bool = True) -> None:
    print("=" * 70)
    print(" TENSORRT ENGINE COMPILATION (OPTIONAL OPTIMIZATION)")
    print(" Target: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89)")
    print("=" * 70)

    try:
        import tensorrt as trt
    except ImportError:
        print("[!] TensorRT Python package is not installed.")
        print("    Note: TensorRT is optional (Section 25). ONNX Runtime CUDA is the primary engine.")
        print("    To install on Windows: download TensorRT 10.x zip from NVIDIA Developer Portal.")
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not Path(onnx_path).exists():
        print(f"[!] ONNX model file not found at: {onnx_path}")
        return

    print(f"Parsing ONNX model: {onnx_path}...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # 2GB workspace limit for RTX 4050 6GB VRAM budget
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    if fp16 and builder.platform_has_fast_fp16:
        print("[+] Enabling FP16 precision for Ada Lovelace Tensor Cores.")
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building serialized TensorRT engine (this may take 2-4 minutes)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[!] Failed to build TensorRT engine.")
        return

    with open(output_engine, "wb") as f:
        f.write(plan)

    print(f"[+] Engine successfully written to: {output_engine}")
    print("[*] Note: Engine is locked to your specific RTX 4050 GPU and driver version.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/inswapper_128.onnx")
    parser.add_argument("--engine", default="models/inswapper_128_rtx4050.engine")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, fp16=not args.no_fp16)
