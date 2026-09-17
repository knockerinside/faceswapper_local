"""Standalone Image-to-Image Face Swap Verification Tool.

Requirements (Section 70):
- Takes: Source Image + Target Image -> Swapped Image Output.
- Verifies model weights and affine alignment completely independently of camera, OBS, or GUI.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from realtime_faceswap.detection.scrfd import SCRFDDetector
from realtime_faceswap.alignment.face_alignment import FaceAligner
from realtime_faceswap.swapping.inswapper import InSwapperBackend
from realtime_faceswap.compositing.masks import MaskGenerator
from realtime_faceswap.compositing.blender import FaceBlender


def test_image_swap(source_path: str, target_path: str, output_path: str = "swapped_output.jpg") -> None:
    print("=" * 70)
    print(" STANDALONE MODEL VERIFICATION (IMAGE-TO-IMAGE)")
    print("=" * 70)

    if not HAS_CV2:
        print("[!] OpenCV is required for image file I/O.")
        return

    src_img = cv2.imread(source_path)
    tgt_img = cv2.imread(target_path)

    if src_img is None:
        print(f"[!] Could not load source image: {source_path}")
        return
    if tgt_img is None:
        print(f"[!] Could not load target image: {target_path}")
        return

    print(f"Loaded source image: {src_img.shape} from {source_path}")
    print(f"Loaded target image: {tgt_img.shape} from {target_path}")

    # Initialize subsystems
    detector = SCRFDDetector()
    detector.load("models/scrfd_10g_bnkps.onnx")

    aligner = FaceAligner(crop_size=128)
    swapper = InSwapperBackend()
    swapper.load("models/inswapper_128.onnx")

    # 1. Detect and align source face
    print("Detecting source face...")
    src_faces = detector.detect(src_img, max_num=1)
    if not src_faces:
        print("[!] No face found in source image.")
        return
    M_src, _ = aligner.get_alignment_matrix(src_faces[0].landmarks)
    src_aligned = aligner.crop_face(src_img, M_src)
    swapper.set_source(src_aligned, src_faces[0].embedding)
    print("[+] Source face aligned and embedding set.")

    # 2. Detect and align target face
    print("Detecting target face...")
    tgt_faces = detector.detect(tgt_img, max_num=1)
    if not tgt_faces:
        print("[!] No face found in target image.")
        return
    target_face = tgt_faces[0]
    M_tgt, M_tgt_inv = aligner.get_alignment_matrix(target_face.landmarks)
    tgt_aligned = aligner.crop_face(tgt_img, M_tgt)

    # 3. Swap
    print("Running InSwapper inference on 128x128 crop...")
    swapped_128 = swapper.process(tgt_aligned)

    # 4. Mask and Blend
    mask_gen = MaskGenerator(crop_size=128)
    mask = mask_gen.generate_mask(feather_px=15, erosion_px=4)

    blender = FaceBlender()
    final_output = blender.blend(
        original_frame=tgt_img,
        aligned_target_face=tgt_aligned,
        swapped_face_128=swapped_128,
        mask_128=mask,
        M_inv=M_tgt_inv,
        blend_strength=1.0,
        enable_color_correction=True,
    )

    cv2.imwrite(output_path, final_output)
    print(f"[+] Swap completed successfully! Saved output image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/source/source.jpg")
    parser.add_argument("--target", default="models/source/target.jpg")
    parser.add_argument("--output", default="swapped_output.jpg")
    args = parser.parse_args()
    test_image_swap(args.source, args.target, args.output)
