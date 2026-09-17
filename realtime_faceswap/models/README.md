# AI Model Weights Directory

Place your ONNX model weight files directly in this folder.

## Required Models:
1. **`scrfd_10g_bnkps.onnx`** (~16.5 MB)
   - SCRFD-10G Face Detector
   - Auto-download: Run `.\download_models.ps1`
   - Manual download: `https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx`

2. **`inswapper_128.onnx`** (~529 MB)
   - InsightFace InSwapper-128 Face Swapper
   - Auto-download: Run `.\download_models.ps1`
   - Manual download: `https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx`

## Optional Models:
3. **`bisenet_face.onnx`** (~53 MB)
   - Face and hair boundary semantic segmentation (used when Hair/Bangs preservation is toggled ON).

## Source Identity Photos:
- Place source reference face photos inside `models/source/` (e.g. `source.jpg`). You can also load any photo at runtime via the in-app "LOAD SOURCE FACE" button.
