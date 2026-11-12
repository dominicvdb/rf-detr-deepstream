# rfdetr-deepstream

Run [RF-DETR](https://github.com/roboflow/rf-detr) detection models in NVIDIA DeepStream, with detections that match RF-DETR's own `predict()`.

```bash
rfdetr-deepstream build nano --frame-size 1280x720 -o bundle/
make -C bundle/parser        # once, on the target (DeepStream container or Jetson)
# then use bundle/config_infer_rfdetr.txt as the nvinfer config-file-path
```

## Results

RF-DETR Nano, FP32, DeepStream 7.0 (TensorRT 8.6), RTX 4060 under WSL2. 720p sample video, 1442 frames, compared per frame with the Python reference pipeline (which matches `RFDETR.predict()`):

| Preprocessing | Matched detections | Mean box IoU | Mean \|Δscore\| | FPS (end to end) |
|---|---|---|---|---|
| nvinfer's own (bilinear scaler, single-scale normalization) | 90.1% | 0.870 | 0.036 | ~146 |
| **Built into the ONNX (`--frame-size`)** | **99.2%** | **0.9946** | **0.0028** | **~151** |

Nearly all remaining unmatched detections score within 0.05 of the threshold.

Why nvinfer's own preprocessing falls short:
- Its scaler uses a different pixel-center convention than torchvision, which shifts boxes by about half a network pixel. With bilinear filtering it **resamples even at scale 1** (a 2×2 blur plus a half-pixel shift).
- It has one scale factor for all channels, while RF-DETR normalizes with a per-channel ImageNet std.

With `--frame-size`, the ONNX takes the full RGB frame and does the resize and normalization itself (exactly like torchvision). nvinfer runs with `scaling-filter=0`, which at scale 1 copies pixels unchanged.

## Install

```bash
pip install -e ".[torch]"     # needed to build from a pretrained name or a .pth checkpoint
pip install -e .              # enough for building from an .onnx
```

## Usage

```bash
rfdetr-deepstream build MODEL -o OUT_DIR [--frame-size WxH] [--class-names FILE | --coco]
                                         [--threshold 0.5] [--precision fp32|fp16] [--parser-lib PATH]
```

`MODEL` can be:
- a pretrained name: `nano`, `small`, `medium`, `large`, `base`
- an RF-DETR checkpoint (`.pth` / `.ckpt`), including fine-tuned models. Class names and the class layout come from the checkpoint.
- an ONNX export (`.onnx`). Pass `--class-names` (one per line) or `--coco` for the official COCO weights.

`--frame-size` should match `nvstreammux` `width`/`height`. The engine is then tied to that resolution. Without it, nvinfer approximates the preprocessing (see Results).

The bundle contains:

```
<model>_<W>x<H>.onnx       model, preprocessing built in
labels.txt                 one class name per logits slot
config_infer_rfdetr.txt    nvinfer config (relative paths)
parser/                    C++ parser source + Makefile -> libnvds_rfdetr.so
README.txt
```

TensorRT engines aren't portable between GPUs or TensorRT versions, so nvinfer builds the engine from the ONNX on first run and caches it next to the ONNX.

## Supported

- Detection models (Nano, Small, Medium, Large, Base, and fine-tuned checkpoints). Segmentation and keypoints are not supported yet.
- FP32. FP16 works, but some transformer layers overflow and degrade detections (fix planned: keep those layers in FP32).
- Tested on DeepStream 7.0 (x86, dGPU).

## Development

`scripts/` has the parity tests (`parity_deepstream.py`, `check_reference.py`, `check_onnx_patch.py`) and the diagnostics used to find the preprocessing differences (`debug_*.py`).
