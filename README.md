# rfdetr-deepstream

Run [RF-DETR](https://github.com/roboflow/rf-detr) detection and keypoint (pose) models in NVIDIA DeepStream, with results that match RF-DETR's own `predict()`.

```bash
rfdetr-deepstream build nano --frame-size 1280x720 -o bundle/
make -C bundle/parser        # once, on the target (DeepStream container or Jetson)
# then use bundle/config_infer_rfdetr.txt as the nvinfer config-file-path

rfdetr-deepstream build keypoint --frame-size 1920x1080 -o pose_bundle/
make -C pose_bundle/pose     # see pose_bundle/README.txt for the pipeline
```

## Results: detection

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

## Results: keypoints (RF-DETR KeypointPreview)

The official `RFDETRKeypointPreview` checkpoint (COCO person, 17 keypoints, 576×576, XLarge backbone), FP32, same setup. `sample_walk.mov` (1920×1080, 288 frames):

| Comparison | Matched people | Mean box IoU | Keypoint distance | Mean \|Δscore\| |
|---|---|---|---|---|
| Same input tensor (DeepStream's), TF32 on | 96.7% | 0.9995 | mean 0.06 px | 0.0031 |
| **Same input tensor, TF32 off** | **100%** | **1.0000** | **mean 0.0014 px** | **0.0001** |
| End to end (video → detections), TF32 off | 97.6% | 0.9949 | median 0.25 px, mean 0.58 px | 0.0105 |

~31 FPS end to end (TF32 off, including decode and encode); TensorRT compute 27.6 ms/frame with TF32, 33.7 ms without.

How it works and what was needed:
- **TensorRT 8.6 can't build the exported model**: `Could not find any implementation for node {ForeignNode[transformer.keypoint_query_initializer...]}`. That subgraph doesn't depend on the image, so the CLI constant-folds it (ONNX Runtime), after which TensorRT 8.6 builds it. The folded model gives bit-identical outputs. TensorRT 10.9 (DeepStream 8.0) builds the unfolded model too; the CLI folds anyway, so one bundle works on every version.
- nvinfer can't return keypoints from a parse function, so it only attaches the raw tensors (`network-type=100`, `output-tensor-meta=1`). `libnvds_rfdetr_pose.so`, an [nvdsvideotemplate](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsvideotemplate.html) custom library, decodes them like `predict()`, including RF-DETR's keypoint-uncertainty score fusion. It attaches standard `NvDsObjectMeta`, keypoints as object user meta (`pose/nvds_rfdetr_keypoints_meta.h`), and skeleton display meta for `nvdsosd`.
- **TF32**, which TensorRT enables by default on Ampere and newer, measurably changes the fused scores. Building the engine with `NVIDIA_TF32_OVERRIDE=0` gives an exact match, at ~20% lower throughput.

## DeepStream versions

The same source works on every tested version, without code changes. Measured with `scripts/test_deepstream_version.py` (x86, RTX 4060, WSL2, FP32, TF32 off); full numbers in [`results/`](results/).

| | DeepStream 7.0 (TensorRT 8.6) | DeepStream 8.0 (TensorRT 10.9) | DeepStream 9.0 (TensorRT 10.14) |
|---|---|---|---|
| Parser and pose library build | ✅ 0 warnings | ✅ 0 warnings | ✅ 0 warnings |
| Detection, end to end | 99.2% match, IoU 0.9946, 138 FPS | 99.2% match, IoU 0.9946, 147 FPS | 99.2% match, IoU 0.9946, 152 FPS |
| Detection, same input tensor | 100%, IoU 0.99998 | 100%, IoU 0.99998 | 100%, IoU 0.99998 |
| nvinfer passes the frame unchanged (`scaling-filter=0`) | ✅ | ✅ | ✅ |
| Keypoints, end to end | 97.6%, median 0.25 px, 30 FPS | 97.6%, median 0.25 px, 31 FPS | 97.6%, median 0.25 px, 31 FPS |
| Keypoints, same input tensor | 100%, 0.0014 px | 100%, 0.0007 px | 100%, 0.0007 px |
| Unfolded keypoint ONNX builds | ❌ (constant folding needed) | ✅ | ✅ |

Planned: 9.2, then 7.1, plus arm64 (Jetson) compile checks.

**Encoding to a file:** keep an explicit `nvvideoconvert ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc` (as deepstream-app does, and as the generated bundle READMEs show). On DeepStream 8.0 the negotiated NV12 path into the encoder hangs at 1920×1080. I420 and NV12 carry the same 4:2:0 data, so there's no quality or measurable speed cost.

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
- a pretrained name: `nano`, `small`, `medium`, `large`, `base`, `keypoint`
- an RF-DETR checkpoint (`.pth` / `.ckpt`), including fine-tuned models. Class names, the class layout and (for keypoint models) the keypoint schema come from the checkpoint.
- an ONNX export (`.onnx`). Detection: pass `--class-names` (one per line) or `--coco` for the official COCO weights. Keypoints: `--keypoints-per-class` (default `0,17`, the official model) and optionally `--class-names`.

Keypoint models are detected automatically. Their bundle has `pose/` instead of `parser/`, and its README.txt contains the exact `nvdsvideotemplate` pipeline.

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

- Detection models (Nano, Small, Medium, Large, Base, and fine-tuned checkpoints).
- Keypoint models (`RFDETRKeypointPreview`, and fine-tuned keypoint checkpoints via their schema). Only the official checkpoint has been tested.
- Segmentation is not supported yet.
- FP32. FP16 works, but some transformer layers overflow and degrade detections (fix planned: keep those layers in FP32).
- Tested on DeepStream 7.0, 8.0 and 9.0 (x86, dGPU). Jetson not tested yet.

## Development

`scripts/` has:
- the parity tests: `parity_deepstream.py`, `parity_keypoints.py`, `check_reference.py`, `check_keypoint_reference.py`, `check_onnx_patch.py`, `same_input_check.py`, `scaler_check.py`
- the diagnostics used to find the preprocessing differences: `debug_*.py`
- **`test_deepstream_version.py`**, which runs everything against one DeepStream image:
  ```bash
  python scripts/test_deepstream_version.py --tag ds8.0 --image nvcr.io/nvidia/deepstream:8.0-triton-multiarch
  ```
  It builds the C++ libraries in that container, runs both pipelines, all parity checks and the scaler check, and writes `results/<tag>.json`. Engines, libraries and outputs go to `build/<tag>/`.
