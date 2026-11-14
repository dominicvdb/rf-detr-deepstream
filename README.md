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

RF-DETR Nano, FP32, DeepStream 9.2 (TensorRT 10.16), RTX 4060 under WSL2. 720p sample video, 1442 frames, compared per frame with the Python reference pipeline on the same frames (see [Exact frames](#exact-frames-video-decoding-and-colour-conversion)):

| Preprocessing | Matched detections | Mean box IoU | Mean \|Δscore\| |
|---|---|---|---|
| nvinfer's own (bilinear scaler, single-scale normalization) | 90.2% | 0.869 | 0.036 |
| **Built into the ONNX (`--frame-size`)** | **100%** (13458 of 13458) | **0.999995** | **0.0000025** |

That is as close as two runtimes get. On the first 300 frames, compared with `RFDETR.predict()` itself:

| | Matched | Mean box IoU | Mean \|Δscore\| |
|---|---|---|---|
| `predict()` on CPU vs `predict()` on GPU (TF32 off) | 100% (+1 borderline) | 0.999995 | 4.2e-6 |
| `predict()` on CPU vs ONNX Runtime | 100% | 0.999995 | 2.5e-6 |
| **`predict()` on CPU vs DeepStream** | **100%** | **0.999996** | **2.7e-6** |

Bit-identical outputs aren't possible between different runtimes (PyTorch CPU and GPU aren't bit-identical either): they sum in a different order, and float rounding depends on the order.

Why nvinfer's own preprocessing falls short:
- Its scaler uses a different pixel-center convention than torchvision, which shifts boxes by about half a network pixel. With bilinear filtering it **resamples even at scale 1** (a 2×2 blur plus a half-pixel shift).
- It has one scale factor for all channels, while RF-DETR normalizes with a per-channel ImageNet std.

With `--frame-size`, the ONNX takes the full RGB frame and does the resize and normalization itself (exactly like torchvision). nvinfer runs with `scaling-filter=0`, which at scale 1 copies pixels unchanged.

## Exact frames: video decoding and colour conversion

Comparing DeepStream with Python on a video only makes sense when both see the same pixels. H.264/H.265 decoding is bit-exact, so FFmpeg gives the same YUV planes as DeepStream's hardware decoder. The conversion to RGB is not: OpenCV's differs from nvinfer's in 53–57% of all pixel values (by 1–2 levels), which is enough to flip detections near the score threshold. Compared against OpenCV-decoded frames, the same DeepStream output looks like 99.2% (detection) and 97.6% (keypoints).

nvinfer converts NV12 to RGB with BT.601 limited range and truncation, chroma repeated over 2×2 pixels:
- DeepStream 8.0 and later: exact BT.601 coefficients.
- DeepStream 7.x: 4-decimal coefficients in float32 (1.1644, 1.596, 0.3918, 0.813, 2.0172).

`rfdetr_deepstream.video.read_frames(path, deepstream_version=...)` decodes with PyAV (`pip install av`) and applies that conversion. It matches nvinfer's input tensor dump on every pixel, on DeepStream 7.0, 7.1, 8.0, 9.0 and 9.2 (720p and 1080p H.264). Use it to feed `predict()` the frames DeepStream sees:

```python
from rfdetr import RFDETRNano
from rfdetr_deepstream.video import read_frames

model = RFDETRNano()
for rgb in read_frames("video.mp4", deepstream_version="9.2"):
    detections = model.predict(rgb, threshold=0.5)
```

## Results: keypoints (RF-DETR KeypointPreview)

The official `RFDETRKeypointPreview` checkpoint (COCO person, 17 keypoints, 576×576, XLarge backbone), FP32, same setup. `sample_walk.mov` (1920×1080, 288 frames):

| Comparison | Matched people | Mean box IoU | Keypoint distance | Mean \|Δscore\| |
|---|---|---|---|---|
| Same input tensor (DeepStream's), TF32 on | 96.7% | 0.9995 | mean 0.06 px | 0.0031 |
| Same input tensor, TF32 off | 100% | 0.99999 | mean 0.0007 px | 0.00001 |
| **End to end (video → detections), TF32 off** | **100%** (254 of 254) | **0.99999** | **median 0.0004 px** | **0.000006** |

TF32-on row measured on DeepStream 7.0, the others on 9.2. ~30 FPS end to end (TF32 off, including decode, drawing and encode); TensorRT compute 27.6 ms/frame with TF32, 33.7 ms without.

How it works and what was needed:
- **TensorRT 8.6 can't build the exported model**: `Could not find any implementation for node {ForeignNode[transformer.keypoint_query_initializer...]}`. That subgraph doesn't depend on the image, so the CLI constant-folds it (ONNX Runtime), after which TensorRT 8.6 builds it. The folded model gives bit-identical outputs. TensorRT 10.3 and later (DeepStream 7.1+) build the unfolded model too; the CLI folds anyway, so one bundle works on every version. ONNX Runtime writes default attributes out explicitly when it saves the folded model, e.g. `reduction="none"` on ScatterND, which TensorRT 10.3 rejects (`Attribute reduction is not supported`), so the CLI removes attributes that equal their default afterwards.
- nvinfer can't return keypoints from a parse function, so it only attaches the raw tensors (`network-type=100`, `output-tensor-meta=1`). `libnvds_rfdetr_pose.so`, an [nvdsvideotemplate](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsvideotemplate.html) custom library, decodes them like `predict()`, including RF-DETR's keypoint-uncertainty score fusion. It attaches standard `NvDsObjectMeta`, keypoints as object user meta (`pose/nvds_rfdetr_keypoints_meta.h`), and skeleton display meta for `nvdsosd`.
- **TF32**, which TensorRT enables by default on Ampere and newer, measurably changes the fused scores. Building the engine with `NVIDIA_TF32_OVERRIDE=0` gives an exact match, at ~20% lower throughput.

## DeepStream versions

The same source works on every tested version, without code changes. Measured with `scripts/test_deepstream_version.py` (x86, RTX 4060, WSL2, FP32, TF32 off); full numbers in [`results/`](results/).

| | DeepStream 7.0 (TensorRT 8.6) | DeepStream 7.1 (TensorRT 10.3) | DeepStream 8.0 (TensorRT 10.9) | DeepStream 9.0 (TensorRT 10.14) | DeepStream 9.2 (TensorRT 10.16) |
|---|---|---|---|---|---|
| Parser and pose library build | ✅ 0 warnings | ✅ 0 warnings | ✅ 0 warnings | ✅ 0 warnings | ✅ 0 warnings |
| Detection, end to end | 100%, IoU 0.999996 | 100%, IoU 0.999996 | 100%, IoU 0.999996 | 100%, IoU 0.999995 | 100%, IoU 0.999995 |
| Detection, same input tensor | 100%, IoU 0.99998 | 100%, IoU 0.99999 | 100%, IoU 0.99998 | 100%, IoU 0.99999 | 100%, IoU 0.99998 |
| nvinfer passes the frame unchanged (`scaling-filter=0`), bit-identical | ✅ | ✅ | ✅ | ✅ | ✅ |
| Keypoints, end to end | 100%, median 0.0004 px | 100%, median 0.0004 px | 100%, median 0.0004 px | 100%, median 0.0004 px | 100%, median 0.0004 px |
| Keypoints, same input tensor | 100%, 0.0014 px | 100%, 0.0016 px | 100%, 0.0007 px | 100%, 0.0007 px | 100%, 0.0007 px |
| Unfolded keypoint ONNX builds | ❌ (constant folding needed) | ✅ | ✅ | ✅ | ✅ |

End to end is compared against the exact frames each version produces (see [Exact frames](#exact-frames-video-decoding-and-colour-conversion)). On 7.0, 7.1 and 9.0, DeepStream reports one extra detection out of ~13,450, right at the threshold.

**Speed.** Absolute FPS on this machine varies by up to ±20% between sessions, so versions are compared in alternating runs within one session:
- Detection: no difference between versions. Steady state 120–185 FPS depending on session and sinks.
- Keypoints: ~30 FPS on 7.0–9.0; on 9.2, 15–30% lower. The cause is `nvdsosd`: on 9.2 it needs about 6 ms more GPU time per frame to draw the keypoint display meta, and a `queue` doesn't hide it because the keypoint model already keeps the GPU busy. Decoding, inference, the pose library and encoding are as fast as on 9.0. Without `nvdsosd` (e.g. when only the metadata is needed), 9.2 runs at the same speed.
- deepstream-app's running "(Avg)" FPS includes startup, which differs between versions (with file output, 9.2 has a stall of a few seconds before frames flow). `test_deepstream_version.py` reports the steady-state median instead.

**Jetson (arm64):** compile-checked only, not run on a device. In the DeepStream 7.0 and 7.1 Jetson images (`--platform linux/arm64`, QEMU), the parser builds with 0 warnings, and the pose library compiles with 0 warnings. Its final link needs `libnvbufsurface`, `libnvbufsurftransform` and `libnvdsbufferpool`, which the Jetson images only contain as links into `/usr/lib/aarch64-linux-gnu/tegra/`; the device fills that directory at run time. Linked without them, the only unresolved symbol is `gst_nvds_buffer_pool_new` (from `libnvdsbufferpool`), so `make` is expected to work on a Jetson.

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
- Tested on DeepStream 7.0, 7.1, 8.0, 9.0 and 9.2 (x86, dGPU). Jetson: compiles in the DeepStream 7.0 and 7.1 Jetson images, not run on a device yet.

## Development

`scripts/` has:
- the parity tests: `parity_deepstream.py`, `parity_keypoints.py`, `check_reference.py`, `check_keypoint_reference.py`, `check_onnx_patch.py`, `same_input_check.py`, `scaler_check.py`. The end-to-end tests read frames with `rfdetr_deepstream.video` (pass `--ds-version`), so they compare the model and parser, not two colour conversions.
- the diagnostics used to find the preprocessing differences: `debug_*.py`
- **`test_deepstream_version.py`**, which runs everything against one DeepStream image:
  ```bash
  python scripts/test_deepstream_version.py --tag ds8.0 --image nvcr.io/nvidia/deepstream:8.0-triton-multiarch
  ```
  It builds the C++ libraries in that container, runs both pipelines, all parity checks and the scaler check, and writes `results/<tag>.json`. Engines, libraries and outputs go to `build/<tag>/`.
