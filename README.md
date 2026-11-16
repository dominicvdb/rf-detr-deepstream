# rfdetr-deepstream

Run [RF-DETR](https://github.com/roboflow/rf-detr) detection and keypoint (pose) models in NVIDIA DeepStream.

`rfdetr-deepstream build` turns an RF-DETR model into a self-contained DeepStream folder: an ONNX model with RF-DETR's preprocessing built in, an nvinfer config, labels, and the C++ source of the output parser. Copy the folder to the target, run `make`, and point nvinfer at the config. In FP32, detections match RF-DETR's own `predict()` exactly; FP16 (DeepStream 7.1 and later) is 2.4–4.5× faster and keeps 99.5–100% of the detections.

```bash
rfdetr-deepstream build nano --frame-size 1280x720 -o bundle/
make -C bundle/parser                 # once, on the target (DeepStream container or Jetson)
# then use bundle/config_infer_rfdetr.txt as the nvinfer config-file-path

rfdetr-deepstream build keypoint --frame-size 1920x1080 -o pose_bundle/
make -C pose_bundle/pose              # see pose_bundle/README.txt for the pipeline
```

## Install

```bash
pip install -e ".[torch]"     # to build from a pretrained name or a .pth checkpoint
pip install -e .              # enough to build from an .onnx export
```

The machine that builds the bundle needs no DeepStream or GPU. The target needs DeepStream (with `make` and `g++`, as in NVIDIA's DeepStream containers).

## Building a bundle

```bash
rfdetr-deepstream build MODEL -o OUT_DIR --frame-size WxH [options]
```

`MODEL` is one of:
- a pretrained name: `nano`, `small`, `medium`, `large`, `base`, `keypoint`
- an RF-DETR checkpoint (`.pth` / `.ckpt`), including fine-tuned models. Class names and, for keypoint models, the keypoint layout are read from the checkpoint.
- an ONNX export (`.onnx`). Detection: add `--class-names FILE` (one name per line) or `--coco` for the official COCO weights. Keypoints: `--keypoints-per-class` (default `0,17`, the official model).

Keypoint models are detected automatically. The input normalisation (mean/std) is read from the model; an ONNX doesn't record it, so RF-DETR's fixed ImageNet values are used. Only 3-channel (RGB) models are supported, since DeepStream feeds RGB frames.

| Option | |
|---|---|
| `--frame-size WxH` | Size of the frames nvinfer receives (the `nvstreammux` width and height). Builds the preprocessing into the ONNX, so DeepStream gives the same results as `predict()`. Strongly recommended; without it nvinfer approximates the preprocessing and detections differ noticeably. |
| `--batch-size N` | Maximum number of streams per inference call (default 1). Above 1 the ONNX gets a dynamic batch dimension. |
| `--static-batch` | Fixed batch of `--batch-size` instead of a dynamic one. Needed for keypoint models with batch > 1 on DeepStream 7.0. |
| `--threshold 0.5` | Detection threshold. |
| `--class-names FILE` | Class names, one per line. Overrides a checkpoint's names. |
| `--precision fp32\|fp16` | TensorRT precision (default FP32). FP16 needs TensorRT 10 (DeepStream 7.1 or later), see [FP16](#fp16). `--calibration-images IMG ...` measures the FP16 ranges on your frames instead of synthetic ones; `--no-fp16-fix` builds plain FP16 (to compare). |
| `--parser-lib PATH` | Detection: use an already built parser library instead of shipping its source. |

### FP16

`rfdetr-deepstream build MODEL --frame-size WxH --precision fp16 -o DIR` gives a bundle that runs 2.4–4.5× faster than FP32 with nearly the same detections.

**DeepStream versions:**
- **9.0 and later:** nothing to do.
- **7.1 and 8.0:** build the engine once on the target with `trtexec --stronglyTyped`; the bundle's README.txt has the exact command. Without it nvinfer builds an FP32 engine: same detections as FP32, no speedup.
- **7.0:** not supported. TensorRT 8.6 builds FP16 engines that lose almost all detections.

**Measured** on an RTX 4060 with all pretrained detection models (1280×720 bundles, DeepStream 7.1, 8.0, 9.0 and 9.2, first 100 frames of `sample_720p.mp4`), compared frame by frame with RF-DETR in ONNX Runtime (FP32):

| Model | Detections matched (FP16) | Box IoU (FP16) | GPU time FP32 | GPU time FP16 | Speedup |
|---|---|---|---|---|---|
| Nano | 99.5–99.9% | 0.995 | 5.2 ms | 2.1 ms | 2.4× |
| Small | 99.7–99.8% | 0.994 | 9.8 ms | 3.3 ms | 3.0× |
| Medium | 99.7–100% | 0.994 | 13.2 ms | 3.9 ms | 3.4× |
| Base | 99.8–100% | 0.994 | 13.5 ms | 3.9 ms | 3.4× |
| Large | 99.6–99.8% | 0.993 | 21.6 ms | 4.8 ms | 4.5× |

The range is over the four DeepStream versions; GPU time per frame from `trtexec`. FP32 matches 100% (checked on Nano and Medium over the full 1442 frames). The detections FP16 misses are close to the threshold: rounding moves their score across it. An FP16 engine is about half the size of an FP32 one, a quick check that the engine really is FP16.

**How it works.** The bundle's ONNX is converted to FP16 (only the built-in preprocessing stays FP32) and built **strongly typed**: TensorRT runs every layer in the type the ONNX gives it. In nvinfer's normal FP16 mode TensorRT picks FP16 or FP32 per layer by measured speed, so the same ONNX can give an FP16 engine on one build and a mostly FP32 one on the next. Before converting, the build runs the model in FP32 on calibration frames (synthetic, or `--calibration-images`) and:
- divides the input of each LayerNorm whose squared deviations would overflow FP16 (above 65,504) by a power of two. A LayerNorm gives the same result for a scaled input, so the FP32 model is unchanged. The pretrained models have 8 such LayerNorms (Base: 2). A LayerNorm whose small values would lose FP16 precision after scaling stays FP32 instead;
- stops with an error if any other tensor gets within 4× of FP16's maximum (none does in the pretrained models).

TensorRT refuses the FP16 flag on a strongly typed network, so the config has `network-mode=0` and `strongly-typed=1`, and the engine file name ends in `_fp32` although the engine is FP16. `scripts/fp16_ranges.py` shows the LayerNorm ranges of a model.

## What's in a bundle

```
<model>_<W>x<H>.onnx       the model, preprocessing built in
labels.txt                 one class name per output slot
config_infer_rfdetr.txt    nvinfer config (paths relative to the bundle)
parser/                    detection: C++ source + Makefile of the nvinfer parser   -> libnvds_rfdetr.so
pose/                      keypoints: C++ source + Makefile of the pose library      -> libnvds_rfdetr_pose.so
README.txt                 how to build it and the exact pipeline to run it
```

No TensorRT engine is included: engines only work on the GPU and TensorRT version that built them. nvinfer builds one from the ONNX on the first run (this takes a few minutes) and caches it next to the ONNX.

**Detection** uses nvinfer's standard custom parser interface, so the output is normal `NvDsObjectMeta` that trackers, analytics and `nvdsosd` understand.

**Keypoints:** nvinfer can't return keypoints from a parser, so it passes the raw output tensors on, and `libnvds_rfdetr_pose.so` decodes them inside NVIDIA's `nvdsvideotemplate` element. It attaches standard `NvDsObjectMeta` per person, the keypoints as object user meta (the struct is in `pose/nvds_rfdetr_keypoints_meta.h`), and skeleton display meta for `nvdsosd`.

## Running it

Build the library once on the target:

```bash
make -C bundle/parser     # detection
make -C bundle/pose       # keypoints
```

One stream, detection:

```bash
gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
  nvstreammux name=m batch-size=1 width=1280 height=720 ! \
  nvinfer config-file-path=bundle/config_infer_rfdetr.txt ! nvvideoconvert ! nvdsosd ! \
  nvvideoconvert ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4
```

Several streams (bundle built with `--batch-size 4`):

```bash
gst-launch-1.0 nvstreammux name=m batch-size=4 width=1280 height=720 ! \
  nvinfer config-file-path=bundle/config_infer_rfdetr.txt ! nvmultistreamtiler width=1920 height=1080 ! \
  nvvideoconvert ! nvdsosd ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc ! \
  h264parse ! qtmux ! filesink location=out.mp4 \
  filesrc location=cam0.mp4 ! qtdemux ! h264parse ! nvv4l2decoder num-extra-surfaces=4 ! m.sink_0 \
  filesrc location=cam1.mp4 ! qtdemux ! h264parse ! nvv4l2decoder num-extra-surfaces=4 ! m.sink_1 \
  ...
```

Each bundle's README.txt has the same pipelines for its model, including the `nvdsvideotemplate` properties for keypoints. The config also works in `deepstream-app` (`[primary-gie] config-file=...`).

Things to know:
- **Frame size:** `nvstreammux` `width`/`height` must equal `--frame-size`. All streams are scaled to that size by the muxer.
- **Batching:** set `nvstreammux batch-size` to the number of streams (at most the bundle's `--batch-size`). Keep `num-extra-surfaces=4` on the decoders: when a batch isn't full (a stream ended or dropped out), the muxer holds frames while it waits and a decoder with the default number of buffers can run out, which stalls the pipeline.
- **Keypoints:** build the engine with `NVIDIA_TF32_OVERRIDE=0` set for results that match `predict()`. TF32, TensorRT's default on Ampere and newer GPUs, changes the keypoint confidence noticeably (about 20% faster).
- **Encoding to a file:** keep the explicit `format=I420` caps before `nvv4l2h264enc`. On DeepStream 8.0, the negotiated NV12 path hangs at 1920×1080.

## Supported

| | 7.0 | 7.1 | 8.0 | 9.0 | 9.2 |
|---|---|---|---|---|---|
| Detection | ✅ | ✅ | ✅ | ✅ | ✅ |
| Keypoints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Detection, batch > 1 | ✅ | | | | ✅ |
| Keypoints, batch > 1 | ✅ `--static-batch`¹ | | | | ✅ |
| Detection, FP16 | ❌ | ✅ `trtexec`² | ✅ `trtexec`² | ✅ | ✅ |
| Keypoints, FP16 | ❌ | engine builds³ | engine builds³ | engine builds³ | engine builds³ |

Tested on x86 with an RTX 4060 (WSL2). Empty cells weren't tested; they use the same code as the versions around them. On Jetson, both libraries compile in the DeepStream 7.0 and 7.1 Jetson images but haven't been run on a device yet.

¹ TensorRT 8.6 can't build the keypoint model with a dynamic batch. With `--static-batch`, the batch-2 engine builds on an 8 GB GPU (not yet run end to end); batch 4 runs out of memory there.

² Engine built with the `trtexec --stronglyTyped` command from the bundle's README.txt. FP16 detection was tested with all pretrained detection models (Nano, Small, Medium, Large, Base), batch 1.

³ The FP16 engine builds (strongly typed, FP16 size); keypoint accuracy in FP16 hasn't been measured yet.

Rough pipeline speed on an RTX 4060 (`deepstream-app`, one 720p stream): Nano ~150 FPS in FP32 and ~300 FPS in FP16, Medium ~75 FPS in FP32 and ~200 FPS in FP16. Nano FP32 with 4 streams: ~190 FPS in total. The keypoint model (XLarge backbone): ~30 FPS at 1080p in FP32.

## Limitations

- **FP16** needs TensorRT 10 (DeepStream 7.1 or later), and on 7.1 and 8.0 a `trtexec` step. Keypoint accuracy and batch > 1 haven't been measured in FP16 yet. FP16 results are close to FP32 but not identical (see [FP16](#fp16)).
- **Fixed input size.** The bundle is built for one frame size; all streams go through `nvstreammux` at that size.
- **Segmentation models** aren't supported yet.
- **Tested models:** all pretrained detection models (Nano, Small, Medium, Large, Base) in FP16, Nano and Medium also in FP32, and the official `RFDETRKeypointPreview` checkpoint. Fine-tuned checkpoints go through the same code but haven't been run in DeepStream yet.

## Development

`pytest` runs the test suite in a few seconds, without DeepStream or a GPU (`pip install -e ".[dev]"`; the parser tests also need a C++ compiler). It uses small synthetic ONNX models with RF-DETR's interface and covers the reference decoder (against rfdetr's own), the preprocessing built into the ONNX (against `predict()`'s), bundle generation and the nvinfer config, the C++ parser (compiled with a stand-in for DeepStream's header, against the reference), the video colour conversion, and the FP16 conversion (LayerNorm scaling, range checks, strongly typed ONNX).

`scripts/test_deepstream_version.py --tag ds9.2 --image <DeepStream image>` builds the libraries in a DeepStream container, runs detection and keypoint pipelines, and compares every frame with RF-DETR in Python. Results per version are in [`results/`](results/). The other scripts in `scripts/` are the individual checks and diagnostics it uses. To run a single bundle, `scripts/ds_run.py --image <DeepStream image> --bundle DIR --name RUN` runs it through `deepstream-app` (KITTI output, steady-state FPS); `scripts/parity_deepstream.py` then compares that output with the Python reference.
