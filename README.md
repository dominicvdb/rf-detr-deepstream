# rfdetr-deepstream

Run [RF-DETR](https://github.com/roboflow/rf-detr) detection and keypoint (pose) models in NVIDIA DeepStream.

`rfdetr-deepstream build` turns an RF-DETR model into a self-contained DeepStream folder: an ONNX model with RF-DETR's preprocessing built in, an nvinfer config, labels, and the C++ source of the output parser. Copy the folder to the target, run `make`, and point nvinfer at the config. Detections match RF-DETR's own `predict()`.

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

Keypoint models are detected automatically.

| Option | |
|---|---|
| `--frame-size WxH` | Size of the frames nvinfer receives (the `nvstreammux` width and height). Builds the preprocessing into the ONNX, so DeepStream gives the same results as `predict()`. Strongly recommended; without it nvinfer approximates the preprocessing and detections differ noticeably. |
| `--batch-size N` | Maximum number of streams per inference call (default 1). Above 1 the ONNX gets a dynamic batch dimension. |
| `--static-batch` | Fixed batch of `--batch-size` instead of a dynamic one. Needed for keypoint models with batch > 1 on DeepStream 7.0. |
| `--threshold 0.5` | Detection threshold. |
| `--class-names FILE` | Class names, one per line. Overrides a checkpoint's names. |
| `--precision fp32\|fp16` | TensorRT precision (default FP32). FP16 makes the model FP16-safe first, see [FP16](#fp16). `--no-fp16-fix` skips that (to compare); `--calibration-images IMG ...` measures on your frames instead of synthetic ones. |
| `--parser-lib PATH` | Detection: use an already built parser library instead of shipping its source. |

### FP16

Plain RF-DETR breaks in FP16. A few LayerNorms (decoder layer 0 `norm1`, the projector's `cv1/bn`, and some backbone layers) get inputs whose values fit FP16 but whose squared deviations, used in the variance, go far beyond FP16's maximum of 65,504. The variance overflows and those layers' outputs collapse, which can cost most detections. With `--precision fp16`, the build divides the input of each at-risk LayerNorm by a power of two. A LayerNorm gives the same result for a scaled input, apart from the ε term (10⁻⁶ × s², negligible), so the FP32 model is unchanged, but the squares stay at least 16× below the limit. The at-risk layers are found by running the model on synthetic frames (or `--calibration-images`). The largest values come from the weights and appear for any input.

`scripts/fp16_ranges.py` shows the LayerNorm ranges of a model. `scripts/fp16_simulate.py` simulates FP16 in ONNX Runtime (every tensor rounded to FP16, LayerNorms computed step by step in FP16) and compares plain FP16, FP16 with the fix, and FP32 on labelled frames. This is a simulation of FP16, not TensorRT itself. It hasn't been run on a GPU yet (see Limitations).

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

Tested on x86 with an RTX 4060 (WSL2). Empty cells weren't tested; they use the same code as the versions around them. On Jetson, both libraries compile in the DeepStream 7.0 and 7.1 Jetson images but haven't been run on a device yet.

¹ TensorRT 8.6 can't build the keypoint model with a dynamic batch. With `--static-batch`, the batch-2 engine builds on an 8 GB GPU (not yet run end to end); batch 4 runs out of memory there.

Rough speed on an RTX 4060, FP32: RF-DETR Nano ~150 FPS for one 720p stream, ~190 FPS in total with 4 streams; the keypoint model (XLarge backbone) ~30 FPS at 1080p.

## Limitations

- **FP16 hasn't been run in TensorRT yet.** The fix is verified in an FP16 simulation (see [FP16](#fp16)). Until it's checked on a GPU, FP32 is the safe choice.
- **Fixed input size.** The bundle is built for one frame size; all streams go through `nvstreammux` at that size.
- **Segmentation models** aren't supported yet.
- **Tested models:** RF-DETR Nano and the official `RFDETRKeypointPreview` checkpoint. The other sizes and fine-tuned checkpoints go through the same code but haven't been run in DeepStream yet.

## Development

`pytest` runs the test suite in a few seconds, without DeepStream or a GPU (`pip install -e ".[dev]"`; the parser tests also need a C++ compiler). It uses small synthetic ONNX models with RF-DETR's interface and covers the reference decoder (against rfdetr's own), the preprocessing built into the ONNX (against `predict()`'s), bundle generation and the nvinfer config, the C++ parser (compiled with a stand-in for DeepStream's header, against the reference), the video colour conversion, and the FP16 LayerNorm fix.

`scripts/test_deepstream_version.py --tag ds9.2 --image <DeepStream image>` builds the libraries in a DeepStream container, runs detection and keypoint pipelines, and compares every frame with RF-DETR in Python. Results per version are in [`results/`](results/). The other scripts in `scripts/` are the individual checks and diagnostics it uses.
