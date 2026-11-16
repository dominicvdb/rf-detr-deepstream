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
| `--resize bilinear\|antialias` | Resize built into the ONNX (needs `--frame-size`). `bilinear` (default) matches `predict()` exactly. `antialias` smooths before downscaling, as rfdetr training (PIL) and cfg-detect evaluation do. See [Resize](#resize-bilinear-or-antialias). |
| `--batch-size N` | Maximum number of streams per inference call (default 1). Above 1 the ONNX gets a dynamic batch dimension. |
| `--static-batch` | Fixed batch of `--batch-size` instead of a dynamic one. Needed for keypoint models with batch > 1 on DeepStream 7.0. |
| `--threshold 0.5` | Detection threshold. |
| `--class-names FILE` | Class names, one per line. Overrides a checkpoint's names. |
| `--precision fp32\|fp16` | TensorRT precision. Use FP32 for now (see Limitations). |
| `--parser-lib PATH` | Detection: use an already built parser library instead of shipping its source. |

### Resize: bilinear or antialias

RF-DETR trains on images resized with PIL, which smooths before shrinking (antialiasing). `predict()` resizes without smoothing. At a large downscale (1920×1080 to 384×384 is 5×) the two give visibly different inputs. The default `--resize bilinear` matches `predict()`. `--resize antialias` matches training instead: a fixed-size antialiased resize is two matrix multiplications with precomputed weights (+4.6 MB in the ONNX at 1920×1080), equal to torchvision's `resize(antialias=True)` to float precision and within 1 grey level of PIL's 8-bit resize.

Which one scores better depends on the model, and it isn't always antialias. Run `scripts/check_resize_local.py` on a new model's test set to see whether it matters. It tends to matter more for small or thin objects.

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

- **FP32 only for now.** FP16 runs, but some transformer layers overflow and detections degrade. A fix (keeping those layers in FP32) is planned.
- **`--resize antialias` hasn't been run in DeepStream yet** (TensorRT MatMul; checked with ONNX Runtime only).
- **Fixed input size.** The bundle is built for one frame size; all streams go through `nvstreammux` at that size.
- **Segmentation models** aren't supported yet.
- **Tested models:** RF-DETR Nano and the official `RFDETRKeypointPreview` checkpoint. The other sizes and fine-tuned checkpoints go through the same code but haven't been run in DeepStream yet.

## Development

`scripts/test_deepstream_version.py --tag ds9.2 --image <DeepStream image>` builds the libraries in a DeepStream container, runs detection and keypoint pipelines, and compares every frame with RF-DETR in Python. Results per version are in [`results/`](results/). The other scripts in `scripts/` are the individual checks and diagnostics it uses.
