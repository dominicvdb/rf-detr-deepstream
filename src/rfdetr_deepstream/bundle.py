"""Build a DeepStream bundle from an RF-DETR detection or keypoint model.

A bundle is a self-contained folder:
    <name>.onnx               the model (with preprocessing built in when a frame size is given)
    labels.txt                one line per logits slot
    config_infer_rfdetr.txt   nvinfer config
    parser/ or pose/          C++ source + Makefile; build once per platform with `make -C parser` (or pose)
    README.txt                how to use it

Detection models: nvinfer parses boxes with libnvds_rfdetr.so (parser/).
Keypoint models: nvinfer attaches the raw tensors and libnvds_rfdetr_pose.so (pose/), loaded by nvdsvideotemplate,
decodes boxes and keypoints.

The model can be a pretrained name ("nano", ..., "keypoint"), an RF-DETR checkpoint (.pth/.ckpt), or an ONNX export.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import onnx

from rfdetr_deepstream.deepstream import NvinferConfig, coco_sparse_labels, write_labels, write_nvinfer_config
from rfdetr_deepstream.fp16 import Fp16Error, make_fp16_safe
from rfdetr_deepstream.fp16_typed import convert_to_fp16
from rfdetr_deepstream.onnx_patch import add_preprocessing, fold_constants

PRETRAINED = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
    "keypoint": "RFDETRKeypointPreview",
}
PARSER_FILES = ("nvdsparsebbox_rfdetr.cpp", "Makefile")
PARSER_LIB_NAME = "libnvds_rfdetr.so"
POSE_FILES = ("nvdsrfdetr_pose.cpp", "nvds_rfdetr_keypoints_meta.h", "Makefile")
POSE_LIB_NAME = "libnvds_rfdetr_pose.so"
COCO_KEYPOINT_SCHEMA = [0, 17]
COCO_SKELETON = "15-13;13-11;16-14;14-12;11-12;5-11;6-12;5-6;5-7;6-8;7-9;8-10;1-2;0-1;0-2;1-3;2-4;3-5;4-6"


class BundleError(Exception):
    """A problem with the input model or options, with a message meant for the user."""


@dataclass
class ClassLayout:
    names: list[str]                          # one per logits slot; "" = unused slot
    background_class_id: int | None = None    # detection: slot to exclude, or None
    keypoints_per_class: list[int] = field(default_factory=list)  # keypoint models only
    trace_alpha: float = 0.2                  # keypoint models only


@dataclass
class OnnxInterface:
    input_name: str
    input_shape: list[int]            # [1, 3, H, W]
    outputs: dict[str, list[int]]     # name -> shape


def read_onnx_interface(path: Path) -> OnnxInterface:
    graph = onnx.load(str(path), load_external_data=False).graph

    def shape(value_info):
        return [d.dim_value if d.dim_value else d.dim_param for d in value_info.type.tensor_type.shape.dim]

    inp = graph.input[0]
    return OnnxInterface(inp.name, shape(inp), {o.name: shape(o) for o in graph.output})


def check_onnx(iface: OnnxInterface) -> tuple[str, int]:
    """Validate an RF-DETR export. Returns (task, number of logits slots); task is "detect" or "keypoints"."""
    if "masks" in iface.outputs:
        raise BundleError("Segmentation models are not supported yet (the ONNX has a 'masks' output).")
    if "dets" not in iface.outputs or "labels" not in iface.outputs:
        raise BundleError(f"Expected ONNX outputs 'dets' and 'labels', got {list(iface.outputs)}. "
                          "Is this an RF-DETR export?")
    shape = iface.input_shape
    if (len(shape) != 4 or not all(isinstance(d, int) and d > 0 for d in shape[1:])
            or not (isinstance(shape[0], str) or shape[0] > 0)):
        raise BundleError(f"Expected an input of shape [batch, 3, H, W] (dynamic batch) or [N, 3, H, W], got {shape}.")
    num_slots = int(iface.outputs["labels"][-1])
    if "keypoints" in iface.outputs:
        kp = iface.outputs["keypoints"]
        if len(kp) != 4 or kp[-1] < 7:
            raise BundleError(f"Unexpected keypoints output shape {kp}, expected [1, Q, C*K, D>=7].")
        return "keypoints", num_slots
    return "detect", num_slots


# --------------------------------------------------------------------------------------------------
# Class layout
# --------------------------------------------------------------------------------------------------
def layout_from_rfdetr(model, num_slots: int, task: str) -> ClassLayout:
    """Use rfdetr's own mapping, so DeepStream labels match what predict() returns."""
    from rfdetr.utilities.class_names import is_coco_pretrained, prediction_labels

    labels = prediction_labels(model)
    names = [labels.class_id_to_name.get(slot, "") for slot in range(num_slots)]
    if task == "keypoints":
        schema = list(labels.num_keypoints_per_class)
        if len(schema) != num_slots:
            raise BundleError(f"Keypoint schema {schema} doesn't match the model's {num_slots} class slots.")
        names = [n or ("background" if schema[s] == 0 else f"class_{s}") for s, n in enumerate(names)]
        trace_alpha = float(getattr(model.model_config, "postprocess_trace_alpha", 0.2))
        return ClassLayout(names, None, schema, trace_alpha)
    if is_coco_pretrained(labels.class_names, labels.num_classes):
        background = None  # sparse COCO IDs: every slot is a real category (slot 90 = toothbrush)
    elif (num_slots - 1) not in labels.class_id_to_name:
        background = -1    # contiguous IDs: the extra final slot is the untrained no-object slot
    else:
        background = None
    return ClassLayout(names, background)


def layout_from_names_file(path: Path, num_slots: int) -> ClassLayout:
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(names) == num_slots:
        return ClassLayout(names, None)
    if len(names) == num_slots - 1:
        # The usual fine-tuned layout: classes 0..N-1, plus a background slot at the end.
        return ClassLayout(names + [""], -1)
    raise BundleError(f"{path} has {len(names)} class names, but the model has {num_slots} logits slots. "
                      f"Expected {num_slots - 1} names (fine-tuned model, background slot last) or {num_slots}.")


def keypoint_layout_for_onnx(num_slots: int, keypoints_per_class: list[int] | None,
                             class_names_file: Path | None) -> ClassLayout:
    schema = keypoints_per_class
    if schema is None:
        if num_slots != len(COCO_KEYPOINT_SCHEMA):
            raise BundleError("Pass --keypoints-per-class for this keypoint ONNX (one count per class slot).")
        schema = COCO_KEYPOINT_SCHEMA
    if len(schema) != num_slots:
        raise BundleError(f"--keypoints-per-class has {len(schema)} entries, the model has {num_slots} class slots.")
    if class_names_file:
        names = [line.strip() for line in Path(class_names_file).read_text().splitlines() if line.strip()]
        if len(names) != num_slots:
            raise BundleError(f"{class_names_file} must have one name per class slot ({num_slots}).")
    elif schema == COCO_KEYPOINT_SCHEMA:
        names = ["background", "person"]
    else:
        names = ["background" if n == 0 else f"class_{s}" for s, n in enumerate(schema)]
    return ClassLayout(names, None, list(schema))


# --------------------------------------------------------------------------------------------------
# Getting an ONNX
# --------------------------------------------------------------------------------------------------
def load_rfdetr(model: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.jit deprecation notices from rfdetr's imports
        import rfdetr

        if model.lower() in PRETRAINED:
            return getattr(rfdetr, PRETRAINED[model.lower()])()
        return rfdetr.from_checkpoint(model)


def export_onnx(rf_model, out_path: Path, batch_size: int = 1, static_batch: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        # PyTorch's TracerWarnings say "this shape becomes a constant": exactly what a static engine needs.
        warnings.simplefilter("ignore")
        if static_batch:
            # Fixed batch: for TensorRT 8.6 (DeepStream 7.0), which can't build keypoint models with a dynamic batch.
            rf_model.export(format="onnx", output_dir=tmp, verbose=False, batch_size=batch_size)
        else:
            # Dynamic batch, everything else static, opset 17. nvinfer's batch-size sets the engine's maximum batch.
            rf_model.export(format="onnx", output_dir=tmp, verbose=False, dynamic_batch=True)
        exported = sorted(Path(tmp).glob("*.onnx"))
        if not exported:
            raise BundleError("rfdetr export produced no .onnx file.")
        shutil.move(str(exported[0]), out_path)
    if not static_batch:
        _static_output_dims(out_path)


def _static_output_dims(path: Path) -> None:
    """A dynamic-batch export leaves every output dim symbolic ('Concatdets_dim_1', ...). Only the batch varies:
    run once with batch 1 and write the other dims back as numbers."""
    import numpy as np
    import onnxruntime as ort

    model = onnx.load(str(path))
    inp = model.graph.input[0]
    batch_dim = inp.type.tensor_type.shape.dim[0].dim_param or "batch"
    shape = [1] + [d.dim_value for d in inp.type.tensor_type.shape.dim[1:]]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    results = session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
    for out, value in zip(model.graph.output, results):
        dims = out.type.tensor_type.shape.dim
        for i, size in enumerate(value.shape):
            if i == 0:
                dims[i].dim_param = batch_dim
            else:
                dims[i].dim_value = size
    onnx.save(model, str(path))


def _copy_package_dir(subdir: str, files: tuple[str, ...], dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    src = resources.files("rfdetr_deepstream") / subdir
    for name in files:
        (dest / name).write_bytes((src / name).read_bytes())


# --------------------------------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------------------------------
def build_bundle(
    model: str,
    output_dir: Path,
    frame_size: tuple[int, int] | None = None,   # (width, height)
    class_names_file: Path | None = None,
    coco: bool = False,
    threshold: float = 0.5,
    precision: str = "fp32",
    parser_lib: Path | None = None,
    keypoints_per_class: list[int] | None = None,
    batch_size: int = 1,
    static_batch: bool = False,
    fp16_fix: bool = True,
    calibration_images: list[Path] | None = None,
    log=print,
) -> Path:
    if batch_size < 1:
        raise BundleError("--batch-size must be at least 1.")
    # Batch 1 stays a fixed-batch export: TensorRT 8.6 can't build keypoint models with a dynamic batch at all.
    static_batch = static_batch or batch_size == 1
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_onnx = model.lower().endswith(".onnx")
    exported_here: Path | None = None

    # 1. Get the ONNX, the task and the class layout.
    if is_onnx:
        base_onnx = Path(model)
        if not base_onnx.exists():
            raise BundleError(f"{base_onnx} does not exist.")
        task, num_slots = check_onnx(read_onnx_interface(base_onnx))
        if task == "keypoints":
            layout = keypoint_layout_for_onnx(num_slots, keypoints_per_class, class_names_file)
        elif coco:
            layout = ClassLayout(coco_sparse_labels(num_slots), None)
        elif class_names_file:
            layout = layout_from_names_file(class_names_file, num_slots)
        else:
            raise BundleError("An ONNX file has no class names. Pass --class-names FILE, or --coco for the "
                              "official COCO-pretrained weights.")
        stem = base_onnx.stem
    else:
        if model.lower() not in PRETRAINED and not Path(model).exists():
            raise BundleError(f"{model} is neither a pretrained name ({', '.join(PRETRAINED)}) nor an existing file.")
        log(f"Loading RF-DETR model: {model}")
        rf_model = load_rfdetr(model)
        stem = f"rfdetr-{model.lower()}" if model.lower() in PRETRAINED else Path(model).stem
        base_onnx = exported_here = output_dir / f"{stem}_export.onnx"
        log(f"Exporting to ONNX ({f'fixed batch {batch_size}' if static_batch else 'dynamic batch'}) ...")
        export_onnx(rf_model, base_onnx, batch_size, static_batch)
        task, num_slots = check_onnx(read_onnx_interface(base_onnx))
        layout = layout_from_rfdetr(rf_model, num_slots, task)
        if class_names_file and task == "detect":
            layout = layout_from_names_file(class_names_file, num_slots)

    iface = read_onnx_interface(base_onnx)
    onnx_batch, _, model_h, model_w = iface.input_shape
    if not isinstance(onnx_batch, str) and onnx_batch != batch_size:
        raise BundleError(f"--batch-size {batch_size} doesn't fit {base_onnx.name}, which has a fixed batch of "
                          f"{onnx_batch}. Use an ONNX with a dynamic batch (dynamic_batch=True), or pass the .pth / "
                          "pretrained name.")
    if task == "keypoints":
        log(f"Keypoint model: input {model_w}x{model_h}, {num_slots} class slots, "
            f"keypoints per class {layout.keypoints_per_class}")
    else:
        log(f"Detection model: input {model_w}x{model_h}, {num_slots} class slots, "
            f"background slot: {layout.background_class_id}")

    # 2. The ONNX that goes into the bundle.
    with tempfile.TemporaryDirectory() as tmp:
        source = base_onnx
        if task == "keypoints":
            source = Path(tmp) / "folded.onnx"
            fold_constants(base_onnx, source)
            log("Constant subgraphs folded (required for the keypoint model on TensorRT 8.6).")
        if precision == "fp16" and fp16_fix:
            safe = Path(tmp) / "fp16_safe.onnx"
            try:
                plan = make_fp16_safe(source, safe, calibration_images)
            except Fp16Error as e:
                raise BundleError(f"{e}. Use --precision fp32 for this model.") from e
            source = safe
            calibrated = f"{len(calibration_images)} images" if calibration_images else "synthetic frames"
            log(f"FP16: scaled the input of {len(plan.scales)} LayerNorm(s) at risk of FP16 overflow in their "
                f"variance (calibrated on {calibrated}; smallest median deviation after scaling "
                f"{plan.smallest_median:.2g}):")
            for name, divisor in plan.scales.items():
                log(f"  1/{divisor:<4d} {name}")
            if plan.keep_fp32:
                log(f"{len(plan.keep_fp32)} LayerNorm(s) stay in FP32 (no scale keeps their small values precise in "
                    f"FP16): {', '.join(plan.keep_fp32)}")
        if frame_size:
            frame_w, frame_h = frame_size
            bundle_onnx = output_dir / f"{stem}_{frame_w}x{frame_h}.onnx"
            add_preprocessing(source, bundle_onnx, frame_height=frame_h, frame_width=frame_w)
            log(f"Preprocessing built into the ONNX for {frame_w}x{frame_h} frames.")
        else:
            bundle_onnx = output_dir / f"{stem}.onnx"
            if source.resolve() != bundle_onnx.resolve():
                shutil.copyfile(source, bundle_onnx)
            log("WARNING: no --frame-size given, so nvinfer approximates the preprocessing. Expect noticeably "
                "different detections than predict() (measured: 90% match, box IoU 0.87). Use --frame-size.")
        strongly_typed = precision == "fp16" and fp16_fix
        if strongly_typed:
            keep_names = (r"^pre_",) + tuple(f"^{re.escape(n)}$" for n in plan.keep_fp32)
            convert_to_fp16(bundle_onnx, bundle_onnx, keep_fp32_ops=(), keep_fp32_names=keep_names)
            log("FP16: ONNX converted to FP16 for a strongly typed TensorRT build (only the preprocessing"
                f"{' and the LayerNorms above' if plan.keep_fp32 else ''} in FP32).")
    if exported_here is not None and exported_here.exists():
        exported_here.unlink()

    if precision == "fp16" and not fp16_fix:
        log("WARNING: plain FP16 (--no-fp16-fix): TensorRT picks FP16 or FP32 per layer by speed, so engines can come "
            "out mostly FP32, and the LayerNorms are not scaled.")
    if precision == "fp16" and task == "keypoints":
        log("WARNING: FP16 has not been verified for keypoint models yet.")

    # 3. C++ source (built on the target), labels, config, README.
    write_labels(output_dir / "labels.txt", layout.names)
    config = NvinferConfig(
        onnx_file=bundle_onnx,
        labels_file=output_dir / "labels.txt",
        parser_lib=Path(parser_lib or output_dir / "parser" / PARSER_LIB_NAME),
        num_classes=num_slots,
        threshold=threshold,
        precision=precision,
        strongly_typed=strongly_typed,
        background_class_id=layout.background_class_id,
        preprocessing_in_model=frame_size is not None,
        task=task,
        batch_size=batch_size,
    )
    if task == "keypoints":
        _copy_package_dir("pose", POSE_FILES, output_dir / "pose")
        props = _pose_props(layout, threshold)
        _write_keypoint_readme(output_dir, bundle_onnx.name, frame_size, props, batch_size,
                               not isinstance(onnx_batch, str))
    else:
        if parser_lib is None:
            _copy_package_dir("parser", PARSER_FILES, output_dir / "parser")
        _write_detection_readme(output_dir, bundle_onnx.name, frame_size, batch_size, not isinstance(onnx_batch, str))
    write_nvinfer_config(output_dir / "config_infer_rfdetr.txt", config)
    if strongly_typed:
        readme = output_dir / "README.txt"
        readme.write_text(readme.read_text() + _fp16_note(bundle_onnx, batch_size, isinstance(onnx_batch, str)))
    log(f"Bundle written to {output_dir}/")
    return output_dir


def _fp16_note(onnx_path: Path, batch_size: int, dynamic_batch: bool) -> str:
    inp = onnx.load(str(onnx_path), load_external_data=False).graph.input[0]
    _, c, h, w = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    # network-mode=0: nvinfer names the engine after it (see deepstream.write_nvinfer_config).
    engine = f"{onnx_path.name}_b{batch_size}_gpu0_fp32.engine"
    shapes = (f" \\\n       --minShapes={inp.name}:1x{c}x{h}x{w} --optShapes={inp.name}:{batch_size}x{c}x{h}x{w}"
              f" --maxShapes={inp.name}:{batch_size}x{c}x{h}x{w}") if dynamic_batch else ""
    return f"""
FP16
  The ONNX is FP16 (only the preprocessing in FP32) and is built strongly typed: TensorRT keeps exactly these
  types instead of choosing FP16 or FP32 per layer. TensorRT refuses the FP16 flag for strongly typed networks, so
  the config has network-mode=0 and the engine file name ends in _fp32, although the engine is FP16.

  DeepStream 9.0 and later: nothing to do (strongly-typed=1 in the config).
  DeepStream 7.1 and 8.0: nvinfer has no strongly-typed option. Build the engine once on the target, in this
  folder; nvinfer then loads it:
     /usr/src/tensorrt/bin/trtexec --onnx={onnx_path.name} --stronglyTyped \\
       --saveEngine={engine}{shapes}
  Without that engine, nvinfer builds an FP32 engine (same detections as FP32, no speedup).
  DeepStream 7.0 (TensorRT 8.6): FP16 doesn't work (RF-DETR loses almost all detections). Use an FP32 bundle.

  Check: an FP16 engine is about half the size of an FP32 one.
"""


def _pose_props(layout: ClassLayout, threshold: float) -> list[str]:
    props = [
        f"threshold:{threshold}",
        f"trace-alpha:{layout.trace_alpha}",
        f"keypoints-per-class:{','.join(str(n) for n in layout.keypoints_per_class)}",
        f"labels:{';'.join(layout.names)}",
    ]
    if max(layout.keypoints_per_class) == 17:
        props.append(f"skeleton:{COCO_SKELETON}")
    return props


_ENCODE_TAIL = ("nvvideoconvert ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc ! h264parse ! qtmux ! "
                "filesink location=out.mp4")
_ENCODE_NOTE = """   Keep the explicit format=I420 before nvv4l2h264enc (as deepstream-app does): on DeepStream 8.0 the
   negotiated NV12 path hangs at 1920x1080."""

_ENGINE_NOTE = """nvinfer builds a TensorRT engine from the ONNX on first run (this takes a while) and caches it next to it
as {onnx}_b{batch}_gpu0_<precision>.engine. Engines only work on the GPU and TensorRT version that built them,
so never copy them between machines."""

_BATCH_NOTE = """   Batch size {batch}: up to {batch} streams per inference call. {batch_kind}
   Keep num-extra-surfaces on the decoders: when a batch isn't full (fewer streams than batch-size, or a stream that
   ended or dropped out), nvstreammux holds frames while it waits, and a decoder with the default number of
   surfaces can run out and stall the pipeline (seen with H.264 Main profile)."""


def _batch_note(batch: int, static: bool) -> str:
    if static:
        kind = (f"The ONNX has a fixed batch of {batch}\n   (--static-batch): keep batch-size={batch} in "
                "config_infer_rfdetr.txt and nvstreammux.")
    else:
        kind = (f"The ONNX has a dynamic batch dimension and\n   batch-size={batch} in config_infer_rfdetr.txt sets "
                f"the engine's maximum; set nvstreammux batch-size to the number\n   of streams (at most {batch}). "
                "To change the maximum, edit batch-size in the config and delete the cached engine.")
    return "\n" + _BATCH_NOTE.format(batch=batch, batch_kind=kind)


def _sources(batch: int, w, h) -> str:
    """gst-launch fragment: one decoder branch per stream into nvstreammux."""
    n = min(batch, 3)
    lines = [f"nvstreammux name=m batch-size={batch} width={w} height={h} ! \\"]
    tail = []
    for i in range(n):
        tail.append(f"     filesrc location=video{i}.mp4 ! qtdemux ! h264parse ! nvv4l2decoder num-extra-surfaces=4 ! "
                    f"m.sink_{i} \\")
    if batch > n:
        tail.append(f"     ... (one branch per stream, up to m.sink_{batch - 1})")
    else:
        tail[-1] = tail[-1].removesuffix(" \\")
    return lines, tail


def _write_detection_readme(output_dir: Path, onnx_name: str, frame_size, batch_size: int = 1,
                            static_batch: bool = False) -> None:
    mux = (f"Set nvstreammux width={frame_size[0]} height={frame_size[1]}: the model expects exactly that frame size."
           if frame_size else "Any nvstreammux size works; nvinfer resizes frames to the model input.")
    w, h = frame_size if frame_size else ("W", "H")
    if batch_size == 1:
        pipeline = f"""     gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \\
       nvstreammux name=m batch-size=1 width={w} height={h} ! \\
       nvinfer config-file-path=config_infer_rfdetr.txt ! nvvideoconvert ! nvdsosd ! \\
       {_ENCODE_TAIL}"""
        batch_note = ""
    else:
        head, tail = _sources(batch_size, w, h)
        pipeline = (f"""     gst-launch-1.0 {head[0]}
       nvinfer config-file-path=config_infer_rfdetr.txt ! nvmultistreamtiler width=1920 height=1080 ! \\
       nvvideoconvert ! nvdsosd ! {_ENCODE_TAIL} \\
""" + "\n".join(tail))
        batch_note = _batch_note(batch_size, static_batch)
    (output_dir / "README.txt").write_text(f"""RF-DETR detection bundle for DeepStream, generated by rfdetr-deepstream.

Files
  {onnx_name}       the model
  labels.txt                one class name per logits slot
  config_infer_rfdetr.txt   nvinfer config (paths are relative to this folder)
  parser/                   C++ source of the output parser

{_ENGINE_NOTE.format(onnx=onnx_name, batch=batch_size)}

1. Build the parser once on the target (inside a DeepStream container, or on a Jetson):
     make -C parser

2. Use the config in a pipeline, for example:
{pipeline}

   {mux}
{_ENCODE_NOTE}{batch_note}

Detection threshold: pre-cluster-threshold in config_infer_rfdetr.txt ([class-attrs-all], or per class
in [class-attrs-<id>]).
""")


def _write_keypoint_readme(output_dir: Path, onnx_name: str, frame_size, props: list[str], batch_size: int = 1,
                           static_batch: bool = False) -> None:
    w, h = frame_size if frame_size else ("W", "H")
    mux = (f"nvstreammux width={w} height={h} must match: the model expects exactly that frame size."
           if frame_size else "Any nvstreammux size works; nvinfer resizes frames to the model input.")
    props_args = " \\\n       ".join(f'customlib-props="{p}"' for p in props)
    infer_and_pose = f"""nvinfer config-file-path=config_infer_rfdetr.txt ! \\
       nvdsvideotemplate customlib-name=pose/{POSE_LIB_NAME} \\
       {props_args} ! \\"""
    if batch_size == 1:
        pipeline = f"""     gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \\
       nvstreammux name=m batch-size=1 width={w} height={h} ! \\
       {infer_and_pose}
       nvvideoconvert ! nvdsosd ! \\
       {_ENCODE_TAIL}"""
        batch_note = ""
    else:
        head, tail = _sources(batch_size, w, h)
        pipeline = (f"""     gst-launch-1.0 {head[0]}
       {infer_and_pose}
       nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! \\
       {_ENCODE_TAIL} \\
""" + "\n".join(tail))
        batch_note = _batch_note(batch_size, static_batch)
    (output_dir / "README.txt").write_text(f"""RF-DETR keypoint (pose) bundle for DeepStream, generated by rfdetr-deepstream.

Files
  {onnx_name}       the model (constant subgraphs folded for TensorRT 8.6)
  labels.txt                one class name per logits slot
  config_infer_rfdetr.txt   nvinfer config: no parsing, raw tensors attached as metadata (output-tensor-meta=1)
  pose/                     C++ source of libnvds_rfdetr_pose.so, an nvdsvideotemplate custom library that decodes
                            boxes + keypoints, attaches NvDsObjectMeta with keypoint user meta
                            (pose/nvds_rfdetr_keypoints_meta.h) and draws the skeleton via display meta

{_ENGINE_NOTE.format(onnx=onnx_name, batch=batch_size)}
For exact parity with RF-DETR's predict(), build the engine with TF32 disabled (about 20% slower):
     export NVIDIA_TF32_OVERRIDE=0
TF32 (the TensorRT default on Ampere and newer) noticeably changes the keypoint-uncertainty score fusion.

1. Build the library once on the target (inside a DeepStream container, or on a Jetson):
     make -C pose

2. Run from this folder, for example:
{pipeline}

   {mux}
{_ENCODE_NOTE}{batch_note}

Other customlib-props: keypoint-threshold:<float> (minimum confidence to draw a keypoint, default 0.5),
draw:0 (no display meta), gie-id:<int> (default 1), dump-dir:<path> (one text file per frame).
""")
