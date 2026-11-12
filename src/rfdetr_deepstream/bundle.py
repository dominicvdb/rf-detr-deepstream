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

import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import onnx

from rfdetr_deepstream.deepstream import NvinferConfig, coco_sparse_labels, write_labels, write_nvinfer_config
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
    if len(shape) != 4 or not all(isinstance(d, int) and d > 0 for d in shape) or shape[0] != 1:
        raise BundleError(f"Expected a static input of shape [1, 3, H, W], got {shape}. "
                          "Export with batch_size=1 and dynamic_batch=False.")
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


def export_onnx(rf_model, out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        # PyTorch's TracerWarnings say "this shape becomes a constant": exactly what a static engine needs.
        warnings.simplefilter("ignore")
        rf_model.export(format="onnx", output_dir=tmp, verbose=False)  # batch 1, static shapes, opset 17
        exported = sorted(Path(tmp).glob("*.onnx"))
        if not exported:
            raise BundleError("rfdetr export produced no .onnx file.")
        shutil.move(str(exported[0]), out_path)


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
    log=print,
) -> Path:
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
        log("Exporting to ONNX ...")
        export_onnx(rf_model, base_onnx)
        task, num_slots = check_onnx(read_onnx_interface(base_onnx))
        layout = layout_from_rfdetr(rf_model, num_slots, task)
        if class_names_file and task == "detect":
            layout = layout_from_names_file(class_names_file, num_slots)

    iface = read_onnx_interface(base_onnx)
    _, _, model_h, model_w = iface.input_shape
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
    if exported_here is not None and exported_here.exists():
        exported_here.unlink()

    if precision == "fp16":
        log("WARNING: FP16 is known to overflow in some transformer layers of RF-DETR, which degrades detections.")

    # 3. C++ source (built on the target), labels, config, README.
    write_labels(output_dir / "labels.txt", layout.names)
    config = NvinferConfig(
        onnx_file=bundle_onnx,
        labels_file=output_dir / "labels.txt",
        parser_lib=Path(parser_lib or output_dir / "parser" / PARSER_LIB_NAME),
        num_classes=num_slots,
        threshold=threshold,
        precision=precision,
        background_class_id=layout.background_class_id,
        preprocessing_in_model=frame_size is not None,
        task=task,
    )
    if task == "keypoints":
        _copy_package_dir("pose", POSE_FILES, output_dir / "pose")
        props = _pose_props(layout, threshold)
        _write_keypoint_readme(output_dir, bundle_onnx.name, frame_size, props)
    else:
        if parser_lib is None:
            _copy_package_dir("parser", PARSER_FILES, output_dir / "parser")
        _write_detection_readme(output_dir, bundle_onnx.name, frame_size)
    write_nvinfer_config(output_dir / "config_infer_rfdetr.txt", config)
    log(f"Bundle written to {output_dir}/")
    return output_dir


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


_ENGINE_NOTE = """nvinfer builds a TensorRT engine from the ONNX on first run (this takes a while) and caches it next to it
as {onnx}_b1_gpu0_<precision>.engine. Engines only work on the GPU and TensorRT version that built them,
so never copy them between machines."""


def _write_detection_readme(output_dir: Path, onnx_name: str, frame_size) -> None:
    mux = (f"Set nvstreammux width={frame_size[0]} height={frame_size[1]}: the model expects exactly that frame size."
           if frame_size else "Any nvstreammux size works; nvinfer resizes frames to the model input.")
    (output_dir / "README.txt").write_text(f"""RF-DETR detection bundle for DeepStream, generated by rfdetr-deepstream.

Files
  {onnx_name}       the model
  labels.txt                one class name per logits slot
  config_infer_rfdetr.txt   nvinfer config (paths are relative to this folder)
  parser/                   C++ source of the output parser

{_ENGINE_NOTE.format(onnx=onnx_name)}

1. Build the parser once on the target (inside a DeepStream container, or on a Jetson):
     make -C parser

2. Use the config in a pipeline, for example:
     gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \\
       nvstreammux name=m batch-size=1 width=W height=H ! \\
       nvinfer config-file-path=config_infer_rfdetr.txt ! nvvideoconvert ! nvdsosd ! \\
       nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4

   {mux}

Detection threshold: pre-cluster-threshold in config_infer_rfdetr.txt ([class-attrs-all], or per class
in [class-attrs-<id>]).
""")


def _write_keypoint_readme(output_dir: Path, onnx_name: str, frame_size, props: list[str]) -> None:
    w, h = frame_size if frame_size else ("W", "H")
    mux = (f"nvstreammux width={w} height={h} must match: the model expects exactly that frame size."
           if frame_size else "Any nvstreammux size works; nvinfer resizes frames to the model input.")
    props_args = " \\\n       ".join(f'customlib-props="{p}"' for p in props)
    (output_dir / "README.txt").write_text(f"""RF-DETR keypoint (pose) bundle for DeepStream, generated by rfdetr-deepstream.

Files
  {onnx_name}       the model (constant subgraphs folded for TensorRT 8.6)
  labels.txt                one class name per logits slot
  config_infer_rfdetr.txt   nvinfer config: no parsing, raw tensors attached as metadata (output-tensor-meta=1)
  pose/                     C++ source of libnvds_rfdetr_pose.so, an nvdsvideotemplate custom library that decodes
                            boxes + keypoints, attaches NvDsObjectMeta with keypoint user meta
                            (pose/nvds_rfdetr_keypoints_meta.h) and draws the skeleton via display meta

{_ENGINE_NOTE.format(onnx=onnx_name)}
For exact parity with RF-DETR's predict(), build the engine with TF32 disabled (about 20% slower):
     export NVIDIA_TF32_OVERRIDE=0
TF32 (the TensorRT default on Ampere and newer) noticeably changes the keypoint-uncertainty score fusion.

1. Build the library once on the target (inside a DeepStream container, or on a Jetson):
     make -C pose

2. Run from this folder, for example:
     gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \\
       nvstreammux name=m batch-size=1 width={w} height={h} ! \\
       nvinfer config-file-path=config_infer_rfdetr.txt ! \\
       nvdsvideotemplate customlib-name=pose/{POSE_LIB_NAME} \\
       {props_args} ! \\
       nvvideoconvert ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4

   {mux}

Other customlib-props: keypoint-threshold:<float> (minimum confidence to draw a keypoint, default 0.5),
draw:0 (no display meta), gie-id:<int> (default 1), dump-dir:<path> (one text file per frame).
""")
