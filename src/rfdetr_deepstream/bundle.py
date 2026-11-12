"""Build a DeepStream bundle from an RF-DETR detection model.

A bundle is a self-contained folder:
    <name>.onnx               the model (with preprocessing built in when a frame size is given)
    labels.txt                one line per logits slot
    config_infer_rfdetr.txt   nvinfer config
    parser/                   C++ parser source + Makefile; build once per platform with `make -C parser`
    README.txt                how to use it

The model can be a pretrained variant name ("nano", ...), an RF-DETR checkpoint (.pth/.ckpt), or an ONNX export.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import onnx

from rfdetr_deepstream.deepstream import NvinferConfig, coco_sparse_labels, write_labels, write_nvinfer_config
from rfdetr_deepstream.onnx_patch import add_preprocessing

PRETRAINED = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}
PARSER_FILES = ("nvdsparsebbox_rfdetr.cpp", "Makefile")
PARSER_LIB_NAME = "libnvds_rfdetr.so"


class BundleError(Exception):
    """A problem with the input model or options, with a message meant for the user."""


@dataclass
class ClassLayout:
    names: list[str]                  # one per logits slot; "" = unused slot
    background_class_id: int | None   # slot to exclude, or None


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


def check_detection_onnx(iface: OnnxInterface) -> int:
    """Validate an RF-DETR detection export and return its number of logits slots."""
    if "masks" in iface.outputs:
        raise BundleError("Segmentation models are not supported yet (the ONNX has a 'masks' output).")
    if "keypoints" in iface.outputs:
        raise BundleError("Keypoint models are not supported yet (the ONNX has a 'keypoints' output).")
    if "dets" not in iface.outputs or "labels" not in iface.outputs:
        raise BundleError(f"Expected ONNX outputs 'dets' and 'labels', got {list(iface.outputs)}. "
                          "Is this an RF-DETR export?")
    shape = iface.input_shape
    if len(shape) != 4 or not all(isinstance(d, int) and d > 0 for d in shape) or shape[0] != 1:
        raise BundleError(f"Expected a static input of shape [1, 3, H, W], got {shape}. "
                          "Export with batch_size=1 and dynamic_batch=False.")
    return int(iface.outputs["labels"][-1])


# --------------------------------------------------------------------------------------------------
# Class layout: which name belongs to which logits slot, and which slot (if any) is background.
# --------------------------------------------------------------------------------------------------
def layout_from_rfdetr(model, num_slots: int) -> ClassLayout:
    """Use rfdetr's own mapping, so DeepStream labels match what predict() returns."""
    from rfdetr.utilities.class_names import is_coco_pretrained, prediction_labels

    labels = prediction_labels(model)
    if labels.num_keypoints_per_class:
        raise BundleError("Keypoint models are not supported yet.")
    names = [labels.class_id_to_name.get(slot, "") for slot in range(num_slots)]
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
    log=print,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_onnx = model.lower().endswith(".onnx")

    # 1. Get the ONNX and the class layout.
    if is_onnx:
        source = Path(model)
        if not source.exists():
            raise BundleError(f"{source} does not exist.")
        if not (class_names_file or coco):
            raise BundleError("An ONNX file has no class names. Pass --class-names FILE, or --coco for the "
                              "official COCO-pretrained weights.")
        num_slots = check_detection_onnx(read_onnx_interface(source))
        layout = (ClassLayout(coco_sparse_labels(num_slots), None) if coco
                  else layout_from_names_file(class_names_file, num_slots))
        stem = source.stem
        base_onnx = source
    else:
        if not model.lower() in PRETRAINED and not Path(model).exists():
            raise BundleError(f"{model} is neither a pretrained name ({', '.join(PRETRAINED)}) nor an existing file.")
        log(f"Loading RF-DETR model: {model}")
        rf_model = load_rfdetr(model)
        stem = f"rfdetr-{model.lower()}" if model.lower() in PRETRAINED else Path(model).stem
        base_onnx = output_dir / f"{stem}.onnx"
        log("Exporting to ONNX ...")
        export_onnx(rf_model, base_onnx)
        num_slots = check_detection_onnx(read_onnx_interface(base_onnx))
        layout = layout_from_rfdetr(rf_model, num_slots)
        if class_names_file:
            layout = layout_from_names_file(class_names_file, num_slots)

    iface = read_onnx_interface(base_onnx)
    _, _, model_h, model_w = iface.input_shape
    log(f"Model: input {model_w}x{model_h}, {num_slots} class slots, "
        f"background slot: {layout.background_class_id}")

    # 2. The ONNX that goes into the bundle.
    if frame_size:
        frame_w, frame_h = frame_size
        bundle_onnx = output_dir / f"{stem}_{frame_w}x{frame_h}.onnx"
        add_preprocessing(base_onnx, bundle_onnx, frame_height=frame_h, frame_width=frame_w)
        if base_onnx.parent.resolve() == output_dir.resolve() and base_onnx != bundle_onnx:
            base_onnx.unlink()  # we exported it ourselves; only the patched model belongs in the bundle
        log(f"Preprocessing built into the ONNX for {frame_w}x{frame_h} frames.")
    else:
        bundle_onnx = output_dir / f"{stem}.onnx"
        if base_onnx.resolve() != bundle_onnx.resolve():
            shutil.copyfile(base_onnx, bundle_onnx)
        log("WARNING: no --frame-size given, so nvinfer approximates the preprocessing. Expect noticeably different "
            "detections than predict() (measured: 90% match, box IoU 0.87). Use --frame-size for ~99% parity.")

    # 3. Parser: ship the source, unless the user points at a library they already built.
    if parser_lib is None:
        parser_dir = output_dir / "parser"
        parser_dir.mkdir(exist_ok=True)
        src = resources.files("rfdetr_deepstream") / "parser"
        for name in PARSER_FILES:
            (parser_dir / name).write_bytes((src / name).read_bytes())
        parser_lib = parser_dir / PARSER_LIB_NAME

    if precision == "fp16":
        log("WARNING: FP16 is known to overflow in some transformer layers of RF-DETR, which degrades detections.")

    # 4. Labels, config, README.
    write_labels(output_dir / "labels.txt", layout.names)
    config_path = output_dir / "config_infer_rfdetr.txt"
    write_nvinfer_config(
        config_path,
        NvinferConfig(
            onnx_file=bundle_onnx,
            labels_file=output_dir / "labels.txt",
            parser_lib=Path(parser_lib),
            num_classes=num_slots,
            threshold=threshold,
            precision=precision,
            background_class_id=layout.background_class_id,
            preprocessing_in_model=frame_size is not None,
        ),
    )
    _write_readme(output_dir, bundle_onnx.name, frame_size)
    log(f"Bundle written to {output_dir}/")
    return output_dir


def _write_readme(output_dir: Path, onnx_name: str, frame_size) -> None:
    mux = (f"Set nvstreammux width={frame_size[0]} height={frame_size[1]}: the model expects exactly that frame size."
           if frame_size else "Any nvstreammux size works; nvinfer resizes frames to the model input.")
    (output_dir / "README.txt").write_text(f"""RF-DETR DeepStream bundle, generated by rfdetr-deepstream.

Files
  {onnx_name}
      The model. nvinfer builds a TensorRT engine from it on first run (this takes a while) and caches it
      next to it as {onnx_name}_b1_gpu0_<precision>.engine. Engines only work on the GPU and TensorRT
      version that built them, so never copy them between machines.
  labels.txt                one class name per logits slot
  config_infer_rfdetr.txt   nvinfer config (paths are relative to this folder)
  parser/                   C++ source of the output parser

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
