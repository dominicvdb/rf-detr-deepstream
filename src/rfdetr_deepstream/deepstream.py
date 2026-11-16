"""Generate the DeepStream files for an RF-DETR ONNX model: labels file and nvinfer config.

The config encodes RF-DETR's preprocessing (see reference.preprocess) in nvinfer terms:
  RGB, stretch resize (no aspect ratio), bilinear, (x/255 - mean) / std per channel.
nvinfer computes  y = net-scale-factor * (x - offset_c)  with ONE scale for all channels, so:
  offsets          = 255 * mean_c                        (exact)
  net-scale-factor = 1 / (255 * average(std))            (approximation, up to ~1.3% per channel)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD

PARSER_FUNCTION = "NvDsInferParseRfdetr"
SEG_PARSER_FUNCTION = "NvDsInferParseRfdetrSeg"
NETWORK_MODE = {"fp32": 0, "int8": 1, "fp16": 2}


# COCO 2017 category IDs -> names. The IDs are "sparse": 1..90 with gaps (12, 26, 29, 30, 45, 66, 68, 69, 71, 83).
# The official RF-DETR checkpoints use these IDs directly as logits slots.
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def coco_sparse_labels(num_slots: int = 91) -> list[str]:
    """Label per logits slot for the official COCO checkpoints (slot index = sparse COCO category ID)."""
    return [COCO_CATEGORIES.get(slot, "") for slot in range(num_slots)]


def write_labels(path: Path, names: list[str]) -> None:
    """One line per logits slot. nvinfer looks up line N for classId N, so every slot needs a line:
    unused slots (gaps in the COCO IDs, background) get a placeholder instead of an empty line."""
    lines = [name if name else f"unused_{slot}" for slot, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


@dataclass
class NvinferConfig:
    onnx_file: Path
    labels_file: Path
    parser_lib: Path
    num_classes: int                         # logits slots, e.g. 91
    threshold: float = 0.5
    precision: str = "fp32"                  # fp32 | fp16 | int8
    batch_size: int = 1
    background_class_id: int | None = None   # slot to exclude; None for official COCO weights
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    # True when the ONNX has the preprocessing built in (onnx_patch.add_preprocessing): nvinfer then only
    # converts the frame to RGB 0-255, without scaling (network size == frame size) or normalising.
    preprocessing_in_model: bool = False
    # "detect": nvinfer parses boxes with our parser. "segment": the same, plus one mask per object
    # (instance segmentation: network-type=3). "keypoints": nvinfer only attaches the raw output tensors
    # (network-type=100, output-tensor-meta=1); libnvds_rfdetr_pose.so in nvdsvideotemplate decodes them.
    task: str = "detect"
    # The ONNX carries the precision (FP16 with FP32 parts); TensorRT builds it exactly (DeepStream 9.0+).
    strongly_typed: bool = False


def scale_and_offsets(mean, std) -> tuple[float, list[float]]:
    offsets = [255.0 * m for m in mean]
    scale = 1.0 / (255.0 * (sum(std) / len(std)))
    return scale, offsets


def _rel(target: Path, config_dir: Path) -> str:
    """nvinfer resolves relative paths against the config file's folder, which keeps a bundle relocatable."""
    return os.path.relpath(Path(target).resolve(), Path(config_dir).resolve())


TASK_NAMES = {"detect": "detection", "segment": "segmentation", "keypoints": "keypoint"}


def write_nvinfer_config(path: Path, cfg: NvinferConfig) -> None:
    path = Path(path)
    cfg_dir = path.parent
    scale, offsets = scale_and_offsets(cfg.mean, cfg.std)
    onnx_rel = _rel(cfg.onnx_file, cfg_dir)
    # Strongly typed: TensorRT refuses the FP16 flag, so network-mode=0 even though the engine is FP16.
    mode = "fp32" if cfg.strongly_typed else cfg.precision
    # nvinfer's own naming scheme for the engine it builds; pointing at it makes nvinfer reuse the cache.
    engine_rel = f"{onnx_rel}_b{cfg.batch_size}_gpu0_{mode}.engine"

    lines = [
        f"# Generated by rfdetr_deepstream. nvinfer config for an RF-DETR "
        f"{TASK_NAMES[cfg.task]} model.",
        "[property]",
        "gpu-id=0",
        f"onnx-file={onnx_rel}",
        f"model-engine-file={engine_rel}",
        f"labelfile-path={_rel(cfg.labels_file, cfg_dir)}",
        f"batch-size={cfg.batch_size}",
        f"network-mode={NETWORK_MODE[mode]}",
        "process-mode=1",                      # primary: run on full frames
        "gie-unique-id=1",
    ]
    if cfg.strongly_typed:
        lines += [
            "# FP16 ONNX built strongly typed (DeepStream 9.0+; for 7.1/8.0 see README.txt, FP16).",
            "strongly-typed=1",
        ]
    if cfg.task == "keypoints":
        lines += [
            "network-type=100",                # other: no parsing in nvinfer
            "output-tensor-meta=1",            # raw tensors for libnvds_rfdetr_pose.so (nvdsvideotemplate)
            "output-blob-names=dets;labels;keypoints",
        ]
    elif cfg.task == "segment":
        lines += [
            "network-type=3",                  # instance segmentation: boxes + one mask per object
            f"num-detected-classes={cfg.num_classes}",
            "output-blob-names=dets;labels;masks",
            "output-instance-mask=1",
            "# Masks hold the probability per pixel; > 0.5 is exactly RF-DETR's mask logit > 0.",
            "segmentation-threshold=0.5",
        ]
    else:
        lines += [
            "network-type=0",                  # detector
            f"num-detected-classes={cfg.num_classes}",
            "output-blob-names=dets;labels",
        ]
    lines += [
        "",
        "model-color-format=0",                # RGB
        "network-input-order=0",               # NCHW
        "maintain-aspect-ratio=0",             # stretch, like torchvision resize([h, w])
    ]
    if cfg.preprocessing_in_model:
        lines += [
            "# Preprocessing is inside the ONNX (resize + normalise). nvinfer only delivers RGB 0-255.",
            "# The model input size must equal the frame size (nvstreammux width/height).",
            "# nvinfer's bilinear scaler resamples even at scale 1 (half-pixel offset: blur + shift, measured);",
            "# nearest at scale 1 copies pixels unchanged.",
            "scaling-filter=0",
            "net-scale-factor=1.0",
        ]
    else:
        lines += [
            "scaling-filter=1",                # bilinear
            "# Preprocessing approximated by nvinfer, matching RFDETR.predict() as closely as nvinfer allows",
            f"offsets={';'.join(f'{o:.4f}' for o in offsets)}",
            f"net-scale-factor={scale:.10f}",  # single-scale approximation of the per-channel std
        ]
    if cfg.task == "keypoints":
        path.write_text("\n".join(lines) + "\n")
        return
    lines += [
        "",
        "# Postprocessing: our parser. DETR models need no NMS.",
        f"custom-lib-path={_rel(cfg.parser_lib, cfg_dir)}",
        (f"parse-bbox-instance-mask-func-name={SEG_PARSER_FUNCTION}" if cfg.task == "segment"
         else f"parse-bbox-func-name={PARSER_FUNCTION}"),
        "cluster-mode=4",                      # 4 = no clustering
        "",
        "[class-attrs-all]",
        f"pre-cluster-threshold={cfg.threshold}",
    ]
    if cfg.background_class_id is not None:
        bg = cfg.background_class_id % cfg.num_classes
        lines += [
            "",
            "# Background slot: a threshold of 1.0 can never be reached, so the parser skips this class.",
            f"[class-attrs-{bg}]",
            "pre-cluster-threshold=1.0",
        ]
    path.write_text("\n".join(lines) + "\n")
