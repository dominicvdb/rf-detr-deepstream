"""Reference (ground-truth) pre- and post-processing for RF-DETR ONNX models, in Python.

This is the specification the C++ DeepStream parser has to match. It mirrors rfdetr 1.12.1:
  - preprocessing:  rfdetr/export/_runtime/preprocess.py  (preprocess_to_nchw)
  - decoding:       rfdetr/export/_runtime/decode.py      (decode_detections)
                    rfdetr/export/_topk.py                (_select_topk_multiclass)

The decoder steps are numbered; the C++ parser (parser/nvdsparsebbox_rfdetr.cpp) uses the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints: Pillow is not needed to build bundles
    from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------------------
# Preprocessing: image -> model input tensor.  In DeepStream, nvinfer does this (config keys), not the parser.
# --------------------------------------------------------------------------------------------------
def preprocess(image: Image.Image, height: int, width: int) -> np.ndarray:
    """PIL image -> float32 tensor of shape (1, 3, height, width), exactly like RFDETR.predict().

    Steps: RGB -> scale to [0, 1] -> bilinear stretch-resize (no antialias, aspect ratio not kept)
    -> ImageNet normalisation per channel.
    """
    import torch
    import torchvision.transforms.functional as F  # noqa: N812

    with torch.no_grad():
        t = F.to_tensor(image.convert("RGB"))  # (3, H, W) float32 in [0, 1]
        t = F.resize(t, [height, width], antialias=False)
        t = F.normalize(t, list(IMAGENET_MEAN), list(IMAGENET_STD))
    return t.unsqueeze(0).numpy().astype(np.float32)  # add the batch dim -> (1, 3, h, w)


# --------------------------------------------------------------------------------------------------
# Decoding: raw output tensors -> detections.  In DeepStream, this is the parser's job (C++).
# --------------------------------------------------------------------------------------------------
@dataclass
class Detections:
    xyxy: np.ndarray        # (N, 4) float32, pixel corners x1, y1, x2, y2 in the ORIGINAL image
    confidence: np.ndarray  # (N,)   float32, sigmoid score
    class_id: np.ndarray    # (N,)   int64, slot index in the logits tensor (sparse COCO ID for official weights)
    query: np.ndarray       # (N,)   int64, which of the queries this came from (can repeat)


def decode(
    dets: np.ndarray,
    labels: np.ndarray,
    image_width: int,
    image_height: int,
    threshold: float = 0.5,
    num_select: int | None = None,
    background_class_id: int | None = None,
) -> Detections:
    """Decode ONE image's raw outputs.

    Args:
        dets:   (Q, 4) boxes as normalised cx, cy, w, h  (the ONNX output "dets", batch dim removed)
        labels: (Q, C) raw class logits                  (the ONNX output "labels", batch dim removed)
        image_width, image_height: size to scale boxes to (the original image).
        threshold: keep pairs with score strictly greater than this.
        num_select: how many query/class pairs to keep before thresholding (default: Q, i.e. 300).
        background_class_id: logits slot that means "no object" and must be ignored.
            None for the official COCO weights (every slot is a real class), -1 for most fine-tuned models.
    """
    num_queries, num_classes = labels.shape

    # Step 1: logits -> probabilities with an independent SIGMOID per class (not softmax!).
    #         Clipping to +-88 keeps exp() from overflowing float32.
    scores = 1.0 / (1.0 + np.exp(-np.clip(labels.astype(np.float32), -88.0, 88.0)))  # (Q, C)

    # Step 2: ignore the background slot, if this checkpoint has one, by forcing its score to "never selected".
    if background_class_id is not None:
        scores[:, background_class_id % num_classes] = -np.inf

    # Step 3: treat every (query, class) pair as a candidate: flatten (Q, C) -> (Q*C,).
    #         Flat index k corresponds to query = k // C and class = k % C.
    flat = scores.reshape(-1)

    # Step 4: take the num_select highest-scoring pairs. Ties: lower flat index first (stable sort),
    #         matching rfdetr. One query can appear more than once (e.g. "car" AND "truck").
    k = num_queries if num_select is None else num_select
    order = np.argsort(-flat, kind="stable")[:k]

    # Step 5: threshold AFTER selecting (same order as rfdetr).
    order = order[flat[order] > threshold]

    query = order // num_classes
    class_id = order % num_classes
    confidence = flat[order]

    # Step 6: boxes: normalised (cx, cy, w, h) -> pixel corners (x1, y1, x2, y2), clipped to the image.
    cx, cy, w, h = dets[query].astype(np.float32).T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scale = np.array([image_width, image_height, image_width, image_height], dtype=np.float32)
    xyxy = np.clip(xyxy * scale, 0.0, scale)

    return Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, query=query)


def run_onnx(session, image: Image.Image, **decode_kwargs) -> Detections:
    """Full reference pipeline on one image with an onnxruntime InferenceSession."""
    inp = session.get_inputs()[0]
    _, _, height, width = inp.shape
    tensor = preprocess(image, height, width)
    names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(names, session.run(None, {inp.name: tensor})))  # match outputs BY NAME
    return decode(outputs["dets"][0], outputs["labels"][0], image.width, image.height, **decode_kwargs)
