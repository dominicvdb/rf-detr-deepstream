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


# --------------------------------------------------------------------------------------------------
# Segmentation models (RFDETRSeg*). Mirrors rfdetr 1.12.1 models/postprocess.py (_postprocess_masks): the selected
# queries' mask logits are bilinearly upsampled to the original image (align_corners=False) and thresholded at > 0.
# --------------------------------------------------------------------------------------------------
@dataclass
class SegDetections(Detections):
    masks: np.ndarray = None  # (N, H, W) bool over the whole ORIGINAL image


def _bilinear_axis(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weights along one axis, exactly as PyTorch's bilinear upsample with align_corners=False:
    src = (dst + 0.5) * in/out - 0.5, clamped at 0; i1 = i0 + 1 except at the last input index."""
    scale = np.float32(n_in) / np.float32(n_out)
    src = np.maximum(scale * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5), 0)
    i0 = src.astype(np.int64)
    i1 = np.where(i0 < n_in - 1, i0 + 1, i0)
    lam1 = (src - i0).astype(np.float32)
    return i0, i1, (np.float32(1) - lam1), lam1


def upsample_mask_logits(logits: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """(Hm, Wm) mask logits -> (out_h, out_w), torch.nn.functional.interpolate(mode="bilinear", align_corners=False).
    Same summation order as PyTorch's CPU kernel: h0 * (w0 * x00 + w1 * x01) + h1 * (w0 * x10 + w1 * x11)."""
    hm, wm = logits.shape
    y0, y1, hl0, hl1 = _bilinear_axis(out_h, hm)
    x0, x1, wl0, wl1 = _bilinear_axis(out_w, wm)
    x = logits.astype(np.float32)
    top = wl0 * x[y0][:, x0] + wl1 * x[y0][:, x1]
    bottom = wl0 * x[y1][:, x0] + wl1 * x[y1][:, x1]
    return hl0[:, None] * top + hl1[:, None] * bottom


def mask_box_region(xyxy, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """The pixels a box's DeepStream mask covers: every pixel the box touches, clipped to the image, at least 1x1.
    Returns (x0, y0, x1, y1) with x1/y1 exclusive. The C++ parser uses the same rule (maskRegion)."""
    x0 = min(max(int(np.floor(xyxy[0])), 0), image_width - 1)
    y0 = min(max(int(np.floor(xyxy[1])), 0), image_height - 1)
    x1 = min(max(int(np.ceil(xyxy[2])), x0 + 1), image_width)
    y1 = min(max(int(np.ceil(xyxy[3])), y0 + 1), image_height)
    return x0, y0, x1, y1


def clip_masks_to_boxes(d: SegDetections) -> SegDetections:
    """What DeepStream keeps of each mask: only the pixels inside mask_box_region() of its box (DeepStream's
    instance masks cover the box). RF-DETR's masks can reach slightly past their box, usually by a thin border."""
    clipped = np.zeros_like(d.masks)
    h, w = d.masks.shape[1:]
    for i, box in enumerate(d.xyxy):
        x0, y0, x1, y1 = mask_box_region(box, w, h)
        clipped[i, y0:y1, x0:x1] = d.masks[i, y0:y1, x0:x1]
    return SegDetections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id, query=d.query, masks=clipped)


def decode_masks(
    dets: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    image_width: int,
    image_height: int,
    threshold: float = 0.5,
    num_select: int | None = None,
    background_class_id: int | None = None,
) -> SegDetections:
    """Decode ONE image's raw segmentation-model outputs: decode() for the boxes (steps 1-6), then

    Step 7: each kept detection's mask = its query's (Hm, Wm) mask logits, upsampled bilinearly to the original
            image size, > 0 (i.e. probability > 0.5).
    """
    d = decode(dets, labels, image_width, image_height, threshold, num_select, background_class_id)
    out = np.zeros((len(d.query), image_height, image_width), dtype=bool)
    for i, q in enumerate(d.query):
        out[i] = upsample_mask_logits(masks[q], image_height, image_width) > 0
    return SegDetections(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id, query=d.query, masks=out)


# --------------------------------------------------------------------------------------------------
# Keypoint models (RFDETRKeypointPreview). Mirrors rfdetr 1.12.1 models/postprocess.py (_postprocess_keypoints,
# _apply_keypoint_trace_fusion, _keypoint_log_mean_trace) and the threshold filter in _prediction/core.py.
# --------------------------------------------------------------------------------------------------
@dataclass
class KeypointDetections(Detections):
    keypoints: np.ndarray = None  # (N, K, 3) float32: x, y in ORIGINAL image pixels, confidence (sigmoid findable)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def keypoint_log_mean_trace(kp: np.ndarray) -> float:
    """kp: (K, D) raw keypoint predictions of one detection's class. Log of the findability-weighted mean trace
    of the covariance implied by the precision-Cholesky parameters (columns 4:7)."""
    kp = kp.astype(np.float32)
    log_l11, l21, log_l22 = kp[:, 4], kp[:, 5], kp[:, 6]
    w_find = 1.0 / (1.0 + np.exp(-kp[:, 2]))
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = 2.0 * np.log(np.maximum(np.abs(l21), 1e-12)) + log_t1 + log_t2
    log_trace = _logsumexp(np.stack([log_t1, log_t2, log_t3], axis=-1), axis=-1)
    log_w = np.log(np.maximum(w_find, 1e-12))
    return float(_logsumexp(log_trace + log_w, axis=0) - _logsumexp(log_w, axis=0))


def decode_keypoints(
    dets: np.ndarray,
    labels: np.ndarray,
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
    num_keypoints_per_class: list[int],
    threshold: float = 0.5,
    trace_alpha: float = 0.2,
    num_select: int | None = None,
) -> KeypointDetections:
    """Decode ONE image's raw keypoint-model outputs.

    Args:
        dets:      (Q, 4) normalised cx, cy, w, h
        labels:    (Q, C) raw logits; C = len(num_keypoints_per_class)
        keypoints: (Q, C * Kmax, D) raw keypoints, padded per class slot; D >= 7
        num_keypoints_per_class: e.g. [0, 17] (background-first: slot 0 has no keypoints)
    """
    num_queries, num_classes = labels.shape
    kmax = max(num_keypoints_per_class)
    d = keypoints.shape[-1]
    kp_by_class = keypoints.reshape(num_queries, num_classes, kmax, d)

    # Steps 1-4 as in decode(): sigmoid per slot (background included), top-k pairs, no threshold yet.
    scores = 1.0 / (1.0 + np.exp(-np.clip(labels.astype(np.float32), -88.0, 88.0)))
    flat = scores.reshape(-1)
    k = num_queries if num_select is None else num_select
    order = np.argsort(-flat, kind="stable")[:k]
    query, class_id, score = order // num_classes, order % num_classes, flat[order]

    # Step 7: keypoints of the predicted class, and score fusion with keypoint uncertainty.
    fused = score.copy()
    out_kp = np.zeros((len(order), kmax, 3), dtype=np.float32)
    max_score = np.nextafter(np.float32(1.0), np.float32(0.0))
    for i, (q, c) in enumerate(zip(query, class_id)):
        if c >= len(num_keypoints_per_class):
            continue
        n = num_keypoints_per_class[c]
        log_mean_trace = 0.0
        if n > 0:
            a = kp_by_class[q, c, :n]
            out_kp[i, :n, 0] = a[:, 0] * image_width
            out_kp[i, :n, 1] = a[:, 1] * image_height
            out_kp[i, :n, 2] = 1.0 / (1.0 + np.exp(-a[:, 2]))
            if trace_alpha > 0 and d >= 7:
                log_mean_trace = keypoint_log_mean_trace(a)
        if trace_alpha > 0 and d >= 7:
            z = np.log(score[i]) - trace_alpha * log_mean_trace
            fused[i] = min(np.float32(1.0 / (1.0 + np.exp(-z))), max_score)

    # Step 5 (moved): threshold on the FUSED score, as predict() does.
    keep = fused > threshold
    query, class_id, fused, out_kp = query[keep], class_id[keep], fused[keep], out_kp[keep]

    # Step 6: boxes to pixel corners.
    cx, cy, w, h = dets[query].astype(np.float32).T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scale = np.array([image_width, image_height, image_width, image_height], dtype=np.float32)
    xyxy = np.clip(xyxy * scale, 0.0, scale)
    return KeypointDetections(xyxy=xyxy, confidence=fused.astype(np.float32), class_id=class_id, query=query,
                              keypoints=out_kp)


def run_onnx_keypoints(session, image: Image.Image, num_keypoints_per_class, **decode_kwargs) -> KeypointDetections:
    inp = session.get_inputs()[0]
    _, _, height, width = inp.shape
    names = [o.name for o in session.get_outputs()]
    out = dict(zip(names, session.run(None, {inp.name: preprocess(image, height, width)})))
    return decode_keypoints(out["dets"][0], out["labels"][0], out["keypoints"][0], image.width, image.height,
                            num_keypoints_per_class, **decode_kwargs)


def run_onnx(session, image: Image.Image, **decode_kwargs) -> Detections:
    """Full reference pipeline on one image with an onnxruntime InferenceSession."""
    inp = session.get_inputs()[0]
    _, _, height, width = inp.shape
    tensor = preprocess(image, height, width)
    names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(names, session.run(None, {inp.name: tensor})))  # match outputs BY NAME
    return decode(outputs["dets"][0], outputs["labels"][0], image.width, image.height, **decode_kwargs)
