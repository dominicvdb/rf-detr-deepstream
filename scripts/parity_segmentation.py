"""Parity test for segmentation: instance masks DeepStream (nvinfer + our parser) attaches vs the Python reference.

Prerequisites:
  1. a mask_probe run (scripts/mask_probe) on the video: objects.txt + masks.bin from nvinfer's src pad
  2. the plain seg ONNX (scripts/export_seg_reference.py) and the same video locally

Reference per frame: the plain ONNX on the frame (torchvision preprocessing, as predict()), decode_masks() and
clip_masks_to_boxes() (a DeepStream mask covers its box). DeepStream's mask is pasted into a frame-size canvas
at the box region and thresholded at mask_params.threshold.

    python scripts/parity_segmentation.py --dump DIR --onnx build/seg_nano_reference.onnx --labels LABELS [--frames N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.dumps import read_mask_dump
from rfdetr_deepstream.reference import clip_masks_to_boxes, decode_masks, mask_box_region, preprocess
from rfdetr_deepstream.video import read_frames

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_720p.mp4")
parser.add_argument("--ds-version", default="7.0", help="DeepStream version: selects nvinfer's colour conversion")
parser.add_argument("--dump", type=Path, required=True, help="folder with objects.txt and masks.bin")
parser.add_argument("--onnx", required=True, help="plain seg ONNX")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--background", type=int, default=None, help="background logits slot (None: COCO)")
parser.add_argument("--frames", type=int, default=0, help="0 = all frames")
parser.add_argument("--stream", type=int, default=0)
parser.add_argument("--json", type=Path)
args = parser.parse_args()

dump = read_mask_dump(args.dump / "objects.txt", args.dump / "masks.bin")
session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
model_h, model_w = inp.shape[2:]
names = [o.name for o in session.get_outputs()]

stats = ParityStats()
mask_ious: list[float] = []
same_box: list[bool] = []  # per mask IoU: the two boxes agree (IoU >= 0.99), so the difference is the mask head's
shape_mismatch = no_mask = frames = 0
for rgb in read_frames(args.video, args.frames, args.ds_version):
    h, w = rgb.shape[:2]
    out = dict(zip(names, session.run(None, {inp.name: preprocess(Image.fromarray(rgb), model_h, model_w)})))
    ref = clip_masks_to_boxes(decode_masks(out["dets"][0], out["labels"][0], out["masks"][0], w, h,
                                           threshold=args.threshold, background_class_id=args.background))
    objs = dump.get((args.stream, frames), [])
    xyxy = np.array([o["xyxy"] for o in objs]).reshape(-1, 4)
    cls = np.array([o["class_id"] for o in objs], dtype=int)
    conf = np.array([o["confidence"] for o in objs])
    stats.add(ref.xyxy, ref.class_id, ref.confidence, xyxy, cls, conf)

    for i, j, box_iou in match(ref.xyxy, ref.class_id, xyxy, cls):
        mask = objs[j]["mask"]
        if mask is None:
            no_mask += 1
            continue
        x0, y0, x1, y1 = mask_box_region(xyxy[j], w, h)
        if mask.shape != (y1 - y0, x1 - x0):
            shape_mismatch += 1
            continue
        canvas = np.zeros((h, w), dtype=bool)
        canvas[y0:y1, x0:x1] = mask > objs[j]["threshold"]
        union = np.logical_or(canvas, ref.masks[i]).sum()
        mask_ious.append(float(np.logical_and(canvas, ref.masks[i]).sum() / union) if union else 1.0)
        same_box.append(box_iou >= 0.99)
    frames += 1
    if frames % 200 == 0:
        print(f"  {frames} frames...", flush=True)

ious = np.array(mask_ious)
print(f"\nFrames compared: {frames}  (threshold {args.threshold})")
print(stats.report("python", "deepstream", threshold=args.threshold))
print(f"mask IoU:   mean={ious.mean():.5f}  p5={np.percentile(ious, 5):.5f}  min={ious.min():.5f}  "
      f"(n={len(ious)}, shape mismatches={shape_mismatch}, objects without mask={no_mask})")
agree = ious[np.array(same_box, dtype=bool)]
print(f"mask IoU where the boxes agree (IoU >= 0.99): mean={agree.mean():.5f}  p5={np.percentile(agree, 5):.5f}  "
      f"min={agree.min():.5f}  (n={len(agree)})")
if args.json:
    args.json.write_text(json.dumps({"frames": frames, **stats.summary(args.threshold),
                                     "mask_iou_same_box_mean": float(agree.mean()),
                                     "mask_iou_same_box_p5": float(np.percentile(agree, 5)),
                                     "mask_iou_same_box_min": float(agree.min()),
                                     "mask_iou_mean": float(ious.mean()), "mask_iou_p5": float(np.percentile(ious, 5)),
                                     "mask_iou_min": float(ious.min()), "mask_shape_mismatch": shape_mismatch}, indent=2))
