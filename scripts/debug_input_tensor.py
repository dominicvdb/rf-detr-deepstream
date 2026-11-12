"""Debug with DeepStream's own input tensor (nvinfer dump-input-tensor=1 -> data/ds_dump/ip_tensor_dump.bin).

1. Parser check: DeepStream input -> ONNX Runtime -> Python decode, vs DeepStream's detections.
   Same input on both sides, so any difference is TensorRT or our C++ parser.
2. Preprocessing check: DeepStream input vs our Python preprocessing, converted back to 0-255 pixel values.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats
from rfdetr_deepstream.deepstream import scale_and_offsets
from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD, decode, preprocess

N = 300
H = W = 384
FRAME_W, FRAME_H = 1280, 720
labels = {name: i for i, name in enumerate(Path("models/nano/labels.txt").read_text().splitlines())}
dump = np.memmap("data/ds_dump/ip_tensor_dump.bin", dtype=np.float32, mode="r").reshape(-1, 3, H, W)
print(f"dumped frames: {dump.shape[0]}")


def read_kitti(frame):
    xyxy, cls, conf = [], [], []
    for line in Path(f"data/ds_dump/kitti/00_000_{frame:06d}.txt").read_text().splitlines():
        t = line.split()
        nums = [float(v) for v in t[-15:]]
        xyxy.append(nums[3:7])
        cls.append(labels[" ".join(t[:-15])])
        conf.append(nums[14])
    return np.array(xyxy).reshape(-1, 4), np.array(cls), np.array(conf)


session = ort.InferenceSession("models/nano/rfdetr-nano.onnx", providers=["CPUExecutionProvider"])

# ---- 1. Parser check ------------------------------------------------------------------------------
stats = ParityStats()
for f in range(N):
    dets, logits = session.run(["dets", "labels"], {"input": np.ascontiguousarray(dump[f : f + 1])})
    r = decode(dets[0], logits[0], FRAME_W, FRAME_H, threshold=0.5)
    stats.add(r.xyxy, r.class_id, r.confidence, *read_kitti(f))
print(f"\n1. SAME INPUT (DeepStream tensor): python decode vs deepstream, {N} frames")
print(stats.report("python", "deepstream", threshold=0.5))

# ---- 2. Preprocessing check -----------------------------------------------------------------------
scale, offsets = scale_and_offsets(IMAGENET_MEAN, IMAGENET_STD)
std = np.array(IMAGENET_STD, np.float32)[:, None, None]
mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
cap = cv2.VideoCapture("data/sample_720p.mp4")
diffs = []
for f in range(30):
    ok, bgr = cap.read()
    ours = preprocess(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), H, W)[0]
    ds_pix = dump[f] / scale + np.array(offsets, np.float32)[:, None, None]  # undo nvinfer: y = s * (x - offset)
    our_pix = (ours * std + mean) * 255.0                                    # undo ImageNet normalisation
    diffs.append(ds_pix - our_pix)
d = np.stack(diffs)  # (frames, 3, H, W)
print("\n2. PREPROCESSING: deepstream input - python input, in 0-255 pixel units (30 frames)")
print(f"   mean |diff| per channel R,G,B = {np.abs(d).mean(axis=(0, 2, 3)).round(2)}")
print(f"   mean signed diff R,G,B        = {d.mean(axis=(0, 2, 3)).round(2)}   (colour/level bias)")
print(f"   99th percentile |diff|        = {np.percentile(np.abs(d), 99):.1f}")
print(f"   DS pixel value range           = [{(dump[:30] / scale).min() + min(offsets):.1f}, {(dump[:30] / scale).max() + max(offsets):.1f}]")
print("   effect of the single-scale approximation alone: normalised-value error per channel =",
      ((1 / (255 * std.ravel())) / scale - 1).round(4))
