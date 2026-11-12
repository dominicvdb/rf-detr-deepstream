"""Debug: does DeepStream's scaler use a different pixel-center convention than torchvision?

Runs the Python reference with two resize variants and compares each to the DeepStream detections:
  half_pixel: torchvision / RFDETR.predict()  ->  src = (dst + 0.5) * scale - 0.5
  corner:     hypothesis for NvBufSurfTransform ->  src = dst * scale
If "corner" matches DeepStream much better, the box shift comes from the scaler, not from our parser.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from rfdetr_deepstream.compare import match
from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD, decode

N = 150
labels = {name: i for i, name in enumerate(Path("models/nano/labels.txt").read_text().splitlines())}


def read_kitti(frame):
    xyxy, cls = [], []
    for line in Path(f"data/ds_out/kitti/00_000_{frame:06d}.txt").read_text().splitlines():
        t = line.split()
        xyxy.append([float(v) for v in t[-15:][3:7]])
        cls.append(labels[" ".join(t[:-15])])
    return np.array(xyxy).reshape(-1, 4), np.array(cls)


def bilinear(img, out_h, out_w, half_pixel: bool):
    """img: (H, W, 3) float32. Plain bilinear without antialiasing, with a choice of pixel-center convention."""
    in_h, in_w = img.shape[:2]
    sy, sx = in_h / out_h, in_w / out_w
    ys, xs = np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32)
    if half_pixel:
        ys, xs = (ys + 0.5) * sy - 0.5, (xs + 0.5) * sx - 0.5
    else:
        ys, xs = ys * sy, xs * sx
    ys, xs = np.clip(ys, 0, in_h - 1), np.clip(xs, 0, in_w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, in_h - 1), np.minimum(x0 + 1, in_w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def preprocess(rgb, half_pixel):
    x = bilinear(rgb.astype(np.float32) / 255.0, 384, 384, half_pixel)
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


session = ort.InferenceSession("models/nano/rfdetr-nano.onnx", providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture("data/sample_720p.mp4")
results = {True: [], False: []}
for f in range(N):
    ok, bgr = cap.read()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    d_xyxy, d_cls = read_kitti(f)
    for half_pixel in (True, False):
        dets, logits = session.run(["dets", "labels"], {"input": preprocess(rgb, half_pixel)})
        r = decode(dets[0], logits[0], rgb.shape[1], rgb.shape[0], threshold=0.5)
        for i, j, v in match(r.xyxy, r.class_id, d_xyxy, d_cls):
            results[half_pixel].append([*(d_xyxy[j] - r.xyxy[i]), v])

for half_pixel, name in ((True, "half_pixel (torchvision)"), (False, "corner (hypothesis)")):
    a = np.array(results[half_pixel])
    print(f"{name:26s} matched={len(a):5d}  mean IoU={a[:, 4].mean():.4f}  "
          f"mean shift ds-py x={a[:, [0, 2]].mean():+.2f}px y={a[:, [1, 3]].mean():+.2f}px")
