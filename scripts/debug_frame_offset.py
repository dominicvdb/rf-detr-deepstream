"""Debug: is DeepStream frame N the same picture as OpenCV frame N? Compare with offsets -2..+2."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import match
from rfdetr_deepstream.reference import run_onnx

N = 150
labels = {name: i for i, name in enumerate(Path("models/nano/labels.txt").read_text().splitlines())}


def read_kitti(frame):
    xyxy, cls = [], []
    for line in Path(f"data/ds_out/kitti/00_000_{frame:06d}.txt").read_text().splitlines():
        t = line.split()
        xyxy.append([float(v) for v in t[-15:][3:7]])
        cls.append(labels[" ".join(t[:-15])])
    return np.array(xyxy).reshape(-1, 4), np.array(cls)


session = ort.InferenceSession("models/nano/rfdetr-nano.onnx", providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture("data/sample_720p.mp4")
print("OpenCV frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), " fps:", cap.get(cv2.CAP_PROP_FPS))
refs = []
for _ in range(N + 2):
    ok, bgr = cap.read()
    refs.append(run_onnx(session, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), threshold=0.5))

for offset in (-2, -1, 0, 1, 2):
    ious = []
    for f in range(2, N):
        r = refs[f]
        d_xyxy, d_cls = read_kitti(f + offset)
        ious += [p[2] for p in match(r.xyxy, r.class_id, d_xyxy, d_cls)]
    print(f"python frame f vs deepstream frame f{offset:+d}: matched={len(ious):5d}  mean IoU={np.mean(ious):.4f}")

# --- Systematic or random? Signed box differences (deepstream - python) at offset 0, by box size.
rows = []
for f in range(2, N):
    r = refs[f]
    d_xyxy, d_cls = read_kitti(f)
    for i, j, v in match(r.xyxy, r.class_id, d_xyxy, d_cls):
        p, d = r.xyxy[i], d_xyxy[j]
        rows.append([*(d - p), (p[2] - p[0]) * (p[3] - p[1]) ** 0.5 if False else np.sqrt((p[2] - p[0]) * (p[3] - p[1])), v])
rows = np.array(rows)
print("\nsigned delta (ds - py) in pixels:  mean x1,y1,x2,y2 =", np.round(rows[:, :4].mean(0), 2), " std =", np.round(rows[:, :4].std(0), 2))
for lo, hi in ((0, 40), (40, 80), (80, 160), (160, 2000)):
    m = (rows[:, 4] >= lo) & (rows[:, 4] < hi)
    if m.any():
        print(f"box size {lo:4d}-{hi:4d}px: n={m.sum():4d}  mean IoU={rows[m, 5].mean():.4f}  mean |delta|={np.abs(rows[m, :4]).mean():.2f}px")
