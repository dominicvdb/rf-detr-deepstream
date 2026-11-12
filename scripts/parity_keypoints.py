"""Keypoint parity test: DeepStream (TensorRT + libnvds_rfdetr_pose.so) vs the Python reference (ONNX Runtime).

Prerequisite: a DeepStream run with customlib-props="dump-dir:<dir>" on the same video.

    python scripts/parity_keypoints.py [--video data/sample_walk.mov] [--dump data/kp_out/dump]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.reference import run_onnx_keypoints

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_walk.mov")
parser.add_argument("--dump", default="data/kp_out/dump")
parser.add_argument("--onnx", default="models/keypoint/rfdetr-keypoint-preview_folded.onnx")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--frames", type=int, default=0)
args = parser.parse_args()
SCHEMA = [0, 17]


def read_dump(path: Path):
    xyxy, cls, conf, kps = [], [], [], []
    for line in path.read_text().splitlines():
        v = line.split()
        cls.append(int(v[0]))
        conf.append(float(v[1]))
        xyxy.append([float(x) for x in v[2:6]])
        n = int(v[6])
        kps.append(np.array([float(x) for x in v[7 : 7 + 3 * n]]).reshape(n, 3))
    return np.array(xyxy).reshape(-1, 4), np.array(cls, dtype=int), np.array(conf), kps


session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture(args.video)
stats = ParityStats()
kp_dist, kp_dconf, frame = [], [], 0
while True:
    ok, bgr = cap.read()
    if not ok or (args.frames and frame >= args.frames):
        break
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ref = run_onnx_keypoints(session, image, SCHEMA, threshold=args.threshold)
    ds_xyxy, ds_cls, ds_conf, ds_kps = read_dump(Path(args.dump) / f"00_{frame:06d}.txt")
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ds_xyxy, ds_cls, ds_conf)
    for i, j, _ in match(ref.xyxy, ref.class_id, ds_xyxy, ds_cls):
        n = len(ds_kps[j])
        if n:
            kp_dist.append(np.linalg.norm(ref.keypoints[i, :n, :2] - ds_kps[j][:, :2], axis=-1))
            kp_dconf.append(np.abs(ref.keypoints[i, :n, 2] - ds_kps[j][:, 2]))
    frame += 1

d = np.concatenate(kp_dist)
c = np.concatenate(kp_dconf)
print(f"Frames compared: {frame}  (threshold {args.threshold})")
print(stats.report("python", "deepstream", threshold=args.threshold))
print(f"keypoint distance (px): mean={d.mean():.3f}  median={np.median(d):.3f}  p95={np.percentile(d, 95):.3f}  "
      f"max={d.max():.2f}   ({len(d)} keypoints)")
print(f"keypoint |dconf|:       mean={c.mean():.4f}  p95={np.percentile(c, 95):.4f}  max={c.max():.4f}")
