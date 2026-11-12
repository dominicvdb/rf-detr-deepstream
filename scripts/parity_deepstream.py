"""Parity test: DeepStream (TensorRT + our C++ parser) vs the Python reference (ONNX Runtime), per video frame.

Prerequisites:
  1. deepstream-app -c /work/configs/deepstream_app_nano.txt   (writes data/ds_out/kitti/*.txt)
  2. the same video at data/sample_720p.mp4

Run from ~/rfdetr-deepstream with the venv active:
    python scripts/parity_deepstream.py [--frames N]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats
from rfdetr_deepstream.reference import run_onnx

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_720p.mp4")
parser.add_argument("--kitti", default="data/ds_out/kitti")
parser.add_argument("--onnx", default="models/nano/rfdetr-nano.onnx")
parser.add_argument("--labels", default="models/nano/labels.txt")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--frames", type=int, default=0, help="0 = all frames")
args = parser.parse_args()

label_to_id = {name: i for i, name in enumerate(Path(args.labels).read_text().splitlines())}


def read_kitti(path: Path):
    """KITTI line: '<label, may contain spaces> 0.0 0 0.0 left top right bottom 0.0 x7 confidence'."""
    xyxy, cls, conf = [], [], []
    for line in path.read_text().splitlines():
        tokens = line.split()
        label, nums = " ".join(tokens[:-15]), [float(t) for t in tokens[-15:]]
        xyxy.append(nums[3:7])
        cls.append(label_to_id[label])
        conf.append(nums[14])
    return np.array(xyxy).reshape(-1, 4), np.array(cls, dtype=int), np.array(conf)


session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture(args.video)
stats = ParityStats()
frame = 0
while True:
    ok, bgr = cap.read()
    if not ok or (args.frames and frame >= args.frames):
        break
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ref = run_onnx(session, image, threshold=args.threshold)
    ds_xyxy, ds_cls, ds_conf = read_kitti(Path(args.kitti) / f"00_000_{frame:06d}.txt")
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ds_xyxy, ds_cls, ds_conf)
    frame += 1
    if frame % 200 == 0:
        print(f"  {frame} frames...", flush=True)

print(f"\nFrames compared: {frame}  (threshold {args.threshold})")
print(stats.report("python", "deepstream", threshold=args.threshold))
