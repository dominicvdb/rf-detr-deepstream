"""Parity test: DeepStream (TensorRT + our C++ parser) vs the Python reference (ONNX Runtime), per video frame.

Prerequisites:
  1. a deepstream-app run with gie-kitti-output-dir (e.g. configs/deepstream_app_nano_1280x720.txt)
  2. the same video locally (e.g. data/sample_720p.mp4)

Run from ~/rfdetr-deepstream with the venv active:
    python scripts/parity_deepstream.py --kitti DIR --labels FILE [--frames N] [--json OUT]
"""

import argparse
import json
from pathlib import Path

import cv2
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats
from rfdetr_deepstream.dumps import kitti_path, read_kitti, read_labels
from rfdetr_deepstream.reference import run_onnx

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_720p.mp4")
parser.add_argument("--kitti", default="data/ds_out/kitti")
parser.add_argument("--onnx", default="models/nano/rfdetr-nano.onnx")
parser.add_argument("--labels", default="models/nano/labels.txt")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--frames", type=int, default=0, help="0 = all frames")
parser.add_argument("--json", type=Path, help="write the summary as JSON")
args = parser.parse_args()

label_to_id = read_labels(args.labels)
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
    ds_xyxy, ds_cls, ds_conf = read_kitti(kitti_path(args.kitti, frame), label_to_id)
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ds_xyxy, ds_cls, ds_conf)
    frame += 1
    if frame % 200 == 0:
        print(f"  {frame} frames...", flush=True)

print(f"\nFrames compared: {frame}  (threshold {args.threshold})")
print(stats.report("python", "deepstream", threshold=args.threshold))
if args.json:
    args.json.write_text(json.dumps({"frames": frame, **stats.summary(args.threshold)}, indent=2))
