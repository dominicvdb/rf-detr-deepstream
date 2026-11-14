"""Keypoint parity test: DeepStream (TensorRT + libnvds_rfdetr_pose.so) vs the Python reference (ONNX Runtime).

Prerequisite: a DeepStream run with customlib-props="dump-dir:<dir>" on the same video.

    python scripts/parity_keypoints.py [--video data/sample_walk.mov] [--dump data/kp_out/dump] [--json OUT]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.dumps import pose_dump_path, read_pose_dump
from rfdetr_deepstream.reference import run_onnx_keypoints
from rfdetr_deepstream.video import read_frames

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_walk.mov")
parser.add_argument("--ds-version", default="9.2", help="DeepStream version: selects nvinfer's colour conversion")
parser.add_argument("--dump", default="data/kp_out/dump")
parser.add_argument("--onnx", default="models/keypoint/rfdetr-keypoint-preview_folded.onnx")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--frames", type=int, default=0)
parser.add_argument("--stream", type=int, default=0, help="source index in a multi-stream run (dump file prefix)")
parser.add_argument("--json", type=Path, help="write the summary as JSON")
args = parser.parse_args()
SCHEMA = [0, 17]

session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
stats = ParityStats()
kp_dist, kp_dconf, frame = [], [], 0
for rgb in read_frames(args.video, args.frames, args.ds_version):
    ref = run_onnx_keypoints(session, Image.fromarray(rgb), SCHEMA, threshold=args.threshold)
    ds_xyxy, ds_cls, ds_conf, ds_kps = read_pose_dump(pose_dump_path(args.dump, frame, args.stream))
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ds_xyxy, ds_cls, ds_conf)
    for i, j, _ in match(ref.xyxy, ref.class_id, ds_xyxy, ds_cls):
        n = len(ds_kps[j])
        if n:
            kp_dist.append(np.linalg.norm(ref.keypoints[i, :n, :2] - ds_kps[j][:, :2], axis=-1))
            kp_dconf.append(np.abs(ref.keypoints[i, :n, 2] - ds_kps[j][:, 2]))
    frame += 1

d = np.concatenate(kp_dist) if kp_dist else np.zeros(0)
c = np.concatenate(kp_dconf) if kp_dconf else np.zeros(0)
print(f"Frames compared: {frame}  (threshold {args.threshold})")
print(stats.report("python", "deepstream", threshold=args.threshold))
if len(d):
    print(f"keypoint distance (px): mean={d.mean():.3f}  median={np.median(d):.3f}  p95={np.percentile(d, 95):.3f}  "
          f"max={d.max():.2f}   ({len(d)} keypoints)")
    print(f"keypoint |dconf|:       mean={c.mean():.4f}  p95={np.percentile(c, 95):.4f}  max={c.max():.4f}")
if args.json:
    summary = {"frames": frame, **stats.summary(args.threshold)}
    if len(d):
        summary.update(kp_dist_mean=float(d.mean()), kp_dist_median=float(np.median(d)),
                       kp_dist_p95=float(np.percentile(d, 95)), kp_dconf_mean=float(c.mean()))
    args.json.write_text(json.dumps(summary, indent=2))
