"""Scaler check: does nvinfer hand the frame to the model unchanged? (network size == frame size, scaling-filter=0)

Compares nvinfer's dumped input tensor (net-scale-factor=1, RGB 0-255) with OpenCV's decode of the same video.
On DeepStream 7.0 the result is no spatial shift and a small constant rounding offset (mean |diff| < 1 level).
A shifted or blurred frame (the bilinear behaviour) shows up as a better match at a +-1 pixel shift and a
mean |diff| of several levels.

    python scripts/scaler_check.py --bin ip_tensor_dump.bin --video data/sample_720p_100.mp4 [--frames 30] [--json OUT]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=Path, required=True)
parser.add_argument("--video", type=Path, required=True)
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--json", type=Path)
args = parser.parse_args()

cap = cv2.VideoCapture(str(args.video))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dump = np.memmap(args.bin, dtype=np.float32, mode="r").reshape(-1, 3, h, w)
n = min(args.frames, dump.shape[0])
ocv = np.stack([cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
                for _ in range(n)])
ds = np.asarray(dump[:n])
d = ds - ocv

shifts = {}
for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
    a = ds[:, :, 2:-2, 2:-2]
    b = ocv[:, :, 2 + dy : h - 2 + dy, 2 + dx : w - 2 + dx]
    shifts[f"{dy:+d},{dx:+d}"] = float(np.abs(a - b).mean())
best = min(shifts, key=shifts.get)
summary = {
    "frames": n,
    "mean_abs_diff_rgb": np.abs(d).mean(axis=(0, 2, 3)).round(3).tolist(),
    "mean_signed_diff_rgb": d.mean(axis=(0, 2, 3)).round(3).tolist(),
    "p99_abs_diff": float(np.percentile(np.abs(d), 99)),
    "shift_mean_abs_diff": shifts,
    "best_shift": best,
    "unchanged": best == "+0,+0" and float(np.abs(d).mean()) < 1.5,
}
print(json.dumps(summary, indent=2))
if args.json:
    args.json.write_text(json.dumps(summary, indent=2))
