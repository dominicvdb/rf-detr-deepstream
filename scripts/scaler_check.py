"""Scaler check: does nvinfer hand the frame to the model unchanged? (network size == frame size, scaling-filter=0)

Compares nvinfer's dumped input tensor (net-scale-factor=1, RGB 0-255) with the same frames decoded by PyAV and
converted like nvinfer (rfdetr_deepstream.video). Unchanged means bit-identical. A shifted or blurred frame (the
bilinear behaviour) shows up as a better match at a +-1 pixel shift and a mean |diff| of several levels.

    python scripts/scaler_check.py --bin ip_tensor_dump.bin --video data/sample_720p_100.mp4 --ds-version 9.2 \
        [--frames 30] [--json OUT]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rfdetr_deepstream.video import read_frames

parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=Path, required=True)
parser.add_argument("--video", type=Path, required=True)
parser.add_argument("--ds-version", default="9.2", help="DeepStream version: selects nvinfer's colour conversion")
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--json", type=Path)
args = parser.parse_args()

ref = np.stack([f.transpose(2, 0, 1).astype(np.float32) for f in read_frames(args.video, args.frames, args.ds_version)])
_, _, h, w = ref.shape
dump = np.memmap(args.bin, dtype=np.float32, mode="r").reshape(-1, 3, h, w)
n = min(len(ref), dump.shape[0])
ds, ref = np.asarray(dump[:n]), ref[:n]
d = ds - ref

shifts = {}
for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
    a = ds[:, :, 2:-2, 2:-2]
    b = ref[:, :, 2 + dy : h - 2 + dy, 2 + dx : w - 2 + dx]
    shifts[f"{dy:+d},{dx:+d}"] = float(np.abs(a - b).mean())
best = min(shifts, key=shifts.get)
exact = float((d == 0).mean())
summary = {
    "frames": n,
    "exact_match_rate": exact,
    "mean_abs_diff_rgb": np.abs(d).mean(axis=(0, 2, 3)).round(4).tolist(),
    "max_abs_diff": float(np.abs(d).max()),
    "shift_mean_abs_diff": shifts,
    "best_shift": best,
    "unchanged": exact == 1.0,
}
print(json.dumps(summary, indent=2))
if args.json:
    args.json.write_text(json.dumps(summary, indent=2))
