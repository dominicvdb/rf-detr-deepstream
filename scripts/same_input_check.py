"""Same-input check: DeepStream's own input tensor (nvinfer dump-input-tensor=1) -> ONNX Runtime -> Python decoder,
vs what DeepStream produced from that same input. Isolates TensorRT + our C++ decoding from preprocessing.

Needs a bundle with preprocessing in the ONNX (input = full RGB frame, 0-255).

    python scripts/same_input_check.py --task detect --bin ip_tensor_dump.bin --onnx X.onnx \
        --out-dir kitti/ --labels labels.txt [--json OUT]
    python scripts/same_input_check.py --task keypoints --bin ip_tensor_dump.bin --onnx X.onnx --out-dir dump/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.dumps import kitti_path, pose_dump_path, read_kitti, read_labels, read_pose_dump
from rfdetr_deepstream.reference import decode, decode_keypoints

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["detect", "keypoints"], required=True)
parser.add_argument("--bin", type=Path, required=True)
parser.add_argument("--onnx", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True, help="KITTI dir (detect) or pose dump dir (keypoints)")
parser.add_argument("--labels", type=Path, help="labels.txt (detect)")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--json", type=Path)
args = parser.parse_args()

session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
_, c, h, w = inp.shape
frames = np.memmap(args.bin, dtype=np.float32, mode="r").reshape(-1, c, h, w)
label_to_id = read_labels(args.labels) if args.labels else None

stats, kp_dist = ParityStats(), []
for f in range(frames.shape[0]):
    out = dict(zip([o.name for o in session.get_outputs()],
                   session.run(None, {inp.name: np.ascontiguousarray(frames[f : f + 1])})))
    if args.task == "detect":
        ref = decode(out["dets"][0], out["labels"][0], w, h, threshold=args.threshold)
        ds_xyxy, ds_cls, ds_conf = read_kitti(kitti_path(args.out_dir, f), label_to_id)
    else:
        ref = decode_keypoints(out["dets"][0], out["labels"][0], out["keypoints"][0], w, h, [0, 17],
                               threshold=args.threshold)
        ds_xyxy, ds_cls, ds_conf, ds_kps = read_pose_dump(pose_dump_path(args.out_dir, f))
        for i, j, _ in match(ref.xyxy, ref.class_id, ds_xyxy, ds_cls):
            n = len(ds_kps[j])
            if n:
                kp_dist.append(np.linalg.norm(ref.keypoints[i, :n, :2] - ds_kps[j][:, :2], axis=-1))
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ds_xyxy, ds_cls, ds_conf)

print(f"SAME INPUT ({args.task}), {frames.shape[0]} frames")
print(stats.report("python", "deepstream", threshold=args.threshold))
summary = {"frames": int(frames.shape[0]), **stats.summary(args.threshold)}
if kp_dist:
    d = np.concatenate(kp_dist)
    print(f"keypoint distance px: mean={d.mean():.4f}  max={d.max():.3f}")
    summary.update(kp_dist_mean=float(d.mean()), kp_dist_max=float(d.max()))
if args.json:
    args.json.write_text(json.dumps(summary, indent=2))
