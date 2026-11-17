"""Readers for DeepStream output files used in parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_labels(path: Path) -> dict[str, int]:
    return {name: i for i, name in enumerate(Path(path).read_text().splitlines())}


def read_kitti(path: Path, label_to_id: dict[str, int]):
    """deepstream-app gie-kitti-output-dir file. Line:
    '<label, may contain spaces> 0.0 0 0.0 left top right bottom 0.0 x7 confidence' (frame pixels)."""
    xyxy, cls, conf = [], [], []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        label, nums = " ".join(tokens[:-15]), [float(t) for t in tokens[-15:]]
        xyxy.append(nums[3:7])
        cls.append(label_to_id[label])
        conf.append(nums[14])
    return np.array(xyxy).reshape(-1, 4), np.array(cls, dtype=int), np.array(conf)


def kitti_path(kitti_dir: Path, frame: int, stream: int = 0) -> Path:
    return Path(kitti_dir) / f"00_{stream:03d}_{frame:06d}.txt"


def read_pose_dump(path: Path):
    """libnvds_rfdetr_pose.so dump-dir file. Line:
    'class_id score x1 y1 x2 y2 num_keypoints (x y confidence) * num_keypoints' (frame pixels)."""
    xyxy, cls, conf, kps = [], [], [], []
    for line in Path(path).read_text().splitlines():
        v = line.split()
        cls.append(int(v[0]))
        conf.append(float(v[1]))
        xyxy.append([float(x) for x in v[2:6]])
        n = int(v[6])
        kps.append(np.array([float(x) for x in v[7 : 7 + 3 * n]]).reshape(n, 3))
    return np.array(xyxy).reshape(-1, 4), np.array(cls, dtype=int), np.array(conf), kps


def pose_dump_path(dump_dir: Path, frame: int, stream: int = 0) -> Path:
    return Path(dump_dir) / f"{stream:02d}_{frame:06d}.txt"


def read_mask_dump(objects_txt: Path, masks_bin: Path) -> dict[tuple[int, int], list[dict]]:
    """scripts/mask_probe output: {(source_id, frame_num): [object, ...]}. Each object has xyxy (nvinfer's
    rect_params), class_id, confidence, threshold and mask: a float32 (mask_height, mask_width) array, or None."""
    blob = np.fromfile(masks_bin, dtype=np.uint8)
    frames: dict[tuple[int, int], list[dict]] = {}
    for line in Path(objects_txt).read_text().splitlines():
        t = line.split()
        source, frame, cls = int(t[0]), int(t[1]), int(t[2])
        conf, left, top, width, height = (float(v) for v in t[3:8])
        mask_w, mask_h, threshold, offset = int(t[8]), int(t[9]), float(t[10]), int(t[11])
        mask = None
        if mask_w and mask_h:
            mask = blob[offset:offset + 4 * mask_w * mask_h].view(np.float32).reshape(mask_h, mask_w)
        frames.setdefault((source, frame), []).append(
            {"xyxy": [left, top, left + width, top + height], "class_id": cls, "confidence": conf,
             "threshold": threshold, "mask": mask})
    return frames
