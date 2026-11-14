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
