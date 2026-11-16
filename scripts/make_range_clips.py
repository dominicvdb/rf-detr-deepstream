"""Synthetic H.264 test clips to find out how nvinfer converts full-range video (no camera footage needed).

Writes two 1280x720 BT.709 clips with the same YUV test pattern (luma ramps over the whole 0-255 range, chroma sweeps,
a moving block so frames differ):

  data/range_full.mp4     full range (yuvj420p, VUI video_full_range_flag=1), like many IP cameras
  data/range_limited.mp4  limited range (Y 16-235, C 16-240), the control: nvinfer's conversion for this is verified

Run nvinfer on both with dump-input-tensor=1 and compare with scripts/check_color_range.py.

    python scripts/make_range_clips.py [--frames 30] [--out data]
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

W, H = 1280, 720


def pattern(frame: int, full: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y (H, W) and U, V (H/2, W/2) uint8."""
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    luma = np.where(y < H // 2, x * 255 // (W - 1), 255 - x * 255 // (W - 1))         # ramps both ways
    luma = np.broadcast_to(luma, (H, W)).copy()
    luma[(y // 40 % 2 == 0).repeat(W, 1) & (x // 40 % 2 == 0).repeat(H, 0)] //= 2     # checker detail
    bx = (frame * 16) % (W - 128)
    luma[300:428, bx:bx + 128] = 255 - luma[300:428, bx:bx + 128]                      # moving block
    cx = np.arange(W // 2)[None, :]
    cy = np.arange(H // 2)[:, None]
    u = np.broadcast_to(cx * 255 // (W // 2 - 1), (H // 2, W // 2)).copy()
    v = np.broadcast_to(cy * 255 // (H // 2 - 1), (H // 2, W // 2)).copy()
    if not full:  # squeeze into limited range
        luma = 16 + luma * 219 // 255
        u, v = 16 + u * 224 // 255, 16 + v * 224 // 255
    return luma.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)


def write(path: Path, full: bool, frames: int) -> None:
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=Fraction(20, 1))
        s.width, s.height = W, H
        s.pix_fmt = "yuvj420p" if full else "yuv420p"
        s.options = {"crf": "8", "preset": "medium", "profile": "high",
                     "x264-params": f"fullrange={'on' if full else 'off'}:colorprim=bt709:transfer=bt709:"
                                    "colormatrix=bt709"}
        for i in range(frames):
            yp, up, vp = pattern(i, full)
            f = av.VideoFrame(W, H, "yuvj420p" if full else "yuv420p")
            for plane, data in zip(f.planes, (yp, up, vp)):
                buf = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)
                buf[: data.shape[0], : data.shape[1]] = data
            f.pts = i
            for packet in s.encode(f):
                out.mux(packet)
        for packet in s.encode():
            out.mux(packet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, full in (("range_full.mp4", True), ("range_limited.mp4", False)):
        write(args.out / name, full, args.frames)
        print(f"written {args.out / name}")


if __name__ == "__main__":
    main()
