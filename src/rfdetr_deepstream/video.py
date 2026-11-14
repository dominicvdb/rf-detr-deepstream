"""Video frames exactly as DeepStream's nvinfer sees them.

H.264/H.265 decoding is bit-exact, so FFmpeg (PyAV) gives the same YUV planes as nvv4l2decoder. nvinfer then converts
NV12 to RGB with BT.601 limited range and truncation, chroma repeated over 2x2 pixels:
  - DeepStream 8.0 and later: exact BT.601 coefficients;
  - DeepStream 7.x: 4-decimal coefficients in float32 (1.1644, 1.596, 0.3918, 0.813, 2.0172).
Reproduced bit-exactly against nvinfer's dump-input-tensor (net-scale-factor=1, network size == frame size) on
DeepStream 7.0, 7.1, 8.0, 9.0 and 9.2. OpenCV's conversion differs in about half of all pixel values (by up to 2),
which is enough to flip detections that sit near the score threshold.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

_F = np.float32
_KY, _KRV, _KGU, _KGV, _KBU = _F(1.1644), _F(1.596), _F(0.3918), _F(0.813), _F(2.0172)


def _fma(a: np.float32, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # a*b + c rounded once to float32, like CUDA's fmaf. Exact here: the float64 sum of these operands
    # (8-bit integers times 24-bit constants) never needs more than 53 bits.
    return (np.float64(a) * b.astype(np.float64) + c.astype(np.float64)).astype(_F)


_KR, _KB = 0.299, 0.114
_EXACT = (255 / 219, 2 * (1 - _KR) * 255 / 224, 2 * (1 - _KB) * _KB / (1 - _KR - _KB) * 255 / 224,
          2 * (1 - _KR) * _KR / (1 - _KR - _KB) * 255 / 224, 2 * (1 - _KB) * 255 / 224)


def deepstream_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray, deepstream_version: str = "9.2") -> np.ndarray:
    """Full-resolution Y and half-resolution U, V planes (uint8) -> H x W x 3 RGB uint8, as nvinfer converts NV12."""
    h, w = y.shape
    u = u.repeat(2, 0).repeat(2, 1)[:h, :w]
    v = v.repeat(2, 0).repeat(2, 1)[:h, :w]
    if int(deepstream_version.split(".")[0]) >= 8:
        ky, krv, kgu, kgv, kbu = _EXACT
        yc, uc, vc = y - 16.0, u - 128.0, v - 128.0
        r = ky * yc + krv * vc
        g = ky * yc - kgu * uc - kgv * vc
        b = ky * yc + kbu * uc
    else:
        yc = y.astype(_F) - _F(16)
        uc = u.astype(_F) - _F(128)
        vc = v.astype(_F) - _F(128)
        ky = _KY * yc
        # R as a fused multiply-add, G as separate multiplies and subtractions: matches every pixel.
        r = _fma(_KRV, vc, ky)
        g = ky - _KGU * uc - _KGV * vc
        b = ky + _KBU * uc
    return np.stack([np.clip(np.trunc(x), 0, 255) for x in (r, g, b)], axis=-1).astype(np.uint8)


def read_frames(path: str | Path, max_frames: int = 0, deepstream_version: str = "9.2") -> Iterator[np.ndarray]:
    """Decode a video with PyAV and yield RGB uint8 frames (H x W x 3) as nvinfer receives them."""
    import av

    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if max_frames and i >= max_frames:
                break
            if frame.format.name not in ("yuv420p", "nv12"):
                raise ValueError(f"unsupported pixel format {frame.format.name}; expected 8-bit 4:2:0, limited range")
            if frame.format.name == "nv12":
                frame = frame.reformat(format="yuv420p")
            h, w = frame.height, frame.width
            planes = [np.frombuffer(p, np.uint8).reshape(-1, p.line_size) for p in frame.planes]
            yield deepstream_rgb(planes[0][:h, :w], planes[1][:(h + 1) // 2, :(w + 1) // 2],
                                 planes[2][:(h + 1) // 2, :(w + 1) // 2], deepstream_version)
