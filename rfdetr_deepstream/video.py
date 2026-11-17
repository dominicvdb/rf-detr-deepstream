"""Video frames exactly as DeepStream's nvinfer sees them.

H.264/H.265 decoding is bit-exact, so FFmpeg (PyAV) gives the same YUV planes as nvv4l2decoder. nvinfer then converts
NV12 to RGB in float32 with truncation, chroma repeated over 2x2 pixels. The colour matrix comes from the colour
description in the video bitstream (the hardware decoder ignores container metadata such as a MOV 'colr' atom):
BT.709 if the bitstream says so, BT.601 otherwise. Coefficients:
  - DeepStream 8.0 and later: exact;
  - DeepStream 7.x: rounded to 4 decimals (BT.601: 1.1644, 1.596, 0.3918, 0.813, 2.0172).
Reproduced bit-exactly against nvinfer's dump-input-tensor (net-scale-factor=1, network size == frame size) on
DeepStream 7.0, 7.1, 8.0, 9.0 and 9.2, for untagged and BT.709-tagged limited-range H.264. OpenCV's conversion
differs in about half of all pixel values (by up to 2), which is enough to flip detections near the score threshold.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

_F = np.float32
_MATRICES = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}
_AVCOL_SPC_BT709 = 1
_AVCOL_SPC_BT601 = (2, 5, 6)   # unspecified, BT470BG, SMPTE170M
_AVCOL_RANGE_FULL = 2


def _coefficients(matrix: str, deepstream_version: str) -> tuple[float, ...]:
    kr, kb = _MATRICES[matrix]
    kg = 1 - kr - kb
    ys, cs = 255 / 219, 255 / 224   # limited range: Y 16-235, Cb/Cr 16-240
    c = (ys, 2 * (1 - kr) * cs, 2 * (1 - kb) * kb / kg * cs, 2 * (1 - kr) * kr / kg * cs, 2 * (1 - kb) * cs)
    if int(deepstream_version.split(".")[0]) < 8:
        c = tuple(round(x, 4) for x in c)
    return c


def _fma(a: np.float32, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # a*b + c rounded once to float32, like CUDA's fmaf. Exact here: the float64 sum of these operands
    # (8-bit integers times 24-bit constants) never needs more than 53 bits.
    return (np.float64(a) * b.astype(np.float64) + c.astype(np.float64)).astype(_F)


def deepstream_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray, deepstream_version: str = "9.2",
                   matrix: str = "bt601") -> np.ndarray:
    """Full-resolution Y and half-resolution U, V planes (uint8, limited range) -> H x W x 3 RGB uint8,
    as nvinfer converts NV12."""
    h, w = y.shape
    ky, krv, kgu, kgv, kbu = (_F(x) for x in _coefficients(matrix, deepstream_version))
    yc = y.astype(_F) - _F(16)
    uc = u.repeat(2, 0).repeat(2, 1)[:h, :w].astype(_F) - _F(128)
    vc = v.repeat(2, 0).repeat(2, 1)[:h, :w].astype(_F) - _F(128)
    k = ky * yc
    # R as a fused multiply-add, G and B as separate multiplies and adds: matches every pixel on every version.
    r = _fma(krv, vc, k)
    g = k - kgu * uc - kgv * vc
    b = k + kbu * uc
    return np.stack([np.clip(np.trunc(x), 0, 255) for x in (r, g, b)], axis=-1).astype(np.uint8)


def frame_matrix(frame) -> str:
    """Colour matrix nvinfer uses for a decoded frame (from the bitstream's colour description)."""
    if frame.color_range == _AVCOL_RANGE_FULL:
        raise ValueError("full-range video: nvinfer's conversion for it has not been verified")
    if frame.colorspace == _AVCOL_SPC_BT709:
        return "bt709"
    if frame.colorspace in _AVCOL_SPC_BT601:
        return "bt601"
    raise ValueError(f"unsupported colour matrix (AVColorSpace {frame.colorspace}); verified: BT.601, BT.709")


def read_frames(path: str | Path, max_frames: int = 0, deepstream_version: str = "9.2") -> Iterator[np.ndarray]:
    """Decode a video with PyAV and yield RGB uint8 frames (H x W x 3) as nvinfer receives them."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # A decoder of our own, fed only the bitstream: like the hardware decoder, it doesn't see container colour
        # metadata, so frame.colorspace is what the bitstream says.
        decoder = av.CodecContext.create(stream.codec_context.name, "r")
        decoder.extradata = stream.codec_context.extradata
        count = 0
        for packet in container.demux(stream):
            for frame in decoder.decode(packet):  # the final empty packet flushes the decoder
                if max_frames and count >= max_frames:
                    return
                yield _to_rgb(frame, deepstream_version)
                count += 1


def _to_rgb(frame, deepstream_version: str) -> np.ndarray:
    if frame.format.name not in ("yuv420p", "nv12"):
        raise ValueError(f"unsupported pixel format {frame.format.name}; expected 8-bit 4:2:0")
    matrix = frame_matrix(frame)
    if frame.format.name == "nv12":
        frame = frame.reformat(format="yuv420p")
    h, w = frame.height, frame.width
    planes = [np.frombuffer(p, np.uint8).reshape(-1, p.line_size) for p in frame.planes]
    return deepstream_rgb(planes[0][:h, :w], planes[1][:(h + 1) // 2, :(w + 1) // 2],
                          planes[2][:(h + 1) // 2, :(w + 1) // 2], deepstream_version, matrix)
