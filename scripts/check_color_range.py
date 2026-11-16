"""Which YUV -> RGB conversion does nvinfer apply? Compares nvinfer's dumped input tensor (dump-input-tensor=1, bundle
with preprocessing in the ONNX: net-scale-factor=1, network size == frame size, so the dump is the RGB 0-255 frame)
with the same frames decoded by PyAV and converted in several ways:

  limited-bt709, limited-bt601   rfdetr_deepstream.video.deepstream_rgb (verified for limited-range video)
  full-bt709, full-bt601         the same arithmetic with full-range scaling (Y 0-255, C centred at 128)

Each with exact and with DeepStream 7.x-rounded coefficients. The best match tells how nvinfer treats the clip's
range flag. Run it on data/range_limited.mp4 first: limited-bt709 must match exactly (the control).

    python scripts/check_color_range.py --bin ip_tensor_dump.bin --video data/range_full.mp4 [--frames 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np

from rfdetr_deepstream.video import _fma, deepstream_rgb

_F = np.float32


def full_range_rgb(y, u, v, matrix: str, round4: bool) -> np.ndarray:
    """Full-range YUV -> RGB, same arithmetic as video.deepstream_rgb (chroma repeated 2x2, R as FMA, truncation)."""
    kr, kb = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}[matrix]
    kg = 1 - kr - kb
    c = (1.0, 2 * (1 - kr), 2 * (1 - kb) * kb / kg, 2 * (1 - kr) * kr / kg, 2 * (1 - kb))
    if round4:
        c = tuple(round(x, 4) for x in c)
    ky, krv, kgu, kgv, kbu = (_F(x) for x in c)
    h, w = y.shape
    yc = y.astype(_F)
    uc = u.repeat(2, 0).repeat(2, 1)[:h, :w].astype(_F) - _F(128)
    vc = v.repeat(2, 0).repeat(2, 1)[:h, :w].astype(_F) - _F(128)
    k = ky * yc
    r = _fma(krv, vc, k)
    g = k - kgu * uc - kgv * vc
    b = k + kbu * uc
    return np.stack([np.clip(np.trunc(x), 0, 255) for x in (r, g, b)], axis=-1).astype(np.uint8)


def decoded_planes(path: Path, n: int):
    """YUV planes as the hardware decoder sees them (bitstream only, no container colour metadata)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        dec = av.CodecContext.create(stream.codec_context.name, "r")
        dec.extradata = stream.codec_context.extradata
        out = []
        for packet in container.demux(stream):
            for f in dec.decode(packet):
                if len(out) == n:
                    return out
                h, w = f.height, f.width
                p = [np.frombuffer(pl, np.uint8).reshape(-1, pl.line_size) for pl in f.planes]
                out.append((p[0][:h, :w], p[1][: (h + 1) // 2, : (w + 1) // 2], p[2][: (h + 1) // 2, : (w + 1) // 2],
                            {"format": f.format.name, "color_range": int(f.color_range),
                             "colorspace": int(f.colorspace)}))
        return out


CANDIDATES = {
    "limited-bt709": lambda y, u, v: deepstream_rgb(y, u, v, "9.2", "bt709"),
    "limited-bt709-ds7": lambda y, u, v: deepstream_rgb(y, u, v, "7.0", "bt709"),
    "limited-bt601": lambda y, u, v: deepstream_rgb(y, u, v, "9.2", "bt601"),
    "limited-bt601-ds7": lambda y, u, v: deepstream_rgb(y, u, v, "7.0", "bt601"),
    "full-bt709": lambda y, u, v: full_range_rgb(y, u, v, "bt709", False),
    "full-bt709-ds7": lambda y, u, v: full_range_rgb(y, u, v, "bt709", True),
    "full-bt601": lambda y, u, v: full_range_rgb(y, u, v, "bt601", False),
    "full-bt601-ds7": lambda y, u, v: full_range_rgb(y, u, v, "bt601", True),
}


def compare(dump: np.ndarray, planes) -> dict:
    """dump: (N, 3, H, W) float RGB 0-255."""
    n = min(len(planes), dump.shape[0])
    results = {}
    for name, convert in CANDIDATES.items():
        exact, diff = [], []
        for i in range(n):
            rgb = convert(*planes[i][:3]).transpose(2, 0, 1).astype(np.float32)
            d = np.abs(np.asarray(dump[i]) - rgb)
            exact.append(float((d == 0).mean()))
            diff.append(float(d.mean()))
        results[name] = {"exact_match_rate": float(np.mean(exact)), "mean_abs_diff": float(np.mean(diff))}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True, help="nvinfer ip_tensor_dump.bin")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    planes = decoded_planes(args.video, args.frames)
    h, w = planes[0][0].shape
    dump = np.memmap(args.bin, dtype=np.float32, mode="r").reshape(-1, 3, h, w)
    print(f"{args.video}: {planes[0][3]}, {len(planes)} frames; dump has {dump.shape[0]} frames")
    results = compare(dump, planes)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["mean_abs_diff"]):
        print(f"  {name:20s} exact {r['exact_match_rate'] * 100:7.3f}%   mean |diff| {r['mean_abs_diff']:.4f}")
    best = min(results, key=lambda k: results[k]["mean_abs_diff"])
    verdict = f"nvinfer converts this clip as {best}" + (" (bit-exact)" if results[best]["exact_match_rate"] == 1.0
                                                         else " (closest, not bit-exact)")
    print(verdict)
    if args.json:
        args.json.write_text(json.dumps({"video": str(args.video), "flags": planes[0][3], "results": results,
                                         "best": best}, indent=2))


if __name__ == "__main__":
    main()
