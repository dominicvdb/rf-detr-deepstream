"""Is an RF-DETR model FP16-safe, and does the LayerNorm fix help? Without a GPU: simulates FP16 in ONNX Runtime
(rfdetr_deepstream.fp16.simulate_fp16) and compares detections with FP32 on labelled frames.

Variants:
  fp32               the model as is (reference)
  fp16               every tensor rounded to FP16, LayerNorms computed in FP16
  fp16_ln_fp32       the same, but the at-risk LayerNorms in FP32 (= nvinfer layer-device-precision)
  fp16_scaled        the at-risk LayerNorm inputs scaled by 1/2^k (the fix built into the ONNX), all in FP16
  fp32_scaled        the scaled model in FP32: must equal fp32 (the fix doesn't change the model)

The at-risk LayerNorms and their scales come from calibration frames (the first --calibration frames).

    python scripts/fp16_simulate.py MODEL.onnx --annotations annotations.coco.json --images DIR --threshold 0.6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.fp16 import (layernorm_deviations, layernorm_scales, scale_layernorm_inputs, simulate_fp16)
from rfdetr_deepstream.reference import decode, preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="plain RF-DETR ONNX (input = model size)")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--background", default="-1")
    ap.add_argument("--calibration", type=int, default=12, help="frames used to find at-risk LayerNorms")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    background = None if args.background.lower() == "none" else int(args.background)
    thr = args.threshold

    ann = json.loads(args.annotations.read_text())
    images = [im for im in ann["images"] if im.get("split", "test") == "test"][: args.limit or None]
    model = onnx.load(str(args.model))
    _, _, h, w = [d.dim_value or 1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    tensors = [preprocess(Image.open(args.images / im["file_name"]).convert("RGB"), h, w) for im in images]

    # Calibration on evenly spread frames
    step = max(len(tensors) // args.calibration, 1)
    dev = layernorm_deviations(args.model, tensors[::step][: args.calibration])
    scales = layernorm_scales(dev)
    print(f"{len(dev)} LayerNorms; at risk (largest |x - mean| > 64 on {args.calibration} calibration frames):")
    for name, s in scales.items():
        print(f"  {dev[name]:8.1f}  -> divide by {s:4d}   {name}")

    scaled = scale_layernorm_inputs(onnx.ModelProto.FromString(model.SerializeToString()), scales)
    variants = {
        "fp32": model,
        "fp16": simulate_fp16(model),
        "fp16_ln_fp32": simulate_fp16(model, keep_fp32=set(scales)),
        "fp16_scaled": simulate_fp16(scaled),
        "fp32_scaled": scaled,
    }
    sessions = {k: ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
                for k, m in variants.items()}
    dets = {k: [] for k in variants}
    for n, (im, x) in enumerate(zip(images, tensors)):
        for k, s in sessions.items():
            out = dict(zip([o.name for o in s.get_outputs()], s.run(None, {s.get_inputs()[0].name: x})))
            dets[k].append(decode(out["dets"][0], out["labels"][0], im["width"], im["height"], threshold=0.0,
                                  background_class_id=background))
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")

    def at(d, t):
        k = d.confidence > t
        return d.xyxy[k], d.class_id[k], d.confidence[k]

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {**ann, "images": images}
        gt.createIndex()
    print(f"\n{'variant':14s} {'vs fp32: matched':>18s} {'IoU mean':>9s} {'|dscore| mean/max':>18s}   "
          f"{'mAP50-95':>8s}  tp/fp/fn @{thr}")
    results = {"scales": scales, "deviations": dev}
    for k in variants:
        s = ParityStats()
        for a, b in zip(dets["fp32"], dets[k]):
            s.add(*at(a, thr), *at(b, thr))
        r = s.summary(thr)
        rows = [{"image_id": im["id"], "category_id": int(c), "score": float(sc),
                 "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
                for im, d in zip(images, dets[k]) for b, c, sc in zip(*at(d, 0.0))]
        mAP = float("nan")
        if rows:
            with contextlib.redirect_stdout(io.StringIO()):
                e = COCOeval(gt, gt.loadRes(rows), "bbox")
                e.evaluate(); e.accumulate(); e.summarize()  # noqa: E702
            mAP = e.stats[0]
        tp = fp = fn = 0
        for im, d in zip(images, dets[k]):
            g = [a for a in ann["annotations"] if a["image_id"] == im["id"]]
            gx = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                           for a in g]).reshape(-1, 4)
            px, pc, _ = at(d, thr)
            m = len(match(gx, [a["category_id"] for a in g], px, pc))
            tp, fp, fn = tp + m, fp + len(px) - m, fn + len(gx) - m
        iou = f"{r['iou_mean']:.4f}" if r["iou_mean"] is not None else "-"
        ds = f"{r['dscore_mean']:.4f}/{r['dscore_max']:.3f}" if r["dscore_mean"] is not None else "-"
        print(f"{k:14s} {r['matched']:>8d}/{r['n_a']:<4d} (+{r['only_b']:<3d}) {iou:>9s} {ds:>18s}   "
              f"{mAP:8.4f}  {tp}/{fp}/{fn}")
        results[k] = {**r, "map50_95": mAP, "tp": tp, "fp": fp, "fn": fn}
    Path("build").mkdir(exist_ok=True)
    Path("build/fp16_simulate.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nwritten: build/fp16_simulate.json")


if __name__ == "__main__":
    main()
