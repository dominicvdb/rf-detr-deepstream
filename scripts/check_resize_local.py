"""Does the resize matter for this model? Compares, on labelled frames (COCO json), without DeepStream:

  training    PIL resize of the frame (antialiased, 8-bit), as rfdetr trains: torchvision resize on a PIL image,
              then to_tensor + normalize, through the plain RF-DETR ONNX
  antialias   bundle built with --resize antialias, on raw RGB 0-255 frames (as nvinfer delivers them)
  bilinear    bundle built with the default --resize bilinear (= predict())

Reports detection parity of both bundles against the training preprocessing, the input-tensor difference, and
box mAP + precision/recall against the annotations.

    python scripts/check_resize_local.py --original model.onnx --antialias aa/model_1920x1080.onnx \
        --bilinear bl/model_1920x1080.onnx --annotations annotations.coco.json --images DIR --threshold 0.6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD, decode


def outputs(session: ort.InferenceSession, x: np.ndarray, capture: str | None = None) -> dict[str, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, {session.get_inputs()[0].name: x})))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True, help="plain RF-DETR ONNX (input = model size)")
    ap.add_argument("--antialias", type=Path, required=True, help="bundle ONNX built with --resize antialias")
    ap.add_argument("--bilinear", type=Path, required=True, help="bundle ONNX built with --resize bilinear")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder the file_name entries are relative to")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--background", default="-1", help="background logits slot (-1 fine-tuned, none: COCO)")
    args = ap.parse_args()
    background = None if args.background.lower() == "none" else int(args.background)
    thr = args.threshold

    ann = json.loads(args.annotations.read_text())
    images = [im for im in ann["images"] if im.get("split", "test") == "test"]
    sessions = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for k, p in (("training", args.original), ("antialias", args.antialias), ("bilinear", args.bilinear))}
    model_h, model_w = sessions["training"].get_inputs()[0].shape[2:]

    # The antialias bundle's own resized + normalised tensor, to compare with training's input tensor.
    import onnx
    probe = onnx.load(str(args.antialias))
    old_input = next(n.output[0] for n in probe.graph.node if n.name == "pre_div_std")
    probe.graph.output.insert(0, onnx.helper.make_tensor_value_info(old_input, onnx.TensorProto.FLOAT, None))
    probe_session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])

    dets = {k: [] for k in sessions}
    tensor_diff = []
    for n, im in enumerate(images):
        pil = Image.open(args.images / im["file_name"]).convert("RGB")
        w, h = pil.size
        x_train = F.normalize(F.to_tensor(F.resize(pil, [model_h, model_w])), list(IMAGENET_MEAN),
                              list(IMAGENET_STD)).unsqueeze(0).numpy()
        frame = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)[None].copy()
        for k in sessions:
            o = outputs(sessions[k], x_train if k == "training" else frame)
            dets[k].append(decode(o["dets"][0], o["labels"][0], w, h, threshold=0.0, background_class_id=background))
        if n < 20:
            x_aa = probe_session.run([old_input], {"image": frame})[0]
            # in grey levels of the 0-255 image: undo the normalisation
            tensor_diff.append(np.abs(x_aa - x_train) * np.array(IMAGENET_STD).reshape(1, 3, 1, 1) * 255)
        if n % 50 == 0:
            print(f"  {n}/{len(images)}")

    td = np.concatenate([t.ravel() for t in tensor_diff])
    print(f"\ninput tensor, antialias bundle vs training (20 frames, in grey levels): mean {td.mean():.3f}  "
          f"max {td.max():.3f}  > 0.5: {np.mean(td > 0.5) * 100:.2f}%")

    def at(d, t):
        k = d.confidence > t
        return d.xyxy[k], d.class_id[k], d.confidence[k]

    results = {}
    for k in ("antialias", "bilinear"):
        s = ParityStats()
        for a, b in zip(dets["training"], dets[k]):
            s.add(*at(a, thr), *at(b, thr))
        print(f"\n== training vs {k} bundle (threshold {thr})")
        print(s.report("training", k, thr))
        results[f"training_vs_{k}"] = s.summary(thr)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {**ann, "images": images}
        gt.createIndex()
    print(f"\n== accuracy vs annotations ({len(images)} frames)")
    for k in sessions:
        rows = [{"image_id": im["id"], "category_id": int(c), "score": float(sc),
                 "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
                for im, d in zip(images, dets[k]) for b, c, sc in zip(*at(d, 0.0))]
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(gt, gt.loadRes(rows), "bbox")
            e.evaluate(); e.accumulate(); e.summarize()  # noqa: E702
        tp = fp = fn = 0
        for im, d in zip(images, dets[k]):
            g = [a for a in ann["annotations"] if a["image_id"] == im["id"]]
            gx = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                           for a in g]).reshape(-1, 4)
            px, pc, _ = at(d, thr)
            m = len(match(gx, [a["category_id"] for a in g], px, pc))
            tp, fp, fn = tp + m, fp + len(px) - m, fn + len(gx) - m
        results[f"accuracy_{k}"] = {"map50_95": e.stats[0], "map50": e.stats[1], "map75": e.stats[2],
                                    "tp": tp, "fp": fp, "fn": fn}
        print(f"  {k:10s} mAP50-95 {e.stats[0]:.4f}  mAP50 {e.stats[1]:.4f}  mAP75 {e.stats[2]:.4f}   "
              f"@{thr}: tp {tp} fp {fp} fn {fn}")
    Path("build").mkdir(exist_ok=True)
    Path("build/resize_local.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nwritten: build/resize_local.json")


if __name__ == "__main__":
    main()
