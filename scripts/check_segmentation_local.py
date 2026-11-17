"""Check a segmentation bundle without DeepStream: on a folder of frames (COCO annotations), compare

  rfdetr     the plain RF-DETR ONNX + predict()'s preprocessing + rfdetr's own PostProcess (masks included)
  bundle     the bundle ONNX (preprocessing built in) on raw RGB 0-255 frames, as nvinfer delivers them,
             + reference.decode_masks()
  parser     the bundle's C++ parser (NvDsInferParseRfdetrSeg) on the bundle ONNX's raw outputs, through
             scripts/parser_harness (build it first: make -C scripts/parser_harness)

and, with --gt, report box and mask mAP against the annotations: the bundle as DeepStream returns it (masks
clipped to their boxes), the bundle's full-frame masks (= predict()), and the plain model with an antialiased resize
(how cfg-detect / rfdetr train and evaluate) for comparison.

    python scripts/check_segmentation_local.py --original model.onnx --bundle bundle/model_1920x1080.onnx \
        --annotations annotations.coco.json --images DIR --threshold 0.5 --gt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image

from rfdetr_deepstream.compare import ParityStats, match
from rfdetr_deepstream.reference import (IMAGENET_MEAN, IMAGENET_STD, clip_masks_to_boxes, decode_masks,
                                         mask_box_region)

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "build/parser_harness/harness"


def run(session: ort.InferenceSession, x: np.ndarray) -> dict[str, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(None, {session.get_inputs()[0].name: x})))


def model_input(pil: Image.Image, size: tuple[int, int], antialias: bool) -> np.ndarray:
    t = F.resize(F.to_tensor(pil), list(size), antialias=antialias)
    return F.normalize(t, list(IMAGENET_MEAN), list(IMAGENET_STD)).unsqueeze(0).numpy()


def rfdetr_postprocess(out: dict, width: int, height: int, threshold: float, background: int | None):
    """rfdetr 1.12.1's own PostProcess (top-k, boxes, mask upsample > 0) on raw ONNX outputs."""
    from rfdetr.models.postprocess import PostProcess

    logits = torch.from_numpy(out["labels"]).clone()
    if background is not None:  # predict() never returns the background slot; exclude it the same way as decode()
        logits[..., background] = -torch.inf
    res = PostProcess()({"pred_logits": logits, "pred_boxes": torch.from_numpy(out["dets"]),
                         "pred_masks": torch.from_numpy(out["masks"])},
                        torch.tensor([[height, width]]), score_threshold=threshold)[0]
    return res["boxes"].numpy(), res["labels"].numpy(), res["scores"].numpy(), res["masks"][:, 0].numpy()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True, help="plain RF-DETR seg ONNX (input = model size)")
    ap.add_argument("--bundle", type=Path, required=True, help="bundle ONNX (preprocessing built in)")
    ap.add_argument("--annotations", type=Path, required=True, help="COCO json listing the frames")
    ap.add_argument("--images", type=Path, required=True, help="folder the file_name entries are relative to")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--background", type=int, default=-1, help="background logits slot (-1 fine-tuned, none: COCO)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", action="store_true", help="also report box and mask mAP against the annotations")
    args = ap.parse_args()
    background = None if str(args.background).lower() == "none" else args.background
    thr = args.threshold

    ann = json.loads(args.annotations.read_text())
    images = [im for im in ann["images"] if im.get("split", "test") == "test"][: args.limit or None]
    original = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
    bundle = ort.InferenceSession(str(args.bundle), providers=["CPUExecutionProvider"])
    model_hw = tuple(original.get_inputs()[0].shape[2:])
    frame_hw = tuple(bundle.get_inputs()[0].shape[2:])
    print(f"{len(images)} frames, model input {model_hw}, bundle input {frame_hw}, threshold {thr}")

    stats = {k: ParityStats() for k in ("rfdetr_vs_bundle", "bundle_vs_parser", "antialiased_vs_bundle")}
    mask_ious = {k: [] for k in stats}
    raw, bundle_dets, aa_dets = [], [], []
    for i, im in enumerate(images):
        pil = Image.open(args.images / im["file_name"]).convert("RGB")
        w, h = pil.size
        assert (h, w) == frame_hw, f"{im['file_name']} is {w}x{h}, the bundle expects {frame_hw[1]}x{frame_hw[0]}"

        ref = rfdetr_postprocess(run(original, model_input(pil, model_hw, False)), w, h, thr, background)
        out = run(bundle, np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)[None].copy())
        raw.append(out)
        b = decode_masks(out["dets"][0], out["labels"][0], out["masks"][0], w, h, 0.0 if args.gt else thr,
                         background_class_id=background)
        bundle_dets.append(b)
        keep = b.confidence > thr
        bx, bc, bs, bm = b.xyxy[keep], b.class_id[keep], b.confidence[keep], b.masks[keep]

        stats["rfdetr_vs_bundle"].add(ref[0], ref[1], ref[2], bx, bc, bs)
        mask_ious["rfdetr_vs_bundle"] += [mask_iou(ref[3][a], bm[c]) for a, c, _ in match(ref[0], ref[1], bx, bc)]

        aa_out = run(original, model_input(pil, model_hw, True))
        a = decode_masks(aa_out["dets"][0], aa_out["labels"][0], aa_out["masks"][0], w, h, 0.0 if args.gt else thr,
                         background_class_id=background)
        aa_dets.append(a)
        ak = a.confidence > thr
        stats["antialiased_vs_bundle"].add(a.xyxy[ak], a.class_id[ak], a.confidence[ak], bx, bc, bs)
        mask_ious["antialiased_vs_bundle"] += [mask_iou(a.masks[ak][p], bm[q])
                                               for p, q, _ in match(a.xyxy[ak], a.class_id[ak], bx, bc)]
        if i % 50 == 0:
            print(f"  {i}/{len(images)}")

    # The C++ parser on the same raw outputs, as nvinfer would call it (network size = frame size).
    q, c = raw[0]["labels"].shape[1:]
    hm, wm = raw[0]["masks"].shape[2:]
    num_classes_thr = [str(thr)] * c
    if background is not None:
        num_classes_thr[background % c] = "1.0"
    pixel_diff = pixel_total = 0
    outside_fraction = []
    with tempfile.TemporaryDirectory() as tmp:
        blob, masks_out = Path(tmp) / "in.bin", Path(tmp) / "masks.bin"
        with open(blob, "wb") as f:
            for out in raw:
                for k in ("dets", "labels", "masks"):
                    f.write(out[k][0].astype(np.float32).tobytes())
        lines = subprocess.run([str(HARNESS), "seg", str(blob), str(len(raw)), str(q), str(c), str(hm), str(wm),
                                str(frame_hw[1]), str(frame_hw[0]), str(masks_out), *num_classes_thr],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        mask_data = np.fromfile(masks_out, dtype=np.float32)

    per_image = [([], [], [], []) for _ in raw]
    offset = 0
    for line in lines:
        img, cls, left, top, bw, bh, score, mw, mh = line.split()
        left, top, bw, bh = map(float, (left, top, bw, bh))
        mw, mh = int(mw), int(mh)
        m = mask_data[offset: offset + mw * mh].reshape(mh, mw)
        offset += mw * mh
        p = per_image[int(img)]
        p[0].append([left, top, left + bw, top + bh]); p[1].append(int(cls)); p[2].append(float(score)); p[3].append(m)
    assert offset == mask_data.size

    for out, b, p in zip(raw, bundle_dets, per_image):
        keep = b.confidence > thr
        bx, bc, bs, bm = b.xyxy[keep], b.class_id[keep], b.confidence[keep], b.masks[keep]
        px = np.array(p[0]).reshape(-1, 4)
        stats["bundle_vs_parser"].add(bx, bc, bs, px, p[1], p[2])
        for i, j, _ in match(bx, bc, px, p[1]):
            x0, y0, x1, y1 = mask_box_region(px[j], frame_hw[1], frame_hw[0])
            assert p[3][j].shape == (y1 - y0, x1 - x0), (p[3][j].shape, (x0, y0, x1, y1))
            crop = bm[i][y0:y1, x0:x1]
            parser_mask = p[3][j] > 0.5
            pixel_diff += int((crop != parser_mask).sum())
            pixel_total += crop.size
            mask_ious["bundle_vs_parser"].append(mask_iou(crop, parser_mask))
            # The parser's mask covers the box only (DeepStream's convention): what predict() has outside it is lost.
            outside_fraction.append(float(bm[i].sum() - crop.sum()) / max(int(bm[i].sum()), 1))

    results = {}
    for name, s in stats.items():
        a, b = name.split("_vs_")
        print(f"\n== {a} vs {b}")
        print(s.report(a, b, thr))
        mi = np.array(mask_ious[name])
        if len(mi):
            print(f"mask IoU:   mean={mi.mean():.6f}  min={mi.min():.6f}  (matched detections)")
        results[name] = {**s.summary(thr), "mask_iou_mean": float(mi.mean()) if len(mi) else None,
                         "mask_iou_min": float(mi.min()) if len(mi) else None}
    of = np.array(outside_fraction)
    print(f"parser mask pixels differing from the reference inside the boxes: {pixel_diff} of {pixel_total}")
    print(f"mask area outside its box (dropped by DeepStream's box-shaped masks): mean {of.mean() * 100:.3f}%  "
          f"max {of.max() * 100:.2f}%")
    results["bundle_vs_parser"].update(mask_pixels_differing=pixel_diff, mask_pixels=pixel_total,
                                       mask_area_outside_box_mean=float(of.mean()),
                                       mask_area_outside_box_max=float(of.max()))

    if args.gt:
        results["accuracy"] = {"deepstream": coco_eval(ann, images, [clip_masks_to_boxes(d) for d in bundle_dets]),
                               "predict": coco_eval(ann, images, bundle_dets),
                               "antialiased": coco_eval(ann, images, aa_dets)}
        print("\n== accuracy vs annotations (all scores, top 100)\n"
              "deepstream = bundle, masks clipped to boxes; predict = bundle, full-frame masks; "
              "antialiased = plain model, antialiased resize (training)")
        for k, r in results["accuracy"].items():
            print(f"{k:12s} box mAP50-95={r['bbox'][0]:.4f} mAP50={r['bbox'][1]:.4f}   "
                  f"mask mAP50-95={r['segm'][0]:.4f} mAP50={r['segm'][1]:.4f}")
    print(json.dumps(results, indent=1, default=float), file=open(ROOT / "build/segmentation_local.json", "w"))
    print("\nwritten: build/segmentation_local.json")


def coco_eval(ann: dict, images: list[dict], dets: list) -> dict:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rows = []
    for im, d in zip(images, dets):
        order = np.argsort(-d.confidence, kind="stable")[:100]
        for k in order:
            x1, y1, x2, y2 = (float(v) for v in d.xyxy[k])
            rle = mask_utils.encode(np.asfortranarray(d.masks[k].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            rows.append({"image_id": im["id"], "category_id": int(d.class_id[k]), "score": float(d.confidence[k]),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": rle})
    ids = {im["id"] for im in images}
    gt_json = {**ann, "images": images, "annotations": [a for a in ann["annotations"] if a["image_id"] in ids]}
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_json
        gt.createIndex()
        res = gt.loadRes(rows)
        for kind in ("bbox", "segm"):
            e = COCOeval(gt, res, kind)
            e.evaluate(); e.accumulate(); e.summarize()  # noqa: E702
            out[kind] = [float(v) for v in e.stats[:3]]
    return out


if __name__ == "__main__":
    main()
