"""Check our Python reference decoder (ONNX Runtime) against RF-DETR's own PyTorch predict().

Run from ~/rfdetr-deepstream with the venv active:
    python scripts/check_reference.py
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

from rfdetr_deepstream.compare import match as match_boxes
from rfdetr_deepstream.reference import run_onnx

ONNX = Path("models/nano/rfdetr-nano.onnx")
IMG_DIR = Path("data/coco_samples")
COCO_IDS = ["000000039769", "000000000139", "000000000724", "000000000785", "000000001000", "000000000632"]
THRESHOLD = 0.5


def match(ref, ours):
    return match_boxes(ref.xyxy, ref.class_id, ours.xyxy, ours.class_id)


IMG_DIR.mkdir(parents=True, exist_ok=True)
for cid in COCO_IDS:
    path = IMG_DIR / f"{cid}.jpg"
    if not path.exists():
        urlretrieve(f"http://images.cocodataset.org/val2017/{cid}.jpg", path)

model = RFDETRNano()
session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

all_ious, all_dscore, total_ref, total_ours, total_matched = [], [], 0, 0, 0
for cid in COCO_IDS:
    image = Image.open(IMG_DIR / f"{cid}.jpg").convert("RGB")
    ref = model.predict(image, threshold=THRESHOLD)  # PyTorch, rfdetr's own pre/post-processing
    ours = run_onnx(session, image, threshold=THRESHOLD)  # ONNX Runtime + our reference decoder

    pairs = match(ref, ours)
    ious = [p[2] for p in pairs]
    dscore = [abs(float(ref.confidence[i]) - float(ours.confidence[j])) for i, j, _ in pairs]
    all_ious += ious
    all_dscore += dscore
    total_ref, total_ours, total_matched = total_ref + len(ref), total_ours + len(ours.xyxy), total_matched + len(pairs)

    names = sorted({COCO_CLASSES.get(int(c), f"?{c}") for c in ours.class_id})
    print(f"{cid}: predict={len(ref):2d}  ours={len(ours.xyxy):2d}  matched={len(pairs):2d}  "
          f"min IoU={min(ious, default=float('nan')):.4f}  max |dscore|={max(dscore, default=float('nan')):.4f}  "
          f"classes={names}")

print(f"\nTOTAL predict={total_ref} ours={total_ours} matched={total_matched}")
print(f"IoU     mean={np.mean(all_ious):.5f}  min={np.min(all_ious):.5f}")
print(f"|dscore| mean={np.mean(all_dscore):.5f}  max={np.max(all_dscore):.5f}")
