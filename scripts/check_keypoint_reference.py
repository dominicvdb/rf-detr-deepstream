"""Check the Python keypoint reference (ONNX Runtime) against RFDETRKeypointPreview.predict(),
and the constant-folded ONNX against the original export.

    python scripts/check_keypoint_reference.py
"""

import warnings
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
from PIL import Image

warnings.simplefilter("ignore")
from rfdetr import RFDETRKeypointPreview  # noqa: E402

from rfdetr_deepstream.compare import match  # noqa: E402
from rfdetr_deepstream.reference import preprocess, run_onnx_keypoints  # noqa: E402

ORIGINAL = "models/keypoint/rfdetr-keypoint-preview.onnx"
FOLDED = "models/keypoint/rfdetr-keypoint-preview_folded.onnx"
SCHEMA = [0, 17]
IMG_DIR = Path("data/coco_samples")
COCO_IDS = ["000000000785", "000000001000", "000000000139", "000000000872", "000000000885", "000000001268",
            "000000001296", "000000001490", "000000002153", "000000002261"]
THRESHOLD = 0.5

IMG_DIR.mkdir(parents=True, exist_ok=True)
images = []
for cid in COCO_IDS:
    path = IMG_DIR / f"{cid}.jpg"
    if not path.exists():
        try:
            urlretrieve(f"http://images.cocodataset.org/val2017/{cid}.jpg", path)
        except HTTPError:
            continue
    images.append((cid, Image.open(path).convert("RGB")))

# 1. Folded vs original ONNX: raw outputs on one image.
orig = ort.InferenceSession(ORIGINAL, providers=["CPUExecutionProvider"])
fold = ort.InferenceSession(FOLDED, providers=["CPUExecutionProvider"])
x = preprocess(images[0][1], 576, 576)
for name, a, b in zip(["dets", "labels", "keypoints"], orig.run(None, {"input": x}), fold.run(None, {"input": x})):
    print(f"folded vs original  {name:9s} max |diff| = {np.abs(a - b).max():.2e}")

# 2. Reference (folded ONNX) vs predict().
model = RFDETRKeypointPreview()
n_ref = n_ours = n_match = 0
ious, dscore, dkp, dconf = [], [], [], []
for cid, image in images:
    ref = model.predict(image, threshold=THRESHOLD)
    ours = run_onnx_keypoints(fold, image, SCHEMA, threshold=THRESHOLD)
    ref_xyxy = np.asarray(ref.data["xyxy"]).reshape(-1, 4)
    pairs = match(ref_xyxy, ref.class_id, ours.xyxy, ours.class_id)
    n_ref, n_ours, n_match = n_ref + len(ref_xyxy), n_ours + len(ours.xyxy), n_match + len(pairs)
    for i, j, v in pairs:
        ious.append(v)
        dscore.append(abs(float(ref.detection_confidence[i]) - float(ours.confidence[j])))
        dkp.append(np.linalg.norm(ref.xy[i] - ours.keypoints[j, :, :2], axis=-1).max())
        dconf.append(np.abs(ref.keypoint_confidence[i] - ours.keypoints[j, :, 2]).max())
    print(f"{cid}: predict={len(ref_xyxy):2d} ours={len(ours.xyxy):2d} matched={len(pairs):2d}")

print(f"\nTOTAL predict={n_ref} ours={n_ours} matched={n_match}")
print(f"box IoU            mean={np.mean(ious):.5f}  min={np.min(ious):.5f}")
print(f"|dscore| (fused)   mean={np.mean(dscore):.5f}  max={np.max(dscore):.5f}")
print(f"keypoint distance  max per person: mean={np.mean(dkp):.3f}px  max={np.max(dkp):.3f}px")
print(f"|dconf| keypoint   mean={np.mean(dconf):.5f}  max={np.max(dconf):.5f}")
