"""Check that the ONNX with built-in preprocessing gives the same detections as the original pipeline
(torchvision preprocessing + original ONNX), on the first video frames.

    python scripts/check_onnx_patch.py
"""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from rfdetr_deepstream.compare import ParityStats
from rfdetr_deepstream.reference import decode, run_onnx

N = 100
original = ort.InferenceSession("models/nano/rfdetr-nano.onnx", providers=["CPUExecutionProvider"])
patched = ort.InferenceSession("models/nano_1280x720/rfdetr-nano_1280x720.onnx", providers=["CPUExecutionProvider"])
print("patched input:", [(i.name, i.shape) for i in patched.get_inputs()])

cap = cv2.VideoCapture("data/sample_720p.mp4")
stats = ParityStats()
for _ in range(N):
    ok, bgr = cap.read()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ref = run_onnx(original, Image.fromarray(rgb), threshold=0.5)
    # The patched model takes what nvinfer will give it: RGB, NCHW, float 0-255, full frame.
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    dets, logits = patched.run(["dets", "labels"], {"image": x})
    ours = decode(dets[0], logits[0], rgb.shape[1], rgb.shape[0], threshold=0.5)
    stats.add(ref.xyxy, ref.class_id, ref.confidence, ours.xyxy, ours.class_id, ours.confidence)
print(stats.report("torchvision+original", "patched", threshold=0.5))
