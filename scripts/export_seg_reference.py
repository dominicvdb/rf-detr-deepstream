"""Export the plain RF-DETR segmentation ONNX (no preprocessing built in): the reference for parity_segmentation.py.

    python scripts/export_seg_reference.py seg-nano build/seg_nano_reference.onnx
"""

import sys
from pathlib import Path

from rfdetr_deepstream.bundle import export_onnx, load_rfdetr

export_onnx(load_rfdetr(sys.argv[1]), Path(sys.argv[2]), batch_size=1, static_batch=True)
print(f"wrote {sys.argv[2]}")
