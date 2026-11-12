"""Write a DeepStream bundle (labels.txt, nvinfer config, optionally a patched ONNX) for the exported Nano ONNX.

Run from ~/rfdetr-deepstream with the venv active (after scripts/export_nano.py):
    python scripts/make_nano_bundle.py                          # models/nano/: nvinfer does the preprocessing
    python scripts/make_nano_bundle.py --frame-size 1280x720    # models/nano_1280x720/: preprocessing in the ONNX
"""

import argparse
import shutil
from pathlib import Path

from rfdetr_deepstream.deepstream import NvinferConfig, coco_sparse_labels, write_labels, write_nvinfer_config
from rfdetr_deepstream.onnx_patch import add_preprocessing

SOURCE_ONNX = Path("models/nano/rfdetr-nano.onnx")

parser = argparse.ArgumentParser()
parser.add_argument("--frame-size", help="WIDTHxHEIGHT of the frames nvinfer receives; builds preprocessing into the ONNX")
args = parser.parse_args()

if args.frame_size:
    frame_w, frame_h = (int(v) for v in args.frame_size.lower().split("x"))
    bundle = Path(f"models/nano_{frame_w}x{frame_h}")
    bundle.mkdir(parents=True, exist_ok=True)
    onnx_file = bundle / f"rfdetr-nano_{frame_w}x{frame_h}.onnx"
    add_preprocessing(SOURCE_ONNX, onnx_file, frame_height=frame_h, frame_width=frame_w)
else:
    bundle = SOURCE_ONNX.parent
    onnx_file = SOURCE_ONNX

names = coco_sparse_labels(91)
write_labels(bundle / "labels.txt", names)
write_nvinfer_config(
    bundle / "config_infer_rfdetr.txt",
    NvinferConfig(
        onnx_file=onnx_file,
        labels_file=bundle / "labels.txt",
        parser_lib=Path("parser/libnvds_rfdetr.so"),
        num_classes=len(names),
        threshold=0.5,
        precision="fp32",
        preprocessing_in_model=bool(args.frame_size),
    ),
)
print(f"Bundle written to {bundle}/:")
for f in sorted(bundle.iterdir()):
    print(f"  {f.name}")
print()
print((bundle / "config_infer_rfdetr.txt").read_text())
