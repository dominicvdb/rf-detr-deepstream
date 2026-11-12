"""Export RF-DETR Nano (COCO pretrained) to ONNX and print its interface.

Run from ~/rfdetr-deepstream with the venv active:
    python scripts/export_nano.py
"""

from pathlib import Path

import onnx
from rfdetr import RFDETRNano

OUT_DIR = Path("models/nano")

# 1. Export. Downloads the pretrained COCO weights on first run.
#    Defaults: batch_size=1, dynamic_batch=False, opset 17 -> a fully static graph.
model = RFDETRNano()
model.export(format="onnx", output_dir=str(OUT_DIR))

# 2. Inspect what was written.
print("\nFiles written:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.1f} MB")

onnx_path = next(OUT_DIR.glob("*.onnx"))
graph = onnx.load(str(onnx_path)).graph


def describe(value_info):
    t = value_info.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(t.elem_type)
    return f"{value_info.name:10s} {dtype:8s} {dims}"


print(f"\nONNX interface of {onnx_path.name}:")
for i in graph.input:
    print("  INPUT ", describe(i))
for o in graph.output:
    print("  OUTPUT", describe(o))

# 3. Which ONNX operations does the graph use? TensorRT 8.6 must support all of them.
ops = sorted({node.op_type for node in graph.node})
print(f"\n{len(graph.node)} nodes, {len(ops)} distinct op types:")
print("  " + ", ".join(ops))
