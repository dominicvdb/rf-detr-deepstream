"""Export RF-DETR KeypointPreview to ONNX and print its interface and keypoint schema.

    python scripts/export_keypoint.py
"""

import warnings
from pathlib import Path

import onnx

warnings.simplefilter("ignore")
from rfdetr import RFDETRKeypointPreview  # noqa: E402
from rfdetr.utilities.class_names import prediction_labels  # noqa: E402

OUT_DIR = Path("models/keypoint")

model = RFDETRKeypointPreview()
cfg = model.model_config
labels = prediction_labels(model)
print("class names:", labels.class_names)
print("logit slots (num_classes):", labels.num_classes)
print("num_keypoints_per_class:", labels.num_keypoints_per_class)
print("class_id_to_name:", labels.class_id_to_name)
print("resolution:", getattr(cfg, "resolution", None), " trace_alpha:", getattr(cfg, "postprocess_trace_alpha", 0.2))
for attr in ("keypoint_names", "skeleton", "flip_pairs", "num_select"):
    print(f"{attr}:", getattr(cfg, attr, getattr(model.model, attr, None)))

model.export(format="onnx", output_dir=str(OUT_DIR), verbose=False)
onnx_path = next(OUT_DIR.glob("*.onnx"))
graph = onnx.load(str(onnx_path)).graph


def describe(v):
    t = v.type.tensor_type
    return f"{v.name:10s} {onnx.TensorProto.DataType.Name(t.elem_type):8s} {[d.dim_value or d.dim_param for d in t.shape.dim]}"


print(f"\n{onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
for i in graph.input:
    print("  INPUT ", describe(i))
for o in graph.output:
    print("  OUTPUT", describe(o))
ops = sorted({n.op_type for n in graph.node})
print(f"{len(graph.node)} nodes, ops: {', '.join(ops)}")
