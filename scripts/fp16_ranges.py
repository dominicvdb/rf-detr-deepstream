"""Which layers of an RF-DETR ONNX produce values FP16 can't hold? Runs the model in FP32 (ONNX Runtime) on real
frames with every intermediate tensor exposed, and reports per node the largest |value|, and for LayerNorms also the
largest input |value| and squared deviation (what an FP16 variance computation has to hold).

FP16's largest finite value is 65504; squares overflow from |x| > 256.

    python scripts/fp16_ranges.py MODEL.onnx IMAGE [IMAGE ...] [--json out.json]

MODEL is the plain export (input = model size) or a bundle ONNX (input = frame size, preprocessing built in).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

FP16_MAX = 65504.0


def model_input(image: Image.Image, shape) -> np.ndarray:
    _, _, h, w = shape
    if image.size == (w, h):  # bundle ONNX: raw RGB 0-255, preprocessing inside
        return np.asarray(image, np.float32).transpose(2, 0, 1)[None].copy()
    from rfdetr_deepstream.reference import preprocess
    return preprocess(image, h, w)


def float_tensors(model: onnx.ModelProto) -> list[str]:
    """Every non-constant float tensor produced by a node."""
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    const = {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
    return [o for n in model.graph.node if n.op_type != "Constant" for o in n.output
            if o and o not in const and types.get(o) == onnx.TensorProto.FLOAT]


def analyse(model_path: Path, images: list[Path]) -> dict:
    model = onnx.load(str(model_path))
    names = float_tensors(model)
    existing = {o.name for o in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    producer = {o: n for n in model.graph.node for o in n.output}
    ln_inputs = {n.input[0]: n.name for n in model.graph.node if n.op_type == "LayerNormalization"}

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep every node's output
    session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out_names = [o.name for o in session.get_outputs()]

    peak = defaultdict(float)
    ln_dev2 = defaultdict(float)
    for path in images:
        values = dict(zip(out_names, session.run(None, {inp.name: model_input(Image.open(path).convert("RGB"),
                                                                                  inp.shape)})))
        for name in names:
            v = values.get(name)
            if v is None or v.size == 0:
                continue
            peak[name] = max(peak[name], float(np.abs(v).max()))
            if name in ln_inputs:
                dev = v - v.mean(axis=-1, keepdims=True)
                ln_dev2[name] = max(ln_dev2[name], float((dev * dev).max()))
    nodes = []
    for name in names:
        n = producer[name]
        nodes.append({"node": n.name, "op": n.op_type, "tensor": name, "max_abs": peak[name],
                      "feeds_layernorm": ln_inputs.get(name), "ln_max_sq_dev": ln_dev2.get(name)})
    return {"model": str(model_path), "images": len(images), "nodes": nodes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    r = analyse(args.model, args.images)
    if args.json:
        args.json.write_text(json.dumps(r, indent=1))
    nodes = r["nodes"]
    over = [n for n in nodes if n["max_abs"] > FP16_MAX]
    print(f"{len(nodes)} float tensors, {r['images']} frames. Over FP16's max (65504): {len(over)}")
    for n in sorted(nodes, key=lambda n: -n["max_abs"])[: args.top]:
        flag = "OVER " if n["max_abs"] > FP16_MAX else "     "
        print(f"  {flag}{n['max_abs']:12.1f}  {n['op']:20s} {n['node']}")
    lns = [n for n in nodes if n["feeds_layernorm"]]
    print(f"\nLayerNorm inputs ({len(lns)}): largest |x| and squared deviation from the row mean")
    for n in sorted(lns, key=lambda n: -n["ln_max_sq_dev"])[: args.top]:
        flag = "OVER " if n["ln_max_sq_dev"] > FP16_MAX else "     "
        print(f"  {flag}|x| {n['max_abs']:10.1f}   (x-mean)^2 {n['ln_max_sq_dev']:14.1f}   {n['feeds_layernorm']}")


if __name__ == "__main__":
    main()
