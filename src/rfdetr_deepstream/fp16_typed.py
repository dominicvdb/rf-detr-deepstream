"""FP16 ONNX for a strongly typed TensorRT build: the precision of every layer is fixed in the ONNX.

Every float tensor becomes FP16, except the outputs of nodes on the keep-FP32 list (by default the LayerNorms and the
preprocessing built into the ONNX). Casts are inserted where FP16 and FP32 meet. The model input and outputs stay
FP32, so nvinfer's input and the parser are unchanged.

With nvinfer strongly-typed=1 (DeepStream 9.0+) or trtexec --stronglyTyped, TensorRT runs each layer in exactly the
type the ONNX gives it. Without strong typing TensorRT picks FP16 or FP32 per layer by measured speed, so the same
ONNX can give an FP16 engine on one build and a mostly FP32 one on the next.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FLOAT, FLOAT16 = TensorProto.FLOAT, TensorProto.FLOAT16

# LayerNorm variance overflows in FP16 (RF-DETR issues #176/#352); pre_*: the preprocessing in the ONNX (0-255 input).
DEFAULT_KEEP_FP32_OPS = ("LayerNormalization",)
DEFAULT_KEEP_FP32_NAMES = (r"^pre_",)


def _float_tensors(model: onnx.ModelProto) -> set[str]:
    inferred = onnx.shape_inference.infer_shapes(model)
    g = inferred.graph
    names = {v.name for v in [*g.value_info, *g.input, *g.output] if v.type.tensor_type.elem_type == FLOAT}
    names |= {t.name for t in g.initializer if t.data_type == FLOAT}
    for node in g.node:
        if node.op_type == "Constant":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is not None and value.data_type == FLOAT:
                names.add(node.output[0])
    return names


def convert_to_fp16(onnx_in: Path, onnx_out: Path, keep_fp32_ops=DEFAULT_KEEP_FP32_OPS,
                    keep_fp32_names=DEFAULT_KEEP_FP32_NAMES) -> list[str]:
    """Write an FP16 copy of onnx_in. Returns the names of the nodes kept in FP32."""
    model = onnx.load(str(onnx_in))
    graph = model.graph
    floats = _float_tensors(model)
    name_res = [re.compile(p) for p in keep_fp32_names]
    keep = {n.name for n in graph.node
            if n.op_type in keep_fp32_ops or any(r.search(n.name) for r in name_res)}

    def wants_fp32(node) -> bool:
        return node.name in keep

    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    # Precision each float tensor is produced in.
    precision: dict[str, int] = {}
    for inp in graph.input:
        if inp.name in floats:
            precision[inp.name] = FLOAT
    constants = {t.name: t for t in graph.initializer}
    constant_nodes = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    for name in [*constants, *constant_nodes]:
        if name in floats:
            users = consumers.get(name, [])
            precision[name] = FLOAT if users and all(wants_fp32(u) for u in users) else FLOAT16
    for node in graph.node:
        if node.op_type == "Constant":
            continue
        for out in node.output:
            if out in floats:
                precision[out] = FLOAT if wants_fp32(node) else FLOAT16

    # Constants and type-setting attributes follow their tensor's precision.
    for t in graph.initializer:
        if precision.get(t.name) == FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), t.name))
    for node in graph.node:
        out_prec = precision.get(node.output[0]) if node.output else None
        if out_prec != FLOAT16:
            continue
        for a in node.attribute:
            if node.op_type == "Cast" and a.name == "to" and a.i == FLOAT:
                a.i = FLOAT16
            elif a.name == "value" and a.t.data_type == FLOAT:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16), a.t.name))

    # Casts where a node reads a float tensor in the other precision.
    casts: dict[tuple[str, int], str] = {}
    new_nodes = []

    def cast(name: str, to: int) -> str:
        key = (name, to)
        if key not in casts:
            casts[key] = f"{name}__{'fp32' if to == FLOAT else 'fp16'}"
            new_nodes.append(helper.make_node("Cast", [name], [casts[key]], to=to,
                                              name=f"cast_{len(casts)}_{'fp32' if to == FLOAT else 'fp16'}"))
        return casts[key]

    ordered = []
    for node in graph.node:
        need = FLOAT if wants_fp32(node) else FLOAT16
        new_nodes.clear()
        for i, name in enumerate(node.input):
            if name in precision and precision[name] != need and not (node.op_type == "Cast"):
                node.input[i] = cast(name, need)
        ordered.extend(new_nodes)
        ordered.append(node)

    # Graph outputs stay FP32.
    for out in graph.output:
        if precision.get(out.name) == FLOAT16:
            inner = f"{out.name}__fp16"
            for node in ordered:
                node.output[:] = [inner if o == out.name else o for o in node.output]
                node.input[:] = [inner if i == out.name else i for i in node.input]
            ordered.append(helper.make_node("Cast", [inner], [out.name], to=FLOAT, name=f"cast_out_{out.name}"))

    del graph.node[:]
    graph.node.extend(ordered)
    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_out))
    return sorted(keep)
