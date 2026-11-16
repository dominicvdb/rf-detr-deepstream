"""FP16 safety for RF-DETR ONNX models, and an FP16 simulation to check it without a GPU.

Problem: a LayerNorm computes its variance from squared deviations (x - mean)^2. A few RF-DETR LayerNorm inputs
(e.g. decoder layer 0 norm1, the projector's cv1/bn) reach values whose squares go far beyond FP16's maximum of 65504,
although the values themselves fit. When TensorRT runs such a LayerNorm in FP16 the variance becomes inf and the
normalised output collapses to the bias. scripts/fp16_ranges.py shows the ranges for a model.

Fix (scale_layernorm_inputs): divide the input of each at-risk LayerNorm by a power of two s. LayerNorm is scale
invariant: LN(x / s) = (x - m) / sqrt(var + eps * s^2), so the only change is eps * s^2 (1e-6 * 1024 = 1e-3 against a
variance of hundreds or more). A power of two is exact in floating point. This works on every TensorRT version and
needs no per-layer precision settings in nvinfer.

Simulation (simulate_fp16): every float tensor is rounded to FP16 (Cast to float16 and back: FP16 storage with range
overflow to inf), and each LayerNorm can be decomposed into its steps, each rounded to FP16, which is what an FP16
LayerNorm kernel risks. ONNX Runtime then runs it on the CPU. It is a model of TensorRT's FP16 mode, not a copy: real
kernels may accumulate in FP32 and choose FP32 for some layers. It shows whether a model is FP16-safe in principle.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16_MAX = 65504.0
# Scale LayerNorm inputs so that the measured largest |x - mean| becomes at most this. Squares are then at most 4096:
# 16x headroom below FP16's maximum for frames with larger values than the calibration frames.
TARGET_MAX_DEVIATION = 64.0


# --------------------------------------------------------------------------------------------------
# Calibration: largest |x - mean| at each LayerNorm input
# --------------------------------------------------------------------------------------------------
def layernorm_deviations(model_path: Path, inputs: list[np.ndarray]) -> dict[str, float]:
    """LayerNorm node name -> largest |x - row mean| of its input over the given model inputs (FP32, ONNX Runtime)."""
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    lns = [n for n in model.graph.node if n.op_type == "LayerNormalization"]
    existing = {o.name for o in model.graph.output}
    for n in lns:
        if n.input[0] not in existing:
            model.graph.output.append(helper.make_empty_tensor_value_info(n.input[0]))
            existing.add(n.input[0])
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    inp = session.get_inputs()[0].name
    dev = {n.name: 0.0 for n in lns}
    for x in inputs:
        values = dict(zip(names, session.run(None, {inp: x})))
        for n in lns:
            axis = next((a.i for a in n.attribute if a.name == "axis"), -1)
            v = values[n.input[0]]
            dev[n.name] = max(dev[n.name], float(np.abs(v - v.mean(axis=axis, keepdims=True)).max()))
    return dev


def calibration_inputs(height: int, width: int, batch: int = 1, images: list[Path] | None = None,
                       seed: int = 0) -> list[np.ndarray]:
    """Model-input tensors (normalised, NCHW) for calibration. Without images: synthetic frames (grey, noise,
    coarse blocks). The largest deviations come from the weights and appear for any input; the input-dependent ones
    can be somewhat larger on real frames, which the 16x headroom of TARGET_MAX_DEVIATION covers. Pass images to
    calibrate on real frames instead."""
    from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD

    if images:
        from PIL import Image

        rgbs = [np.asarray(Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)) for p in images]
    else:
        rng = np.random.default_rng(seed)
        blocks = rng.integers(0, 256, (height // 32 + 1, width // 32 + 1, 3))
        rgbs = [np.full((height, width, 3), 128, np.uint8),
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
                np.kron(blocks, np.ones((32, 32, 1)))[:height, :width].astype(np.uint8)]
    mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
    out = []
    for rgb in rgbs:
        x = (rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0 - mean) / std
        out.append(np.repeat(x, batch, axis=0).astype(np.float32))
    return out


def make_fp16_safe(onnx_in: Path, onnx_out: Path, images: list[Path] | None = None) -> dict[str, int]:
    """Calibrate onnx_in (plain model, input = model size) and write onnx_out with the at-risk LayerNorm inputs
    scaled. Returns {LayerNorm node: divisor}."""
    model = onnx.load(str(onnx_in))
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    batch = dims[0] or 1
    scales = layernorm_scales(layernorm_deviations(onnx_in, calibration_inputs(dims[2], dims[3], batch, images)))
    scale_layernorm_inputs(model, scales)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_out))
    return scales


def layernorm_scales(deviations: dict[str, float], target: float = TARGET_MAX_DEVIATION) -> dict[str, int]:
    """Power-of-two divisor per LayerNorm that brings its largest deviation to at most `target`; only those > 1."""
    scales = {}
    for name, d in deviations.items():
        if d > target:
            scales[name] = 2 ** math.ceil(math.log2(d / target))
    return scales


# --------------------------------------------------------------------------------------------------
# The fix
# --------------------------------------------------------------------------------------------------
def scale_layernorm_inputs(model: onnx.ModelProto, scales: dict[str, int]) -> onnx.ModelProto:
    """Insert x * (1/s) before each named LayerNorm (in place; returns the model)."""
    by_name = {n.name: n for n in model.graph.node}
    missing = set(scales) - set(by_name)
    if missing:
        raise ValueError(f"LayerNorm nodes not in the model: {sorted(missing)}")
    new_nodes = []
    for node in model.graph.node:
        s = scales.get(node.name)
        if s is not None:
            if node.op_type != "LayerNormalization":
                raise ValueError(f"{node.name} is a {node.op_type}, not a LayerNormalization")
            const = f"{node.name}/fp16_safe_scale"
            scaled = f"{node.input[0]}/fp16_safe_{node.name}"
            model.graph.initializer.append(numpy_helper.from_array(np.array(1.0 / s, np.float32), const))
            new_nodes.append(helper.make_node("Mul", [node.input[0], const], [scaled], name=f"{node.name}/fp16_safe"))
            node.input[0] = scaled
        new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


# --------------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------------
def _float_outputs(model: onnx.ModelProto) -> set[str]:
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    return {o for n in model.graph.node if n.op_type != "Constant" for o in n.output
            if o and types.get(o) == TensorProto.FLOAT}


def simulate_fp16(model: onnx.ModelProto, layernorm_in_fp16: bool = True,
                  keep_fp32: set[str] = frozenset()) -> onnx.ModelProto:
    """A copy of the model that rounds every float tensor to FP16. With layernorm_in_fp16, LayerNorms not in
    keep_fp32 are decomposed and every step is rounded to FP16 too (the variance can overflow)."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    floats = _float_outputs(model)
    nodes = []
    counter = [0]

    def rounded(tensor: str, out: str | None = None) -> str:
        """Cast tensor to FP16 and back; returns the name of the rounded tensor."""
        counter[0] += 1
        out = out or f"{tensor}/fp16sim{counter[0]}"
        half = f"{out}/half"
        nodes.append(helper.make_node("Cast", [tensor], [half], to=TensorProto.FLOAT16))
        nodes.append(helper.make_node("Cast", [half], [out], to=TensorProto.FLOAT))
        return out

    for node in model.graph.node:
        if node.op_type == "LayerNormalization" and layernorm_in_fp16 and node.name not in keep_fp32:
            x, scale = node.input[0], node.input[1]
            bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
            axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
            eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
            p = f"{node.name}/fp16sim"
            eps_name = f"{p}/eps"
            model.graph.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), eps_name))

            def step(op, ins, name, **attrs):
                nodes.append(helper.make_node(op, ins, [f"{p}/{name}_raw"], **attrs))
                return rounded(f"{p}/{name}_raw", f"{p}/{name}")

            mean = step("ReduceMean", [x], "mean", axes=[axis], keepdims=1)
            d = step("Sub", [x, mean], "dev")
            sq = step("Mul", [d, d], "sq")                     # overflows to inf for |dev| > 256
            var = step("ReduceMean", [sq], "var", axes=[axis], keepdims=1)
            std = step("Sqrt", [step("Add", [var, eps_name], "var_eps")], "std")
            y = step("Mul", [step("Div", [d, std], "norm"), scale], "scaled")
            out = node.output[0]
            if bias:
                nodes.append(helper.make_node("Add", [y, bias], [f"{out}/fp16sim_raw"]))
            else:
                nodes.append(helper.make_node("Identity", [y], [f"{out}/fp16sim_raw"]))
            rounded(f"{out}/fp16sim_raw", out)
            continue
        outs = list(node.output)
        for i, o in enumerate(outs):
            if o in floats:
                node.output[i] = f"{o}/fp32"
        nodes.append(node)
        for o in outs:
            if o in floats:
                rounded(f"{o}/fp32", o)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model
