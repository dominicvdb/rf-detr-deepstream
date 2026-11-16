"""FP16 safety for RF-DETR ONNX models, and an FP16 simulation to check it without a GPU.

Problem: a LayerNorm computes its variance from squared deviations (x - mean)^2. A few RF-DETR LayerNorm inputs
(e.g. decoder layer 0 norm1, the projector's cv1/bn) reach values whose squares go far beyond FP16's maximum of 65504,
although the values themselves fit. When TensorRT runs such a LayerNorm in FP16 the variance becomes inf and the
normalised output collapses to the bias. scripts/fp16_ranges.py shows the ranges for a model.

Fix (scale_layernorm_inputs): divide the input of each at-risk LayerNorm by a power of two s. LayerNorm is scale
invariant: LN(x / s) = (x - m) / sqrt(var + eps * s^2), so the only change is eps * s^2 (1e-6 * 1024 = 1e-3 against a
variance of hundreds or more). A power of two is exact in floating point. The same calibration run checks that no
other tensor comes near FP16's maximum, and that the scaling keeps small values above FP16's subnormal range.
fp16_typed then converts the ONNX to FP16 for a strongly typed TensorRT build.

Simulation (simulate_fp16): every float tensor is rounded to FP16 (Cast to float16 and back: FP16 storage with range
overflow to inf), and each LayerNorm can be decomposed into its steps, each rounded to FP16, which is what an FP16
LayerNorm kernel risks. ONNX Runtime then runs it on the CPU. It is a model of TensorRT's FP16 mode, not a copy: real
kernels may accumulate in FP32 and choose FP32 for some layers. It shows whether a model is FP16-safe in principle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16_MAX = 65504.0
# Scale LayerNorm inputs so that the measured largest |x - mean| becomes at most this. Squares are then at most 4096:
# 16x headroom below FP16's maximum for frames with larger values than the calibration frames.
TARGET_MAX_DEVIATION = 64.0
# After scaling, the median |x - mean| at a LayerNorm input stays at least this (well above FP16's subnormals).
MIN_MEDIAN_DEVIATION = 1e-3
# Any tensor above this on the calibration inputs is an overflow risk: 4x headroom for frames with larger values.
OVERFLOW_LIMIT = FP16_MAX / 4


# --------------------------------------------------------------------------------------------------
# Calibration: LayerNorm input deviations and the largest value of every tensor
# --------------------------------------------------------------------------------------------------
@dataclass
class Calibration:
    max_deviation: dict[str, float]       # LayerNorm -> largest |x - row mean| of its input
    median_deviation: dict[str, float]    # LayerNorm -> median (over inputs) of the median |x - row mean|
    tensor_max: dict[str, tuple[float, str]] = field(default_factory=dict)  # tensor -> (largest |x|, producing node)


def calibrate(model_path: Path, inputs: list[np.ndarray]) -> Calibration:
    """Run the model in FP32 (ONNX Runtime) on the given model inputs and record the FP16-relevant ranges."""
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    lns = [n for n in model.graph.node if n.op_type == "LayerNormalization"]
    floats = _float_outputs(model)
    producer = {o: n.name for n in model.graph.node for o in n.output}
    existing = {o.name for o in model.graph.output}
    for name in sorted(floats | {n.input[0] for n in lns}):
        if name not in existing:
            model.graph.output.append(helper.make_empty_tensor_value_info(name))
            existing.add(name)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    inp = session.get_inputs()[0].name
    cal = Calibration({n.name: 0.0 for n in lns}, {})
    medians: dict[str, list[float]] = {n.name: [] for n in lns}
    for x in inputs:
        values = dict(zip(names, session.run(None, {inp: x})))
        for n in lns:
            axis = next((a.i for a in n.attribute if a.name == "axis"), -1)
            v = values[n.input[0]]
            dev = np.abs(v - v.mean(axis=axis, keepdims=True))
            cal.max_deviation[n.name] = max(cal.max_deviation[n.name], float(dev.max()))
            medians[n.name].append(float(np.median(dev)))
        for name in floats:
            m = float(np.abs(values[name]).max()) if values[name].size else 0.0
            if m > cal.tensor_max.get(name, (-1.0, ""))[0]:
                cal.tensor_max[name] = (m, producer.get(name, ""))
    # Median over the inputs, without flat ones (all deviations 0: nothing to lose precision on).
    cal.median_deviation = {name: float(np.median([m for m in v if m > 0])) if any(m > 0 for m in v) else math.inf
                            for name, v in medians.items()}
    return cal


def layernorm_deviations(model_path: Path, inputs: list[np.ndarray]) -> dict[str, float]:
    """LayerNorm node name -> largest |x - row mean| of its input over the given model inputs (FP32, ONNX Runtime)."""
    return calibrate(model_path, inputs).max_deviation


def overflow_risks(cal: Calibration, limit: float = OVERFLOW_LIMIT) -> list[tuple[str, str, float]]:
    """(tensor, node, largest |x|) for every tensor above `limit` on the calibration inputs, largest first."""
    risky = [(t, node, m) for t, (m, node) in cal.tensor_max.items() if m > limit]
    return sorted(risky, key=lambda r: -r[2])


def calibration_inputs(height: int, width: int, batch: int = 1, images: list[Path] | None = None,
                       seed: int = 0, mean=None, std=None) -> list[np.ndarray]:
    """Model-input tensors (normalised with the model's mean/std, ImageNet by default; NCHW) for calibration.
    Without images: synthetic frames (grey, noise, coarse blocks). The largest deviations come from the weights and
    appear for any input; the input-dependent ones can be somewhat larger on real frames, which the 16x headroom of
    TARGET_MAX_DEVIATION covers. Pass images to calibrate on real frames instead."""
    from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD

    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    if images:
        from PIL import Image

        rgbs = [np.asarray(Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)) for p in images]
    else:
        rng = np.random.default_rng(seed)
        blocks = rng.integers(0, 256, (height // 32 + 1, width // 32 + 1, 3))
        rgbs = [np.full((height, width, 3), 128, np.uint8),
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
                np.kron(blocks, np.ones((32, 32, 1)))[:height, :width].astype(np.uint8)]
    mean = np.array(mean, np.float32).reshape(1, 3, 1, 1)
    std = np.array(std, np.float32).reshape(1, 3, 1, 1)
    out = []
    for rgb in rgbs:
        x = (rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0 - mean) / std
        out.append(np.repeat(x, batch, axis=0).astype(np.float32))
    return out


class Fp16Error(ValueError):
    """The model can't be made FP16-safe (a tensor outside the LayerNorms gets too close to FP16's maximum)."""


@dataclass
class Fp16Plan:
    scales: dict[str, int]       # LayerNorm -> power-of-two divisor of its input
    keep_fp32: list[str]         # LayerNorms that no scale makes safe without losing precision
    smallest_median: float       # smallest median |x - mean| at a LayerNorm input after scaling


def make_fp16_safe(onnx_in: Path, onnx_out: Path, images: list[Path] | None = None, mean=None,
                   std=None) -> Fp16Plan:
    """Calibrate onnx_in (plain model, input = model size, normalised with mean/std) and write onnx_out with the
    at-risk LayerNorm inputs scaled. Raises Fp16Error when a tensor outside the LayerNorms would overflow FP16."""
    model = onnx.load(str(onnx_in))
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    batch = dims[0] or 1
    cal = calibrate(onnx_in, calibration_inputs(dims[2], dims[3], batch, images, mean=mean, std=std))
    risky = overflow_risks(cal)
    if risky:
        lines = "; ".join(f"{node or t}: {m:.0f}" for t, node, m in risky[:5])
        raise Fp16Error(f"FP16 overflow risk outside the LayerNorms (largest |value| on the calibration frames, "
                        f"limit {OVERFLOW_LIMIT:.0f}, FP16 max {FP16_MAX:.0f}): {lines}")
    scales, keep = layernorm_scales(cal.max_deviation, medians=cal.median_deviation)
    scale_layernorm_inputs(model, scales)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_out))
    medians = [m / scales.get(n, 1) for n, m in cal.median_deviation.items() if n not in keep]
    return Fp16Plan(scales, keep, min(medians, default=math.inf))


def layernorm_scales(deviations: dict[str, float], target: float = TARGET_MAX_DEVIATION,
                     medians: dict[str, float] | None = None,
                     min_median: float = MIN_MEDIAN_DEVIATION) -> tuple[dict[str, int], list[str]]:
    """Power-of-two divisor per LayerNorm that brings its largest deviation to at most `target` (only those > 1),
    and the LayerNorms for which that divisor would push the median deviation below `min_median` (FP16 loses
    precision for small values: subnormals start at 6e-5). Those get no divisor and belong in FP32."""
    scales, keep = {}, []
    for name, d in deviations.items():
        if d <= target:
            continue
        s = 2 ** math.ceil(math.log2(d / target))
        if medians is not None and medians[name] / s < min_median:
            keep.append(name)
        else:
            scales[name] = s
    return scales, keep


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
