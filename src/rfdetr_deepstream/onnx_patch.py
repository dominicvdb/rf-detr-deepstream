"""Build RF-DETR's preprocessing into the ONNX graph, so DeepStream doesn't have to approximate it.

nvinfer's own preprocessing differs from RFDETR.predict() in three ways (measured with scripts/debug_*.py):
  1. its scaler uses a different pixel-center convention than torchvision (boxes shift ~half a pixel),
  2. it has one scale for all channels, while ImageNet normalisation needs one std per channel,
  3. its bilinear filter is not documented to match torchvision's.

The patched model instead takes the full video frame, RGB, values 0-255 (what nvinfer delivers with
net-scale-factor=1 and no offsets, at network size == frame size, so nvinfer does no scaling), and does:
  x / 255  ->  Resize(bilinear, half_pixel, no antialias) to the model resolution  ->  (x - mean) / std
That is exactly torchvision's to_tensor -> resize(antialias=False) -> normalize, i.e. RFDETR.predict().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD

NEW_INPUT = "image"


def fold_constants(onnx_in: Path, onnx_out: Path) -> None:
    """Pre-compute every subgraph that doesn't depend on the input (ONNX Runtime basic optimisations, standard
    ONNX ops only). Needed for the keypoint model on TensorRT 8.6, which fails on the constant
    keypoint_query_initializer subgraph ("Could not find any implementation for node ForeignNode[...]")."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(onnx_out)
    ort.InferenceSession(str(onnx_in), options, providers=["CPUExecutionProvider"])
    _undo_ort_rewrites(onnx_out)


def _undo_ort_rewrites(path: Path) -> None:
    """ONNX Runtime writes default attribute values explicitly and lists every opset domain it knows. TensorRT 10.3
    (DeepStream 7.1) rejects ScatterND with an explicit reduction="none", so drop attributes that equal their
    schema default and opset imports no node uses. Same semantics as before."""
    model = onnx.load(str(path))
    versions = {o.domain: o.version for o in model.opset_import}
    for node in model.graph.node:
        try:
            schema = onnx.defs.get_schema(node.op_type, versions.get(node.domain, 1), node.domain)
        except onnx.defs.SchemaError:
            continue
        keep = []
        for attr in node.attribute:
            spec = schema.attributes.get(attr.name)
            default = spec.default_value if spec is not None else None
            if (default is not None and default.type != onnx.AttributeProto.UNDEFINED
                    and helper.get_attribute_value(attr) == helper.get_attribute_value(default)):
                continue
            keep.append(attr)
        del node.attribute[:]
        node.attribute.extend(keep)
    used = {node.domain for node in model.graph.node} | {""}
    kept = [o for o in model.opset_import if o.domain in used]
    del model.opset_import[:]
    model.opset_import.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def add_preprocessing(
    onnx_in: Path,
    onnx_out: Path,
    frame_height: int,
    frame_width: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> None:
    model = onnx.load(str(onnx_in))
    graph = model.graph

    old_input = graph.input[0]
    old_name = old_input.name  # "input" for RF-DETR exports
    # [1, 3, 384, 384], or ["batch", 3, 384, 384] for a dynamic-batch export
    dims = [d.dim_value or d.dim_param for d in old_input.type.tensor_type.shape.dim]
    batch, channels, model_h, model_w = dims
    if not all(isinstance(d, int) and d > 0 for d in dims[1:]) or not batch:
        raise ValueError(f"expected an input shape [N or batch, C, H, W] with static C, H, W, got {dims}")

    c = lambda name, arr: numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)  # noqa: E731
    graph.initializer.extend([
        c("pre_inv255", 1.0 / 255.0),
        c("pre_mean", np.array(mean).reshape(1, channels, 1, 1)),
        c("pre_std", np.array(std).reshape(1, channels, 1, 1)),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "pre_zero"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "pre_two"),
        numpy_helper.from_array(np.array([model_h, model_w], dtype=np.int64), "pre_hw"),
    ])

    nodes = [
        # Resize target [N, C, model_h, model_w], with N taken from the input so any batch size works.
        helper.make_node("Shape", [NEW_INPUT], ["pre_in_shape"], name="pre_shape"),
        helper.make_node("Slice", ["pre_in_shape", "pre_zero", "pre_two"], ["pre_nc"], name="pre_slice_nc"),
        helper.make_node("Concat", ["pre_nc", "pre_hw"], ["pre_sizes"], name="pre_sizes", axis=0),
        helper.make_node("Mul", [NEW_INPUT, "pre_inv255"], ["pre_scaled"], name="pre_scale"),
        # Inputs: X, roi (unused), scales (unused, empty), sizes. half_pixel + linear = torchvision bilinear,
        # align_corners=False. Opset 17 Resize has no antialiasing, same as antialias=False.
        helper.make_node(
            "Resize", ["pre_scaled", "", "", "pre_sizes"], ["pre_resized"], name="pre_resize",
            mode="linear", coordinate_transformation_mode="half_pixel",
        ),
        helper.make_node("Sub", ["pre_resized", "pre_mean"], ["pre_centered"], name="pre_sub_mean"),
        # Output under the OLD input name, so every existing node keeps reading from it unchanged.
        helper.make_node("Div", ["pre_centered", "pre_std"], [old_name], name="pre_div_std"),
    ]
    # Nodes must be topologically sorted: put the new ones in front.
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + existing)

    graph.input.remove(old_input)
    graph.input.insert(
        0, helper.make_tensor_value_info(NEW_INPUT, TensorProto.FLOAT, [batch, channels, frame_height, frame_width])
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_out))
