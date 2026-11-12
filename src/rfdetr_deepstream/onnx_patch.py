"""Build RF-DETR's preprocessing into the ONNX graph, so DeepStream doesn't have to approximate it.

nvinfer's own preprocessing differs from RFDETR.predict() in three ways (measured, see docs):
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
    dims = [d.dim_value for d in old_input.type.tensor_type.shape.dim]  # [1, 3, 384, 384]
    batch, channels, model_h, model_w = dims
    if 0 in dims:
        raise ValueError(f"expected a static input shape, got {dims}")

    c = lambda name, arr: numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)  # noqa: E731
    graph.initializer.extend([
        c("pre_inv255", 1.0 / 255.0),
        c("pre_mean", np.array(mean).reshape(1, channels, 1, 1)),
        c("pre_std", np.array(std).reshape(1, channels, 1, 1)),
        numpy_helper.from_array(np.array([batch, channels, model_h, model_w], dtype=np.int64), "pre_sizes"),
    ])

    nodes = [
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
