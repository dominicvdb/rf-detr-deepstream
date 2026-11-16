"""Shared fixtures: tiny synthetic ONNX models with RF-DETR's interface, so tests need no downloads or GPU."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
OPSET = 17


def _const(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def make_detr_onnx(path: Path, size: tuple[int, int] = (32, 32), queries: int = 10, classes: int = 3,
                   batch: int | str = 1, keypoints: tuple[int, int] | None = None, masks: bool = False,
                   seed: int = 0, channels: int = 3) -> Path:
    """An ONNX with RF-DETR's interface: input [N, 3, H, W] -> dets [N, Q, 4], labels [N, Q, C]
    (+ keypoints [N, Q, C*K, D] or masks [N, Q, H/4, W/4]). The outputs are fixed tensors plus a term that depends on
    the input's mean, so a changed input is visible in the outputs."""
    rng = np.random.default_rng(seed)
    h, w = size
    out_shapes = {"dets": [batch, queries, 4], "labels": [batch, queries, classes]}
    consts = {"dets": rng.uniform(0.2, 0.8, (1, queries, 4)), "labels": rng.normal(0, 3, (1, queries, classes))}
    if keypoints:
        k, d = keypoints
        out_shapes["keypoints"] = [batch, queries, classes * k, d]
        consts["keypoints"] = rng.normal(0, 1, (1, queries, classes * k, d))
    if masks:
        out_shapes["masks"] = [batch, queries, h // 4, w // 4]
        consts["masks"] = rng.normal(0, 1, (1, queries, h // 4, w // 4))

    init = [_const("gain", 0.01)]
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=1),  # [N, 1, 1, 1]
        helper.make_node("Mul", ["mean", "gain"], ["signal4"]),
    ]
    outputs = []
    for name, shape in out_shapes.items():
        init.append(_const(f"{name}_base", consts[name]))
        signal = "signal4"
        if len(shape) == 3:
            init.append(numpy_helper.from_array(np.array([1], dtype=np.int64), f"{name}_axes"))
            nodes.append(helper.make_node("Squeeze", ["signal4", f"{name}_axes"], [f"{name}_signal"]))
            signal = f"{name}_signal"
        nodes.append(helper.make_node("Add", [f"{name}_base", signal], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    graph = helper.make_graph(nodes, "fake_rfdetr",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, channels, h, w])],
                              outputs, init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def make_identity_onnx(path: Path, size: tuple[int, int] = (32, 48), batch: int | str = 1) -> Path:
    """input -> out unchanged: after add_preprocessing, 'out' is exactly the preprocessed tensor."""
    h, w = size
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["out"])], "identity",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3, h, w])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [batch, 3, h, w])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.fixture
def detr_onnx(tmp_path):
    return make_detr_onnx(tmp_path / "fake.onnx")


@pytest.fixture(scope="session")
def parser_harness(tmp_path_factory) -> Path:
    """The real nvinfer parser source, compiled with a stand-in for DeepStream's header (tests/cpp)."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("harness") / "harness"
    subprocess.run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'tests/cpp'}",
                    str(ROOT / "src/rfdetr_deepstream/parser/nvdsparsebbox_rfdetr.cpp"),
                    str(ROOT / "tests/cpp/harness.cpp"), "-o", str(out)], check=True)
    return out
