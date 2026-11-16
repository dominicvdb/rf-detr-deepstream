"""Preprocessing built into the ONNX must equal reference.preprocess() (= RFDETR.predict())."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import OPSET, make_identity_onnx
from rfdetr_deepstream.onnx_patch import NEW_INPUT, add_preprocessing, fold_constants


def _run(path, feed):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, feed)


@pytest.mark.parametrize("frame", [(720, 1280), (1080, 1920), (100, 60)])
def test_preprocessing_matches_reference(tmp_path, frame):
    pytest.importorskip("torch")
    from PIL import Image

    from rfdetr_deepstream.reference import preprocess

    model_hw = (32, 48)
    add_preprocessing(make_identity_onnx(tmp_path / "id.onnx", model_hw), tmp_path / "p.onnx",
                      frame_height=frame[0], frame_width=frame[1])
    rgb = np.random.default_rng(0).integers(0, 256, (*frame, 3), dtype=np.uint8)
    (ours,) = _run(tmp_path / "p.onnx", {NEW_INPUT: rgb.astype(np.float32).transpose(2, 0, 1)[None].copy()})
    ref = preprocess(Image.fromarray(rgb), *model_hw)
    # ONNX Runtime's Resize and PyTorch's interpolate round differently: up to ~2.5e-4 here (0.01 grey level)
    # on random noise, the worst case. Real frames: 100% detection match (scripts/check_reference.py et al.).
    np.testing.assert_allclose(ours, ref, atol=5e-4)
    assert np.abs(ours - ref).mean() < 1e-5


def test_preprocessing_input_shape_and_dynamic_batch(tmp_path):
    add_preprocessing(make_identity_onnx(tmp_path / "id.onnx", (16, 16), batch="batch"), tmp_path / "p.onnx",
                      frame_height=40, frame_width=60)
    inp = onnx.load(str(tmp_path / "p.onnx")).graph.input[0]
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    assert inp.name == NEW_INPUT and dims == ["batch", 3, 40, 60]
    x = np.random.default_rng(0).uniform(0, 255, (3, 3, 40, 60)).astype(np.float32)
    (batched,) = _run(tmp_path / "p.onnx", {NEW_INPUT: x})
    (single,) = _run(tmp_path / "p.onnx", {NEW_INPUT: x[1:2]})
    np.testing.assert_allclose(batched[1:2], single, atol=1e-6)


def test_fold_constants_keeps_outputs(tmp_path):
    # out = input * (2 + 3): the constant branch should be folded away.
    init = [numpy_helper.from_array(np.float32(2), "a"), numpy_helper.from_array(np.float32(3), "b")]
    nodes = [helper.make_node("Add", ["a", "b"], ["c"]), helper.make_node("Mul", ["input", "c"], ["out"])]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "in.onnx"))
    fold_constants(tmp_path / "in.onnx", tmp_path / "out.onnx")
    folded = onnx.load(str(tmp_path / "out.onnx"))
    assert [n.op_type for n in folded.graph.node] == ["Mul"]
    x = np.arange(4, dtype=np.float32)[None]
    np.testing.assert_allclose(_run(tmp_path / "out.onnx", {"input": x})[0], x * 5)
