"""FP16 LayerNorm fix: same results in FP32, no overflow in (simulated) FP16, wired into the bundle."""

import configparser

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import OPSET, make_detr_onnx
from rfdetr_deepstream.bundle import build_bundle
from rfdetr_deepstream.fp16 import (layernorm_deviations, layernorm_scales, make_fp16_safe, scale_layernorm_inputs,
                                    simulate_fp16)


def make_layernorm_onnx(path, gain=1000.0):
    """image-like input -> x * gain -> LayerNorm over the last axis -> out. With gain 1000 the squared deviations
    reach ~1e6: they overflow FP16."""
    init = [numpy_helper.from_array(np.float32(gain), "gain"),
            numpy_helper.from_array(np.linspace(0.5, 1.5, 8).astype(np.float32), "ln_scale"),
            numpy_helper.from_array(np.linspace(-0.1, 0.1, 8).astype(np.float32), "ln_bias")]
    nodes = [helper.make_node("Mul", ["input", "gain"], ["big"]),
             helper.make_node("LayerNormalization", ["big", "ln_scale", "ln_bias"], ["out"], name="ln", axis=-1,
                              epsilon=1e-6)]
    graph = helper.make_graph(nodes, "ln", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 8, 8])], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _run(model, x):
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, {"input": x})[0]


@pytest.fixture
def x():
    return np.random.default_rng(0).normal(0, 1, (1, 3, 8, 8)).astype(np.float32)


def test_layernorm_overflows_in_simulated_fp16(tmp_path, x):
    model = onnx.load(str(make_layernorm_onnx(tmp_path / "ln.onnx")))
    fp32 = _run(model, x)
    fp16 = _run(simulate_fp16(model), x)
    bias = np.linspace(-0.1, 0.1, 8).astype(np.float32)
    np.testing.assert_allclose(fp16, np.broadcast_to(bias, fp16.shape), atol=1e-3)  # variance inf: output = bias
    assert np.abs(fp32 - fp16).max() > 0.5


def test_scaling_is_exact_in_fp32_and_safe_in_fp16(tmp_path, x):
    path = make_layernorm_onnx(tmp_path / "ln.onnx")
    model = onnx.load(str(path))
    dev = layernorm_deviations(path, [x])
    scales = layernorm_scales(dev)
    assert scales == {"ln": 64}                                      # ~2400 / 64 -> power of two >= 37.5
    fixed = scale_layernorm_inputs(onnx.load(str(path)), scales)
    np.testing.assert_allclose(_run(fixed, x), _run(model, x), atol=1e-5)
    np.testing.assert_allclose(_run(simulate_fp16(fixed), x), _run(model, x), atol=2e-2)  # FP16 rounding only


def test_scales_are_powers_of_two_above_target():
    s = layernorm_scales({"a": 10.0, "b": 64.0, "c": 64.1, "d": 1000.0})
    assert s == {"c": 2, "d": 16}


def test_layernorm_kept_in_fp32_in_simulation(tmp_path, x):
    model = onnx.load(str(make_layernorm_onnx(tmp_path / "ln.onnx")))
    np.testing.assert_allclose(_run(simulate_fp16(model, keep_fp32={"ln"}), x), _run(model, x), atol=2e-2)


def test_make_fp16_safe_with_synthetic_calibration(tmp_path, x):
    path = make_layernorm_onnx(tmp_path / "ln.onnx")
    scales = make_fp16_safe(path, tmp_path / "safe.onnx")
    assert set(scales) == {"ln"} and scales["ln"] >= 16
    safe = onnx.load(str(tmp_path / "safe.onnx"))
    np.testing.assert_allclose(_run(simulate_fp16(safe), x), _run(onnx.load(str(path)), x), atol=2e-2)


def test_unknown_layernorm_name_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not in the model"):
        scale_layernorm_inputs(onnx.load(str(make_layernorm_onnx(tmp_path / "ln.onnx"))), {"nope": 2})


@pytest.mark.parametrize("fix", [True, False])
def test_fp16_bundle_config(tmp_path, fix):
    out = build_bundle(model=str(make_detr_onnx(tmp_path / "m.onnx")), output_dir=tmp_path / "b",
                       frame_size=(64, 48), coco=True, precision="fp16", fp16_fix=fix, log=lambda *_: None)
    cp = configparser.ConfigParser()
    cp.read(out / "config_infer_rfdetr.txt")
    assert cp["property"]["network-mode"] == "2"
    assert cp["property"]["model-engine-file"].endswith("_b1_gpu0_fp16.engine")
