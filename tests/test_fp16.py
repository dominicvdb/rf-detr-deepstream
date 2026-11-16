"""FP16 LayerNorm fix: same results in FP32, no overflow in (simulated) FP16, wired into the bundle."""

import configparser

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from conftest import OPSET, make_detr_onnx
from rfdetr_deepstream.bundle import build_bundle
from rfdetr_deepstream.fp16 import (Fp16Error, calibrate, layernorm_deviations, layernorm_scales, make_fp16_safe,
                                    overflow_risks, scale_layernorm_inputs, simulate_fp16)
from rfdetr_deepstream.fp16_typed import convert_to_fp16


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
    scales, keep = layernorm_scales(dev)
    assert scales == {"ln": 64} and keep == []                                    # ~2400 / 64 -> power of two >= 37.5
    fixed = scale_layernorm_inputs(onnx.load(str(path)), scales)
    np.testing.assert_allclose(_run(fixed, x), _run(model, x), atol=1e-5)
    np.testing.assert_allclose(_run(simulate_fp16(fixed), x), _run(model, x), atol=2e-2)  # FP16 rounding only


def test_scales_are_powers_of_two_above_target():
    s, keep = layernorm_scales({"a": 10.0, "b": 64.0, "c": 64.1, "d": 1000.0})
    assert s == {"c": 2, "d": 16} and keep == []


def test_layernorm_with_small_median_is_kept_in_fp32():
    # d needs 1/16; its median 0.01 would become 6e-4, below the 1e-3 minimum: no scale, keep FP32.
    s, keep = layernorm_scales({"c": 64.1, "d": 1000.0}, medians={"c": 0.5, "d": 0.01})
    assert s == {"c": 2} and keep == ["d"]


def test_overflow_outside_layernorm_is_refused(tmp_path):
    with pytest.raises(Fp16Error, match="overflow risk"):
        make_fp16_safe(make_layernorm_onnx(tmp_path / "ln.onnx", gain=60000.0), tmp_path / "safe.onnx")


def test_calibration_records_tensor_ranges(tmp_path, x):
    cal = calibrate(make_layernorm_onnx(tmp_path / "ln.onnx"), [x])
    big, node = cal.tensor_max["big"]
    assert big == pytest.approx(1000 * np.abs(x).max(), rel=1e-5) and node == ""
    assert 0 < cal.median_deviation["ln"] < cal.max_deviation["ln"]
    assert overflow_risks(cal) == []


def test_layernorm_kept_in_fp32_in_simulation(tmp_path, x):
    model = onnx.load(str(make_layernorm_onnx(tmp_path / "ln.onnx")))
    np.testing.assert_allclose(_run(simulate_fp16(model, keep_fp32={"ln"}), x), _run(model, x), atol=2e-2)


def test_make_fp16_safe_with_synthetic_calibration(tmp_path, x):
    path = make_layernorm_onnx(tmp_path / "ln.onnx")
    plan = make_fp16_safe(path, tmp_path / "safe.onnx")
    assert set(plan.scales) == {"ln"} and plan.scales["ln"] >= 16 and plan.keep_fp32 == []
    assert plan.smallest_median > 1e-3
    safe = onnx.load(str(tmp_path / "safe.onnx"))
    np.testing.assert_allclose(_run(simulate_fp16(safe), x), _run(onnx.load(str(path)), x), atol=2e-2)


def test_unknown_layernorm_name_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not in the model"):
        scale_layernorm_inputs(onnx.load(str(make_layernorm_onnx(tmp_path / "ln.onnx"))), {"nope": 2})


def _fp16_bundle(tmp_path, fix, **kw):
    out = build_bundle(model=str(make_detr_onnx(tmp_path / "m.onnx", **kw)), output_dir=tmp_path / "b",
                       frame_size=(64, 48), coco=True, precision="fp16", fp16_fix=fix, log=lambda *_: None,
                       batch_size=4 if kw.get("batch") else 1)
    cp = configparser.ConfigParser()
    cp.read(out / "config_infer_rfdetr.txt")
    return out, cp["property"]


def test_fp16_bundle_is_strongly_typed(tmp_path):
    out, p = _fp16_bundle(tmp_path, fix=True)
    assert p["network-mode"] == "0" and p["strongly-typed"] == "1"
    assert p["model-engine-file"] == "m_64x48.onnx_b1_gpu0_fp32.engine"
    model = onnx.load(str(out / "m_64x48.onnx"))
    io = [v.type.tensor_type.elem_type for v in [*model.graph.input, *model.graph.output]]
    assert io == [TensorProto.FLOAT] * 3                                   # parser input/outputs unchanged
    assert {t.data_type for t in model.graph.initializer if not t.name.startswith("pre_")} >= {TensorProto.FLOAT16}
    init = {t.name: t.data_type for t in model.graph.initializer}
    assert init["pre_mean"] == init["pre_std"] == TensorProto.FLOAT       # preprocessing stays FP32
    readme = (out / "README.txt").read_text()
    assert "--stronglyTyped" in readme and "--saveEngine=m_64x48.onnx_b1_gpu0_fp32.engine" in readme

    fp32_ref = build_bundle(model=str(tmp_path / "m.onnx"), output_dir=tmp_path / "ref", frame_size=(64, 48),
                            coco=True, log=lambda *_: None)
    x = np.random.default_rng(0).uniform(0, 255, (1, 3, 48, 64)).astype(np.float32)
    a = ort.InferenceSession(str(fp32_ref / "m_64x48.onnx"), providers=["CPUExecutionProvider"]).run(None, {"image": x})
    b = ort.InferenceSession(str(out / "m_64x48.onnx"), providers=["CPUExecutionProvider"]).run(None, {"image": x})
    for u, v in zip(a, b):
        np.testing.assert_allclose(v, u, atol=2e-2, rtol=1e-2)            # FP16 rounding only


def test_fp16_readme_has_shapes_for_dynamic_batch(tmp_path):
    out, p = _fp16_bundle(tmp_path, fix=True, batch="batch")
    readme = (out / "README.txt").read_text()
    assert "--minShapes=image:1x3x48x64 --optShapes=image:4x3x48x64 --maxShapes=image:4x3x48x64" in readme
    assert p["model-engine-file"] == "m_64x48.onnx_b4_gpu0_fp32.engine"


def test_plain_fp16_bundle(tmp_path):
    out, p = _fp16_bundle(tmp_path, fix=False)
    assert p["network-mode"] == "2" and "strongly-typed" not in p
    assert p["model-engine-file"].endswith("_b1_gpu0_fp16.engine")
    assert "--stronglyTyped" not in (out / "README.txt").read_text()


def test_convert_to_fp16_keeps_named_nodes_in_fp32(tmp_path, x):
    path = make_layernorm_onnx(tmp_path / "ln.onnx", gain=1.0)
    kept = convert_to_fp16(path, tmp_path / "half.onnx", keep_fp32_ops=(), keep_fp32_names=(r"^ln$",))
    model = onnx.load(str(tmp_path / "half.onnx"))
    assert kept == ["ln"]
    init = {t.name: t.data_type for t in model.graph.initializer}
    assert init["ln_scale"] == TensorProto.FLOAT and init["gain"] == TensorProto.FLOAT16
    np.testing.assert_allclose(_run(model, x), _run(onnx.load(str(path)), x), atol=1e-2)
