"""Bundle generation from an ONNX: files, class layout, nvinfer config, errors."""

import configparser
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper

from conftest import make_detr_onnx
from rfdetr_deepstream.bundle import (RFDETR_VERSIONS, BundleError, ClassLayout, build_bundle, check_onnx,
                                      check_rfdetr_version, normalization_from_rfdetr, read_onnx_interface)
from rfdetr_deepstream.reference import IMAGENET_MEAN, IMAGENET_STD


def _config(path):
    cp = configparser.ConfigParser(comment_prefixes=("#",), inline_comment_prefixes=None)
    cp.read(path)
    return cp


def _build(tmp_path, onnx_path, **kw):
    return build_bundle(model=str(onnx_path), output_dir=tmp_path / "bundle", log=lambda *_: None, **kw)


def test_fine_tuned_detection_bundle(tmp_path):
    onnx_path = make_detr_onnx(tmp_path / "m.onnx", classes=3)
    names = tmp_path / "names.txt"
    names.write_text("cat\ndog\n")  # 2 names for 3 slots: background slot last
    out = _build(tmp_path, onnx_path, frame_size=(64, 48), class_names_file=names, threshold=0.4)

    assert sorted(p.name for p in out.iterdir()) == ["README.txt", "config_infer_rfdetr.txt", "labels.txt",
                                                     "m_64x48.onnx", "parser"]
    assert sorted(p.name for p in (out / "parser").iterdir()) == ["Makefile", "nvdsparsebbox_rfdetr.cpp"]
    assert (out / "labels.txt").read_text().splitlines() == ["cat", "dog", "unused_2"]

    cfg = _config(out / "config_infer_rfdetr.txt")
    p = cfg["property"]
    assert p["onnx-file"] == "m_64x48.onnx"
    assert p["model-engine-file"] == "m_64x48.onnx_b1_gpu0_fp32.engine"
    assert p["num-detected-classes"] == "3"
    assert p["network-type"] == "0" and p["network-mode"] == "0"
    assert p["output-blob-names"] == "dets;labels"
    assert p["parse-bbox-func-name"] == "NvDsInferParseRfdetr"
    assert p["custom-lib-path"] == "parser/libnvds_rfdetr.so"
    assert p["net-scale-factor"] == "1.0" and "offsets" not in p   # preprocessing is in the model
    assert cfg["class-attrs-all"]["pre-cluster-threshold"] == "0.4"
    assert cfg["class-attrs-2"]["pre-cluster-threshold"] == "1.0"  # background never passes

    s = ort.InferenceSession(str(out / "m_64x48.onnx"), providers=["CPUExecutionProvider"])
    assert s.get_inputs()[0].name == "image" and s.get_inputs()[0].shape == [1, 3, 48, 64]
    assert [o.name for o in s.get_outputs()] == ["dets", "labels"]


def test_coco_bundle_has_sparse_labels_and_no_background(tmp_path):
    out = _build(tmp_path, make_detr_onnx(tmp_path / "m.onnx", classes=91), frame_size=(64, 48), coco=True)
    labels = (out / "labels.txt").read_text().splitlines()
    assert len(labels) == 91 and labels[1] == "person" and labels[12] == "unused_12" and labels[90] == "toothbrush"
    cfg = _config(out / "config_infer_rfdetr.txt")
    assert [s for s in cfg.sections() if s.startswith("class-attrs-") and s != "class-attrs-all"] == []


def test_without_frame_size_nvinfer_does_the_preprocessing(tmp_path):
    out = _build(tmp_path, make_detr_onnx(tmp_path / "m.onnx"), coco=True)
    p = _config(out / "config_infer_rfdetr.txt")["property"]
    assert p["onnx-file"] == "m.onnx"
    assert len(p["offsets"].split(";")) == 3 and float(p["net-scale-factor"]) == pytest.approx(1 / (255 * 0.226),
                                                                                              rel=1e-3)


def test_dynamic_batch_bundle(tmp_path):
    onnx_path = make_detr_onnx(tmp_path / "m.onnx", batch="batch")
    out = _build(tmp_path, onnx_path, frame_size=(64, 48), coco=True, batch_size=4)
    assert _config(out / "config_infer_rfdetr.txt")["property"]["batch-size"] == "4"
    s = ort.InferenceSession(str(out / "m_64x48.onnx"), providers=["CPUExecutionProvider"])
    assert s.get_inputs()[0].shape[0] == "batch"
    x = np.random.default_rng(0).uniform(0, 255, (3, 3, 48, 64)).astype(np.float32)
    assert s.run(None, {"image": x})[1].shape[0] == 3


def test_keypoint_bundle(tmp_path):
    onnx_path = make_detr_onnx(tmp_path / "kp.onnx", classes=2, keypoints=(17, 7))
    out = _build(tmp_path, onnx_path, frame_size=(64, 48))
    assert sorted(p.name for p in (out / "pose").iterdir()) == ["Makefile", "nvds_rfdetr_keypoints_meta.h",
                                                                "nvdsrfdetr_pose.cpp"]
    assert not (out / "parser").exists()
    p = _config(out / "config_infer_rfdetr.txt")["property"]
    assert p["network-type"] == "100" and p["output-tensor-meta"] == "1"
    assert p["output-blob-names"] == "dets;labels;keypoints"
    readme = (out / "README.txt").read_text()
    assert "keypoints-per-class:0,17" in readme and "labels:background;person" in readme


@pytest.mark.parametrize("kwargs, message", [
    ({}, "--class-names"),                                     # an ONNX has no class names
    ({"coco": True, "batch_size": 2}, "fixed batch"),          # batch 2 on a fixed-batch-1 ONNX
    ({"coco": True, "batch_size": 0}, "at least 1"),
])
def test_bundle_errors(tmp_path, detr_onnx, kwargs, message):
    with pytest.raises(BundleError, match=message):
        _build(tmp_path, detr_onnx, frame_size=(64, 48), **kwargs)


def test_class_names_count_must_fit(tmp_path, detr_onnx):
    names = tmp_path / "names.txt"
    names.write_text("a\n")  # 1 name for 3 slots
    with pytest.raises(BundleError, match="3 logits slots"):
        _build(tmp_path, detr_onnx, class_names_file=names)


def test_only_rgb_models(tmp_path):
    with pytest.raises(BundleError, match="3-channel"):
        _build(tmp_path, make_detr_onnx(tmp_path / "gray.onnx", channels=1), coco=True)


def test_normalization_from_rfdetr():
    assert normalization_from_rfdetr(SimpleNamespace()) == (IMAGENET_MEAN, IMAGENET_STD)
    model = SimpleNamespace(means=[0.5, 0.5, 0.5], stds=[0.25, 0.25, 0.25])
    assert normalization_from_rfdetr(model) == ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    with pytest.raises(BundleError, match="positive"):
        normalization_from_rfdetr(SimpleNamespace(means=[0.5] * 3, stds=[0.25, 0.0, 0.25]))


@pytest.fixture
def custom_norm_model(tmp_path, monkeypatch):
    """A 'checkpoint' whose normaliser isn't ImageNet; export gives the synthetic ONNX."""
    import rfdetr_deepstream.bundle as bundle

    model = SimpleNamespace(means=[0.5, 0.4, 0.3], stds=[0.2, 0.25, 0.3])
    monkeypatch.setattr(bundle, "load_rfdetr", lambda name: model)
    monkeypatch.setattr(bundle, "export_onnx", lambda m, out, *a: make_detr_onnx(out))
    monkeypatch.setattr(bundle, "layout_from_rfdetr", lambda m, n, t: ClassLayout(["a", "b", ""], 2))
    return model


def test_model_normalization_in_onnx(tmp_path, custom_norm_model):
    out = build_bundle("nano", tmp_path / "bundle", frame_size=(64, 48), log=lambda *_: None)
    init = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(out / "rfdetr-nano_64x48.onnx")).graph.initializer}
    np.testing.assert_allclose(init["pre_mean"].ravel(), custom_norm_model.means, rtol=1e-6)
    np.testing.assert_allclose(init["pre_std"].ravel(), custom_norm_model.stds, rtol=1e-6)


def test_model_normalization_in_nvinfer_config(tmp_path, custom_norm_model):
    out = build_bundle("nano", tmp_path / "bundle", log=lambda *_: None)
    p = _config(out / "config_infer_rfdetr.txt")["property"]
    np.testing.assert_allclose([float(v) for v in p["offsets"].split(";")], [127.5, 102.0, 76.5], rtol=1e-4)
    assert float(p["net-scale-factor"]) == pytest.approx(1 / (255 * 0.25), rel=1e-4)


def test_check_onnx_tasks(tmp_path):
    assert check_onnx(read_onnx_interface(make_detr_onnx(tmp_path / "d.onnx"))) == ("detect", 3)
    kp = make_detr_onnx(tmp_path / "k.onnx", classes=2, keypoints=(17, 7))
    assert check_onnx(read_onnx_interface(kp)) == ("keypoints", 2)
    assert check_onnx(read_onnx_interface(make_detr_onnx(tmp_path / "s.onnx", masks=True))) == ("segment", 3)


@pytest.mark.parametrize("version", ["1.12.0", "1.12.1", "1.12.10rc1"])
def test_rfdetr_version_supported(version):
    check_rfdetr_version(version)


@pytest.mark.parametrize("version", ["1.11.2", "1.13.0", "2.0.0"])
def test_rfdetr_version_unsupported(version):
    with pytest.raises(BundleError, match="needs rfdetr >=1.12,<1.13"):
        check_rfdetr_version(version)


def test_rfdetr_pin_matches_pyproject():
    """The version check and the pin in pyproject.toml must agree."""
    import re
    from pathlib import Path

    (lo, hi) = RFDETR_VERSIONS
    pins = re.findall(r'"rfdetr\[onnx\]([^"]*)"', (Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert pins and set(pins) == {f">={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]}"}
