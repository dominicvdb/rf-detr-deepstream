"""Bundle generation from an ONNX: files, class layout, nvinfer config, errors."""

import configparser

import numpy as np
import onnxruntime as ort
import pytest

from conftest import make_detr_onnx
from rfdetr_deepstream.bundle import BundleError, build_bundle, check_onnx, read_onnx_interface


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


def test_check_onnx_tasks(tmp_path):
    assert check_onnx(read_onnx_interface(make_detr_onnx(tmp_path / "d.onnx"))) == ("detect", 3)
    kp = make_detr_onnx(tmp_path / "k.onnx", classes=2, keypoints=(17, 7))
    assert check_onnx(read_onnx_interface(kp)) == ("keypoints", 2)
    with pytest.raises(BundleError, match="Segmentation"):
        check_onnx(read_onnx_interface(make_detr_onnx(tmp_path / "s.onnx", masks=True)))
