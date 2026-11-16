"""Smaller units: CLI parsing, matching/parity statistics, the DeepStream-exact YUV->RGB conversion, dump readers."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from rfdetr_deepstream.cli import _frame_size, _int_list, build_parser
from rfdetr_deepstream.compare import ParityStats, iou, match
from rfdetr_deepstream.dumps import read_kitti, read_labels
from rfdetr_deepstream.video import _coefficients, deepstream_rgb, frame_matrix


# ---- CLI -------------------------------------------------------------------------------------------
def test_frame_size_parsing():
    assert _frame_size("1920x1080") == (1920, 1080)
    assert _frame_size("1280X720") == (1280, 720)
    for bad in ("1920", "axb", "0x10", "-5x10"):
        with pytest.raises(argparse.ArgumentTypeError):
            _frame_size(bad)


def test_cli_defaults():
    args = build_parser().parse_args(["build", "nano", "-o", "out"])
    assert (args.threshold, args.precision, args.batch_size, args.frame_size) == (0.5, "fp32", 1, None)
    assert _int_list("0,17") == [0, 17]


# ---- compare ---------------------------------------------------------------------------------------
def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_match_is_greedy_one_to_one_and_per_class():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], float)
    pairs = match(a, [0, 0], b, [1, 0, 0])
    # each box used once; b[0] is class 1 and never matches; the exact match goes first
    assert sorted(j for _, j, _ in pairs) == [1, 2] and len({i for i, _, _ in pairs}) == 2
    assert max(v for _, _, v in pairs) == 1.0


def test_parity_stats():
    s = ParityStats()
    s.add(np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), [0, 0], [0.9, 0.52],
          np.array([[0, 0, 10, 10]]), [0], [0.8])
    r = s.summary(threshold=0.5)
    assert (r["matched"], r["only_a"], r["only_b"]) == (1, 1, 0)
    assert r["dscore_max"] == pytest.approx(0.1)
    assert r["unmatched_near_threshold"] == 1.0  # 0.52 is within 0.05 of 0.5


# ---- video: nvinfer's NV12 -> RGB ---------------------------------------------------------------------
def test_limited_range_black_white_grey():
    y = np.array([[16, 235], [126, 126]], np.uint8)
    u = v = np.full((1, 1), 128, np.uint8)
    rgb = deepstream_rgb(y, u, v)
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert len(set(rgb[1, 0].tolist())) == 1  # no chroma: grey


def test_deepstream7_rounds_coefficients():
    exact, ds7 = _coefficients("bt601", "9.2"), _coefficients("bt601", "7.0")
    assert ds7 == (1.1644, 1.596, 0.3918, 0.813, 2.0172)
    assert exact != ds7 and np.allclose(exact, ds7, atol=1e-4)


def test_chroma_repeated_over_2x2():
    y = np.full((4, 4), 128, np.uint8)
    u = np.array([[100, 160], [128, 128]], np.uint8)
    rgb = deepstream_rgb(y, u, np.full((2, 2), 128, np.uint8))
    assert (rgb[:2, :2] == rgb[0, 0]).all() and not (rgb[0, 0] == rgb[0, 2]).all()


def test_frame_matrix():
    assert frame_matrix(SimpleNamespace(color_range=1, colorspace=1)) == "bt709"
    assert frame_matrix(SimpleNamespace(color_range=1, colorspace=2)) == "bt601"
    with pytest.raises(ValueError, match="full-range"):
        frame_matrix(SimpleNamespace(color_range=2, colorspace=1))


# ---- dumps --------------------------------------------------------------------------------------------
def test_read_kitti_with_spaces_in_label(tmp_path):
    (tmp_path / "labels.txt").write_text("unused_0\ntraffic light\n")
    ids = read_labels(tmp_path / "labels.txt")
    line = "traffic light 0.0 0 0.0 10.0 20.0 30.0 40.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.87\n"
    (tmp_path / "f.txt").write_text(line)
    xyxy, cls, conf = read_kitti(tmp_path / "f.txt", ids)
    assert cls.tolist() == [1] and xyxy.tolist() == [[10, 20, 30, 40]] and conf.tolist() == pytest.approx([0.87])
