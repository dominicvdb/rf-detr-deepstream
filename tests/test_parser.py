"""The C++ nvinfer parser, compiled against a stand-in header, must return exactly what reference.decode() returns."""

import subprocess

import numpy as np
import pytest

from rfdetr_deepstream.reference import decode


def _parse(harness, tmp_path, dets, labels, net_w, net_h, thresholds):
    blob = tmp_path / "in.bin"
    with open(blob, "wb") as f:
        for d, lab in zip(dets, labels):
            f.write(d.astype(np.float32).tobytes())
            f.write(lab.astype(np.float32).tobytes())
    n, q, c = labels.shape
    run = subprocess.run([str(harness), str(blob), str(n), str(q), str(c), str(net_w), str(net_h),
                          *map(str, thresholds)], capture_output=True, text=True)
    per_image = [[] for _ in range(n)]
    for line in run.stdout.splitlines():
        img, cls, left, top, w, h, score = line.split()
        per_image[int(img)].append((int(cls), float(left), float(top), float(left) + float(w),
                                    float(top) + float(h), float(score)))
    return run.returncode, per_image, run.stderr


@pytest.mark.parametrize("background", [None, -1])
def test_parser_matches_reference(parser_harness, tmp_path, background):
    rng = np.random.default_rng(3)
    n, q, c, w, h = 8, 300, 4, 1280, 720
    dets = rng.uniform(-0.1, 1.1, (n, q, 4)).astype(np.float32)   # some boxes outside the image: clipping
    labels = rng.normal(-4, 3, (n, q, c)).astype(np.float32)
    labels[0, :5] = 2.0                                            # ties: lower flat index first
    thresholds = [0.5] * c
    if background is not None:
        thresholds[background] = 1.0                               # how the config excludes the background slot
    code, parsed, err = _parse(parser_harness, tmp_path, dets, labels, w, h, thresholds)
    assert code == 0, err
    for i in range(n):
        ref = decode(dets[i], labels[i], w, h, threshold=0.5, background_class_id=background)
        assert [p[0] for p in parsed[i]] == ref.class_id.tolist()
        if len(ref.class_id):
            np.testing.assert_allclose([p[1:5] for p in parsed[i]], ref.xyxy, atol=2e-3)
            np.testing.assert_allclose([p[5] for p in parsed[i]], ref.confidence, rtol=1e-6)


def test_parser_rejects_class_count_mismatch(parser_harness, tmp_path):
    dets, labels = np.full((1, 2, 4), 0.5), np.zeros((1, 2, 3))
    # The model has 3 class slots, but only 2 per-class thresholds are configured.
    code, _, err = _parse(parser_harness, tmp_path, dets, labels, 100, 100, [0.5, 0.5])
    assert code == 4 and "num-detected-classes" in err
