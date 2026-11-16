"""reference.py is the specification the C++ parser and pose library are checked against."""

import numpy as np
import pytest

from rfdetr_deepstream.reference import decode, decode_keypoints, keypoint_log_mean_trace


def test_decode_sigmoid_threshold_and_boxes():
    dets = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.4, 0.4]], np.float32)
    labels = np.array([[3.0, -5.0], [-5.0, 0.0]], np.float32)  # query 1 class 1: sigmoid(0) = 0.5, not > 0.5
    d = decode(dets, labels, 200, 100, threshold=0.5)
    assert d.class_id.tolist() == [0]
    assert d.confidence[0] == pytest.approx(1 / (1 + np.exp(-3.0)))
    np.testing.assert_allclose(d.xyxy[0], [80, 30, 120, 70])


def test_decode_clips_boxes_to_image():
    d = decode(np.array([[0.05, 0.95, 0.2, 0.2]], np.float32), np.array([[5.0]], np.float32), 100, 100)
    np.testing.assert_allclose(d.xyxy[0], [0, 85, 15, 100])


def test_decode_one_query_can_give_several_classes():
    # num_select defaults to the number of queries (as rfdetr); pairs, not queries, are selected.
    d = decode(np.full((1, 4), 0.5, np.float32), np.array([[4.0, 3.0, -4.0]], np.float32), 10, 10, num_select=3)
    assert d.class_id.tolist() == [0, 1] and d.query.tolist() == [0, 0]


def test_decode_background_slot_excluded():
    labels = np.array([[1.0, 9.0]], np.float32)
    dets = np.full((1, 4), 0.5, np.float32)
    assert decode(dets, labels, 10, 10, num_select=2).class_id.tolist() == [1, 0]
    assert decode(dets, labels, 10, 10, num_select=2, background_class_id=-1).class_id.tolist() == [0]


def test_decode_topk_happens_before_threshold():
    # 3 queries above threshold, but only the best num_select pairs are considered.
    labels = np.array([[5.0], [4.0], [3.0]], np.float32)
    d = decode(np.full((3, 4), 0.5, np.float32), labels, 10, 10, threshold=0.5, num_select=2)
    assert d.query.tolist() == [0, 1]


def test_decode_ties_lower_flat_index_first():
    d = decode(np.full((3, 4), 0.5, np.float32), np.ones((3, 2), np.float32), 10, 10, threshold=0.0)
    # 6 equal pairs, the first Q = 3 by flat index (query * C + class)
    assert list(zip(d.query.tolist(), d.class_id.tolist())) == [(0, 0), (0, 1), (1, 0)]


@pytest.mark.parametrize("background", [None, -1])
@pytest.mark.parametrize("num_select", [None, 50])
def test_decode_matches_rfdetr(background, num_select):
    rf = pytest.importorskip("rfdetr.export._runtime.decode")
    rng = np.random.default_rng(1)
    for _ in range(5):
        dets = rng.uniform(0, 1, (300, 4)).astype(np.float32)
        labels = rng.normal(-3, 3, (300, 5)).astype(np.float32)
        ours = decode(dets, labels, 640, 480, threshold=0.4, num_select=num_select, background_class_id=background)
        theirs = rf.decode_detections(dets, labels, (640, 480), threshold=0.4, num_select=num_select,
                                      background_class_id=background)
        np.testing.assert_array_equal(ours.class_id, theirs.class_id)
        np.testing.assert_array_equal(ours.query, theirs.query_index)
        np.testing.assert_allclose(ours.xyxy, theirs.xyxy, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(ours.confidence, theirs.confidence, rtol=1e-6)


def test_keypoint_decode_scales_and_fuses():
    rng = np.random.default_rng(2)
    q, schema, d = 4, [0, 3], 7
    kp = rng.normal(0, 1, (q, 2 * 3, d)).astype(np.float32)
    kp[..., 0:2] = rng.uniform(0, 1, (q, 6, 2))
    labels = np.array([[-9, 4], [-9, 3], [-9, -9], [-9, -9]], np.float32)
    out = decode_keypoints(np.full((q, 4), 0.5, np.float32), labels, kp, 200, 100, schema, threshold=0.3)
    assert out.keypoints.shape[1:] == (3, 3)
    for i, (query, cls) in enumerate(zip(out.query, out.class_id)):
        a = kp.reshape(q, 2, 3, d)[query, cls]
        np.testing.assert_allclose(out.keypoints[i, :, 0], a[:, 0] * 200, rtol=1e-6)
        np.testing.assert_allclose(out.keypoints[i, :, 1], a[:, 1] * 100, rtol=1e-6)
        score = 1 / (1 + np.exp(-labels[query, cls]))
        z = np.log(score) - 0.2 * keypoint_log_mean_trace(a)
        assert out.confidence[i] == pytest.approx(1 / (1 + np.exp(-z)), rel=1e-5)
