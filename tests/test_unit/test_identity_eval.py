"""Tests for identity evaluation (IDF1 / linked accuracy), no models."""

import numpy as np
import pandas as pd

from octron.yolo_octron import identity_eval as ev

# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------


def test_box_iou():
    assert ev.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert ev.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 0.5 / 1.5
    assert ev.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_match_boxes_one_to_one_by_iou():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    pred = [[51, 50, 61, 60], [1, 0, 11, 10], [100, 100, 110, 110]]
    assert sorted(ev.match_boxes(gt, pred)) == [(0, 1), (1, 0)]
    assert ev.match_boxes([], pred) == []


def test_idf1_perfect_and_fragmented():
    # Perfect: 10 detections of A on track 1, 10 of B on track 2.
    pairs = [("A", 1)] * 10 + [("B", 2)] * 10
    res = ev.idf1(pairs, n_gt=20, n_pred=20)
    assert res["idf1"] == 1.0
    assert res["mapping"] == {"A": "1", "B": "2"}
    # A fragmented into two tracks 6/4: only the larger counts.
    pairs = [("A", 1)] * 6 + [("A", 3)] * 4 + [("B", 2)] * 10
    res = ev.idf1(pairs, n_gt=20, n_pred=20)
    assert res["idtp"] == 16
    assert res["idf1"] == 2 * 16 / 40


def test_idf1_counts_unmatched_as_fp_fn():
    pairs = [("A", 1)] * 5
    res = ev.idf1(pairs, n_gt=8, n_pred=6)
    assert res["idtp"] == 5 and res["idfn"] == 3 and res["idfp"] == 1
    assert res["idf1"] == 10 / 14
    assert np.isnan(ev.idf1([], 0, 0)["idf1"])


# ---------------------------------------------------------------------------
# evaluate_pairs: frame matching + the three id schemes
# ---------------------------------------------------------------------------


def _obs_rows(rows):
    """Rows = (track_id, label, frame, box, identity)."""
    out = []
    for tid, label, f, box, ident in rows:
        out.append(
            {
                "track_id": tid,
                "label": label,
                "frame_idx": f,
                "bbox_x_min": box[0],
                "bbox_y_min": box[1],
                "bbox_x_max": box[2],
                "bbox_y_max": box[3],
                "identity": ident,
            }
        )
    return pd.DataFrame(out)


def test_evaluate_pairs_swap_visible_in_accuracy_not_idf1():
    box_a = np.array([0, 0, 10, 10.0])
    box_b = np.array([50, 50, 60, 60.0])
    gt = {
        f: [("bird_1", "bird", box_a), ("bird_2", "bird", box_b)]
        for f in range(4)
    }
    obs = _obs_rows(
        [(1, "bird", f, box_a, "bird_1") for f in range(4)]
        + [(2, "bird", f, box_b, "bird_2") for f in range(4)]
    )
    # Linking swapped both identities consistently.
    linked = {1: "bird_2", 2: "bird_1"}
    res = ev.evaluate_pairs(gt, obs, linked)
    report = ev.summarise(res)
    assert report["n_gt"] == 8 and report["n_pred"] == 8
    assert report["n_matched"] == 8
    assert report["idf1_tracks"] == 1.0
    assert report["idf1_frame_identity"] == 1.0
    assert report["idf1_linked"] == 1.0  # consistent, just swapped
    assert report["linked_accuracy"] == 0.0
    assert report["per_identity"]["bird_1"] == {
        "n_gt": 4,
        "matched": 4,
        "correct": 0,
    }


def test_evaluate_pairs_fragmented_tracks_fixed_by_link():
    box = np.array([0, 0, 10, 10.0])
    gt = {f: [("bird_1", "bird", box)] for f in range(10)}
    obs = _obs_rows(
        [(1, "bird", f, box, "bird_1") for f in range(6)]
        + [(2, "bird", f, box, "bird_1") for f in range(6, 10)]
    )
    linked = {1: "bird_1", 2: "bird_1"}
    report = ev.summarise(ev.evaluate_pairs(gt, obs, linked))
    assert report["idf1_tracks"] == 2 * 6 / 20
    assert report["idf1_linked"] == 1.0
    assert report["linked_accuracy"] == 1.0


def test_evaluate_pairs_unassigned_and_missed():
    box = np.array([0, 0, 10, 10.0])
    far = np.array([100, 100, 110, 110.0])
    gt = {0: [("bird_1", "bird", box)], 1: [("bird_1", "bird", box)]}
    # Frame 0 matched but unassigned; frame 1 prediction elsewhere.
    obs = _obs_rows([(1, "bird", 0, box, np.nan), (1, "bird", 1, far, np.nan)])
    report = ev.summarise(ev.evaluate_pairs(gt, obs, {1: None}))
    assert report["n_matched"] == 1
    assert report["linked_accuracy"] == 0.0
    assert report["idf1_linked"] == 2 * 1 / 4


def test_evaluate_pairs_matches_within_label_only():
    box = np.array([0, 0, 10, 10.0])
    gt = {0: [("bird_1", "bird", box)]}
    obs = _obs_rows([(1, "mouse", 0, box, "mouse_1")])
    report = ev.summarise(ev.evaluate_pairs(gt, obs, {1: "mouse_1"}))
    assert report["n_matched"] == 0
    assert report["idf1_linked"] == 0.0
