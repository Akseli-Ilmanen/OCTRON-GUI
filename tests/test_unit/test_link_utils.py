"""Tests for the multi-camera identity linking pipeline (octron link).

Covers the worked example from
https://github.com/OCTRON-tracking/OCTRON-GUI/issues/91 plus the
exclusivity/label-restriction edge cases: non-overlapping tracklets may
share an individual, more tracklets than individuals leaves one
unassigned, global vs. per-camera cross-folder exclusivity, label
restriction, output files / overwrite guard, and the
``get_identity_assignment`` loader roundtrip.
"""

import json

import numpy as np
import pandas as pd
import pytest

from octron.tools.link import (
    Tracklet,
    build_score_matrix,
    load_tracklets,
    run_link,
    solve_assignment,
    temporal_overlaps,
)
from octron.yolo_octron.helpers.yolo_results import YOLO_results

# ---------------------------------------------------------------------------
# Fixture helper: write a fake tracklet CSV (7-line metadata header + data)
# ---------------------------------------------------------------------------

_BASE_COLUMNS = [
    "frame_counter",
    "frame_idx",
    "track_id",
    "label",
    "confidence",
    "pos_x",
    "pos_y",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_x_min",
    "bbox_x_max",
    "bbox_y_min",
    "bbox_y_max",
]


def write_tracklet_csv(
    folder,
    *,
    label,
    track_id,
    frame_idx,
    identity_probs=None,
    video_name="video",
):
    """Write a ``<label>_track_<track_id>.csv`` matching the real format.

    Parameters
    ----------
    folder : Path
        Destination directory (created if missing).
    label : str
        Detector label.
    track_id : int
        Track ID (used in the filename and the ``track_id`` column).
    frame_idx : sequence of int
        Frame indices for this tracklet's observations.
    identity_probs : dict, optional
        ``{classname: value}`` where value is either a scalar (applied
        to every frame) or a per-frame sequence. Written as
        ``identity_prob_<classname>`` columns.
    video_name : str
        Value for the ``video_name`` header line.

    Returns
    -------
    Path
        The written CSV path.

    """
    folder.mkdir(parents=True, exist_ok=True)
    frame_idx = list(frame_idx)
    n = len(frame_idx)

    header = [
        f"video_name: {video_name}",
        f"frame_count: {n}",
        f"frame_count_analyzed: {n}",
        "video_height: 480",
        "video_width: 640",
        "created_at: 2024-01-01 00:00:00",
        "",
    ]

    data = {
        "frame_counter": list(range(n)),
        "frame_idx": frame_idx,
        "track_id": [track_id] * n,
        "label": [label] * n,
        "confidence": [0.9] * n,
        "pos_x": [10.0] * n,
        "pos_y": [10.0] * n,
        "bbox_area": [100.0] * n,
        "bbox_aspect_ratio": [1.0] * n,
        "bbox_x_min": [0.0] * n,
        "bbox_x_max": [10.0] * n,
        "bbox_y_min": [0.0] * n,
        "bbox_y_max": [10.0] * n,
    }
    if identity_probs:
        for classname, probs in identity_probs.items():
            if np.isscalar(probs):
                probs = [probs] * n
            assert len(probs) == n
            data[f"identity_prob_{classname}"] = list(probs)

    columns = _BASE_COLUMNS + (
        sorted(f"identity_prob_{c}" for c in identity_probs)
        if identity_probs
        else []
    )
    df = pd.DataFrame(data)[columns]

    path = folder / f"{label}_track_{track_id}.csv"
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False, na_rep="NaN", lineterminator="\n")
    return path


# ---------------------------------------------------------------------------
# 1. Worked example from issue #91
# ---------------------------------------------------------------------------


def _worked_example(tmp_path):
    """cam1: A, B overlapping; cam2: C. Returns (cam1, cam2)."""
    cam1 = tmp_path / "cam1"
    cam2 = tmp_path / "cam2"
    frames = list(range(10))

    write_tracklet_csv(
        cam1,
        label="bird",
        track_id=1,  # A
        frame_idx=frames,
        identity_probs={"bird_1": 0.9, "bird_2": 0.05, "bird_3": 0.05},
    )
    write_tracklet_csv(
        cam1,
        label="bird",
        track_id=2,  # B
        frame_idx=frames,
        identity_probs={"bird_1": 0.55, "bird_2": 0.5, "bird_3": 0.1},
    )
    write_tracklet_csv(
        cam2,
        label="bird",
        track_id=1,  # C
        frame_idx=frames,
        identity_probs={"bird_1": 0.05, "bird_2": 0.05, "bird_3": 0.9},
    )
    return cam1, cam2


def test_worked_example_assigns_and_flags_low_margin(tmp_path):
    cam1, cam2 = _worked_example(tmp_path)
    summary = run_link([cam1, cam2])

    cam1_csv = pd.read_csv(
        cam1 / "identity_assignment.csv", index_col="track_id"
    )
    cam2_csv = pd.read_csv(
        cam2 / "identity_assignment.csv", index_col="track_id"
    )

    a = cam1_csv.loc[1]
    b = cam1_csv.loc[2]
    c = cam2_csv.loc[1]

    assert a["identity"] == "bird_1"
    assert b["identity"] == "bird_2"
    assert c["identity"] == "bird_3"

    assert not bool(a["flagged"])
    assert bool(b["flagged"])
    assert b["reason"] == "low_margin"
    assert not bool(c["flagged"])

    assert summary["n_tracklets"] == 3
    assert summary["n_assigned"] == 3
    assert summary["n_flagged"] == 1


def test_load_tracklets_scores_match_summed_probs(tmp_path):
    cam1, _ = _worked_example(tmp_path)
    tracklets = load_tracklets(cam1)
    assert len(tracklets) == 2
    a = next(t for t in tracklets if t.track_id == 1)
    assert a.scores["bird_1"] == pytest.approx(9.0)
    assert a.scores["bird_2"] == pytest.approx(0.5)
    assert a.n_frames == 10


def test_temporal_overlaps_detects_shared_frames(tmp_path):
    cam1, cam2 = _worked_example(tmp_path)
    tracklets = load_tracklets(cam1) + load_tracklets(cam2)
    pairs = temporal_overlaps(tracklets)
    # A (idx 0) and B (idx 1) share every frame; C (idx 2) is a separate
    # folder but shares frame_idx values too (frame_idx alone doesn't
    # know about folders).
    assert (0, 1) in pairs


# ---------------------------------------------------------------------------
# 2. Non-overlapping tracklets in one camera may share an individual
# ---------------------------------------------------------------------------


def test_non_overlapping_tracklets_may_share_individual(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,  # D
        frame_idx=range(0, 10),
        identity_probs={"bird_1": 0.9, "bird_2": 0.05},
    )
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=2,  # E
        frame_idx=range(20, 30),
        identity_probs={"bird_1": 0.9, "bird_2": 0.05},
    )
    summary = run_link([cam])
    df = pd.read_csv(cam / "identity_assignment.csv", index_col="track_id")
    assert df.loc[1, "identity"] == "bird_1"
    assert df.loc[2, "identity"] == "bird_1"
    assert summary["n_assigned"] == 2


# ---------------------------------------------------------------------------
# 3. More overlapping tracklets than individuals: one unassigned+flagged
# ---------------------------------------------------------------------------


def test_more_tracklets_than_individuals_leaves_one_unassigned(tmp_path):
    cam = tmp_path / "cam"
    frames = range(0, 10)
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,  # F: stronger match
        frame_idx=frames,
        identity_probs={"bird_1": 0.9},
    )
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=2,  # G: weaker match, same (only) individual
        frame_idx=frames,
        identity_probs={"bird_1": 0.5},
    )
    summary = run_link([cam])
    df = pd.read_csv(cam / "identity_assignment.csv", index_col="track_id")

    assert df.loc[1, "identity"] == "bird_1"
    assert pd.isna(df.loc[2, "identity"])
    assert bool(df.loc[2, "flagged"])
    assert df.loc[2, "reason"] == "unassigned"
    assert summary["n_assigned"] == 1
    assert summary["n_flagged"] == 1

    report = json.loads((cam / "link_report.json").read_text())
    assert report["n_tracklets"] == 2
    assert report["n_assigned"] == 1
    flagged_ids = {f["track_id"] for f in report["flagged"]}
    assert flagged_ids == {2}


# ---------------------------------------------------------------------------
# 4. global_exclusive forbids cross-camera reuse; per-camera allows it
# ---------------------------------------------------------------------------


def test_global_exclusive_vs_per_camera_cross_folder_reuse(tmp_path):
    cam1 = tmp_path / "cam1"
    cam2 = tmp_path / "cam2"
    frames = range(0, 10)
    write_tracklet_csv(
        cam1,
        label="bird",
        track_id=1,  # H
        frame_idx=frames,
        identity_probs={"bird_1": 0.9, "bird_2": 0.05},
    )
    write_tracklet_csv(
        cam2,
        label="bird",
        track_id=1,  # I: same frames, different camera
        frame_idx=frames,
        identity_probs={"bird_1": 0.9, "bird_2": 0.05},
    )

    # Per-camera (default): no in-folder pair exists for either lone
    # tracklet, so both may take bird_1.
    summary_local = run_link([cam1, cam2], overwrite=False)
    h_local = pd.read_csv(
        cam1 / "identity_assignment.csv", index_col="track_id"
    )
    i_local = pd.read_csv(
        cam2 / "identity_assignment.csv", index_col="track_id"
    )
    assert h_local.loc[1, "identity"] == "bird_1"
    assert i_local.loc[1, "identity"] == "bird_1"
    assert summary_local["global_exclusive"] is False

    # Global: the cross-camera overlap forbids sharing bird_1.
    summary_global = run_link(
        [cam1, cam2], global_exclusive=True, overwrite=True
    )
    h_global = pd.read_csv(
        cam1 / "identity_assignment.csv", index_col="track_id"
    )
    i_global = pd.read_csv(
        cam2 / "identity_assignment.csv", index_col="track_id"
    )
    identities = {h_global.loc[1, "identity"], i_global.loc[1, "identity"]}
    assert identities == {"bird_1", "bird_2"}
    assert summary_global["global_exclusive"] is True


# ---------------------------------------------------------------------------
# 5. Label restriction: a mouse tracklet never gets a bird_* identity
# ---------------------------------------------------------------------------


def test_label_restriction_mouse_never_gets_bird_identity(tmp_path):
    cam = tmp_path / "cam"
    frames = range(0, 10)
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,
        frame_idx=frames,
        identity_probs={"bird_1": 0.9},
    )
    write_tracklet_csv(
        cam,
        label="mouse",
        track_id=2,  # J: has a (misleading) bird_1 column too
        frame_idx=frames,
        identity_probs={"bird_1": 0.9, "mouse_1": 0.3},
    )
    run_link([cam])
    df = pd.read_csv(cam / "identity_assignment.csv", index_col="track_id")

    mouse_row = df.loc[2]
    assert mouse_row["label"] == "mouse"
    assert mouse_row["identity"] == "mouse_1"
    assert "bird" not in str(mouse_row["identity"])


def test_load_tracklets_excludes_other_label_columns(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(
        cam,
        label="mouse",
        track_id=1,
        frame_idx=range(5),
        identity_probs={"bird_1": 0.9, "mouse_1": 0.3},
    )
    tracklets = load_tracklets(cam)
    (mouse_tracklet,) = tracklets
    assert "bird_1" not in mouse_tracklet.scores
    assert mouse_tracklet.scores["mouse_1"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 6. Output files exist; overwrite guard
# ---------------------------------------------------------------------------


def test_outputs_written_and_overwrite_guard(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,
        frame_idx=range(5),
        identity_probs={"bird_1": 0.9},
    )
    run_link([cam])
    assert (cam / "identity_assignment.csv").exists()
    assert (cam / "link_report.json").exists()

    with pytest.raises(FileExistsError):
        run_link([cam])

    # overwrite=True succeeds and replaces the files.
    run_link([cam], overwrite=True)
    assert (cam / "identity_assignment.csv").exists()


# ---------------------------------------------------------------------------
# 7. get_identity_assignment roundtrip
# ---------------------------------------------------------------------------


def test_get_identity_assignment_roundtrip(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,
        frame_idx=range(5),
        identity_probs={"bird_1": 0.9, "bird_2": 0.1},
    )
    run_link([cam])

    results = YOLO_results(cam, verbose=False)
    df = results.get_identity_assignment()
    assert df is not None
    assert df.index.name == "track_id"
    assert df.loc[1, "identity"] == "bird_1"


def test_get_identity_assignment_none_when_missing(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(
        cam,
        label="bird",
        track_id=1,
        frame_idx=range(5),
        identity_probs={"bird_1": 0.9},
    )
    results = YOLO_results(cam, verbose=False)
    assert results.get_identity_assignment() is None


# ---------------------------------------------------------------------------
# Misc: no-identity-scores tracklets, and build_score_matrix / solve_assignment
# ---------------------------------------------------------------------------


def test_tracklet_without_identity_columns_is_no_identity_scores(tmp_path):
    cam = tmp_path / "cam"
    write_tracklet_csv(cam, label="bird", track_id=1, frame_idx=range(5))
    run_link([cam])
    df = pd.read_csv(cam / "identity_assignment.csv", index_col="track_id")
    assert pd.isna(df.loc[1, "identity"])
    assert df.loc[1, "reason"] == "no_identity_scores"
    assert bool(df.loc[1, "flagged"])


def test_solve_assignment_empty_individuals_returns_all_none():
    t = Tracklet(
        folder=None,
        track_id=1,
        label="bird",
        frame_idx=np.array([0, 1]),
        scores={},
        n_frames=2,
    )
    assert solve_assignment([t], [], []) == [None]


def test_build_score_matrix_zero_for_other_label():
    t = Tracklet(
        folder=None,
        track_id=1,
        label="bird",
        frame_idx=np.array([0]),
        scores={"bird_1": 3.0},
        n_frames=1,
    )
    S = build_score_matrix([t], ["bird_1", "mouse_1"])
    assert S.tolist() == [[3.0, 0.0]]
