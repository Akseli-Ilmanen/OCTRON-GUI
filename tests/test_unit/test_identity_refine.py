"""Tests for identity self-training (octron refine-identity), no models."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from octron.yolo_octron import identity_refine as refine
from tests.test_unit.test_link_utils import write_tracklet_csv


def _assignment(folder, rows):
    """Write identity_assignment.csv.

    rows = (track_id, label, identity, flagged).
    """
    df = pd.DataFrame(
        [
            {
                "track_id": tid,
                "label": label,
                "identity": identity,
                "score": 1.0,
                "runner_up": None,
                "runner_up_score": 0.0,
                "margin_ratio": 5.0,
                "n_frames": 1,
                "first_frame": 0,
                "last_frame": 0,
                "flagged": flagged,
                "reason": "",
            }
            for tid, label, identity, flagged in rows
        ]
    ).set_index("track_id")
    df.to_csv(folder / "identity_assignment.csv")


def _metadata(folder, video_path, camera=None, padding=0.1):
    meta = {
        "video_info": {"original_video_path": str(video_path)},
        "camera": camera,
        "identity": {"padding": padding},
        "identity_classes": {
            "bird_1": {"label": "bird", "suffix": "1"},
            "bird_2": {"label": "bird", "suffix": "2"},
        },
    }
    with open(folder / "prediction_metadata.json", "w") as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# select_pseudo_tracklets: assignment + coexistence rule
# ---------------------------------------------------------------------------


def _obs(*tracks):
    """Tracks = (track_id, label, frames)."""
    rows = []
    for tid, label, frames in tracks:
        for f in frames:
            rows.append(
                {
                    "track_id": tid,
                    "label": label,
                    "frame_idx": f,
                    "bbox_x_min": 0.0,
                    "bbox_y_min": 0.0,
                    "bbox_x_max": 10.0,
                    "bbox_y_max": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _assign_df(rows):
    return pd.DataFrame(
        [
            {"track_id": t, "label": lab, "identity": ident, "flagged": fl}
            for t, lab, ident, fl in rows
        ]
    ).set_index("track_id")


def test_select_takes_every_assigned_tracklet():
    obs = _obs(
        (1, "bird", range(10)), (2, "bird", range(10)), (3, "bird", range(5))
    )
    assignment = _assign_df(
        [
            (1, "bird", "bird_1", False),
            (2, "bird", "bird_2", True),  # flagged still counts
            (3, "bird", np.nan, True),
        ]
    )
    selected, rejected = refine.select_pseudo_tracklets(assignment, obs)
    assert selected == {1: "bird_1", 2: "bird_2"}
    assert rejected == {3: "unassigned"}


# ---------------------------------------------------------------------------
# select_pseudo_frames
# ---------------------------------------------------------------------------


def _track(frames, probs):
    df = _obs((1, "bird", frames))
    df["identity_prob_bird_1"] = probs
    return df


def test_frames_keep_all_but_excluded():
    df = _track(range(6), [0.9, 0.1, 0.9, 0.9, 0.05, 0.9])
    chosen = refine.select_pseudo_frames(
        df, "bird_1", max_per_tracklet=0, exclude_frames={5}
    )
    assert chosen["frame_idx"].tolist() == [0, 1, 2, 3, 4]


def test_frames_subsampled_evenly():
    df = _track(range(100), [1.0] * 100)
    chosen = refine.select_pseudo_frames(df, "bird_1", max_per_tracklet=5)
    assert chosen["frame_idx"].tolist() == [0, 25, 50, 74, 99]


# ---------------------------------------------------------------------------
# split_frames from dataset file names
# ---------------------------------------------------------------------------


def test_split_frames_parses_crop_names(tmp_path):
    for split, names in {
        "val": ["vid_a_3.png", "vid_a_4_cam1.png", "other_9.png"],
        "test": ["vid_a_10.png", "pseudo_x_1_2.png"],
    }.items():
        d = tmp_path / split / "bird_1"
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(b"")
    held = refine.split_frames(tmp_path, "vid_a")
    assert held == {"val": {3, 4}, "test": {10}}


# ---------------------------------------------------------------------------
# export_pseudo_crops end to end with a fake video
# ---------------------------------------------------------------------------


class _FakeVideo:
    def __init__(self, n=30, h=40, w=60):
        self.frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]

    def __getitem__(self, idx):
        return self.frames[idx]


def _project_with_dataset(tmp_path, subfolder="vid"):
    project = tmp_path / "project"
    data = project / "model_identity" / "training_data"
    (data / "train" / "bird_1").mkdir(parents=True)
    (data / "test" / "bird_1").mkdir(parents=True)
    # Frame 2 of this video is a held-out test frame.
    (data / "test" / "bird_1" / f"{subfolder}_2.png").write_bytes(b"")
    # Organizer so find_project_subfolder can match the video stem.
    folder = project / subfolder
    folder.mkdir()
    with open(folder / "object_organizer.json", "w") as f:
        json.dump(
            {
                "entries": {
                    "0": {
                        "label": "bird",
                        "suffix": "1",
                        "label_id": 0,
                        "color": [1, 0, 0, 1],
                        "prediction_layer_metadata": {
                            "zarr_path": f"{subfolder}/bird 1 masks.zarr",
                            "video_file_path": "videos/clip.mp4",
                            "video_hash": "abc",
                            "data_shape": [30, 40, 60],
                        },
                    }
                },
                "settings": {},
            },
            f,
        )
    return project, data


def test_export_pseudo_crops_writes_train_only_and_skips_test_frames(
    tmp_path,
):
    project, data = _project_with_dataset(tmp_path)
    pred = tmp_path / "pred" / "cam"
    pred.mkdir(parents=True)
    write_tracklet_csv(
        pred,
        label="bird",
        track_id=1,
        frame_idx=[0, 1, 2, 3],
        identity_probs={"bird_1": 0.9, "bird_2": 0.1},
    )
    _assignment(pred, [(1, "bird", "bird_1", False)])
    _metadata(
        pred,
        tmp_path / "videos" / "clip.mp4",
        camera={
            "name": "cam",
            "x_min": 10,
            "y_min": 5,
            "x_max": 50,
            "y_max": 35,
        },
    )
    video = _FakeVideo()
    summary = refine.export_pseudo_crops(
        project,
        [pred],
        max_per_tracklet=0,
        coexistence_only=False,
        open_video=lambda p: video,
    )
    out = sorted((data / "train" / "bird_1").glob("pseudo_*.png"))
    names = [p.name for p in out]
    assert names == [
        "pseudo_cam_1_0.png",
        "pseudo_cam_1_1.png",
        "pseudo_cam_1_3.png",
    ]
    assert summary["n_crops"] == {"bird_1": 3}
    assert summary["n_selected"] == 1
    # Crop came from the camera region of frame 3 (all pixels == 3),
    # square and padded: box 10x10 -> 12x12.
    arr = np.asarray(Image.open(out[-1]))
    assert arr.shape == (12, 12, 3)
    assert arr.max() == 3
    assert not list((data / "test" / "bird_1").glob("pseudo_*"))


def test_export_clear_removes_old_pseudo_crops(tmp_path):
    project, data = _project_with_dataset(tmp_path)
    stale = data / "train" / "bird_1" / "pseudo_old_1_1.png"
    stale.write_bytes(b"")
    pred = tmp_path / "pred" / "cam"
    pred.mkdir(parents=True)
    write_tracklet_csv(pred, label="bird", track_id=1, frame_idx=[0])
    _assignment(pred, [(1, "bird", np.nan, True)])
    _metadata(pred, tmp_path / "videos" / "clip.mp4")
    summary = refine.export_pseudo_crops(
        project, [pred], clear=True, open_video=lambda p: _FakeVideo()
    )
    assert summary["n_cleared"] == 1
    assert not stale.exists()
    assert summary["n_crops"] == {}


def test_export_requires_dataset(tmp_path):
    (tmp_path / "project").mkdir()
    with pytest.raises(FileNotFoundError, match="train-identity"):
        refine.export_pseudo_crops(tmp_path / "project", [tmp_path])


def test_load_identity_assignment_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="octron link"):
        refine.load_identity_assignment(tmp_path)


# ---------------------------------------------------------------------------
# coexistence_frames: only frames where every individual is present
# ---------------------------------------------------------------------------


def test_coexistence_frames_requires_all_individuals():
    obs = _obs((1, "bird", range(0, 10)), (2, "bird", range(5, 15)))
    linked = {1: "bird_1", 2: "bird_2"}
    frames = refine.coexistence_frames(
        obs, linked, {"bird": {"bird_1", "bird_2"}}
    )
    assert frames == set(range(5, 10))
    # an unassigned second tracklet does not count
    assert (
        refine.coexistence_frames(
            obs, {1: "bird_1", 2: None}, {"bird": {"bird_1", "bird_2"}}
        )
        == set()
    )
    # a label with a single individual never qualifies
    assert (
        refine.coexistence_frames(obs, linked, {"bird": {"bird_1"}}) == set()
    )
    assert (
        refine.coexistence_frames(
            obs.iloc[0:0], linked, {"bird": {"bird_1", "bird_2"}}
        )
        == set()
    )


def test_export_coexistence_only_uses_shared_frames(tmp_path):
    project, data = _project_with_dataset(tmp_path)
    pred = tmp_path / "pred" / "cam"
    pred.mkdir(parents=True)
    write_tracklet_csv(pred, label="bird", track_id=1, frame_idx=range(0, 10))
    write_tracklet_csv(pred, label="bird", track_id=2, frame_idx=range(6, 20))
    _assignment(
        pred, [(1, "bird", "bird_1", False), (2, "bird", "bird_2", False)]
    )
    _metadata(pred, tmp_path / "videos" / "clip.mp4")
    (data / "train" / "bird_2").mkdir()
    summary = refine.export_pseudo_crops(
        project, [pred], max_per_tracklet=0, open_video=lambda p: _FakeVideo()
    )
    # shared frames 6..9; frame 2 (test) is outside them anyway
    got_1 = sorted(
        int(p.stem.split("_")[-1])
        for p in (data / "train" / "bird_1").glob("pseudo_*")
    )
    got_2 = sorted(
        int(p.stem.split("_")[-1])
        for p in (data / "train" / "bird_2").glob("pseudo_*")
    )
    assert got_1 == [6, 7, 8, 9] and got_2 == [6, 7, 8, 9]
    assert summary["n_coexistence_frames"][pred.as_posix()] == 4
