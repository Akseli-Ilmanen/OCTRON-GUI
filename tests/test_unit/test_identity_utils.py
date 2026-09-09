"""Tests for octron.yolo_octron.identity (no model downloads)."""

import json

import numpy as np
import pytest

from octron.yolo_octron import identity as ident


def test_sanitize_class_name_replaces_whitespace():
    assert ident.sanitize_class_name("bird", "1") == "bird_1"
    assert ident.sanitize_class_name("my bird", " 2 ") == "my_bird_2"
    assert ident.identity_prob_column("bird_1") == "identity_prob_bird_1"


def test_mask_bbox_exclusive_max_and_empty():
    mask = np.zeros((10, 12), dtype=np.int8)
    mask[2:5, 3:9] = 1
    box = ident.mask_bbox(mask)
    assert box.tolist() == [3, 2, 9, 5]
    assert ident.mask_bbox(np.zeros((4, 4))) is None


def test_square_crop_is_square_padded_and_centred():
    frame = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    crop = ident.square_crop(frame, [10, 5, 14, 15], padding=0.0)
    # longer side is 10 -> 10x10 crop centred on (12, 10)
    assert crop.shape == (10, 10, 3)
    np.testing.assert_array_equal(crop, frame[5:15, 7:17])


def test_square_crop_zero_pads_outside_frame():
    frame = np.full((8, 8), 7, dtype=np.uint8)
    crop = ident.square_crop(frame, [0, 0, 4, 4], padding=0.5)
    # side = 4 * 2 = 8, centred on (2, 2) -> left/top = -2
    assert crop.shape == (8, 8)
    assert crop[0, 0] == 0
    assert crop[2, 2] == 7
    assert crop[7, 7] == 7  # inside frame (pixel 5, 5)


def test_square_crop_padding_grows_longer_side():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = ident.square_crop(frame, [10, 10, 30, 20], padding=0.1)
    assert crop.shape[0] == crop.shape[1] == 24


def _write_organizer(project, subfolder, entries):
    folder = project / subfolder
    folder.mkdir(parents=True)
    payload = {"entries": {}, "settings": {}}
    for i, (label, suffix) in enumerate(entries):
        payload["entries"][str(i)] = {
            "label": label,
            "suffix": suffix,
            "label_id": 0,
            "color": [1, 0, 0, 1],
            "prediction_layer_metadata": {
                "zarr_path": f"{subfolder}/{label} {suffix} masks.zarr",
                "video_file_path": "videos/clip.mp4",
                "video_hash": "abc12345",
                "data_shape": [50, 20, 30],
            },
        }
    with open(folder / "object_organizer.json", "w") as f:
        json.dump(payload, f)


def test_collect_identity_entries_skips_missing_suffix(tmp_path):
    _write_organizer(
        tmp_path, "aaaa1111", [("bird", "1"), ("bird", "2"), ("bird", "")]
    )
    _write_organizer(tmp_path, "bbbb2222", [("mouse", "x")])
    entries, n_skipped = ident.collect_identity_entries(tmp_path)
    assert n_skipped == 1
    names = sorted(e["class_name"] for e in entries)
    assert names == ["bird_1", "bird_2", "mouse_x"]
    bird = next(e for e in entries if e["class_name"] == "bird_1")
    assert bird["subfolder"] == "aaaa1111"
    assert bird["num_frames"] == 50
    assert bird["zarr_path"].name == "bird 1 masks.zarr"
    assert bird["video_file_path"].name == "clip.mp4"


def test_build_identity_dataset_requires_suffixes(tmp_path):
    _write_organizer(tmp_path, "aaaa1111", [("bird", "")])
    with pytest.raises(ValueError, match="suffix"):
        ident.build_identity_dataset(tmp_path)


def test_build_identity_dataset_refuses_overwrite(tmp_path):
    _, data_path = ident.identity_paths(tmp_path)
    data_path.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        ident.build_identity_dataset(tmp_path)


def test_ensure_val_examples_moves_one_crop(tmp_path):
    train = tmp_path / "train" / "bird_1"
    train.mkdir(parents=True)
    for i in range(3):
        (train / f"f_{i}.png").write_bytes(b"x")
    ident._ensure_val_examples(tmp_path, {"bird_1": {}})
    assert len(list(train.iterdir())) == 2
    assert len(list((tmp_path / "val" / "bird_1").iterdir())) == 1


def test_ensure_val_examples_keeps_single_crop(tmp_path):
    train = tmp_path / "train" / "bird_1"
    train.mkdir(parents=True)
    (train / "f_0.png").write_bytes(b"x")
    ident._ensure_val_examples(tmp_path, {"bird_1": {}})
    assert len(list(train.iterdir())) == 1
    assert not (tmp_path / "val" / "bird_1").exists()


def _stub_classifier(class_names, classes=None):
    clf = ident.IdentityClassifier.__new__(ident.IdentityClassifier)
    clf.class_names = list(class_names)
    clf.classes = classes or {
        n: {"label": n.rpartition("_")[0], "suffix": n.rpartition("_")[2]}
        for n in class_names
    }
    clf.prob_columns = [ident.identity_prob_column(n) for n in class_names]
    return clf


def test_decide_restricts_to_same_label():
    clf = _stub_classifier(["bird_1", "bird_2", "mouse_1"])
    probs = np.array([0.1, 0.3, 0.6])
    assert clf.decide(probs, "bird") == ("bird_2", pytest.approx(0.3))
    assert clf.decide(probs, "mouse") == ("mouse_1", pytest.approx(0.6))
    name, conf = clf.decide(probs, "fish")
    assert name is None and np.isnan(conf)


def test_candidate_indices_ignore_whitespace_differences():
    clf = _stub_classifier(
        ["my_bird_1", "my_bird_2"],
        classes={
            "my_bird_1": {"label": "my bird", "suffix": "1"},
            "my_bird_2": {"label": "my bird", "suffix": "2"},
        },
    )
    assert clf.candidate_indices("my bird") == [0, 1]
    assert clf.candidate_indices("my_bird") == [0, 1]
    assert clf.candidate_indices("bird") == []


def test_resolve_identity_model_path_uses_model_cache(tmp_path, monkeypatch):
    from octron import config

    monkeypatch.setattr(config, "get_yolo_models_dir", lambda: tmp_path)
    assert ident.resolve_identity_model_path("yolo11n-cls") == (
        tmp_path / "yolo11n-cls.pt"
    )
    existing = tmp_path / "custom.pt"
    existing.write_bytes(b"")
    assert ident.resolve_identity_model_path(existing) == existing
