"""Per-camera handling in the export helpers (identity crops, masks)."""

import json

import numpy as np

from octron.cameras import (
    Camera,
    CameraLayout,
    load_layout_for_folder,
    mask_within_camera,
)
from octron.yolo_octron.identity import identity_crops


def _layout():
    return CameraLayout(
        cameras=[Camera("left", 0, 0, 40, 30), Camera("right", 40, 0, 80, 30)],
        frame_width=80,
        frame_height=30,
    )


def test_mask_within_camera_zeroes_outside():
    mask = np.ones((30, 80), dtype=np.int8)
    left = mask_within_camera(mask, _layout().get("left"))
    assert left[:, :40].all() and not left[:, 40:].any()
    assert left.shape == mask.shape and left.dtype == mask.dtype


def test_load_layout_for_folder(tmp_path):
    assert load_layout_for_folder(tmp_path) is None
    CameraLayout.full_frame(80, 30).save(tmp_path / "cameras.json")
    assert load_layout_for_folder(tmp_path) is None  # single full frame
    _layout().save(tmp_path / "cameras.json", overwrite=True)
    layout = load_layout_for_folder(tmp_path)
    assert layout is not None and layout.names == ["left", "right"]
    with open(tmp_path / "cameras.json") as f:
        assert json.load(f)["frame_width"] == 80


def test_identity_crops_without_layout_is_single_crop():
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    mask[5:10, 60:70] = 1
    frame = np.full((30, 80, 3), 7, dtype=np.uint8)
    crops = list(identity_crops(mask, lambda: frame, None, padding=0.0))
    assert len(crops) == 1
    name, crop = crops[0]
    assert name is None
    assert crop.shape[0] == crop.shape[1] == 65  # box spans both blobs


def test_identity_crops_one_per_camera_and_lazy_frame():
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[5:10, 5:10] = 1  # left
    mask[5:15, 60:70] = 1  # right
    frame = np.zeros((30, 80, 3), dtype=np.uint8)
    frame[:, 40:] = 200
    calls = []

    def get_frame():
        calls.append(1)
        return frame

    crops = dict(identity_crops(mask, get_frame, _layout(), padding=0.0))
    assert sorted(crops) == ["left", "right"]
    assert crops["left"].shape == (5, 5, 3) and crops["left"].max() == 0
    assert crops["right"].shape == (10, 10, 3) and crops["right"].min() == 200
    assert len(calls) == 1  # frame decoded once


def test_identity_crops_skip_cameras_without_mask():
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[2:6, 50:54] = 1
    frame = np.zeros((30, 80, 3), dtype=np.uint8)
    crops = list(identity_crops(mask, lambda: frame, _layout()))
    assert [c[0] for c in crops] == ["right"]


def test_identity_crops_empty_mask_never_decodes():
    mask = np.zeros((30, 80), dtype=np.uint8)

    def boom():
        raise AssertionError("frame should not be decoded")

    assert list(identity_crops(mask, boom, _layout())) == []
    assert list(identity_crops(mask, boom, None)) == []
