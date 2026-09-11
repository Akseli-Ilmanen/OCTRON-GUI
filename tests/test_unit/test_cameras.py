"""Tests for the multi-camera mosaic layout (octron/cameras.py).

Pure stdlib + numpy + loguru module, so these tests need no project,
model, or napari import.
"""

import json

import numpy as np
import pytest

from octron.cameras import (
    CAMERAS_VERSION,
    Camera,
    CameraLayout,
    apply_cameras,
    find_cameras_file,
    resolve_layout,
    sibling_cameras_path,
)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_camera_width_height():
    cam = Camera(name="cam0", x_min=10, y_min=20, x_max=110, y_max=220)
    assert cam.width == 100
    assert cam.height == 200


def test_camera_contains_point_half_open():
    cam = Camera(name="cam0", x_min=0, y_min=0, x_max=10, y_max=10)
    assert cam.contains_point(0, 0)
    assert cam.contains_point(9, 9)
    assert not cam.contains_point(10, 9)  # x_max excluded
    assert not cam.contains_point(9, 10)  # y_max excluded
    assert not cam.contains_point(-1, 5)


def test_camera_crop_2d_and_3d():
    array2d = np.arange(100).reshape(10, 10)
    cam = Camera(name="cam0", x_min=2, y_min=3, x_max=6, y_max=8)
    cropped = cam.crop(array2d)
    assert cropped.shape == (5, 4)  # (height, width)
    np.testing.assert_array_equal(cropped, array2d[3:8, 2:6])

    array3d = np.arange(300).reshape(10, 10, 3)
    cropped3d = cam.crop(array3d)
    assert cropped3d.shape == (5, 4, 3)


def test_camera_shift_box_clips_to_bounds():
    cam = Camera(name="cam0", x_min=100, y_min=200, x_max=300, y_max=400)
    # Fully inside.
    shifted = cam.shift_box([150, 250, 180, 280])
    np.testing.assert_array_equal(shifted, [50, 50, 80, 80])

    # Straddling the camera boundary -> clipped to [0, width]/[0, height].
    shifted = cam.shift_box([50, 150, 350, 450])
    np.testing.assert_array_equal(shifted, [0, 0, 200, 200])


def test_camera_to_dict_from_dict_roundtrip():
    cam = Camera(name="cam0", x_min=1, y_min=2, x_max=3, y_max=4)
    d = cam.to_dict()
    assert d == {
        "name": "cam0",
        "x_min": 1,
        "y_min": 2,
        "x_max": 3,
        "y_max": 4,
    }
    cam2 = Camera.from_dict(d)
    assert cam2 == cam


# ---------------------------------------------------------------------------
# CameraLayout: construction
# ---------------------------------------------------------------------------


def test_full_frame_single_camera():
    layout = CameraLayout.full_frame(640, 480)
    assert len(layout) == 1
    cam = layout.get("cam0")
    assert (cam.x_min, cam.y_min, cam.x_max, cam.y_max) == (0, 0, 640, 480)
    assert layout.is_single_full_frame()


def test_full_frame_custom_name():
    layout = CameraLayout.full_frame(100, 100, name="only")
    assert layout.names == ["only"]


# ---------------------------------------------------------------------------
# CameraLayout: save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=50, y_max=100),
            Camera(name="cam1", x_min=50, y_min=0, x_max=100, y_max=100),
        ],
        frame_width=100,
        frame_height=100,
        video_file_path="video.mp4",
        video_hash="abc123",
    )
    path = layout.save(tmp_path / "cameras.json")
    assert path.exists()

    loaded = CameraLayout.load(path)
    assert loaded.frame_width == 100
    assert loaded.frame_height == 100
    assert loaded.video_file_path == "video.mp4"
    assert loaded.video_hash == "abc123"
    assert loaded.names == ["cam0", "cam1"]
    assert loaded.get("cam1").x_min == 50


def test_save_json_is_indented_sorted_and_versioned(tmp_path):
    layout = CameraLayout.full_frame(10, 10)
    path = layout.save(tmp_path / "cameras.json")
    raw = path.read_text()
    assert "\n" in raw  # indented, not a single line
    data = json.loads(raw)
    assert data["version"] == CAMERAS_VERSION
    # sorted keys at the top level
    assert list(data.keys()) == sorted(data.keys())


def test_save_raises_if_exists_without_overwrite(tmp_path):
    layout = CameraLayout.full_frame(10, 10)
    path = tmp_path / "cameras.json"
    layout.save(path)
    with pytest.raises(FileExistsError):
        layout.save(path)
    # overwrite=True succeeds
    layout.save(path, overwrite=True)


def test_to_dict_from_dict_roundtrip():
    layout = CameraLayout.full_frame(10, 20)
    d = layout.to_dict()
    layout2 = CameraLayout.from_dict(d)
    assert layout2.frame_width == 10
    assert layout2.frame_height == 20
    assert layout2.names == layout.names


# ---------------------------------------------------------------------------
# CameraLayout.from_rectangles
# ---------------------------------------------------------------------------


def test_from_rectangles_snaps_and_names_by_order():
    # napari Shapes gives (row, col) = (y, x) corners, any order.
    rect0 = np.array([[0.4, 0.6], [0.4, 49.2], [99.8, 49.2], [99.8, 0.6]])
    rect1 = np.array([[0, 50], [0, 100], [100, 100], [100, 50]])
    layout = CameraLayout.from_rectangles(
        [rect0, rect1], frame_width=100, frame_height=100
    )
    assert layout.names == ["cam0", "cam1"]
    cam0 = layout.get("cam0")
    # x: floor(min)=0, ceil(max)=50 ; y: floor(min)=0, ceil(max)=100
    assert (cam0.x_min, cam0.x_max) == (0, 50)
    assert (cam0.y_min, cam0.y_max) == (0, 100)


def test_from_rectangles_corner_order_independence():
    corners_orderings = [
        [[0, 0], [0, 10], [10, 10], [10, 0]],
        [[10, 10], [10, 0], [0, 0], [0, 10]],
        [[0, 10], [10, 10], [10, 0], [0, 0]],
    ]
    layouts = [
        CameraLayout.from_rectangles(
            [np.array(c)], frame_width=20, frame_height=20
        )
        for c in corners_orderings
    ]
    bounds = {
        (
            layout.get("cam0").x_min,
            layout.get("cam0").y_min,
            layout.get("cam0").x_max,
            layout.get("cam0").y_max,
        )
        for layout in layouts
    }
    assert bounds == {(0, 0, 10, 10)}


def test_from_rectangles_clips_to_frame():
    rect = np.array([[-5, -5], [-5, 15], [15, 15], [15, -5]])
    layout = CameraLayout.from_rectangles(
        [rect], frame_width=10, frame_height=10
    )
    cam = layout.get("cam0")
    assert (cam.x_min, cam.y_min, cam.x_max, cam.y_max) == (0, 0, 10, 10)


def test_from_rectangles_blank_name_falls_back_to_default():
    rect = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    layout = CameraLayout.from_rectangles(
        [rect], frame_width=10, frame_height=10, names=["   "]
    )
    assert layout.names == ["cam0"]


def test_from_rectangles_zero_area_raises():
    # A degenerate rectangle collapsed to a line after clipping.
    rect = np.array([[0, 20], [0, 30], [10, 30], [10, 20]])
    with pytest.raises(ValueError):
        CameraLayout.from_rectangles([rect], frame_width=10, frame_height=10)


# ---------------------------------------------------------------------------
# CameraLayout.validate
# ---------------------------------------------------------------------------


def test_validate_no_cameras_raises():
    layout = CameraLayout(cameras=[], frame_width=10, frame_height=10)
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_duplicate_name_raises():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=5),
            Camera(name="cam0", x_min=5, y_min=0, x_max=10, y_max=5),
        ],
        frame_width=10,
        frame_height=10,
    )
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_empty_name_raises():
    layout = CameraLayout(
        cameras=[Camera(name="", x_min=0, y_min=0, x_max=5, y_max=5)],
        frame_width=10,
        frame_height=10,
    )
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_zero_area_raises():
    layout = CameraLayout(
        cameras=[Camera(name="cam0", x_min=0, y_min=0, x_max=0, y_max=5)],
        frame_width=10,
        frame_height=10,
    )
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_negative_area_raises():
    layout = CameraLayout(
        cameras=[Camera(name="cam0", x_min=5, y_min=0, x_max=0, y_max=5)],
        frame_width=10,
        frame_height=10,
    )
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_outside_frame_raises():
    layout = CameraLayout(
        cameras=[Camera(name="cam0", x_min=0, y_min=0, x_max=20, y_max=5)],
        frame_width=10,
        frame_height=10,
    )
    with pytest.raises(ValueError):
        layout.validate()


def test_validate_allows_overlap_and_logs_warning():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=8, y_max=10),
            Camera(name="cam1", x_min=5, y_min=0, x_max=10, y_max=10),
        ],
        frame_width=10,
        frame_height=10,
    )
    # Must not raise; overlap is allowed (only logged).
    layout.validate()


def test_validate_non_overlapping_ok():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=10),
            Camera(name="cam1", x_min=5, y_min=0, x_max=10, y_max=10),
        ],
        frame_width=10,
        frame_height=10,
    )
    layout.validate()


def test_is_single_full_frame_false_for_multi():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=10),
            Camera(name="cam1", x_min=5, y_min=0, x_max=10, y_max=10),
        ],
        frame_width=10,
        frame_height=10,
    )
    assert not layout.is_single_full_frame()


def test_is_single_full_frame_false_when_not_covering():
    layout = CameraLayout(
        cameras=[Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=10)],
        frame_width=10,
        frame_height=10,
    )
    assert not layout.is_single_full_frame()


# ---------------------------------------------------------------------------
# CameraLayout.assign_box / assign_boxes
# ---------------------------------------------------------------------------


def test_assign_box_by_centre():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=10, y_max=10),
            Camera(name="cam1", x_min=10, y_min=0, x_max=20, y_max=10),
        ],
        frame_width=20,
        frame_height=10,
    )
    cam = layout.assign_box([1, 1, 3, 3])  # centre (2, 2) -> cam0
    assert cam.name == "cam0"
    cam = layout.assign_box([11, 1, 13, 3])  # centre (12, 2) -> cam1
    assert cam.name == "cam1"


def test_assign_box_none_outside_all_cameras():
    layout = CameraLayout(
        cameras=[Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=5)],
        frame_width=10,
        frame_height=10,
    )
    assert layout.assign_box([6, 6, 9, 9]) is None


def test_assign_box_tie_break_by_intersection_area():
    # Two overlapping cameras both contain the box centre; the box has
    # much larger intersection with cam_big than cam_small.
    layout = CameraLayout(
        cameras=[
            Camera(name="cam_small", x_min=4, y_min=4, x_max=6, y_max=6),
            Camera(name="cam_big", x_min=0, y_min=0, x_max=10, y_max=10),
        ],
        frame_width=10,
        frame_height=10,
    )
    box = [0, 0, 10, 10]  # centre (5,5) inside both
    cam = layout.assign_box(box)
    assert cam.name == "cam_big"


def test_assign_boxes_batch():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=10, y_max=10),
            Camera(name="cam1", x_min=10, y_min=0, x_max=20, y_max=10),
        ],
        frame_width=20,
        frame_height=10,
    )
    boxes = np.array([[1, 1, 3, 3], [11, 1, 13, 3], [100, 100, 105, 105]])
    result = layout.assign_boxes(boxes)
    assert [c.name if c else None for c in result] == ["cam0", "cam1", None]


# ---------------------------------------------------------------------------
# CameraLayout.get / names / __len__ / __iter__
# ---------------------------------------------------------------------------


def test_get_raises_keyerror_for_unknown_name():
    layout = CameraLayout.full_frame(10, 10)
    with pytest.raises(KeyError):
        layout.get("nope")


def test_len_and_iter():
    layout = CameraLayout(
        cameras=[
            Camera(name="cam0", x_min=0, y_min=0, x_max=5, y_max=10),
            Camera(name="cam1", x_min=5, y_min=0, x_max=10, y_max=10),
        ],
        frame_width=10,
        frame_height=10,
    )
    assert len(layout) == 2
    assert [c.name for c in layout] == ["cam0", "cam1"]


# ---------------------------------------------------------------------------
# find_cameras_file / sibling_cameras_path
# ---------------------------------------------------------------------------


def test_sibling_cameras_path_naming(tmp_path):
    video = tmp_path / "myvideo.mp4"
    assert sibling_cameras_path(video) == tmp_path / "myvideo_cameras.json"


def test_find_cameras_file_explicit_must_exist(tmp_path):
    video = tmp_path / "video.mp4"
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        find_cameras_file(video, explicit=missing)


def test_find_cameras_file_explicit_precedence(tmp_path):
    video = tmp_path / "video.mp4"
    sibling = sibling_cameras_path(video)
    CameraLayout.full_frame(10, 10).save(sibling)

    explicit = tmp_path / "explicit_cameras.json"
    CameraLayout.full_frame(20, 20).save(explicit)

    found = find_cameras_file(video, explicit=explicit)
    assert found == explicit


def test_find_cameras_file_falls_back_to_sibling(tmp_path):
    video = tmp_path / "video.mp4"
    sibling = sibling_cameras_path(video)
    CameraLayout.full_frame(10, 10).save(sibling)

    found = find_cameras_file(video)
    assert found == sibling


def test_find_cameras_file_none_when_nothing_found(tmp_path):
    video = tmp_path / "video.mp4"
    assert find_cameras_file(video) is None


# ---------------------------------------------------------------------------
# resolve_layout
# ---------------------------------------------------------------------------


def test_resolve_layout_falls_back_to_full_frame(tmp_path):
    video = tmp_path / "video.mp4"
    layout, explicit = resolve_layout(video, frame_width=64, frame_height=48)
    assert explicit is False
    assert layout.is_single_full_frame()
    assert layout.frame_width == 64
    assert layout.frame_height == 48


def test_resolve_layout_uses_sibling_when_present(tmp_path):
    video = tmp_path / "video.mp4"
    sibling = sibling_cameras_path(video)
    CameraLayout.full_frame(64, 48, name="cam_sib").save(sibling)

    layout, explicit = resolve_layout(video, frame_width=64, frame_height=48)
    assert explicit is True
    assert layout.names == ["cam_sib"]


def test_resolve_layout_accepts_path(tmp_path):
    video = tmp_path / "video.mp4"
    explicit_path = tmp_path / "custom_cameras.json"
    CameraLayout.full_frame(64, 48, name="cam_x").save(explicit_path)

    layout, explicit = resolve_layout(
        video, frame_width=64, frame_height=48, cameras=explicit_path
    )
    assert explicit is True
    assert layout.names == ["cam_x"]


def test_resolve_layout_accepts_dict_and_layout_object(tmp_path):
    video = tmp_path / "video.mp4"
    layout_obj = CameraLayout.full_frame(64, 48, name="cam_obj")
    resolved, explicit = resolve_layout(
        video, frame_width=64, frame_height=48, cameras=layout_obj
    )
    assert explicit is True
    assert resolved is layout_obj

    d = layout_obj.to_dict()
    resolved2, explicit2 = resolve_layout(
        video, frame_width=64, frame_height=48, cameras=d
    )
    assert explicit2 is True
    assert resolved2.names == ["cam_obj"]


def test_resolve_layout_dimension_mismatch_raises(tmp_path):
    video = tmp_path / "video.mp4"
    sibling = sibling_cameras_path(video)
    CameraLayout.full_frame(64, 48).save(sibling)

    with pytest.raises(ValueError):
        resolve_layout(video, frame_width=100, frame_height=100)


# ---------------------------------------------------------------------------
# apply_cameras
# ---------------------------------------------------------------------------


def test_apply_cameras_writes_to_project_subfolders(tmp_path):
    cameras_path = tmp_path / "cameras.json"
    CameraLayout.full_frame(10, 10).save(cameras_path)

    project = tmp_path / "project"
    sub = project / "abc12345"
    sub.mkdir(parents=True)
    (sub / "object_organizer.json").write_text("{}")

    # A subfolder without object_organizer.json must be skipped.
    other = project / "not_a_run"
    other.mkdir()

    written = apply_cameras(cameras_path, project_path=project)
    assert (sub / "cameras.json") in written
    assert (sub / "cameras.json").exists()
    assert not (other / "cameras.json").exists()


def test_apply_cameras_writes_video_siblings(tmp_path):
    cameras_path = tmp_path / "cameras.json"
    CameraLayout.full_frame(10, 10).save(cameras_path)

    video = tmp_path / "vids" / "run1.mp4"
    video.parent.mkdir()

    written = apply_cameras(cameras_path, videos=[video])
    expected = sibling_cameras_path(video)
    assert expected in written
    assert expected.exists()


def test_apply_cameras_skips_existing_unless_force(tmp_path):
    cameras_path = tmp_path / "cameras.json"
    CameraLayout.full_frame(10, 10).save(cameras_path)

    video = tmp_path / "run1.mp4"
    dest = sibling_cameras_path(video)
    dest.write_text("existing content")

    written = apply_cameras(cameras_path, videos=[video])
    assert written == []
    assert dest.read_text() == "existing content"

    written_forced = apply_cameras(cameras_path, videos=[video], force=True)
    assert dest in written_forced
    assert dest.read_text() != "existing content"


def test_apply_cameras_validates_source_before_writing(tmp_path):
    bad_path = tmp_path / "bad_cameras.json"
    bad_layout_dict = {
        "version": 1,
        "frame_width": 10,
        "frame_height": 10,
        "video_file_path": None,
        "video_hash": None,
        "cameras": [],  # invalid: no cameras
    }
    bad_path.write_text(json.dumps(bad_layout_dict))

    video = tmp_path / "run1.mp4"
    with pytest.raises(ValueError):
        apply_cameras(bad_path, videos=[video])
    assert not sibling_cameras_path(video).exists()


def test_split_video_builds_even_crops(tmp_path, monkeypatch):
    from pathlib import Path

    import octron.cameras as cameras_mod

    video = tmp_path / "mosaic.mp4"
    video.write_bytes(b"")
    layout = CameraLayout(
        cameras=[Camera("a", 0, 0, 41, 31), Camera("b", 41, 0, 80, 30)],
        frame_width=80,
        frame_height=31,
    )
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"")

    monkeypatch.setattr(cameras_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "octron.tools._ffmpeg.resolve_encoder", lambda e: "libx264"
    )
    out = cameras_mod.split_video(video, layout, output_dir=tmp_path / "v")
    assert [p.name for p in out] == ["a.mp4", "b.mp4"]
    assert "crop=40:30:0:0" in calls[0]  # odd 41x31 rounded down
    assert "crop=38:30:41:0" in calls[1]
    # existing clips are skipped unless overwrite
    assert (
        cameras_mod.split_video(video, layout, output_dir=tmp_path / "v") == []
    )
    assert (
        len(
            cameras_mod.split_video(
                video, layout, output_dir=tmp_path / "v", overwrite=True
            )
        )
        == 2
    )
