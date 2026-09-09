"""MosaicPredictor: per-camera SAM states behind one mosaic video."""

import numpy as np
import pytest
import torch

from octron.cameras import Camera, CameraLayout
from octron.sam_octron.helpers.mosaic_predictor import (
    LOGIT_BACKGROUND,
    CameraView,
    MosaicPredictor,
)


class FakeImages:
    """Stand-in for OctoZarr: records which indices were fetched."""

    def __init__(self, shape):
        """Remember the crop shape and start with no fetches."""
        self.shape = shape
        self.fetched = []

    def __getitem__(self, indices):
        """Record the request."""
        self.fetched.append(list(indices))
        return None


class FakePredictor:
    """Mimics OCTRON's SAM predictors: state lives on self.

    Every prompt makes a mask that is a filled box around the prompt in
    the camera crop; propagation replays each object's last mask on
    every requested frame.
    """

    image_size = 1024
    device = "cpu"

    def __init__(self):
        """Start uninitialised."""
        self.inference_state = None
        self.images = None
        self.init_calls = []

    def init_state(self, video_data, zarr_store):
        """Create a fresh per-video state (shape from the crop)."""
        n, h, w, _ = video_data.shape
        self.init_calls.append((video_data.shape, zarr_store))
        self.inference_state = {
            "num_frames": n,
            "video_height": h,
            "video_width": w,
            "obj_ids": [],
            "obj_id_to_idx": {},
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "output_dict_per_obj": {},
            "tracking_has_started": False,
            "last_mask": {},
        }
        self.images = FakeImages(video_data.shape)

    def _register(self, obj_id, frame_idx, kind):
        st = self.inference_state
        if obj_id not in st["obj_ids"]:
            st["obj_ids"].append(obj_id)
            st["obj_id_to_idx"][obj_id] = len(st["obj_ids"]) - 1
        idx = st["obj_id_to_idx"][obj_id]
        st[kind].setdefault(idx, {})[frame_idx] = True
        st["output_dict_per_obj"].setdefault(idx, {"cond_frame_outputs": {}})[
            "cond_frame_outputs"
        ][frame_idx] = True

    def _all_masks(self):
        st = self.inference_state
        h, w = st["video_height"], st["video_width"]
        out = torch.full(
            (len(st["obj_ids"]), 1, h, w),
            LOGIT_BACKGROUND,
            dtype=torch.float32,
        )
        for i, oid in enumerate(st["obj_ids"]):
            out[i, 0] = st["last_mask"][oid]
        return out

    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Return a box mask around the prompt (camera coordinates)."""
        st = self.inference_state
        h, w = st["video_height"], st["video_width"]
        if box is None:
            pts = np.asarray(points).reshape(-1, 2)
            x, y = pts[0]
            box = [max(x - 2, 0), max(y - 2, 0), min(x + 3, w), min(y + 3, h)]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, (box, w, h)
        m = torch.full((h, w), LOGIT_BACKGROUND)
        m[y0:y1, x0:x1] = 5.0
        self._register(obj_id, frame_idx, "point_inputs_per_obj")
        st["last_mask"][obj_id] = m
        return frame_idx, list(st["obj_ids"]), self._all_masks()

    def add_new_mask(self, frame_idx, obj_id, mask):
        """Accept a camera-sized mask verbatim."""
        st = self.inference_state
        assert mask.shape == (st["video_height"], st["video_width"])
        m = torch.where(
            torch.from_numpy(np.asarray(mask, dtype=bool)),
            torch.tensor(5.0),
            torch.tensor(LOGIT_BACKGROUND),
        )
        self._register(obj_id, frame_idx, "mask_inputs_per_obj")
        st["last_mask"][obj_id] = m
        return frame_idx, list(st["obj_ids"]), self._all_masks()

    def propagate_in_video(self, processing_order=None, reverse=False):
        """Replay the last masks on every frame."""
        st = self.inference_state
        st["tracking_has_started"] = True
        for f in processing_order:
            yield f, list(st["obj_ids"]), self._all_masks()

    def reset_state(self):
        """Forget every object."""
        st = self.inference_state
        for k in ("obj_ids",):
            st[k].clear()
        for k in (
            "obj_id_to_idx",
            "point_inputs_per_obj",
            "mask_inputs_per_obj",
            "output_dict_per_obj",
            "last_mask",
        ):
            st[k].clear()
        st["tracking_has_started"] = False

    def remove_object(self, obj_id, strict=False, need_output=True):
        """Drop one object."""
        st = self.inference_state
        if obj_id not in st["obj_id_to_idx"]:
            if strict:
                raise RuntimeError("unknown")
            return st["obj_ids"], []
        st["obj_ids"].remove(obj_id)
        st["obj_id_to_idx"] = {o: i for i, o in enumerate(st["obj_ids"])}
        st["last_mask"].pop(obj_id, None)
        return st["obj_ids"], []


@pytest.fixture
def layout():
    return CameraLayout(
        cameras=[
            Camera("left", 0, 0, 40, 30),
            Camera("right", 40, 0, 80, 30),
        ],
        frame_width=80,
        frame_height=30,
    )


@pytest.fixture
def video():
    v = np.zeros((5, 30, 80, 3), dtype=np.uint8)
    v[:, :, 40:, :] = 200  # right half bright
    return v


@pytest.fixture
def mosaic(layout, video):
    inner = FakePredictor()
    m = MosaicPredictor(inner, layout)
    m.init_state(video, {"left": "zl", "right": "zr"})
    return m


def test_camera_view_crops_lazily(video, layout):
    view = CameraView(video, layout.get("right"))
    assert view.shape == (5, 30, 40, 3)
    assert view[0].shape == (30, 40, 3)
    assert view[0][0, 0, 0] == 200
    assert view[[1, 2]].shape == (2, 30, 40, 3)


def test_init_state_creates_one_state_per_camera(mosaic):
    inner = mosaic.predictor
    assert [c[0] for c in inner.init_calls] == [(5, 30, 40, 3), (5, 30, 40, 3)]
    assert [c[1] for c in inner.init_calls] == ["zl", "zr"]
    assert mosaic.is_initialized
    assert mosaic.image_size == 1024  # delegated attribute


def test_point_prompt_routes_to_camera_and_pastes_back(mosaic):
    frame_idx, obj_ids, masks = mosaic.add_new_points_or_box(
        frame_idx=0, obj_id=7, points=[[50, 10]], labels=[1]
    )
    assert (frame_idx, obj_ids) == (0, [7])
    assert masks.shape == (1, 1, 30, 80)
    fg = (masks[0, 0] > 0).numpy()
    assert fg[10, 50] and not fg[10, 10]
    assert fg[:, :40].sum() == 0  # nothing pasted into the left camera
    assert mosaic.last_cameras[0].name == "right"
    # the object only exists in the right camera's state
    assert 7 in mosaic._states["right"][0]["obj_ids"]
    assert 7 not in mosaic._states["left"][0]["obj_ids"]


def test_box_prompt_is_shifted_into_camera_coordinates(mosaic):
    _, _, masks = mosaic.add_new_points_or_box(
        frame_idx=0, obj_id=1, box=[45, 5, 55, 15]
    )
    fg = (masks[0, 0] > 0).numpy()
    ys, xs = np.nonzero(fg)
    assert xs.min() == 45 and xs.max() == 54
    assert ys.min() == 5 and ys.max() == 14


def test_mask_prompt_splits_over_cameras(mosaic):
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[5:10, 5:10] = 1  # left
    mask[20:25, 60:70] = 1  # right
    _, obj_ids, masks = mosaic.add_new_mask(0, obj_id=3, mask=mask)
    assert obj_ids == [3]
    fg = (masks[0, 0] > 0).numpy()
    np.testing.assert_array_equal(fg, mask.astype(bool))
    assert sorted(c.name for c in mosaic.last_cameras) == ["left", "right"]
    assert 3 in mosaic._states["left"][0]["obj_ids"]
    assert 3 in mosaic._states["right"][0]["obj_ids"]


def test_merge_frame_mask_keeps_other_cameras(mosaic):
    existing = np.zeros((30, 80), dtype=np.int8)
    existing[2:4, 2:4] = 1  # left camera blob already there
    mosaic.add_new_points_or_box(0, obj_id=1, points=[[60, 10]], labels=[1])
    new = np.zeros((30, 80), dtype=np.uint8)
    new[8:13, 58:63] = 1
    merged = mosaic.merge_frame_mask(existing, new)
    assert merged[2, 2] == 1  # left untouched
    assert merged[10, 60] == 1  # right replaced


def test_propagate_yields_union_of_cameras(mosaic):
    mosaic.add_new_points_or_box(0, obj_id=1, points=[[10, 10]], labels=[1])
    mosaic.add_new_points_or_box(0, obj_id=2, points=[[60, 20]], labels=[1])
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[1:3, 70:72] = 1
    mosaic.add_new_mask(0, obj_id=3, mask=mask)  # both cameras
    out = list(mosaic.propagate_in_video(processing_order=[1, 2]))
    assert [f for f, _, _ in out] == [1, 2]
    frame, obj_ids, masks = out[0]
    assert sorted(obj_ids) == [1, 2, 3]
    assert masks.shape == (3, 1, 30, 80)
    fg = (masks > 0).numpy()[:, 0]
    i1, i2, i3 = (obj_ids.index(k) for k in (1, 2, 3))
    assert fg[i1][10, 10] and not fg[i1][20, 60]
    assert fg[i2][20, 60] and not fg[i2][10, 10]
    assert fg[i3][1, 1] and fg[i3][1, 70]
    assert mosaic.inference_state["tracking_has_started"]


def test_propagate_skips_cameras_without_inputs(mosaic):
    mosaic.add_new_points_or_box(0, obj_id=1, points=[[10, 10]], labels=[1])
    out = list(mosaic.propagate_in_video(processing_order=[0]))
    assert out[0][1] == [1]
    assert not mosaic._states["right"][0]["tracking_has_started"]


def test_inference_state_view_reports_inputs(mosaic):
    assert not mosaic.inference_state["point_inputs_per_obj"]
    mosaic.add_new_points_or_box(0, obj_id=1, points=[[10, 10]], labels=[1])
    assert mosaic.inference_state["point_inputs_per_obj"]
    assert mosaic.inference_state["obj_ids"] == [1]


def test_reset_and_remove(mosaic):
    mosaic.add_new_points_or_box(0, obj_id=1, points=[[10, 10]], labels=[1])
    mosaic.add_new_points_or_box(0, obj_id=2, points=[[60, 10]], labels=[1])
    mosaic.remove_object(1, strict=True)
    assert mosaic.inference_state["obj_ids"] == [2]
    with pytest.raises(RuntimeError):
        mosaic.remove_object(99, strict=True)
    mosaic.reset_state()
    assert mosaic.inference_state["obj_ids"] == []


def test_images_prefetches_every_camera(mosaic):
    mosaic.images[[0, 1]]
    for name in ("left", "right"):
        assert mosaic._states[name][1].fetched == [[0, 1]]


def test_prompt_outside_cameras_uses_nearest(layout, video):
    layout = CameraLayout(
        cameras=[Camera("a", 0, 0, 30, 30), Camera("b", 50, 0, 80, 30)],
        frame_width=80,
        frame_height=30,
    )
    m = MosaicPredictor(FakePredictor(), layout)
    m.init_state(video, {"a": 1, "b": 2})
    m.add_new_points_or_box(0, obj_id=1, points=[[45, 10]], labels=[1])
    assert m.last_cameras[0].name == "b"


def test_points_of_one_object_in_two_cameras_are_split(mosaic):
    # The GUI re-sends all points of the frame: one in each camera.
    _, obj_ids, masks = mosaic.add_new_points_or_box(
        frame_idx=0,
        obj_id=5,
        points=[[10, 10], [60, 20]],
        labels=[1, 1],
    )
    assert obj_ids == [5]
    fg = (masks[0, 0] > 0).numpy()
    assert fg[10, 10] and fg[20, 60]
    assert sorted(c.name for c in mosaic.last_cameras) == ["left", "right"]
    assert 5 in mosaic._states["left"][0]["obj_ids"]
    assert 5 in mosaic._states["right"][0]["obj_ids"]
    # each camera saw only its own point, in its own coordinates
    left_mask = mosaic._states["left"][0]["last_mask"][5]
    right_mask = mosaic._states["right"][0]["last_mask"][5]
    assert left_mask[10, 10] > 0 and right_mask[20, 20] > 0
