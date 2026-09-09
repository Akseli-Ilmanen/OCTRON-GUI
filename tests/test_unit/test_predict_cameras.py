"""Per-camera tracking inside predict (fake tracker, detection mode)."""

import numpy as np
import pandas as pd
import pytest

from octron.cameras import Camera, CameraLayout
from octron.yolo_octron.identity import IdentityClassifier
from octron.yolo_octron.yolo_octron import YOLO_octron, _CameraTrackingState


class FakeTracker:
    """Tracker that gives every detection a persistent id by x position."""

    def __init__(self):
        """Start with no calls recorded."""
        self.calls = 0

    def update(self, dets, frame):
        """Return BoxMOT-style rows: xyxy, id, conf, cls, det index."""
        self.calls += 1
        rows = []
        for i, det in enumerate(dets):
            track_id = 1 if det[0] < 200 else 2
            rows.append([*det[:4], track_id, det[4], det[5], i])
        return np.asarray(rows, dtype=float).reshape(-1, 8)


def _state(camera, tmp_path):
    video_dict = {
        "num_frames": 10,
        "num_frames_analyzed": 10,
        "height": camera.height,
        "width": camera.width,
        "video_name": "mosaic.mp4",
    }
    return _CameraTrackingState(
        camera=camera,
        save_dir=tmp_path / camera.name,
        video_dict=video_dict,
        tracker=FakeTracker(),
        prediction_store=None,
    )


def _core():
    return YOLO_octron.__new__(YOLO_octron)


def _track(
    core, state, frame_no, frame_idx, boxes, labels, probs=None, clf=None
):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    core._track_camera_frame(
        state,
        frame_no=frame_no,
        frame_idx=frame_idx,
        frame=np.zeros(
            (state.video_dict["height"], state.video_dict["width"], 3),
            np.uint8,
        ),
        boxes=boxes,
        confidences=np.full(n, 0.9),
        classes=np.zeros(n),
        label_names=list(labels),
        masks=None,
        identity_probs=probs,
        model_names={0: "bird"},
        is_segment=False,
        per_class=False,
        one_object_per_label=False,
        iou_thresh=0.7,
        opening_radius=0,
        region_details=False,
        region_properties=None,
        extra_properties=None,
        buffer_size=500,
        identity_clf=clf,
    )


def test_layout_assigns_detections_to_cameras_by_centre():
    layout = CameraLayout(
        cameras=[
            Camera("left", 0, 0, 400, 300),
            Camera("right", 400, 0, 800, 300),
        ],
        frame_width=800,
        frame_height=300,
    )
    boxes = np.array([[10, 10, 50, 50], [500, 10, 540, 50], [380, 0, 430, 40]])
    assigned = layout.assign_boxes(boxes)
    assert [c.name for c in assigned] == ["left", "right", "right"]
    # shift_box expresses the box in camera coordinates
    np.testing.assert_allclose(
        assigned[1].shift_box(boxes[1]), [100, 10, 140, 50]
    )


def test_track_camera_frame_writes_csv_in_camera_coordinates(tmp_path):
    camera = Camera("right", 400, 0, 800, 300)
    state = _state(camera, tmp_path)
    core = _core()
    _track(core, state, 0, 0, [[100, 10, 140, 50]], ["bird"])
    _track(
        core,
        state,
        1,
        1,
        [[110, 10, 150, 50], [300, 20, 340, 60]],
        ["bird", "bird"],
    )
    assert sorted(state.all_ids) == [1, 2]
    state.save_dir.mkdir(parents=True)
    state.save_tracking_csvs()
    csvs = sorted(state.save_dir.glob("*_track_*.csv"))
    assert [c.name for c in csvs] == ["bird_track_1.csv", "bird_track_2.csv"]
    df = pd.read_csv(csvs[0], skiprows=7)
    assert list(df["frame_idx"]) == [0, 1]
    assert df["bbox_x_min"].tolist() == [100, 110]
    assert df["pos_x"].tolist() == [120, 130]
    with open(csvs[0]) as f:
        header = f.readline()
    assert header.strip() == "video_name: mosaic.mp4"


def test_track_camera_frame_adds_identity_columns(tmp_path):
    camera = Camera("cam0", 0, 0, 400, 300)
    state = _state(camera, tmp_path)
    clf = IdentityClassifier.__new__(IdentityClassifier)
    clf.class_names = ["bird_1", "bird_2"]
    clf.classes = {
        "bird_1": {"label": "bird", "suffix": "1"},
        "bird_2": {"label": "bird", "suffix": "2"},
    }
    clf.prob_columns = ["identity_prob_bird_1", "identity_prob_bird_2"]
    probs = np.array([[0.2, 0.8]])
    _track(_core(), state, 0, 0, [[10, 10, 50, 50]], ["bird"], probs, clf)
    df = state.tracking_df_dict[1]
    row = df.loc[(0, 0, 1)]
    assert row["identity"] == "bird_2"
    assert row["identity_conf"] == pytest.approx(0.8)
    assert row["identity_prob_bird_1"] == pytest.approx(0.2)
    assert row["identity_prob_bird_2"] == pytest.approx(0.8)


def test_independent_states_keep_independent_track_ids(tmp_path):
    left = _state(Camera("left", 0, 0, 400, 300), tmp_path)
    right = _state(Camera("right", 400, 0, 800, 300), tmp_path)
    core = _core()
    _track(core, left, 0, 0, [[10, 10, 50, 50]], ["bird"])
    _track(core, right, 0, 0, [[10, 10, 50, 50]], ["bird"])
    assert left.all_ids == [1]
    assert right.all_ids == [1]
    assert left.tracker.calls == 1 and right.tracker.calls == 1


def test_label_change_splits_tracklet(tmp_path):
    state = _state(Camera("cam0", 0, 0, 400, 300), tmp_path)
    core = _core()
    _track(core, state, 0, 0, [[10, 10, 50, 50]], ["bird"])
    _track(core, state, 1, 1, [[10, 10, 50, 50]], ["mouse"])
    labels = {
        tid: df.attrs["label"] for tid, df in state.tracking_df_dict.items()
    }
    assert labels == {1: "bird", 2: "mouse"}
