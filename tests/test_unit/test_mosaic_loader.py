"""load_predictions on a mosaic output folder (per-camera subfolders)."""

import json

import numpy as np
import pandas as pd
import pytest

from octron.cameras import Camera, CameraLayout
from octron.yolo_octron.yolo_octron import YOLO_octron

HEADER = (
    "video_name: mosaic.mp4\nframe_count: 10\nframe_count_analyzed: 10\n"
    "video_height: {h}\nvideo_width: {w}\ncreated_at: now\n\n"
)


def _write_track(folder, label, track_id, frames, x, y, h, w):
    rows = []
    for i, f in enumerate(frames):
        rows.append(
            {
                "frame_counter": i,
                "frame_idx": f,
                "track_id": track_id,
                "label": label,
                "confidence": 0.9,
                "pos_x": x,
                "pos_y": y,
                "bbox_area": 100.0,
                "bbox_aspect_ratio": 1.0,
                "bbox_x_min": x - 5,
                "bbox_x_max": x + 5,
                "bbox_y_min": y - 5,
                "bbox_y_max": y + 5,
            }
        )
    df = pd.DataFrame(rows).set_index(
        ["frame_counter", "frame_idx", "track_id"]
    )
    with open(folder / f"{label}_track_{track_id}.csv", "w") as fh:
        fh.write(HEADER.format(h=h, w=w))
        df.to_csv(fh, lineterminator="\n")


@pytest.fixture
def mosaic_folder(tmp_path):
    root = tmp_path / "octron_predictions" / "mosaic_bytetrack"
    root.mkdir(parents=True)
    layout = CameraLayout(
        cameras=[Camera("left", 0, 0, 40, 30), Camera("right", 40, 0, 80, 30)],
        frame_width=80,
        frame_height=30,
    )
    layout.save(root / "cameras.json")
    for cam in layout:
        d = root / cam.name
        d.mkdir()
        with open(d / "prediction_metadata.json", "w") as f:
            json.dump(
                {
                    "model_classes": {"0": "bird"},
                    "video_info": {
                        "num_frames_original": 10,
                        "height": cam.height,
                        "width": cam.width,
                        "original_video_path": "missing.mp4",
                    },
                },
                f,
            )
        _write_track(
            d, "bird", 1, range(10), 10.0, 20.0, cam.height, cam.width
        )
    pd.DataFrame(
        {
            "track_id": [1],
            "label": ["bird"],
            "identity": ["bird_male"],
            "score": [9.0],
            "runner_up": ["bird_female"],
            "runner_up_score": [1.0],
            "margin_ratio": [9.0],
            "n_frames": [10],
            "first_frame": [0],
            "last_frame": [9],
            "flagged": [False],
            "reason": [""],
        }
    ).to_csv(root / "right" / "identity_assignment.csv", index=False)
    return root


def test_mosaic_tracks_are_offset_per_camera(mosaic_folder):
    core = YOLO_octron.__new__(YOLO_octron)
    core.project_path = None
    out = list(core.load_predictions(mosaic_folder, open_viewer=False))
    labels = sorted(o[0] for o in out)
    assert labels == ["left/bird", "right/bird"]
    by = {o[0]: o[3] for o in out}
    assert np.isclose(by["left/bird"]["pos_x"].iloc[0], 10.0)
    assert np.isclose(by["right/bird"]["pos_x"].iloc[0], 50.0)
    assert np.isclose(by["right/bird"]["pos_y"].iloc[0], 20.0)
    assert all(o[5] is None for o in out)  # detection output, no masks
