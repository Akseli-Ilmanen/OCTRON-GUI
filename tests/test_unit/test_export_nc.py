"""NetCDF export of linked tracks, one file per camera (octron export-nc)."""

import json

import numpy as np
import pandas as pd
import pytest

from octron.cameras import Camera, CameraLayout

xr = pytest.importorskip("xarray")

from octron.tools.export_nc import (  # noqa: E402
    build_camera_dataset,
    build_datasets,
    load_assigned_tracks,
    run_export_nc,
)

HEADER = (
    "video_name: mosaic.mp4\nframe_count: 20\nframe_count_analyzed: 20\n"
    "video_height: 30\nvideo_width: 40\ncreated_at: now\n\n"
)


def _track(folder, label, tid, frames, x, y, conf=0.9, ident_conf=0.8):
    rows = [
        {
            "frame_counter": i,
            "frame_idx": f,
            "track_id": tid,
            "label": label,
            "confidence": conf,
            "pos_x": x,
            "pos_y": y,
            "bbox_area": 100.0,
            "bbox_aspect_ratio": 1.0,
            "bbox_x_min": x - 5,
            "bbox_x_max": x + 5,
            "bbox_y_min": y - 4,
            "bbox_y_max": y + 4,
            "identity_conf": ident_conf,
        }
        for i, f in enumerate(frames)
    ]
    df = pd.DataFrame(rows).set_index(
        ["frame_counter", "frame_idx", "track_id"]
    )
    with open(folder / f"{label}_track_{tid}.csv", "w") as fh:
        fh.write(HEADER)
        df.to_csv(fh, lineterminator="\n")


def _assign(folder, rows):
    cols = [
        "track_id",
        "label",
        "identity",
        "score",
        "runner_up",
        "runner_up_score",
        "margin_ratio",
        "n_frames",
        "first_frame",
        "last_frame",
        "flagged",
        "reason",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(
        folder / "identity_assignment.csv", index=False
    )


def _row(tid, identity, flagged=False, reason=""):
    return [tid, "bird", identity, 9, None, 1, 9, 10, 0, 9, flagged, reason]


@pytest.fixture
def mosaic(tmp_path):
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
                    "video_info": {
                        "original_video_name": "mosaic.mp4",
                        "original_video_path": "C:/x/mosaic.mp4",
                        "num_frames_original": 20,
                        "fps_original": 25.0,
                        "width": cam.width,
                        "height": cam.height,
                        "mosaic_width": 80,
                        "mosaic_height": 30,
                    }
                },
                f,
            )
    left, right = root / "left", root / "right"
    _track(left, "bird", 1, range(0, 10), 10.0, 20.0)
    _track(left, "bird", 2, range(12, 20), 11.0, 21.0)  # same bird later
    _track(left, "bird", 3, range(0, 5), 30.0, 5.0)  # unassigned
    _assign(
        left,
        [
            _row(1, "bird_male"),
            _row(2, "bird_male"),
            _row(3, None, True, "no_identity_scores"),
        ],
    )
    _track(right, "bird", 1, range(0, 20), 15.0, 15.0, conf=0.7)
    _assign(right, [_row(1, "bird_female", True, "low_margin")])
    return root


def test_load_assigned_tracks_skips_unassigned(mosaic):
    table, n_un, n_fl = load_assigned_tracks(mosaic / "left")
    assert n_un == 1 and n_fl == 1
    assert sorted(table["track_id"].unique()) == [1, 2]
    assert set(table["identity"]) == {"bird_male"}


def test_build_datasets_one_per_camera_shared_individuals(mosaic):
    dss = build_datasets(mosaic)
    assert list(dss) == ["left", "right"]
    for name, ds in dss.items():
        assert dict(ds.sizes) == {"time": 20, "space": 2, "individual": 2}
        assert list(ds["individual"].values) == ["bird_female", "bird_male"]
        assert ds.attrs["camera"] == name
        assert ds.attrs["ds_type"] == "bboxes" and ds.attrs["fps"] == 25.0
        assert ds.attrs["time_unit"] == "seconds"
    left = dss["left"]
    assert np.isclose(left["time"].values[1], 0.04)
    male = left["position"].sel(individual="bird_male")
    assert np.isclose(male.sel(space="x").values[0], 10.0)
    assert np.isnan(male.sel(space="x").values[11])  # gap between tracklets
    assert np.isclose(male.sel(space="x").values[15], 11.0)
    assert left["track_id"].sel(individual="bird_male").values[15] == 2
    assert left["track_id"].sel(individual="bird_female").values[0] == -1
    assert left["flagged"].sel(individual="bird_male").values[0] == 0
    assert left.attrs["n_unassigned_tracklets"] == 1
    assert (left.attrs["camera_x_min"], left.attrs["camera_x_max"]) == (0, 40)
    right = dss["right"]
    shape = right["shape"].sel(individual="bird_female")
    assert np.isclose(shape.sel(space="x").values[0], 10.0)
    assert np.isclose(shape.sel(space="y").values[0], 8.0)
    assert right["flagged"].sel(individual="bird_female").values[0] == 1
    assert right.attrs["camera_x_min"] == 40


def test_run_export_nc_writes_one_file_per_camera(mosaic):
    written = run_export_nc(mosaic)
    assert [p.name for p in written] == ["left.nc", "right.nc"]
    with xr.open_dataset(written[1]) as ds:
        assert ds["position"].dims == ("time", "space", "individual")
        assert ds.attrs["source_software"] == "OCTRON"
        assert ds.attrs["camera"] == "right"
    with pytest.raises(FileExistsError):
        run_export_nc(mosaic)
    run_export_nc(mosaic, overwrite=True)


def test_single_camera_folder_exports_itself(mosaic):
    ds = build_camera_dataset(mosaic / "right", ["bird_female"])
    assert ds.attrs["camera"] == "right"
    assert "camera_x_min" not in ds.attrs
    written = run_export_nc(mosaic / "right")
    assert written[0] == mosaic / "right" / "right.nc"
