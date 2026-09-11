"""Export linked multi-camera tracks to one NetCDF dataset.

After ``octron predict --cameras ... --identity ...`` and ``octron link``,
every camera folder holds anonymous tracklets plus an
``identity_assignment.csv``. This module writes one
`movement <https://movement.neuroinformatics.dev>`_-style bounding-box
dataset **per camera**, named ``<camera>.nc``::

    Dimensions:   (time, space, individual)
    position      (time, space, individual)  box centre, pixels
    shape         (time, space, individual)  box width / height
    confidence    (time, individual)         detector confidence
    identity_conf (time, individual)         classifier score
    track_id      (time, individual)         source tracklet, -1 = none
    flagged       (time, individual)         1 = flagged by octron link

Coordinates are ``time`` (seconds when fps is known, else frames, with a
``frame_idx`` coordinate alongside), ``space`` (``x``, ``y``) and
``individual`` (identity class names from the classifier, e.g.
``bird_male``; the same axis in every file of one video). Positions are
in **camera-crop pixels**; the camera name and its rectangle in the
mosaic are stored in the attributes (``camera``, ``camera_x_min`` ...).

Only tracklets that ``link`` assigned are exported; unassigned ones are
counted in the attributes. Assigned-but-flagged tracklets are exported
with ``flagged = 1`` so a consumer can filter them.

Requires ``xarray`` and a NetCDF backend (``pip install octron[export]``).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from octron.cameras import load_layout_for_folder
from octron.tools.link import CSV_HEADER_LINES

SPACE = ["x", "y"]


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "NetCDF export needs xarray and netCDF4: "
            "pip install 'octron[export]'"
        ) from e
    return xr


def _read_metadata(folder):
    path = Path(folder) / "prediction_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _camera_folders(source):
    """Return ``(parent, layout, [(camera_name, folder), ...])``.

    ``source`` is the prediction folder holding ``cameras.json`` and one
    subfolder per camera, or a single camera folder (exported as one
    camera named after the folder).
    """
    source = Path(source)
    layout = load_layout_for_folder(source)
    if layout is not None:
        folders = [
            (cam.name, source / cam.name)
            for cam in layout
            if (source / cam.name).is_dir()
        ]
        if not folders:
            raise FileNotFoundError(
                f"No camera subfolders found under {source}"
            )
        return source, layout, folders
    if list(source.glob("*_track_*.csv")):
        return source.parent, None, [(source.name, source)]
    raise FileNotFoundError(
        f"{source} is neither a mosaic prediction folder (cameras.json + "
        "camera subfolders) nor a folder with tracklet CSVs."
    )


def load_assigned_tracks(folder):
    """Read one camera folder into a table of assigned observations.

    Returns
    -------
    pd.DataFrame
        Columns ``frame_idx, identity, track_id, x, y, w, h,
        confidence, identity_conf`` for every frame of every tracklet
        that ``identity_assignment.csv`` assigned; empty when the
        assignment file is missing.
    n_unassigned : int
        Tracklets present in the folder but not assigned.
    n_flagged : int
        Tracklets flagged for review (assigned or not).

    """
    folder = Path(folder)
    assign_path = folder / "identity_assignment.csv"
    cols = [
        "frame_idx",
        "identity",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "confidence",
        "identity_conf",
        "flagged",
    ]
    if not assign_path.exists():
        logger.warning(
            f"{folder.name}: no identity_assignment.csv, run 'octron "
            "link' first; exporting nothing for this camera."
        )
        return pd.DataFrame(columns=cols), 0, 0
    assignment = pd.read_csv(assign_path)
    assigned = assignment[assignment["identity"].notna()]
    n_unassigned = int(assignment["identity"].isna().sum())
    n_flagged = int(assignment["flagged"].astype(bool).sum())
    lookup = dict(zip(assigned["track_id"], assigned["identity"], strict=True))
    flagged_ids = set(
        assignment.loc[assignment["flagged"].astype(bool), "track_id"]
    )

    frames = []
    for csv_file in sorted(folder.glob("*_track_*.csv")):
        try:
            df = pd.read_csv(csv_file, skiprows=CSV_HEADER_LINES)
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")
            continue
        if df.empty:
            continue
        track_id = int(df.iloc[0]["track_id"])
        if track_id not in lookup:
            continue
        out = pd.DataFrame(
            {
                "frame_idx": df["frame_idx"].astype(int),
                "identity": lookup[track_id],
                "track_id": track_id,
                "x": (df["bbox_x_min"] + df["bbox_x_max"]) / 2,
                "y": (df["bbox_y_min"] + df["bbox_y_max"]) / 2,
                "w": df["bbox_x_max"] - df["bbox_x_min"],
                "h": df["bbox_y_max"] - df["bbox_y_min"],
                "confidence": df["confidence"],
                "identity_conf": df["identity_conf"]
                if "identity_conf" in df.columns
                else np.nan,
                "flagged": track_id in flagged_ids,
            }
        )
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=cols), n_unassigned, n_flagged
    table = pd.concat(frames, ignore_index=True)
    # One observation per (frame, identity): keep the most confident
    # when two tracklets of one identity overlap despite linking.
    table = table.sort_values("confidence", ascending=False)
    table = table.drop_duplicates(["frame_idx", "identity"], keep="first")
    return table.sort_values("frame_idx"), n_unassigned, n_flagged


def _parse_fps(video_info, fps):
    if fps is not None:
        return float(fps)
    raw = video_info.get("fps_original")
    try:
        return float(raw) if raw not in (None, "unknown") else None
    except (TypeError, ValueError):
        return None


def build_camera_dataset(
    folder,
    individuals,
    camera=None,
    fps=None,
    num_frames=None,
):
    """Assemble the movement-style dataset of one camera folder.

    Parameters
    ----------
    folder : str or Path
        Camera folder with tracklet CSVs and ``identity_assignment.csv``.
    individuals : list of str
        Individual names and order (shared across the cameras of one
        video so the files align).
    camera : octron.cameras.Camera, optional
        Rectangle of this camera in the mosaic, stored in the attributes.
    fps : float, optional
        Frame rate override; default from the prediction metadata.
    num_frames : int, optional
        Length of the time axis; default from the prediction metadata.

    Returns
    -------
    xarray.Dataset
        Dimensions ``(time, space, individual)``.

    """
    xr = _require_xarray()
    folder = Path(folder)
    table, n_un, n_fl = load_assigned_tracks(folder)
    meta = _read_metadata(folder)
    video_info = meta.get("video_info", {})
    if num_frames is None:
        num_frames = int(video_info.get("num_frames_original", 0) or 0)
    if num_frames == 0:
        num_frames = int(table["frame_idx"].max() + 1) if len(table) else 0
    fps = _parse_fps(video_info, fps)

    shape3 = (num_frames, len(SPACE), len(individuals))
    shape2 = (num_frames, len(individuals))
    position = np.full(shape3, np.nan)
    box_shape = np.full(shape3, np.nan)
    confidence = np.full(shape2, np.nan)
    identity_conf = np.full(shape2, np.nan)
    track_id = np.full(shape2, -1, dtype=np.int32)
    flagged = np.zeros(shape2, dtype=np.int8)
    if len(table):
        ind_index = {n: i for i, n in enumerate(individuals)}
        t = table[table["identity"].isin(ind_index)]
        t = t[t["frame_idx"] < num_frames]
        f = t["frame_idx"].to_numpy()
        i = t["identity"].map(ind_index).to_numpy()
        position[f, 0, i] = t["x"].to_numpy()
        position[f, 1, i] = t["y"].to_numpy()
        box_shape[f, 0, i] = t["w"].to_numpy()
        box_shape[f, 1, i] = t["h"].to_numpy()
        confidence[f, i] = t["confidence"].to_numpy()
        identity_conf[f, i] = t["identity_conf"].to_numpy(dtype=float)
        track_id[f, i] = t["track_id"].to_numpy()
        flagged[f, i] = t["flagged"].to_numpy(dtype=bool)

    frame_idx = np.arange(num_frames)
    if fps:
        time = frame_idx / fps
        time_unit = "seconds"
    else:
        time = frame_idx.astype(float)
        time_unit = "frames"
    ds = xr.Dataset(
        {
            "position": (("time", "space", "individual"), position),
            "shape": (("time", "space", "individual"), box_shape),
            "confidence": (("time", "individual"), confidence),
            "identity_conf": (("time", "individual"), identity_conf),
            "track_id": (("time", "individual"), track_id),
            "flagged": (("time", "individual"), flagged),
        },
        coords={
            "time": ("time", time),
            "frame_idx": ("time", frame_idx),
            "space": ("space", SPACE),
            "individual": ("individual", list(individuals)),
        },
    )
    ds["position"].attrs["units"] = "pixels (camera crop)"
    ds["shape"].attrs["units"] = "pixels (camera crop)"
    ds["track_id"].attrs["description"] = (
        "OCTRON tracklet id within the camera folder; -1 = no observation"
    )
    ds["flagged"].attrs["description"] = (
        "1 when the source tracklet was flagged for review by octron link "
        "(low identity margin); filter with ds.where(ds.flagged == 0)"
    )
    attrs = {
        "source_software": "OCTRON",
        "ds_type": "bboxes",
        "time_unit": time_unit,
        "camera": camera.name if camera is not None else folder.name,
        "video_name": str(video_info.get("original_video_name", "")),
        "video_path": str(video_info.get("original_video_path", "")),
        "frame_width": int(video_info.get("width", 0) or 0),
        "frame_height": int(video_info.get("height", 0) or 0),
        "mosaic_width": int(video_info.get("mosaic_width", 0) or 0),
        "mosaic_height": int(video_info.get("mosaic_height", 0) or 0),
        "n_unassigned_tracklets": int(n_un),
        "n_flagged_tracklets": int(n_fl),
        "prediction_folder": str(folder),
    }
    if camera is not None:
        attrs.update(
            {
                "camera_x_min": int(camera.x_min),
                "camera_y_min": int(camera.y_min),
                "camera_x_max": int(camera.x_max),
                "camera_y_max": int(camera.y_max),
            }
        )
    if fps:
        attrs["fps"] = float(fps)
    ds.attrs.update(attrs)
    return ds


def build_datasets(source, fps=None, individuals=None):
    """Build one dataset per camera of a prediction folder.

    Parameters
    ----------
    source : str or Path
        Mosaic prediction folder (with ``cameras.json``) or one camera
        folder.
    fps : float, optional
        Frame rate override for the time axes.
    individuals : list of str, optional
        Individual names; default is every identity assigned in any
        camera, sorted, so all files of one video share the same axis.

    Returns
    -------
    dict
        ``{camera_name: xarray.Dataset}`` in layout order.

    """
    parent, layout, folders = _camera_folders(source)
    if individuals is None:
        names = set()
        for _, folder in folders:
            table, _, _ = load_assigned_tracks(folder)
            names.update(str(i) for i in table["identity"].unique())
        individuals = sorted(names)
    cams = {cam.name: cam for cam in layout} if layout is not None else {}
    return {
        name: build_camera_dataset(
            folder, individuals, camera=cams.get(name), fps=fps
        )
        for name, folder in folders
    }


def run_export_nc(source, output_dir=None, fps=None, overwrite=False):
    """Write one NetCDF file per camera for a prediction folder.

    Files are named ``<camera>.nc`` and written next to the camera
    folders (``output_dir`` defaults to the prediction folder; for a
    single camera folder, to that folder).

    Parameters
    ----------
    source : str or Path
        Mosaic prediction folder or single camera folder.
    output_dir : str or Path, optional
        Directory for the ``.nc`` files.
    fps : float, optional
        Frame rate override for the time axis.
    overwrite : bool
        Replace existing files.

    Returns
    -------
    list of Path
        The written files, in camera order.

    """
    source = Path(source)
    datasets = build_datasets(source, fps=fps)
    if output_dir is None:
        output_dir = source
    output_dir = Path(output_dir)
    targets = {name: output_dir / f"{name}.nc" for name in datasets}
    existing = [p for p in targets.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} exists. Pass overwrite=True (--overwrite)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, ds in datasets.items():
        path = targets[name]
        ds.to_netcdf(path)
        written.append(path)
        n_obs = int(np.isfinite(ds["confidence"].values).sum())
        print(
            f"Wrote {path} | {ds.sizes['time']} frames, "
            f"{ds.sizes['individual']} individual(s) "
            f"({', '.join(map(str, ds['individual'].values))}), "
            f"{n_obs} observations; "
            f"{ds.attrs['n_unassigned_tracklets']} unassigned tracklets "
            f"not exported, {ds.attrs['n_flagged_tracklets']} flagged "
            f"(kept, see 'flagged')."
        )
    return written
