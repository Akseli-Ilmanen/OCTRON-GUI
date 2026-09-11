"""Self-training for the identity classifier (``octron refine-identity``).

The hand-annotated crops that train the identity classifier are few and
come from a handful of frames. After ``octron predict --identity`` and
``octron link``, every confidently linked tracklet is a long run of
crops with a known identity. This module turns those into extra
training crops and retrains the classifier on the union, the
"uniqueness feedback" loop of TRex (Walter & Couzin 2021, eLife).

Which frames are used
---------------------
Only *co-existence* frames: frames in which every individual of a
label has been assigned to a tracklet in that camera at the same time
(TRex's "global segments", Walter & Couzin 2021). There the linked
identities are decided by exclusivity and elimination, not by the
classifier's opinion of a single crop, and they are reliable even
where the classifier is systematically wrong. Measured on the Birdpark
mosaic (2026-09-11, all annotated frames): linked accuracy 0.86-1.00
in co-existence frames versus 0.37-0.88 when one bird was alone, where
the lone male was called female most of the time. Training on the
lone-bird frames poisoned one refine round; training on co-existence
frames cannot, and it is balanced by construction (every individual is
present in every used frame).

Frames are spread evenly over each tracklet (``max_per_tracklet``) so
one very long tracklet does not dominate the class. Pseudo crops are
written into the ``train`` split only; ``val`` and ``test`` stay
hand-labelled, and frames of the same video that belong to those
splits are never exported, so ``octron evaluate-identity`` on them
remains honest.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from octron.cameras import Camera
from octron.yolo_octron.identity import (
    DEFAULT_CROP_PADDING,
    IDENTITY_PROB_PREFIX,
    identity_paths,
    square_crop,
)

CSV_HEADER_LINES = 7
PSEUDO_PREFIX = "pseudo_"
IDENTITY_ASSIGNMENT_FILENAME = "identity_assignment.csv"
PREDICTION_METADATA_FILENAME = "prediction_metadata.json"
BOX_COLUMNS = ["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]


# ---------------------------------------------------------------------------
# Reading a prediction folder
# ---------------------------------------------------------------------------


def load_prediction_metadata(folder):
    """Return ``prediction_metadata.json`` of a prediction folder as a dict."""
    path = Path(folder) / PREDICTION_METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {PREDICTION_METADATA_FILENAME} in {folder}"
        )
    with open(path) as f:
        return json.load(f)


def folder_video_and_camera(folder):
    """Return ``(video_path, camera, padding)`` for a prediction folder.

    ``camera`` is a :class:`octron.cameras.Camera` for a mosaic camera
    folder and None for a full-frame prediction. ``padding`` is the
    identity crop padding used at prediction time (falls back to the
    dataset default when ``--identity`` was not used).
    """
    meta = load_prediction_metadata(folder)
    video_path = Path(meta["video_info"]["original_video_path"])
    camera = meta.get("camera")
    camera = Camera.from_dict(camera) if camera else None
    identity = meta.get("identity") or {}
    padding = float(identity.get("padding", DEFAULT_CROP_PADDING))
    return video_path, camera, padding


def load_tracklet_frames(folder):
    """Load every tracklet CSV in ``folder`` with its per-frame boxes.

    Returns
    -------
    pandas.DataFrame
        One row per observation with columns ``track_id, label,
        frame_idx, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max`` plus
        every ``identity_prob_*`` column present. Empty when the folder
        has no tracklets.

    """
    folder = Path(folder)
    frames = []
    for csv_file in sorted(folder.glob("*_track_*.csv")):
        try:
            df = pd.read_csv(csv_file, skiprows=CSV_HEADER_LINES)
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")
            continue
        if df.empty or "track_id" not in df.columns:
            continue
        keep = ["track_id", "label", "frame_idx", *BOX_COLUMNS]
        keep += [c for c in df.columns if c.startswith(IDENTITY_PROB_PREFIX)]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(
            columns=["track_id", "label", "frame_idx", *BOX_COLUMNS]
        )
    out = pd.concat(frames, ignore_index=True)
    out["track_id"] = out["track_id"].astype(int)
    out["frame_idx"] = out["frame_idx"].astype(int)
    return out


def load_identity_assignment(folder):
    """Return ``identity_assignment.csv`` indexed by ``track_id``."""
    path = Path(folder) / IDENTITY_ASSIGNMENT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {IDENTITY_ASSIGNMENT_FILENAME} in {folder}; run "
            "`octron link` first."
        )
    return pd.read_csv(path, index_col="track_id")


# ---------------------------------------------------------------------------
# Selecting trusted tracklets and frames
# ---------------------------------------------------------------------------


def _frame_sets(obs):
    """``{track_id: set(frame_idx)}`` from an observation dataframe."""
    return {
        int(tid): set(g["frame_idx"].tolist())
        for tid, g in obs.groupby("track_id")
    }


def select_pseudo_tracklets(assignment, obs=None):
    """Return the tracklets to train on: every assigned one.

    Parameters
    ----------
    assignment : pandas.DataFrame
        ``identity_assignment.csv`` indexed by ``track_id`` with an
        ``identity`` column.
    obs : pandas.DataFrame or None
        Unused; kept so callers may pass the observations.

    Returns
    -------
    selected : dict
        ``{track_id: identity}`` for every assigned tracklet.
    rejected : dict
        ``{track_id: "unassigned"}`` for the rest.

    """
    selected, rejected = {}, {}
    for tid, row in assignment.iterrows():
        tid = int(tid)
        identity = row.get("identity")
        if identity is None or (
            isinstance(identity, float) and np.isnan(identity)
        ):
            rejected[tid] = "unassigned"
        else:
            selected[tid] = str(identity)
    return selected, rejected


def select_pseudo_frames(
    track_obs,
    identity,
    max_per_tracklet=50,
    exclude_frames=None,
):
    """Choose which frames of one trusted tracklet become crops.

    Parameters
    ----------
    track_obs : pandas.DataFrame
        Observations of a single tracklet (rows of
        :func:`load_tracklet_frames` with one ``track_id``).
    identity : str
        Identity class name assigned to the tracklet (unused, kept for
        symmetry with the caller).
    max_per_tracklet : int
        At most this many frames, spread evenly over the tracklet, so
        one long tracklet does not dominate the class. ``0`` means no
        limit.
    exclude_frames : set of int or None
        Frames never to use (val/test frames of the same video).

    Returns
    -------
    pandas.DataFrame
        The chosen rows, in frame order.

    """
    obs = track_obs.sort_values("frame_idx")
    if exclude_frames:
        obs = obs[~obs["frame_idx"].isin(exclude_frames)]
    if max_per_tracklet and len(obs) > max_per_tracklet:
        pick = (
            np.linspace(0, len(obs) - 1, max_per_tracklet).round().astype(int)
        )
        obs = obs.iloc[np.unique(pick)]
    return obs


# ---------------------------------------------------------------------------
# Matching a prediction folder to the project video and its split
# ---------------------------------------------------------------------------


def find_project_subfolder(project_path, video_path):
    """Return the project subfolder that annotates ``video_path``, or None.

    Matches on the video file stem against every entry returned by
    :func:`octron.yolo_octron.identity.collect_identity_entries`.
    """
    from octron.yolo_octron.identity import collect_identity_entries

    stem = Path(video_path).stem
    entries, _ = collect_identity_entries(project_path)
    for entry in entries:
        if Path(entry["video_file_path"]).stem == stem:
            return entry["subfolder"]
    return None


def split_frames(data_path, subfolder, splits=("val", "test")):
    """Frames of ``subfolder`` that the identity dataset put in ``splits``.

    Recovered from the crop file names
    ``<split>/<class>/<subfolder>_<frame>[_<camera>].png`` written by
    :func:`octron.yolo_octron.identity.build_identity_dataset`.

    Returns
    -------
    dict
        ``{split_name: set(frame_idx)}``.

    """
    data_path = Path(data_path)
    prefix = f"{subfolder}_"
    out = {}
    for split in splits:
        frames = set()
        split_dir = data_path / split
        if split_dir.exists():
            for png in split_dir.glob("*/*.png"):
                name = png.stem
                if not name.startswith(prefix):
                    continue
                token = name[len(prefix) :].split("_")[0]
                if token.isdigit():
                    frames.add(int(token))
        out[split] = frames
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def clear_pseudo_crops(data_path):
    """Delete every ``pseudo_*`` crop from the train split.

    Returns the number of deleted files.
    """
    n = 0
    train = Path(data_path) / "train"
    if not train.exists():
        return 0
    for png in train.glob(f"*/{PSEUDO_PREFIX}*.png"):
        png.unlink()
        n += 1
    return n


def _open_video(video_path):
    from napari_pyav._reader import FastVideoReader

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return FastVideoReader(video_path, read_format="rgb24")


def coexistence_frames(obs, linked, individuals_per_label):
    """Frames in which every individual of a label is assigned at once.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observations (:func:`load_tracklet_frames`) of one camera.
    linked : dict
        ``{track_id: identity or None}`` from ``identity_assignment.csv``.
    individuals_per_label : dict
        ``{label: set of identity class names}`` known to the
        classifier (from ``prediction_metadata.json``).

    Returns
    -------
    set of int
        Frame indices where, for at least one label, all of its
        individuals are present (assigned to distinct tracklets).

    """
    if len(obs) == 0:
        return set()
    df = obs[["frame_idx", "track_id", "label"]].copy()
    df["identity"] = df["track_id"].map(linked)
    df = df.dropna(subset=["identity"])
    frames = set()
    for (frame_idx, label), g in df.groupby(["frame_idx", "label"]):
        wanted = individuals_per_label.get(str(label))
        if not wanted or len(wanted) < 2:
            continue
        if set(g["identity"]) >= wanted:
            frames.add(int(frame_idx))
    return frames


def _individuals_per_label(folder):
    """``{label: set(classname)}`` from ``prediction_metadata.json``."""
    meta = load_prediction_metadata(folder)
    out = {}
    for name, info in (meta.get("identity_classes") or {}).items():
        if isinstance(info, dict) and "label" in info:
            out.setdefault(str(info["label"]), set()).add(name)
    return out


def export_pseudo_crops(
    project_path,
    folders,
    max_per_tracklet=50,
    clear=False,
    coexistence_only=True,
    open_video=_open_video,
):
    """Write crops of assigned tracklets into the identity train split.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project (the identity dataset must exist).
    folders : list of str or Path
        Prediction folders (one per camera) that have been linked.
    max_per_tracklet : int
        See :func:`select_pseudo_frames`.
    clear : bool
        Remove previously exported pseudo crops first.
    coexistence_only : bool
        Use only frames in which every individual of a label is present
        in the camera (see :func:`coexistence_frames`). Default True.
    open_video : callable
        ``open_video(path) -> indexable frames`` (tests inject a stub).

    Returns
    -------
    dict
        ``n_crops`` per class, ``n_selected``/``n_rejected`` tracklets,
        ``rejected`` reasons per folder, ``n_cleared``.

    """
    from PIL import Image

    project_path = Path(project_path)
    _, data_path = identity_paths(project_path)
    if not (data_path / "train").exists():
        raise FileNotFoundError(
            f"No identity dataset at {data_path}; run "
            "`octron train-identity` first."
        )
    if isinstance(folders, (str, Path)):
        folders = [folders]
    folders = [Path(f) for f in folders]

    summary = {
        "n_crops": {},
        "n_selected": 0,
        "n_rejected": 0,
        "rejected": {},
        "n_cleared": clear_pseudo_crops(data_path) if clear else 0,
    }
    summary["n_coexistence_frames"] = {}
    for folder in folders:
        video_path, camera, padding = folder_video_and_camera(folder)
        obs = load_tracklet_frames(folder)
        assignment = load_identity_assignment(folder)
        selected, rejected = select_pseudo_tracklets(assignment, obs)
        summary["n_selected"] += len(selected)
        summary["n_rejected"] += len(rejected)
        summary["rejected"][folder.as_posix()] = rejected
        if not selected:
            logger.info(f"{folder.name}: no assigned tracklets")
            continue

        subfolder = find_project_subfolder(project_path, video_path)
        exclude = set()
        if subfolder is not None:
            held_out = split_frames(data_path, subfolder)
            exclude = held_out["val"] | held_out["test"]

        allowed = None
        if coexistence_only:
            linked = {tid: ident for tid, ident in selected.items()}
            allowed = coexistence_frames(
                obs, linked, _individuals_per_label(folder)
            )
            summary["n_coexistence_frames"][folder.as_posix()] = len(allowed)

        rows = []
        for tid, identity in selected.items():
            track_obs = obs[obs["track_id"] == tid]
            if allowed is not None:
                track_obs = track_obs[track_obs["frame_idx"].isin(allowed)]
            chosen = select_pseudo_frames(
                track_obs,
                identity,
                max_per_tracklet=max_per_tracklet,
                exclude_frames=exclude,
            )
            for _, r in chosen.iterrows():
                rows.append((int(r["frame_idx"]), tid, identity, r))
        rows.sort(key=lambda x: x[0])

        video = open_video(video_path)
        tag = folder.name
        current_idx, frame = None, None
        for frame_idx, tid, identity, r in rows:
            if frame_idx != current_idx:
                frame = np.asarray(video[frame_idx])
                if camera is not None:
                    frame = camera.crop(frame)
                current_idx = frame_idx
            box = r[BOX_COLUMNS].to_numpy(dtype=float)
            crop = square_crop(frame, box, padding)
            out_dir = data_path / "train" / identity
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{PSEUDO_PREFIX}{tag}_{tid}_{frame_idx}.png"
            Image.fromarray(crop).save(out_dir / name)
            summary["n_crops"][identity] = (
                summary["n_crops"].get(identity, 0) + 1
            )
        logger.info(
            f"{folder.name}: {len(selected)} assigned tracklet(s), "
            f"{len(rows)} crop(s)"
            + (
                f" from {len(allowed)} co-existence frame(s)"
                if allowed is not None
                else ""
            )
        )
    return summary
