"""Identity evaluation against the project annotations (IDF1).

``octron evaluate-identity`` scores a linked prediction folder against
the hand-annotated ``(label, suffix)`` masks of the same video. Only
frames of the chosen dataset split (default ``test``) are used, so the
number is not inflated by frames the classifier trained on.

Metrics
-------
IDF1 (Ristani et al. 2016) is the F1 of the best global one-to-one
mapping between ground-truth identities and predicted ids. It rewards
*consistency*: the same animal keeping the same id. It is reported for
three id schemes of increasing processing:

* ``tracks``: raw BoxMOT track ids (how fragmented/contaminated the
  tracker output is),
* ``frame_identity``: the per-frame classifier argmax,
* ``linked``: the identity assigned per tracklet by ``octron link``.

IDF1 is permutation invariant, so it does not notice a systematic
swap (every "male" called "female"). ``linked_accuracy`` covers that:
the fraction of matched ground-truth boxes whose linked identity is the
annotated one.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linear_sum_assignment

from octron.yolo_octron.identity import (
    collect_identity_entries,
    identity_paths,
    mask_bbox,
)
from octron.yolo_octron.identity_refine import (
    BOX_COLUMNS,
    IDENTITY_ASSIGNMENT_FILENAME,
    folder_video_and_camera,
    load_tracklet_frames,
    split_frames,
)

IDENTITY_EVAL_FILENAME = "identity_eval.json"


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------


def box_iou(a, b):
    """IoU between two ``[x_min, y_min, x_max, y_max]`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5):
    """One-to-one match of ground truth to predictions by IoU (Hungarian).

    Returns a list of ``(gt_index, pred_index)`` pairs with IoU at or
    above ``iou_thresh``.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou[i, j] = box_iou(g, p)
    rows, cols = linear_sum_assignment(-iou)
    return [
        (int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
        if iou[i, j] >= iou_thresh
    ]


def idf1(pairs, n_gt, n_pred):
    """IDF1 from matched ``(gt_id, pred_id)`` pairs.

    Parameters
    ----------
    pairs : list of tuple
        One ``(gt_id, pred_id)`` per matched detection.
    n_gt, n_pred : int
        Total ground-truth and predicted detections (matched or not).

    Returns
    -------
    dict
        ``idf1, idtp, idfp, idfn`` and the chosen ``mapping``
        ``{gt_id: pred_id}``.

    """
    if n_gt + n_pred == 0:
        return {
            "idf1": float("nan"),
            "idtp": 0,
            "idfp": 0,
            "idfn": 0,
            "mapping": {},
        }
    counts = Counter(pairs)
    gt_ids = sorted({g for g, _ in counts}, key=str)
    pred_ids = sorted({p for _, p in counts}, key=str)
    idtp = 0
    mapping = {}
    if gt_ids and pred_ids:
        M = np.zeros((len(gt_ids), len(pred_ids)))
        for (g, p), n in counts.items():
            M[gt_ids.index(g), pred_ids.index(p)] = n
        rows, cols = linear_sum_assignment(-M)
        for i, j in zip(rows, cols, strict=True):
            if M[i, j] > 0:
                idtp += int(M[i, j])
                mapping[str(gt_ids[i])] = str(pred_ids[j])
    idfp = n_pred - idtp
    idfn = n_gt - idtp
    return {
        "idf1": 2 * idtp / (n_gt + n_pred),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "mapping": mapping,
    }


# ---------------------------------------------------------------------------
# Ground truth from the project
# ---------------------------------------------------------------------------


def load_ground_truth(project_path, subfolder, frames, camera=None):
    """Ground-truth boxes of one video on ``frames``.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project.
    subfolder : str
        Project video folder whose masks to read.
    frames : iterable of int
        Frames to read.
    camera : octron.cameras.Camera or None
        Restrict masks to this camera and shift boxes into its
        coordinates (mosaic videos).

    Returns
    -------
    dict
        ``{frame_idx: [(class_name, label, box), ...]}``; frames where an
        individual has no pixels (in this camera) contribute nothing
        for it.

    """
    from octron.sam_octron.helpers.sam_zarr import load_image_zarr

    entries, _ = collect_identity_entries(project_path)
    entries = [e for e in entries if e["subfolder"] == subfolder]
    frames = sorted(int(f) for f in frames)
    gt = defaultdict(list)
    for entry in entries:
        masks, status = load_image_zarr(
            entry["zarr_path"],
            entry["num_frames"],
            entry["height"],
            entry["width"],
            num_ch=None,
            verbose=False,
        )
        if not status or masks is None:
            raise RuntimeError(
                f"Could not open mask store {entry['zarr_path']}"
            )
        for frame_idx in frames:
            if frame_idx >= entry["num_frames"]:
                continue
            mask = np.asarray(masks[frame_idx])
            if mask.max() <= 0:
                continue
            if camera is not None:
                mask = camera.crop(mask)
            box = mask_bbox(mask)
            if box is None:
                continue
            gt[frame_idx].append(
                (
                    entry["class_name"],
                    entry["label"],
                    np.asarray(box, dtype=float),
                )
            )
    return dict(gt)


# ---------------------------------------------------------------------------
# Folder evaluation
# ---------------------------------------------------------------------------


def evaluate_pairs(gt, obs, linked, iou_thresh=0.5):
    """Match ground truth to observations frame by frame.

    Parameters
    ----------
    gt : dict
        Output of :func:`load_ground_truth`.
    obs : pandas.DataFrame
        Per-frame observations (:func:`load_tracklet_frames`) with an
        optional ``identity`` column (per-frame classifier argmax).
    linked : dict
        ``{track_id: identity or None}`` from ``identity_assignment.csv``.
    iou_thresh : float
        Minimum IoU for a ground truth / prediction match.

    Returns
    -------
    dict
        ``pairs`` per id scheme (``tracks, frame_identity, linked``),
        ``n_gt``, ``n_pred``, ``n_matched``, ``correct_linked`` and a
        per-identity breakdown ``per_identity``.

    """
    frames = sorted(gt)
    obs_by_frame = (
        {f: g for f, g in obs.groupby("frame_idx")} if len(obs) else {}
    )
    pairs = {"tracks": [], "frame_identity": [], "linked": []}
    n_gt = n_pred = n_matched = correct = 0
    per_identity = defaultdict(lambda: {"n_gt": 0, "matched": 0, "correct": 0})
    for frame_idx in frames:
        gts = gt[frame_idx]
        n_gt += len(gts)
        for cls, _, _ in gts:
            per_identity[cls]["n_gt"] += 1
        preds = obs_by_frame.get(frame_idx)
        if preds is None or preds.empty:
            continue
        n_pred += len(preds)
        # Match within label only.
        for label in {g[1] for g in gts}:
            gi = [i for i, g in enumerate(gts) if g[1] == label]
            p = preds[preds["label"].astype(str) == str(label)]
            if p.empty:
                continue
            gt_boxes = [gts[i][2] for i in gi]
            pred_boxes = p[BOX_COLUMNS].to_numpy(dtype=float)
            for a, b in match_boxes(gt_boxes, pred_boxes, iou_thresh):
                cls = gts[gi[a]][0]
                row = p.iloc[b]
                tid = int(row["track_id"])
                n_matched += 1
                per_identity[cls]["matched"] += 1
                pairs["tracks"].append((cls, tid))
                frame_id = row.get("identity")
                if frame_id is None or (
                    isinstance(frame_id, float) and np.isnan(frame_id)
                ):
                    frame_id = f"none_{tid}"
                pairs["frame_identity"].append((cls, str(frame_id)))
                link_id = linked.get(tid)
                if link_id is None:
                    link_id = f"unassigned_{tid}"
                pairs["linked"].append((cls, str(link_id)))
                if str(link_id) == cls:
                    correct += 1
                    per_identity[cls]["correct"] += 1
    return {
        "pairs": pairs,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "correct_linked": correct,
        "per_identity": dict(per_identity),
    }


def summarise(evaluated):
    """Turn :func:`evaluate_pairs` output into the report dict."""
    n_gt, n_pred = evaluated["n_gt"], evaluated["n_pred"]
    report = {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": evaluated["n_matched"],
        "linked_accuracy": (
            evaluated["correct_linked"] / evaluated["n_matched"]
            if evaluated["n_matched"]
            else float("nan")
        ),
        "per_identity": evaluated["per_identity"],
    }
    for scheme, pairs in evaluated["pairs"].items():
        res = idf1(pairs, n_gt, n_pred)
        report[f"idf1_{scheme}"] = res["idf1"]
        report[f"mapping_{scheme}"] = res["mapping"]
    return report


def evaluate_folder(project_path, folder, split="test", iou_thresh=0.5):
    """Evaluate one linked prediction folder against project annotations.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project that annotates the folder's video.
    folder : str or Path
        Prediction folder (one camera) after ``octron link``.
    split : str
        ``"test"``, ``"val"`` or ``"all"`` annotated frames.
    iou_thresh : float
        Minimum IoU for a ground truth / prediction match.

    Returns
    -------
    dict
        Report (also written to ``identity_eval.json`` in the folder).

    """
    from octron.sam_octron.helpers.sam_zarr import (
        get_annotated_frames,
        load_image_zarr,
    )
    from octron.yolo_octron.identity_refine import find_project_subfolder

    project_path = Path(project_path)
    folder = Path(folder)
    video_path, camera, _ = folder_video_and_camera(folder)
    subfolder = find_project_subfolder(project_path, video_path)
    if subfolder is None:
        raise ValueError(
            f"No annotated individuals for video {video_path.name} in "
            f"project {project_path}"
        )

    entries, _ = collect_identity_entries(project_path)
    entries = [e for e in entries if e["subfolder"] == subfolder]
    annotated = set()
    for entry in entries:
        masks, status = load_image_zarr(
            entry["zarr_path"],
            entry["num_frames"],
            entry["height"],
            entry["width"],
            num_ch=None,
            verbose=False,
        )
        if status and masks is not None:
            annotated.update(int(f) for f in get_annotated_frames(masks))

    _, data_path = identity_paths(project_path)
    if split == "all":
        frames = annotated
    else:
        held = split_frames(data_path, subfolder, splits=(split,))[split]
        frames = annotated & held
        if not frames:
            logger.warning(
                f"No annotated frames of {subfolder} in split '{split}' "
                f"(dataset at {data_path}); nothing to evaluate."
            )
    gt = load_ground_truth(project_path, subfolder, frames, camera)
    obs = load_tracklet_frames(folder)
    obs_ident = _per_frame_identity(folder)
    if obs_ident is not None and len(obs):
        obs = obs.merge(obs_ident, on=["track_id", "frame_idx"], how="left")

    assignment_path = folder / IDENTITY_ASSIGNMENT_FILENAME
    linked = {}
    if assignment_path.exists():
        assignment = pd.read_csv(assignment_path, index_col="track_id")
        for tid, row in assignment.iterrows():
            ident = row.get("identity")
            if isinstance(ident, float) and np.isnan(ident):
                ident = None
            linked[int(tid)] = ident
    else:
        logger.warning(f"No {IDENTITY_ASSIGNMENT_FILENAME} in {folder}")

    evaluated = evaluate_pairs(gt, obs, linked, iou_thresh)
    report = summarise(evaluated)
    report.update(
        {
            "folder": folder.as_posix(),
            "project": project_path.as_posix(),
            "subfolder": subfolder,
            "split": split,
            "iou_thresh": iou_thresh,
            "n_frames": len(gt),
        }
    )
    with open(folder / IDENTITY_EVAL_FILENAME, "w") as f:
        json.dump(report, f, indent=4, default=_json_default)
    return report


def _per_frame_identity(folder):
    """``track_id, frame_idx, identity`` rows from the tracklet CSVs."""
    from octron.yolo_octron.identity_refine import CSV_HEADER_LINES

    parts = []
    for csv_file in sorted(Path(folder).glob("*_track_*.csv")):
        try:
            df = pd.read_csv(csv_file, skiprows=CSV_HEADER_LINES)
        except Exception:
            continue
        if "identity" not in df.columns or df.empty:
            continue
        parts.append(df[["track_id", "frame_idx", "identity"]])
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out["track_id"] = out["track_id"].astype(int)
    out["frame_idx"] = out["frame_idx"].astype(int)
    return out


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")
