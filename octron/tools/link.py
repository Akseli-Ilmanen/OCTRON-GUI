"""OCTRON multi-camera identity linking.

YOLO tracking has no notion that individuals of one label are mutually
exclusive: two tracklets of the same label (e.g. two "bird" tracks) may
each look plausible for the same known individual. The identity
classifier (see ``octron.yolo_octron.identity``) scores every tracked box
against the known individuals of its label, writing per-frame
``identity_prob_<classname>`` columns into the tracking CSVs. This module
runs an exact, post-hoc exclusivity pass over those per-tracklet scores:
every tracklet gets at most one individual, and two tracklets that were
alive at the same time (same ``frame_idx``) never share an individual.

See https://github.com/OCTRON-tracking/OCTRON-GUI/issues/91 for the
worked example this module implements.

Public API
----------
Tracklet          : one tracked object (folder, track_id, label,
                     frame_idx, per-individual summed scores).
load_tracklets     : read every tracklet CSV in a results folder.
temporal_overlaps  : pairs of tracklets sharing at least one frame.
build_score_matrix : dense (n_tracklets, n_individuals) score matrix.
solve_assignment   : exact integer-program identity assignment.
run_link           : end-to-end pipeline; writes CSV/JSON reports.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from octron.yolo_octron.identity import IDENTITY_PROB_PREFIX

CSV_HEADER_LINES = 7
IDENTITY_ASSIGNMENT_FILENAME = "identity_assignment.csv"
LINK_REPORT_FILENAME = "link_report.json"


@dataclass
class Tracklet:
    """One tracked object read from a ``<label>_track_<track_id>.csv``.

    Parameters
    ----------
    folder : Path
        Results folder (one camera) this tracklet was loaded from.
    track_id : int
        Track ID (unique within ``folder``, not across folders).
    label : str
        Detector label (e.g. ``"bird"``).
    frame_idx : numpy.ndarray
        Frame indices (int) at which this tracklet has an observation.
    scores : dict
        ``{classname: summed_probability}`` over all frames, restricted
        to individuals whose label matches ``label``. Empty when the CSV
        has no matching ``identity_prob_*`` columns.
    n_frames : int
        Number of rows (observations) in the CSV.

    """

    folder: Path
    track_id: int
    label: str
    frame_idx: np.ndarray
    scores: dict = field(default_factory=dict)
    n_frames: int = 0


def _load_identity_classes(folder):
    """Return ``{classname: label}`` from ``prediction_metadata.json``.

    Returns an empty dict when the file (or the ``identity_classes`` key)
    is missing.
    """
    meta_path = Path(folder) / "prediction_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {meta_path}: {e}")
        return {}
    identity_classes = metadata.get("identity_classes", {}) or {}
    return {
        name: info.get("label")
        for name, info in identity_classes.items()
        if isinstance(info, dict) and "label" in info
    }


def _classname_label(classname, tracklet_label, identity_classes):
    """Resolve the label a classifier class name belongs to.

    Prefers the folder's ``prediction_metadata.json`` mapping; falls back
    to matching ``classname`` against ``tracklet_label`` as a prefix
    (``"<label>_"``, spaces replaced by underscores) when the mapping is
    missing the class or the metadata file itself is missing.
    """
    label = identity_classes.get(classname)
    if label is not None:
        return label
    prefix = "_".join(str(tracklet_label).split()) + "_"
    if classname.startswith(prefix):
        return tracklet_label
    return None


def load_tracklets(folder):
    """Load every tracklet CSV in a results folder.

    Parameters
    ----------
    folder : str or Path
        Results folder for one video/camera (as written by
        ``octron predict``), containing ``<label>_track_<id>.csv`` files
        and optionally ``prediction_metadata.json``.

    Returns
    -------
    list of Tracklet
        One entry per ``*_track_*.csv`` file, sorted by track_id.

    """
    folder = Path(folder)
    identity_classes = _load_identity_classes(folder)
    csv_files = sorted(folder.glob("*_track_*.csv"))
    tracklets = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, skiprows=CSV_HEADER_LINES)
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")
            continue
        if df.empty or "track_id" not in df.columns:
            logger.warning(
                f"Skipping empty/malformed tracklet CSV: {csv_file.name}"
            )
            continue

        track_id = int(df.iloc[0]["track_id"])
        label = str(df.iloc[0]["label"])
        frame_idx = df["frame_idx"].to_numpy(dtype=int)

        scores = {}
        for col in df.columns:
            if not col.startswith(IDENTITY_PROB_PREFIX):
                continue
            classname = col[len(IDENTITY_PROB_PREFIX) :]
            cls_label = _classname_label(classname, label, identity_classes)
            if cls_label == label:
                scores[classname] = float(df[col].fillna(0).sum())

        tracklets.append(
            Tracklet(
                folder=folder,
                track_id=track_id,
                label=label,
                frame_idx=frame_idx,
                scores=scores,
                n_frames=len(df),
            )
        )
    tracklets.sort(key=lambda t: t.track_id)
    return tracklets


def build_score_matrix(tracklets, individuals):
    """Build the dense (n_tracklets, n_individuals) score matrix.

    Parameters
    ----------
    tracklets : list of Tracklet
        Tracklets to score (row order).
    individuals : list of str
        Ordered individual class names (matrix column order).

    Returns
    -------
    numpy.ndarray
        Shape ``(len(tracklets), len(individuals))``. Entries are 0 for
        individuals of a different label than the tracklet (``Tracklet.
        scores`` already excludes those).

    """
    S = np.zeros((len(tracklets), len(individuals)), dtype=float)
    for t, tracklet in enumerate(tracklets):
        for i, name in enumerate(individuals):
            if name in tracklet.scores:
                S[t, i] = tracklet.scores[name]
    return S


def temporal_overlaps(tracklets, min_overlap=1):
    """Return pairs ``(i, j)``, ``i < j``, sharing ``min_overlap`` frames.

    Parameters
    ----------
    tracklets : list of Tracklet
        Tracklets to check pairwise for shared frame_idx values.
    min_overlap : int
        Minimum number of shared frames for a pair to count as
        co-existing. A tracker hands a bird over from a dying track to
        a new one with a few frames of overlap; treating those as two
        animals would forbid the new track the identity the old one had,
        which can block a very long tracklet on a 3-frame artefact.
        ``1`` (default) keeps every overlap.

    Returns
    -------
    list of tuple of int
        Index pairs into ``tracklets``.

    """
    frame_sets = [set(t.frame_idx.tolist()) for t in tracklets]
    pairs = []
    for i in range(len(tracklets)):
        for j in range(i + 1, len(tracklets)):
            if len(frame_sets[i] & frame_sets[j]) >= min_overlap:
                pairs.append((i, j))
    return pairs


def evidence(scores_row):
    """Net evidence of a score row: best score minus runner-up score.

    With two individuals of one label the per-frame probabilities sum
    to one, so the *total* score of a tracklet is just its frame count;
    only the difference between best and runner-up says how decisive the
    frames were. A 20-frame tracklet at 0.55/0.45 has evidence 2, one at
    0.9/0.1 has 16.
    """
    scores_row = np.asarray(scores_row, dtype=float)
    if scores_row.size == 0:
        return 0.0
    if scores_row.size == 1:
        return float(scores_row[0])
    top = np.sort(scores_row)[-2:]
    return float(top[1] - top[0])


def solve_assignment(tracklets, individuals, overlap_pairs, min_evidence=0.0):
    """Assign at most one individual to each tracklet.

    Exact integer program (``scipy.optimize.milp``): variables
    ``x[t, i] in {0, 1}`` maximise ``sum(S[t, i] * x[t, i])`` subject to
    each tracklet getting at most one individual, and, for every
    temporally overlapping pair and every individual, at most one of the
    two tracklets getting it.

    Tracklets with an all-zero score row (no identity columns, or
    nothing of their label) are excluded from the program and always
    come back unassigned. So are tracklets whose :func:`evidence` (best
    minus runner-up score) is below ``min_evidence``: a short tracklet
    seen from a bad angle is then reported as unknown instead of
    guessed, and it no longer competes with overlapping tracklets.

    Parameters
    ----------
    tracklets : list of Tracklet
        Tracklets to assign (row order of the score matrix).
    individuals : list of str
        Ordered candidate individual class names.
    overlap_pairs : list of tuple of int
        Index pairs (into ``tracklets``) that must not share an
        individual, e.g. from :func:`temporal_overlaps`.
    min_evidence : float
        Minimum best-minus-runner-up summed score for a tracklet to be
        assigned at all. ``0`` (default) assigns every scored tracklet.

    Returns
    -------
    list of int or None
        Index into ``individuals`` assigned to each tracklet, or
        ``None`` if unassigned. Length matches ``tracklets``.

    """
    n_t = len(tracklets)
    n_i = len(individuals)
    result = [None] * n_t
    if n_t == 0 or n_i == 0:
        return result

    S = build_score_matrix(tracklets, individuals)
    has_scores = S.sum(axis=1) > 0
    enough = np.array([evidence(row) >= min_evidence for row in S])
    included = np.flatnonzero(has_scores & enough)
    n_t2 = len(included)
    n_vars = n_t2 * n_i
    if n_vars == 0:
        return result

    pos = {orig: a for a, orig in enumerate(included)}
    c = -S[included].flatten()

    row_idx, col_idx, data, ub = [], [], [], []
    r = 0
    # Each (included) tracklet gets at most one individual.
    for a in range(n_t2):
        for i in range(n_i):
            row_idx.append(r)
            col_idx.append(a * n_i + i)
            data.append(1.0)
        ub.append(1.0)
        r += 1
    # Overlapping tracklets never share an individual.
    for t, u in overlap_pairs:
        if t not in pos or u not in pos:
            continue
        at, au = pos[t], pos[u]
        for i in range(n_i):
            row_idx.append(r)
            col_idx.append(at * n_i + i)
            data.append(1.0)
            row_idx.append(r)
            col_idx.append(au * n_i + i)
            data.append(1.0)
            ub.append(1.0)
            r += 1

    A = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(r, n_vars))
    constraints = LinearConstraint(A, lb=-np.inf, ub=np.array(ub))
    res = milp(
        c=c,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=np.ones(n_vars),
    )
    if not res.success:
        logger.warning(
            f"Identity assignment MILP did not solve: {res.message}"
        )
        return result

    x = res.x.reshape(n_t2, n_i)
    for a, orig in enumerate(included):
        row = x[a]
        if row.max() > 0.5:
            result[orig] = int(np.argmax(row))
    return result


def _folder_overlap_pairs(
    tracklets, global_exclusive, min_overlap=1, exclusive_pairs=()
):
    """Overlap pairs per the exclusivity policy.

    ``global_exclusive=False`` (default): only tracklets from the same
    folder (camera) can conflict, plus tracklets from two folders named
    in ``exclusive_pairs`` (cameras whose fields of view do not
    overlap, so one animal cannot be in both at once). ``True``:
    overlaps are computed across every folder (frame_idx are comparable
    — synchronised camera tiles of one mosaic video). ``min_overlap``
    is passed to :func:`temporal_overlaps`.
    """
    if global_exclusive:
        return temporal_overlaps(tracklets, min_overlap)

    by_folder = defaultdict(list)
    for idx, t in enumerate(tracklets):
        by_folder[t.folder].append(idx)

    pairs = []
    for indices in by_folder.values():
        sub = [tracklets[i] for i in indices]
        for a, b in temporal_overlaps(sub, min_overlap):
            pairs.append((indices[a], indices[b]))

    by_name = {folder.name: indices for folder, indices in by_folder.items()}
    for pair in exclusive_pairs:
        a_name, b_name = tuple(pair)
        if a_name not in by_name or b_name not in by_name:
            continue
        ia, ib = by_name[a_name], by_name[b_name]
        sub = [tracklets[i] for i in ia] + [tracklets[i] for i in ib]
        mapping = ia + ib
        for a, b in temporal_overlaps(sub, min_overlap):
            # keep only cross-folder pairs; same-folder ones are above
            if (a < len(ia)) != (b < len(ia)):
                pairs.append((mapping[a], mapping[b]))
    return pairs


def resolve_exclusive_pairs(folders, exclusive=None):
    """Camera pairs that must not share an individual at the same time.

    Parameters
    ----------
    folders : list of Path
        Camera result folders.
    exclusive : list of tuple or None
        Explicit ``(camera_a, camera_b)`` name pairs (``"*"`` allowed).
        ``None``: read ``non_overlapping`` from the ``cameras.json``
        next to the folders (their common parent), if present.

    Returns
    -------
    set of frozenset
        Unordered camera-name pairs.

    """
    from octron.cameras import CAMERAS_FILENAME, CameraLayout

    names = [Path(f).name for f in folders]
    if exclusive is not None:
        layout = CameraLayout(
            cameras=[], non_overlapping=[[a, b] for a, b in exclusive]
        )
        pairs = set()
        for a, b in layout.non_overlapping:
            left = names if a == "*" else [a]
            right = names if b == "*" else [b]
            for x in left:
                for y in right:
                    if x != y:
                        pairs.add(frozenset((x, y)))
        return pairs
    parents = {Path(f).resolve().parent for f in folders}
    pairs = set()
    for parent in parents:
        path = parent / CAMERAS_FILENAME
        if not path.exists():
            continue
        try:
            layout = CameraLayout.load(path)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue
        pairs |= layout.exclusive_pairs()
    return pairs


def _row_report(
    idx, tracklets, individuals, S, assigned, min_margin, min_evidence=0.0
):
    """Build the per-tracklet output row shared by the CSV and JSON report."""
    tracklet = tracklets[idx]
    scores_row = S[idx]
    a = assigned[idx]
    total = float(scores_row.sum())
    order = np.argsort(scores_row)[::-1]
    net = evidence(scores_row)

    frame_idx = tracklet.frame_idx
    first_frame = int(frame_idx.min()) if len(frame_idx) else None
    last_frame = int(frame_idx.max()) if len(frame_idx) else None

    if total == 0.0:
        identity, score = None, 0.0
        runner_up, runner_up_score = None, 0.0
        margin_ratio = 0.0
        reason = "no_identity_scores"
    else:
        candidates = [i for i in order if i != a]
        runner_up = individuals[candidates[0]] if candidates else None
        runner_up_score = (
            float(scores_row[candidates[0]]) if candidates else 0.0
        )
        if a is None:
            identity, score = None, 0.0
            margin_ratio = 0.0
            reason = "low_evidence" if net < min_evidence else "unassigned"
        else:
            identity = individuals[a]
            score = float(scores_row[a])
            margin_ratio = (
                float("inf")
                if runner_up_score == 0
                else score / runner_up_score
            )
            reason = "low_margin" if margin_ratio < min_margin else ""

    flagged = identity is None or margin_ratio < min_margin
    return {
        "track_id": tracklet.track_id,
        "label": tracklet.label,
        "identity": identity,
        "score": score,
        "runner_up": runner_up,
        "runner_up_score": runner_up_score,
        "margin_ratio": margin_ratio,
        "evidence": net,
        "n_frames": tracklet.n_frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "flagged": bool(flagged),
        "reason": reason,
    }


def run_link(
    folders,
    min_margin=1.5,
    global_exclusive=False,
    overwrite=False,
    min_evidence=0.0,
    min_overlap=1,
    exclusive=None,
):
    """Assign identities to every tracklet across one or more camera folders.

    Loads every tracklet in ``folders``, solves one exact assignment
    (:func:`solve_assignment`) over all of them, and writes an
    ``identity_assignment.csv`` and ``link_report.json`` into each
    folder.

    Parameters
    ----------
    folders : list of str or Path
        Results folders, one per camera. A single path is also accepted.
    min_margin : float
        A tracklet is flagged ``"low_margin"`` when
        ``score / runner_up_score`` falls below this ratio. Default
        ``1.5``.
    global_exclusive : bool
        ``False`` (default): two tracklets may only conflict (share an
        individual exclusion) when they come from the same folder.
        ``True``: tracklets from every folder are pooled and checked for
        temporal overlap across folders too (for synchronised
        multi-camera mosaics).
    overwrite : bool
        Overwrite existing ``identity_assignment.csv`` /
        ``link_report.json`` outputs. Default ``False`` (raises
        ``FileExistsError``).
    min_evidence : float
        Tracklets whose best-minus-runner-up summed score is below this
        are left unassigned (reason ``"low_evidence"``) instead of
        guessed. In units of confident frames; ``0`` disables.
    min_overlap : int
        Two tracklets only exclude each other from sharing an
        individual when they share at least this many frames. Tracker
        hand-overs produce a few frames of overlap between the old and
        the new track of one animal; ``1`` (default) treats those as two
        animals, ``10`` or so ignores them.
    exclusive : list of tuple or None
        Camera-name pairs whose fields of view do not overlap, so a
        tracklet in one and a tracklet in the other alive at the same
        time are different animals. ``None`` (default) reads
        ``non_overlapping`` from the ``cameras.json`` next to the
        folders. Ignored with ``global_exclusive``.

    Returns
    -------
    dict
        Summary: ``min_margin``, ``global_exclusive``, ``n_tracklets``,
        ``n_assigned``, ``n_flagged``, ``individuals``, and a
        ``folders`` dict keyed by folder path with the same per-folder
        counts plus the output file paths.

    """
    if isinstance(folders, (str, Path)):
        folders = [folders]
    folders = [Path(f) for f in folders]

    if not overwrite:
        existing = []
        for folder in folders:
            for name in (IDENTITY_ASSIGNMENT_FILENAME, LINK_REPORT_FILENAME):
                out = folder / name
                if out.exists():
                    existing.append(out)
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing link output(s): "
                + ", ".join(p.as_posix() for p in existing)
                + ". Pass overwrite=True (--overwrite) to replace them."
            )

    all_tracklets = []
    for folder in folders:
        all_tracklets.extend(load_tracklets(folder))
    logger.info(
        f"Loaded {len(all_tracklets)} tracklet(s) from "
        f"{len(folders)} folder(s)"
    )

    individuals = set()
    for t in all_tracklets:
        individuals.update(t.scores.keys())
    for folder in folders:
        individuals.update(_load_identity_classes(folder).keys())
    individuals = sorted(individuals)

    exclusive_pairs = resolve_exclusive_pairs(folders, exclusive)
    if exclusive_pairs and not global_exclusive:
        logger.info(
            "Non-overlapping camera pairs (cross-camera exclusivity): "
            + ", ".join(sorted(":".join(sorted(p)) for p in exclusive_pairs))
        )
    overlap_pairs = _folder_overlap_pairs(
        all_tracklets, global_exclusive, min_overlap, exclusive_pairs
    )
    assigned = solve_assignment(
        all_tracklets, individuals, overlap_pairs, min_evidence=min_evidence
    )
    S = build_score_matrix(all_tracklets, individuals)

    by_folder = defaultdict(list)
    for idx, t in enumerate(all_tracklets):
        by_folder[t.folder].append(idx)

    summary = {
        "min_margin": min_margin,
        "min_evidence": min_evidence,
        "min_overlap": min_overlap,
        "global_exclusive": global_exclusive,
        "exclusive_pairs": sorted(
            ":".join(sorted(p)) for p in exclusive_pairs
        ),
        "n_tracklets": len(all_tracklets),
        "n_assigned": sum(1 for a in assigned if a is not None),
        "n_flagged": 0,
        "individuals": individuals,
        "folders": {},
    }

    for folder in folders:
        indices = by_folder.get(folder, [])
        rows = [
            _row_report(
                idx,
                all_tracklets,
                individuals,
                S,
                assigned,
                min_margin,
                min_evidence,
            )
            for idx in indices
        ]

        csv_path = folder / IDENTITY_ASSIGNMENT_FILENAME
        report_path = folder / LINK_REPORT_FILENAME

        columns = [
            "track_id",
            "label",
            "identity",
            "score",
            "runner_up",
            "runner_up_score",
            "margin_ratio",
            "evidence",
            "n_frames",
            "first_frame",
            "last_frame",
            "flagged",
            "reason",
        ]
        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df = df.set_index("track_id")
        else:
            df = df.set_index(pd.Index([], name="track_id"))
        df.to_csv(csv_path)

        n_assigned = sum(1 for row in rows if row["identity"] is not None)
        n_flagged = sum(1 for row in rows if row["flagged"])
        report = {
            "min_margin": min_margin,
            "min_evidence": min_evidence,
            "min_overlap": min_overlap,
            "global_exclusive": global_exclusive,
            "exclusive_pairs": summary["exclusive_pairs"],
            "n_tracklets": len(rows),
            "n_assigned": n_assigned,
            "n_flagged": n_flagged,
            "individuals": individuals,
            "flagged": [
                {
                    "track_id": row["track_id"],
                    "label": row["label"],
                    "identity": row["identity"],
                    "reason": row["reason"],
                    "margin_ratio": row["margin_ratio"],
                    "evidence": row["evidence"],
                    "n_frames": row["n_frames"],
                }
                for row in rows
                if row["flagged"]
            ],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        summary["n_flagged"] += n_flagged
        summary["folders"][folder.as_posix()] = {
            "n_tracklets": len(rows),
            "n_assigned": n_assigned,
            "n_flagged": n_flagged,
            "identity_assignment_csv": csv_path.as_posix(),
            "link_report_json": report_path.as_posix(),
        }
        logger.info(
            f"{folder.name}: {len(rows)} tracklet(s), {n_assigned} assigned, "
            f"{n_flagged} flagged -> {csv_path.as_posix()}"
        )

    print(
        f"Linked {summary['n_tracklets']} tracklet(s) across {len(folders)} "
        f"folder(s): {summary['n_assigned']} assigned, "
        f"{summary['n_flagged']} flagged (min_margin={min_margin}, "
        f"min_evidence={min_evidence}, min_overlap={min_overlap}, "
        f"global_exclusive={global_exclusive})."
    )
    return summary
