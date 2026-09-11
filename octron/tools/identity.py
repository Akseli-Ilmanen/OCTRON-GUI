"""OCTRON identity-classifier pipeline.

Wraps :func:`octron.yolo_octron.identity.build_identity_dataset` and
:func:`octron.yolo_octron.identity.train_identity` into one callable
used by ``octron train-identity``.

Workflow: the detector is trained on labels only (``bird``), individuals
are annotated as ``<label> <suffix>`` (``bird 1``, ``bird 2``), and this
classifier learns the suffix from square crops of the annotated masks.
``octron predict --identity <best.pt>`` then scores every tracked box and
``octron link`` resolves the per-camera exclusivity between tracklets.
"""

from pathlib import Path


def run_train_identity(
    project_path,
    model="yolo11n-cls",
    imgsz=224,
    epochs=50,
    device=None,
    batch=-1,
    padding=0.1,
    overwrite=False,
    skip_export=False,
    train_fraction=None,
    val_fraction=None,
    seed=None,
    buffer=None,
    verbose=False,
):
    """Export identity crops and train the identity classifier.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project directory.
    model : str or Path
        Classification model name (``yolo11n-cls``) or ``.pt`` path.
    imgsz : int
        Classifier input size.
    epochs : int
        Training epochs.
    device : str or None
        ``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``; None reads
        ``config.yaml``.
    batch : int
        Batch size; -1 lets ultralytics pick.
    padding : float
        Fractional crop padding around each mask box.
    overwrite : bool
        Rebuild the crop dataset and retrain from scratch.
    skip_export : bool
        Reuse an existing crop dataset.
    train_fraction, val_fraction, seed, buffer : optional
        Split settings; None reads ``config.yaml``.
    verbose : bool
        Log per-entry export progress.

    Returns
    -------
    Path
        Path to the trained ``best.pt``.

    """
    from loguru import logger

    from octron import config
    from octron.yolo_octron.identity import (
        build_identity_dataset,
        identity_paths,
        train_identity,
    )

    project_path = Path(project_path)
    if not project_path.is_dir():
        raise FileNotFoundError(f"Project path not found: {project_path}")
    if device is None:
        device = config.get_device()

    root, data_path = identity_paths(project_path)
    best = root / "training" / "weights" / "best.pt"
    if best.exists() and not overwrite:
        print(
            f"Identity classifier already exists at {best}. "
            "Pass --overwrite to retrain."
        )
        return best

    if skip_export and (data_path / "train").exists():
        logger.info(f"Reusing identity dataset at {data_path}")
    else:
        summary = build_identity_dataset(
            project_path,
            padding=padding,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed,
            buffer=buffer,
            overwrite=True,
            verbose=verbose,
        )
        counts = summary["n_crops"]
        print(
            f"Exported {counts['train']} train / {counts['val']} val / "
            f"{counts['test']} test crops for "
            f"{len(summary['classes'])} individual(s): "
            f"{', '.join(sorted(summary['classes']))}"
        )
        if summary["n_skipped_no_suffix"]:
            print(
                f"Skipped {summary['n_skipped_no_suffix']} object(s) "
                "without a suffix."
            )

    best = train_identity(
        project_path,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        device=device,
        batch=batch,
        overwrite=overwrite,
    )
    print(f"Identity classifier saved to {best}")
    print(
        "Next: octron predict <video> --model <detector.pt> "
        f"--identity {best} [--cameras cameras.json], then octron link."
    )
    return best


def run_refine_identity(
    project_path,
    folders,
    min_frame_prob=0.2,
    max_per_tracklet=50,
    clear=False,
    model=None,
    imgsz=224,
    epochs=30,
    device=None,
    batch=-1,
    export_only=False,
):
    """Self-train the identity classifier on linked tracklets.

    Exports crops of every trusted tracklet (assigned, unflagged, no
    unresolved neighbour; see
    :func:`octron.yolo_octron.identity_refine.select_pseudo_tracklets`)
    into the ``train`` split of the identity dataset and retrains the
    classifier on the union of hand-labelled and pseudo-labelled crops.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project directory (identity dataset must exist).
    folders : list of str or Path
        Linked prediction folders (after ``octron link``).
    min_frame_prob : float
        Drop frames whose per-frame probability for the assigned
        identity is below this value.
    max_per_tracklet : int
        Crops per tracklet, spread evenly (0: all).
    clear : bool
        Remove pseudo crops from earlier rounds first.
    model : str or Path or None
        Weights to start from. None continues from the current
        ``best.pt`` when it exists, else ``yolo11n-cls``.
    imgsz, epochs, batch : int
        Training settings.
    device : str or None
        ``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``; None reads
        ``config.yaml``.
    export_only : bool
        Only export the crops, do not retrain.

    Returns
    -------
    Path or None
        Path to the retrained ``best.pt`` (None with ``export_only``).

    """
    import shutil

    from octron import config
    from octron.yolo_octron.identity import identity_paths, train_identity
    from octron.yolo_octron.identity_refine import export_pseudo_crops

    project_path = Path(project_path)
    if device is None:
        device = config.get_device()

    summary = export_pseudo_crops(
        project_path,
        folders,
        min_frame_prob=min_frame_prob,
        max_per_tracklet=max_per_tracklet,
        clear=clear,
    )
    n_crops = sum(summary["n_crops"].values())
    per_class = ", ".join(
        f"{k}: {v}" for k, v in sorted(summary["n_crops"].items())
    )
    print(
        f"Exported {n_crops} pseudo-labelled crop(s) from "
        f"{summary['n_selected']} trusted tracklet(s) "
        f"({summary['n_rejected']} rejected)"
        + (f": {per_class}" if per_class else ".")
    )
    reasons = {}
    for rej in summary["rejected"].values():
        for reason in rej.values():
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        print(
            "Rejected tracklets: "
            + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        )
    if export_only or n_crops == 0:
        if n_crops == 0:
            print("Nothing to train on; classifier unchanged.")
        return None

    root, _ = identity_paths(project_path)
    best = root / "training" / "weights" / "best.pt"
    if model is None:
        if best.exists():
            # train_identity(overwrite=True) wipes the run directory, so
            # keep a copy of the starting weights outside it.
            start = root / "best_before_refine.pt"
            shutil.copy(best, start)
            model = start
        else:
            model = "yolo11n-cls"
    best = train_identity(
        project_path,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        device=device,
        batch=batch,
        overwrite=True,
    )
    print(f"Refined identity classifier saved to {best}")
    print(
        "Next: re-run octron predict --identity with the new weights, "
        "octron link, and octron evaluate-identity to check the gain."
    )
    return best


def run_evaluate_identity(project_path, folders, split="test", iou_thresh=0.5):
    """Score linked prediction folders against the project annotations.

    Prints one line per folder with IDF1 for raw tracks, per-frame
    identity and linked identity plus the linked accuracy, and writes
    ``identity_eval.json`` into each folder.

    Returns
    -------
    dict
        ``{folder: report}``.

    """
    from octron.yolo_octron.identity_eval import evaluate_folder

    if isinstance(folders, (str, Path)):
        folders = [folders]
    reports = {}
    print(
        f"{'folder':<24} {'frames':>6} {'gt':>5} {'IDF1 trk':>9} "
        f"{'IDF1 frm':>9} {'IDF1 lnk':>9} {'acc lnk':>8}"
    )
    for folder in folders:
        folder = Path(folder)
        report = evaluate_folder(
            project_path, folder, split=split, iou_thresh=iou_thresh
        )
        reports[folder.as_posix()] = report
        print(
            f"{folder.name[:24]:<24} {report['n_frames']:>6} "
            f"{report['n_gt']:>5} {report['idf1_tracks']:>9.3f} "
            f"{report['idf1_frame_identity']:>9.3f} "
            f"{report['idf1_linked']:>9.3f} {report['linked_accuracy']:>8.3f}"
        )
    return reports
