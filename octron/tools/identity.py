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
