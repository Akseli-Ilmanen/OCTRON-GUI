"""Identity classification for the species-detector workflow.

OCTRON represents an *individual* as a ``(label, suffix)`` pair. The YOLO
detector is trained on the ``label`` only ("bird"), so every suffix of one
label is pooled into a single detector class. Identity ("bird 1" versus
"bird 2") is decided afterwards by a small YOLO *classification* model
trained on square crops of the annotated masks, one class per
``(label, suffix)`` pair.

This module provides three things:

* :func:`build_identity_dataset` - exports crops from the object organizer
  into the ultralytics classification folder layout under
  ``<project>/model_identity/training_data/``.
* :func:`train_identity` - trains a ``*-cls`` model on that dataset.
* :class:`IdentityClassifier` - inference wrapper used by
  ``YOLO_octron.predict_batch`` to score every tracked box per frame.

Entries whose suffix is empty are treated as "unidentified": they still
train the detector but are skipped here.
"""

import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
from loguru import logger

from octron.cameras import load_layout_for_folder

IDENTITY_DIR = "model_identity"
IDENTITY_DATA_DIR = "training_data"
IDENTITY_CLASSES_FILENAME = "identity_classes.json"
IDENTITY_PROB_PREFIX = "identity_prob_"
DEFAULT_IDENTITY_MODEL = "yolo11n-cls"
DEFAULT_IDENTITY_IMGSZ = 224
DEFAULT_CROP_PADDING = 0.1


def sanitize_class_name(label, suffix):
    """Build the classifier class name for a ``(label, suffix)`` pair.

    Class names double as folder names in the ultralytics dataset, so
    whitespace is replaced by underscores.

    Parameters
    ----------
    label : str
        Label name (detector class), e.g. ``"bird"``.
    suffix : str
        Individual suffix, e.g. ``"1"``.

    Returns
    -------
    str
        ``"<label>_<suffix>"`` with spaces replaced by underscores.

    """
    label = "_".join(str(label).strip().split())
    suffix = "_".join(str(suffix).strip().split())
    return f"{label}_{suffix}"


def identity_prob_column(class_name):
    """Return the tracking CSV column name holding a class probability."""
    return f"{IDENTITY_PROB_PREFIX}{class_name}"


def mask_bbox(mask):
    """Return the ``[x_min, y_min, x_max, y_max]`` box of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D array; non-zero pixels are foreground.

    Returns
    -------
    np.ndarray or None
        Box in pixel coordinates (max exclusive) or None for an empty mask.

    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_idx = np.flatnonzero(rows)
    x_idx = np.flatnonzero(cols)
    return np.array(
        [x_idx[0], y_idx[0], x_idx[-1] + 1, y_idx[-1] + 1], dtype=float
    )


def square_crop(frame, xyxy, padding=DEFAULT_CROP_PADDING):
    """Cut a square, zero-padded crop around a box.

    The crop is centred on the box, its side is the longer box side
    grown by ``padding`` on each side, and pixels outside the frame are
    filled with zeros. A square crop means ultralytics' centre-crop
    classification transform never cuts the animal.

    Parameters
    ----------
    frame : np.ndarray
        Image array ``(H, W)`` or ``(H, W, C)``.
    xyxy : array-like
        ``[x_min, y_min, x_max, y_max]`` in frame pixels.
    padding : float
        Fraction of the longer side added on every side.

    Returns
    -------
    np.ndarray
        Square crop of side ``max(1, round(side))`` with the frame dtype.

    """
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * padding)
    side = max(1, int(round(side)))
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = left + side
    bottom = top + side

    height, width = frame.shape[:2]
    out_shape = (side, side) + frame.shape[2:]
    crop = np.zeros(out_shape, dtype=frame.dtype)
    src_x0, src_y0 = max(left, 0), max(top, 0)
    src_x1, src_y1 = min(right, width), min(bottom, height)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - left, src_y0 - top
        crop[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = frame[src_y0:src_y1, src_x0:src_x1]
    return crop


def identity_crops(mask, get_frame, layout=None, padding=DEFAULT_CROP_PADDING):
    """Yield ``(camera_name, crop)`` for every camera the mask touches.

    Without a layout the whole frame is one camera and ``camera_name``
    is None. With a layout, each camera's part of the mask gives its own
    crop cut from that camera's region only, so context never crosses a
    camera border. ``get_frame`` is called lazily (at most once) so
    frames with an empty mask are never decoded.
    """
    mask = np.asarray(mask)
    frame = None
    if layout is None:
        box = mask_bbox(mask)
        if box is None:
            return
        yield None, square_crop(np.asarray(get_frame()), box, padding)
        return
    for cam in layout:
        box = mask_bbox(cam.crop(mask))
        if box is None:
            continue
        if frame is None:
            frame = np.asarray(get_frame())
        yield cam.name, square_crop(cam.crop(frame), box, padding)


def collect_identity_entries(project_path):
    """Find every ``(label, suffix)`` mask store in a project.

    Walks the per-video ``object_organizer.json`` files (depth 1) the same
    way detector training does, but keeps the suffix.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project root.

    Returns
    -------
    entries : list of dict
        One dict per organizer entry with a non-empty suffix. Keys:
        ``label, suffix, class_name, subfolder, zarr_path,
        video_file_path, video_hash, num_frames, height, width``.
    n_skipped : int
        Number of entries skipped because their suffix was empty.

    """
    from octron.yolo_octron.helpers.training import (
        find_files_with_depth_limit,
        load_object_organizer,
    )

    project_path = Path(project_path)
    entries = []
    n_skipped = 0
    for organizer_json in find_files_with_depth_limit(
        project_path, "object_organizer.json", 1
    ):
        organizer = load_object_organizer(organizer_json)
        for entry in organizer.get("entries", {}).values():
            suffix = str(entry.get("suffix", "") or "").strip()
            if not suffix:
                n_skipped += 1
                continue
            meta = entry["prediction_layer_metadata"]
            num_frames, height, width = meta["data_shape"]
            entries.append(
                {
                    "label": entry["label"],
                    "suffix": suffix,
                    "class_name": sanitize_class_name(entry["label"], suffix),
                    "subfolder": organizer_json.parent.name,
                    "zarr_path": project_path / Path(meta["zarr_path"]),
                    "video_file_path": (
                        project_path / Path(meta["video_file_path"])
                    ).resolve(),
                    "video_hash": meta.get("video_hash", ""),
                    "num_frames": int(num_frames),
                    "height": int(height),
                    "width": int(width),
                }
            )
    return entries, n_skipped


def identity_paths(project_path):
    """Return the identity model directory and its dataset directory."""
    root = Path(project_path) / IDENTITY_DIR
    return root, root / IDENTITY_DATA_DIR


def build_identity_dataset(
    project_path,
    padding=DEFAULT_CROP_PADDING,
    train_fraction=None,
    val_fraction=None,
    seed=None,
    buffer=None,
    overwrite=False,
    verbose=False,
):
    """Export identity crops in the ultralytics classification layout.

    Crops are written to
    ``<project>/model_identity/training_data/<split>/<class>/<name>.png``
    and a ``identity_classes.json`` mapping class name to label and
    suffix is written next to the split folders. Frames are split with
    the same episode-aware block split as detector training, decided
    once per video over the union of all annotated frames so that all
    individuals of one frame land in the same split.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project root.
    padding : float
        Fractional padding around each mask box, see :func:`square_crop`.
    train_fraction, val_fraction : float or None
        Split fractions; None reads ``config.yaml``.
    seed : int or None
        Split seed; None reads ``config.yaml``.
    buffer : int or None
        Split boundary buffer; None reads ``config.yaml``.
    overwrite : bool
        Replace an existing dataset directory.
    verbose : bool
        Log per-entry progress.

    Returns
    -------
    dict
        Summary with ``data_path``, ``classes``, ``n_crops`` per split
        and ``n_skipped_no_suffix``.

    """
    from PIL import Image

    from octron import config
    from octron.yolo_octron.helpers.training import train_test_val

    project_path = Path(project_path)
    _, data_path = identity_paths(project_path)
    if data_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Identity dataset exists at {data_path}. "
                "Pass overwrite=True (--overwrite) to rebuild it."
            )
        shutil.rmtree(data_path)

    entries, n_skipped = collect_identity_entries(project_path)
    if not entries:
        raise ValueError(
            "No annotated objects with a suffix found. Identity training "
            "needs individuals labelled as '<label> <suffix>' "
            "(e.g. 'bird 1', 'bird 2')."
        )
    if n_skipped:
        logger.info(
            f"Skipped {n_skipped} object(s) without a suffix "
            "(unidentified individuals train the detector only)."
        )

    if train_fraction is None or val_fraction is None:
        cfg_train, cfg_val = config.get_split_fractions()
        if train_fraction is None:
            train_fraction = cfg_train
        if val_fraction is None:
            val_fraction = cfg_val
    seed = config.get_split_seed() if seed is None else seed
    buffer = config.get_split_buffer() if buffer is None else buffer

    from napari_pyav._reader import FastVideoReader

    from octron.sam_octron.helpers.sam_zarr import (
        get_annotated_frames,
        load_image_zarr,
    )

    # Group entries per video so one frame is read once and one split
    # decision covers every individual visible in it.
    by_video = {}
    for entry in entries:
        by_video.setdefault(entry["subfolder"], []).append(entry)

    classes = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for subfolder, video_entries in by_video.items():
        video_path = video_entries[0]["video_file_path"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video = FastVideoReader(video_path, read_format="rgb24")
        # Mosaic videos: one crop per camera the individual is visible in
        layout = load_layout_for_folder(project_path / subfolder)

        masks_per_entry = []
        all_frames = []
        for entry in video_entries:
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
            frames = np.asarray(get_annotated_frames(masks), dtype=int)
            masks_per_entry.append((entry, masks, frames))
            all_frames.append(frames)
            classes[entry["class_name"]] = {
                "label": entry["label"],
                "suffix": entry["suffix"],
            }
        union = np.unique(np.concatenate(all_frames)) if all_frames else []
        if len(union) == 0:
            continue
        split = train_test_val(
            union,
            training_fraction=train_fraction,
            validation_fraction=val_fraction,
            random_seed=seed,
            buffer=buffer,
            verbose=verbose,
        )
        frame_to_split = {}
        for split_name in ("train", "val", "test"):
            for f in split[split_name]:
                frame_to_split[int(f)] = split_name

        for entry, masks, frames in masks_per_entry:
            for frame_idx in frames:
                split_name = frame_to_split.get(int(frame_idx))
                if split_name is None:
                    continue  # dropped by the split buffer
                mask = np.asarray(masks[frame_idx])
                get_frame = partial(video.__getitem__, int(frame_idx))
                for cam_name, crop in identity_crops(
                    mask, get_frame, layout, padding
                ):
                    out_dir = data_path / split_name / entry["class_name"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    suffix = f"_{cam_name}" if cam_name else ""
                    Image.fromarray(crop).save(
                        out_dir / f"{subfolder}_{int(frame_idx)}{suffix}.png"
                    )
                    counts[split_name] += 1
            if verbose:
                logger.info(
                    f"{subfolder}: exported {entry['class_name']} "
                    f"({len(frames)} annotated frames)"
                )

    _ensure_val_examples(data_path, classes)
    data_path.mkdir(parents=True, exist_ok=True)
    with open(data_path / IDENTITY_CLASSES_FILENAME, "w") as f:
        json.dump(classes, f, indent=4, sort_keys=True)
    summary = {
        "data_path": data_path,
        "classes": classes,
        "n_crops": counts,
        "n_skipped_no_suffix": n_skipped,
    }
    logger.info(
        f"Identity dataset: {len(classes)} classes, "
        f"{counts['train']} train / {counts['val']} val / "
        f"{counts['test']} test crops at {data_path}"
    )
    return summary


def _ensure_val_examples(data_path, classes):
    """Give every class at least one validation crop.

    ultralytics classification training wants each class present in
    ``val``; a class with very few annotated frames may otherwise end up
    train-only. Moves the last training crop over when needed.
    """
    for class_name in classes:
        val_dir = data_path / "val" / class_name
        train_dir = data_path / "train" / class_name
        if val_dir.exists() and any(val_dir.iterdir()):
            continue
        if not train_dir.exists():
            continue
        train_files = sorted(train_dir.iterdir())
        if len(train_files) < 2:
            continue
        val_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train_files[-1]), str(val_dir / train_files[-1].name))
        logger.warning(
            f"Class {class_name} had no validation crops; moved one "
            "training crop to val."
        )


def resolve_identity_model_path(model):
    """Turn a model name or path into a ``.pt`` path in the model cache.

    A bare name such as ``yolo11n-cls`` resolves to
    ``<model cache>/yolo11n-cls.pt``; ultralytics downloads the official
    weights to that path on first use.
    """
    from octron import config

    model = str(model)
    path = Path(model)
    if path.suffix == ".pt" and path.exists():
        return path
    name = path.name if path.suffix == ".pt" else f"{path.name}.pt"
    return config.get_yolo_models_dir() / name


def train_identity(
    project_path,
    model=DEFAULT_IDENTITY_MODEL,
    imgsz=DEFAULT_IDENTITY_IMGSZ,
    epochs=50,
    device="auto",
    batch=-1,
    overwrite=False,
):
    """Train the identity classifier on the exported crop dataset.

    Parameters
    ----------
    project_path : str or Path
        OCTRON project root (dataset must exist, see
        :func:`build_identity_dataset`).
    model : str or Path
        Classification model name (``yolo11n-cls``) or ``.pt`` path.
    imgsz : int
        Classifier input size.
    epochs : int
        Training epochs.
    device : str
        ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``.
    batch : int
        Batch size; -1 lets ultralytics choose.
    overwrite : bool
        Remove a previous training run first.

    Returns
    -------
    Path
        Path to ``best.pt``.

    """
    from ultralytics import YOLO

    from octron.test_gpu import auto_device

    root, data_path = identity_paths(project_path)
    if not (data_path / "train").exists():
        raise FileNotFoundError(
            f"No identity dataset at {data_path}. Run the dataset export "
            "first (octron train-identity does this automatically)."
        )
    run_dir = root / "training"
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)

    if device == "auto":
        device = auto_device()
    model_path = resolve_identity_model_path(model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model_path.as_posix())
    logger.info(
        f"Training identity classifier {model_path.name} on "
        f"{data_path} ({epochs} epochs, imgsz={imgsz}, device={device})"
    )
    yolo.train(
        data=data_path.resolve().as_posix(),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch if batch > 0 else 16,
        project=root.resolve().as_posix(),
        name=run_dir.name,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best = run_dir / "weights" / "best.pt"
    # ultralytics may place the run elsewhere (e.g. under its runs_dir
    # when the project path is relative); trust the trainer's record.
    trainer_best = getattr(getattr(yolo, "trainer", None), "best", None)
    if not best.exists() and trainer_best and Path(trainer_best).exists():
        best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(trainer_best, best)
        last = Path(trainer_best).with_name("last.pt")
        if last.exists():
            shutil.copy(last, best.with_name("last.pt"))
        logger.info(f"Copied weights from {Path(trainer_best).parent}")
    if not best.exists():
        raise RuntimeError(f"Training finished but {best} is missing.")
    # Keep the class metadata next to the weights so predict can find
    # the label/suffix of every class without the dataset.
    classes_src = data_path / IDENTITY_CLASSES_FILENAME
    if classes_src.exists():
        shutil.copy(classes_src, best.parent / IDENTITY_CLASSES_FILENAME)
    logger.info(f"Identity classifier saved to {best}")
    return best


class IdentityClassifier:
    """Score tracked boxes against the known individuals.

    Parameters
    ----------
    weights : str or Path
        Trained ``*-cls`` weights (``best.pt``).
    device : str
        Inference device.
    imgsz : int
        Classifier input size.
    padding : float
        Crop padding, must match the value used for the dataset.

    """

    def __init__(
        self,
        weights,
        device="cpu",
        imgsz=DEFAULT_IDENTITY_IMGSZ,
        padding=DEFAULT_CROP_PADDING,
    ):
        """Load the classifier weights and class metadata."""
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Identity weights not found: {self.weights}"
            )
        self.device = device
        self.imgsz = imgsz
        self.padding = padding
        self.model = YOLO(self.weights.as_posix())
        task = getattr(self.model, "task", None)
        if task != "classify":
            raise ValueError(
                f"Identity weights must be a classification model, "
                f"got task={task!r}: {self.weights}"
            )
        names = self.model.names
        self.class_names = [names[i] for i in range(len(names))]
        self.classes = self._load_classes()
        self.prob_columns = [
            identity_prob_column(name) for name in self.class_names
        ]

    def _load_classes(self):
        """Load ``identity_classes.json`` or parse names as label_suffix."""
        candidates = [
            self.weights.parent / IDENTITY_CLASSES_FILENAME,
            self.weights.parent.parent.parent
            / IDENTITY_DATA_DIR
            / IDENTITY_CLASSES_FILENAME,
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    classes = json.load(f)
                if all(n in classes for n in self.class_names):
                    return {n: classes[n] for n in self.class_names}
        logger.warning(
            "identity_classes.json not found next to the weights; "
            "deriving label/suffix from class names."
        )
        classes = {}
        for name in self.class_names:
            label, _, suffix = name.rpartition("_")
            classes[name] = {"label": label or name, "suffix": suffix}
        return classes

    def label_of(self, class_name):
        """Return the detector label a classifier class belongs to."""
        return self.classes[class_name]["label"]

    def candidate_indices(self, label):
        """Return indices of classifier classes belonging to ``label``.

        Matching ignores whitespace/underscore differences so a label
        ``"my bird"`` matches class ``"my_bird_1"``.
        """
        key = "_".join(str(label).split())
        return [
            i
            for i, name in enumerate(self.class_names)
            if "_".join(self.label_of(name).split()) == key
        ]

    def classify(self, frame, boxes):
        """Return class probabilities for every box in a frame.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame ``(H, W, 3)``.
        boxes : array-like
            ``(N, 4)`` boxes as ``[x_min, y_min, x_max, y_max]``.

        Returns
        -------
        np.ndarray
            ``(N, n_classes)`` probabilities; empty ``(0, n_classes)``
            when no boxes are given.

        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        n_classes = len(self.class_names)
        if boxes.shape[0] == 0:
            return np.zeros((0, n_classes), dtype=float)
        crops = [square_crop(frame, box, self.padding) for box in boxes]
        results = self.model.predict(
            source=crops,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        probs = np.zeros((len(crops), n_classes), dtype=float)
        for i, result in enumerate(results):
            if result.probs is not None:
                probs[i] = result.probs.data.cpu().numpy()
        return probs

    def decide(self, probs_row, label):
        """Pick the identity for one detection given its detector label.

        Only classes of the same label compete. Returns ``(None, nan)``
        when no class belongs to the label.
        """
        idx = self.candidate_indices(label)
        if not idx:
            return None, float("nan")
        idx = np.asarray(idx)
        best = idx[int(np.argmax(probs_row[idx]))]
        return self.class_names[best], float(probs_row[best])
