r"""Multi-camera mosaic layout for OCTRON.

OCTRON is gaining multi-camera support: the input video is a pre-tiled
mosaic of several synchronised cameras. A per-video ``cameras.json``
(:data:`CAMERAS_FILENAME`) describes the axis-aligned rectangle of each
camera in mosaic pixel coordinates, so downstream code can crop a camera
out of a mosaic frame or assign a detection box (in mosaic coordinates)
to the camera it belongs to.

This module is intentionally stdlib + numpy + loguru only (no napari, no
GUI, no project-layout assumptions) so it can be imported from the CLI, the
GUI, and headless prediction code alike.

Example ``cameras.json``::

    {
        "version": 1,
        "frame_width": 1920,
        "frame_height": 1080,
        "video_file_path": null,
        "video_hash": null,
        "cameras": [
            {
                "name": "cam0",
                "x_min": 0,
                "y_min": 0,
                "x_max": 960,
                "y_max": 1080,
            },
            {
                "name": "cam1",
                "x_min": 960,
                "y_min": 0,
                "x_max": 1920,
                "y_max": 1080,
            },
        ],
    }

"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

CAMERAS_FILENAME = "cameras.json"
CAMERAS_VERSION = 1


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """One camera's axis-aligned rectangle within a mosaic frame.

    Coordinates are half-open pixel intervals: a point ``(x, y)`` belongs
    to the camera when ``x_min <= x < x_max`` and ``y_min <= y < y_max``.

    Attributes
    ----------
    name : str
        Camera identifier, unique within a :class:`CameraLayout`.
    x_min, y_min, x_max, y_max : int
        Rectangle bounds in mosaic pixel coordinates.

    """

    name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Return the rectangle width in pixels (``x_max - x_min``)."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Return the rectangle height in pixels (``y_max - y_min``)."""
        return self.y_max - self.y_min

    def contains_point(self, x, y) -> bool:
        """Return whether ``(x, y)`` falls inside this camera's rectangle.

        Parameters
        ----------
        x, y : float
            Point in mosaic pixel coordinates.

        Returns
        -------
        bool
            ``True`` when ``x_min <= x < x_max`` and
            ``y_min <= y < y_max``.

        """
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def crop(self, array: np.ndarray) -> np.ndarray:
        """Return the slice of ``array`` covered by this camera.

        Parameters
        ----------
        array : np.ndarray
            A ``(H, W)`` or ``(H, W, C)`` array in mosaic pixel
            coordinates (row = y, col = x).

        Returns
        -------
        np.ndarray
            The ``array[y_min:y_max, x_min:x_max, ...]`` view.

        """
        return array[self.y_min : self.y_max, self.x_min : self.x_max, ...]

    def shift_box(self, xyxy) -> np.ndarray:
        """Shift a mosaic-coordinate box into this camera's local frame.

        Parameters
        ----------
        xyxy : array-like
            ``[x0, y0, x1, y1]`` box in mosaic pixel coordinates.

        Returns
        -------
        np.ndarray
            ``[x0 - x_min, y0 - y_min, x1 - x_min, y1 - y_min]``, clipped
            to ``[0, width]`` (x) and ``[0, height]`` (y).

        """
        x0, y0, x1, y1 = np.asarray(xyxy, dtype=float)
        shifted = np.array(
            [
                x0 - self.x_min,
                y0 - self.y_min,
                x1 - self.x_min,
                y1 - self.y_min,
            ]
        )
        shifted[[0, 2]] = np.clip(shifted[[0, 2]], 0, self.width)
        shifted[[1, 3]] = np.clip(shifted[[1, 3]], 0, self.height)
        return shifted

    def to_dict(self) -> dict:
        """Return a plain ``dict`` (JSON-serialisable) for this camera."""
        return {
            "name": self.name,
            "x_min": int(self.x_min),
            "y_min": int(self.y_min),
            "x_max": int(self.x_max),
            "y_max": int(self.y_max),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        """Build a :class:`Camera` from ``d`` (as :meth:`to_dict` produces)."""
        return cls(
            name=d["name"],
            x_min=int(d["x_min"]),
            y_min=int(d["y_min"]),
            x_max=int(d["x_max"]),
            y_max=int(d["y_max"]),
        )


# ---------------------------------------------------------------------------
# CameraLayout
# ---------------------------------------------------------------------------


@dataclass
class CameraLayout:
    """The set of camera rectangles that tile one mosaic video frame.

    Attributes
    ----------
    cameras : list of Camera
        The individual camera rectangles.
    frame_width, frame_height : int
        Dimensions of the full mosaic frame in pixels.
    video_file_path : str, optional
        Provenance: the video this layout was created for.
    video_hash : str, optional
        Provenance: a hash of the video this layout was created for.

    """

    cameras: list = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    video_file_path: str | None = None
    video_hash: str | None = None

    # -- construction ------------------------------------------------

    @classmethod
    def full_frame(cls, width: int, height: int, name: str = "cam0"):
        """Return a single-camera layout covering the whole frame.

        Parameters
        ----------
        width, height : int
            Frame dimensions in pixels.
        name : str
            Name given to the single camera. Default ``"cam0"``.

        Returns
        -------
        CameraLayout
            A layout with one :class:`Camera` spanning
            ``[0, width) x [0, height)``.

        """
        return cls(
            cameras=[
                Camera(name=name, x_min=0, y_min=0, x_max=width, y_max=height)
            ],
            frame_width=width,
            frame_height=height,
        )

    @classmethod
    def from_rectangles(
        cls, rectangles, frame_width: int, frame_height: int, names=None
    ):
        """Build a layout from napari-Shapes-style rectangle corners.

        Parameters
        ----------
        rectangles : iterable of array-like
            Each item is a ``(4, 2)`` array of ``(row, col)`` = ``(y, x)``
            corners, in any corner order, as returned by napari's Shapes
            layer.
        frame_width, frame_height : int
            Dimensions of the full mosaic frame in pixels; each rectangle
            is clipped to ``[0, frame_width]`` x ``[0, frame_height]``.
        names : iterable of str, optional
            One name per rectangle, in order. A blank/``None`` entry (or
            an omitted ``names``) falls back to the default
            ``"cam0"``, ``"cam1"``, ... naming.

        Returns
        -------
        CameraLayout

        Raises
        ------
        ValueError
            If a rectangle has zero area after snapping/clipping.

        """
        names = list(names) if names is not None else []
        cameras = []
        for i, rect in enumerate(rectangles):
            corners = np.asarray(rect, dtype=float)
            rows = corners[:, 0]
            cols = corners[:, 1]
            x_min = int(np.floor(cols.min()))
            x_max = int(np.ceil(cols.max()))
            y_min = int(np.floor(rows.min()))
            y_max = int(np.ceil(rows.max()))

            x_min = int(np.clip(x_min, 0, frame_width))
            x_max = int(np.clip(x_max, 0, frame_width))
            y_min = int(np.clip(y_min, 0, frame_height))
            y_max = int(np.clip(y_max, 0, frame_height))

            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    f"Rectangle {i} has zero area after snapping/clipping "
                    f"to the frame ({x_min}, {y_min}, {x_max}, {y_max})."
                )

            name = names[i] if i < len(names) else None
            if not name or not str(name).strip():
                name = f"cam{i}"
            cameras.append(
                Camera(
                    name=str(name),
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )
            )
        return cls(
            cameras=cameras,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain ``dict`` (JSON-serialisable) for this layout."""
        return {
            "version": CAMERAS_VERSION,
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "video_file_path": self.video_file_path,
            "video_hash": self.video_hash,
            "cameras": [cam.to_dict() for cam in self.cameras],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraLayout":
        """Build a :class:`CameraLayout` from ``d`` (as :meth:`to_dict`)."""
        return cls(
            cameras=[Camera.from_dict(c) for c in d.get("cameras", [])],
            frame_width=int(d["frame_width"]),
            frame_height=int(d["frame_height"]),
            video_file_path=d.get("video_file_path"),
            video_hash=d.get("video_hash"),
        )

    @classmethod
    def load(cls, path) -> "CameraLayout":
        """Load a :class:`CameraLayout` from a ``cameras.json`` file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path, overwrite: bool = False) -> Path:
        """Write this layout to ``path`` as indented, sorted-key JSON.

        Parameters
        ----------
        path : path-like
            Destination file.
        overwrite : bool
            Whether to overwrite an existing file. Default ``False``.

        Returns
        -------
        Path
            The written path.

        Raises
        ------
        FileExistsError
            If ``path`` already exists and ``overwrite`` is ``False``.

        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)
        return path

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Validate this layout, raising on structural errors.

        Raises
        ------
        ValueError
            If there are no cameras, a duplicate or empty camera name, a
            zero/negative-area camera, or a camera extending outside the
            frame.

        Notes
        -----
        Overlapping cameras are allowed but logged as a ``loguru``
        warning listing the overlapping name pairs.

        """
        if not self.cameras:
            raise ValueError("CameraLayout has no cameras.")

        seen_names = set()
        for cam in self.cameras:
            if not cam.name or not str(cam.name).strip():
                raise ValueError("Camera has an empty name.")
            if cam.name in seen_names:
                raise ValueError(f"Duplicate camera name: {cam.name!r}.")
            seen_names.add(cam.name)

            if cam.width <= 0 or cam.height <= 0:
                raise ValueError(
                    f"Camera {cam.name!r} has zero or negative area "
                    f"({cam.width} x {cam.height})."
                )
            if (
                cam.x_min < 0
                or cam.y_min < 0
                or cam.x_max > self.frame_width
                or cam.y_max > self.frame_height
            ):
                raise ValueError(
                    f"Camera {cam.name!r} ({cam.x_min}, {cam.y_min}, "
                    f"{cam.x_max}, {cam.y_max}) lies outside the frame "
                    f"({self.frame_width} x {self.frame_height})."
                )

        overlaps = []
        for i in range(len(self.cameras)):
            for j in range(i + 1, len(self.cameras)):
                a, b = self.cameras[i], self.cameras[j]
                x_overlap = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
                y_overlap = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
                if x_overlap > 0 and y_overlap > 0:
                    overlaps.append((a.name, b.name))
        if overlaps:
            pairs = ", ".join(f"({a}, {b})" for a, b in overlaps)
            logger.warning(f"CameraLayout has overlapping cameras: {pairs}")

    def is_single_full_frame(self) -> bool:
        """Return whether this layout is one camera spanning the frame."""
        if len(self.cameras) != 1:
            return False
        cam = self.cameras[0]
        return (
            cam.x_min == 0
            and cam.y_min == 0
            and cam.x_max == self.frame_width
            and cam.y_max == self.frame_height
        )

    # -- assignment ------------------------------------------------------

    def assign_box(self, xyxy) -> Camera | None:
        """Return the camera a mosaic-coordinate box belongs to.

        Parameters
        ----------
        xyxy : array-like
            ``[x0, y0, x1, y1]`` box in mosaic pixel coordinates.

        Returns
        -------
        Camera or None
            The camera whose rectangle contains the box centre. If
            several cameras contain the centre (overlap), the one with
            the largest intersection area with the box is returned.
            ``None`` if the centre lies inside no camera.

        """
        x0, y0, x1, y1 = np.asarray(xyxy, dtype=float)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        candidates = [
            cam for cam in self.cameras if cam.contains_point(cx, cy)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def _intersection_area(cam: Camera) -> float:
            ix0 = max(x0, cam.x_min)
            iy0 = max(y0, cam.y_min)
            ix1 = min(x1, cam.x_max)
            iy1 = min(y1, cam.y_max)
            return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

        return max(candidates, key=_intersection_area)

    def assign_boxes(self, boxes) -> list:
        """Return :meth:`assign_box` applied to each row of ``boxes``.

        Parameters
        ----------
        boxes : array-like
            ``(N, 4)`` array of ``[x0, y0, x1, y1]`` boxes in mosaic
            pixel coordinates.

        Returns
        -------
        list of (Camera or None)

        """
        boxes = np.asarray(boxes, dtype=float)
        return [self.assign_box(box) for box in boxes]

    # -- lookup / dunder ---------------------------------------------------

    def get(self, name: str) -> Camera:
        """Return the camera named ``name``.

        Raises
        ------
        KeyError
            If no camera with that name exists.

        """
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"No camera named {name!r} in this layout.")

    @property
    def names(self) -> list:
        """Return the list of camera names, in order."""
        return [cam.name for cam in self.cameras]

    def __len__(self) -> int:
        """Return the number of cameras in this layout."""
        return len(self.cameras)

    def __iter__(self):
        """Iterate over the cameras in this layout, in order."""
        return iter(self.cameras)


# ---------------------------------------------------------------------------
# File resolution helpers
# ---------------------------------------------------------------------------


def load_layout_for_folder(folder) -> "CameraLayout | None":
    """Return the multi-camera layout saved in ``folder``, or None.

    None when there is no ``cameras.json`` in the folder, or when it
    describes a single full-frame camera (nothing to split).
    """
    path = Path(folder) / CAMERAS_FILENAME
    if not path.exists():
        return None
    layout = CameraLayout.load(path)
    if len(layout) < 2 or layout.is_single_full_frame():
        return None
    return layout


def mask_within_camera(mask, camera):
    """Return ``mask`` with everything outside ``camera`` set to zero.

    Keeps mosaic coordinates, so downstream code that expects
    full-frame masks (polygon export, bbox export) needs no offsets.
    """
    out = np.zeros_like(mask)
    ys = slice(camera.y_min, camera.y_max)
    xs = slice(camera.x_min, camera.x_max)
    out[ys, xs] = mask[ys, xs]
    return out


def sibling_cameras_path(video_path) -> Path:
    """Return the default ``cameras.json`` sibling of ``video_path``.

    ``<video_parent>/<video_stem>_cameras.json``
    """
    video_path = Path(video_path)
    return video_path.parent / f"{video_path.stem}_cameras.json"


def find_cameras_file(video_path, explicit=None) -> Path | None:
    """Resolve which ``cameras.json`` file (if any) applies to a video.

    Parameters
    ----------
    video_path : path-like
        The video file.
    explicit : path-like, optional
        An explicitly given cameras file.

    Returns
    -------
    Path or None
        ``explicit`` (resolved) when given; otherwise the sibling
        ``<video_stem>_cameras.json`` file if it exists; otherwise
        ``None``.

    Raises
    ------
    FileNotFoundError
        If ``explicit`` is given but does not exist.

    """
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(
                f"Explicit cameras file not found: {explicit}"
            )
        return explicit

    sibling = sibling_cameras_path(video_path)
    if sibling.exists():
        return sibling
    return None


def resolve_layout(
    video_path, frame_width: int, frame_height: int, cameras=None
):
    """Resolve the :class:`CameraLayout` to use for a video.

    Parameters
    ----------
    video_path : path-like
        The video file (used to find a sibling ``cameras.json`` when
        ``cameras`` is ``None`` or a path).
    frame_width, frame_height : int
        Actual dimensions of the mosaic video frame; a loaded layout
        whose dimensions differ raises :class:`ValueError`.
    cameras : None, path-like, dict, or CameraLayout, optional
        - ``None``: look for a sibling ``cameras.json``; fall back to a
          single full-frame camera if none is found.
        - path-like: load the layout from that file.
        - dict: build a layout via :meth:`CameraLayout.from_dict`.
        - :class:`CameraLayout`: used as-is.

    Returns
    -------
    tuple[CameraLayout, bool]
        The resolved layout, and whether it came from an explicit
        source (``False`` only when falling back to a full-frame
        layout).

    Raises
    ------
    ValueError
        If a loaded/given layout's ``frame_width``/``frame_height``
        differ from ``frame_width``/``frame_height``.

    """
    if isinstance(cameras, CameraLayout):
        layout = cameras
        explicit = True
    elif isinstance(cameras, dict):
        layout = CameraLayout.from_dict(cameras)
        explicit = True
    elif cameras is not None:
        layout = CameraLayout.load(cameras)
        explicit = True
    else:
        found = find_cameras_file(video_path)
        if found is not None:
            layout = CameraLayout.load(found)
            explicit = True
        else:
            layout = CameraLayout.full_frame(frame_width, frame_height)
            explicit = False

    if (
        layout.frame_width != frame_width
        or layout.frame_height != frame_height
    ):
        raise ValueError(
            f"CameraLayout frame size ({layout.frame_width} x "
            f"{layout.frame_height}) does not match the video frame "
            f"size ({frame_width} x {frame_height})."
        )

    return layout, explicit


# ---------------------------------------------------------------------------
# apply_cameras
# ---------------------------------------------------------------------------


def apply_cameras(
    cameras_path,
    project_path=None,
    videos=(),
    force: bool = False,
) -> list:
    """Distribute one ``cameras.json`` file to project and video locations.

    Parameters
    ----------
    cameras_path : path-like
        The source ``cameras.json`` file to distribute (validated
        before copying).
    project_path : path-like, optional
        An OCTRON project directory. The file is copied to
        ``<project>/<subfolder>/cameras.json`` for every depth-1
        subfolder containing an ``object_organizer.json``.
    videos : iterable of path-like, optional
        Video files. The file is copied to
        :func:`sibling_cameras_path` of each.
    force : bool
        Overwrite existing destination files. Default ``False``
        (existing files are skipped, with a logged warning).

    Returns
    -------
    list of Path
        The destination paths actually written.

    """
    cameras_path = Path(cameras_path)
    layout = CameraLayout.load(cameras_path)
    layout.validate()

    with open(cameras_path) as f:
        raw_text = f.read()

    written = []

    def _write(dest: Path) -> None:
        if dest.exists() and not force:
            logger.warning(
                f"Skipping existing cameras file (use --force to "
                f"overwrite): {dest}"
            )
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(raw_text)
        written.append(dest)

    if project_path is not None:
        project_path = Path(project_path)
        for sub in sorted(project_path.iterdir()):
            if not sub.is_dir():
                continue
            if not (sub / "object_organizer.json").exists():
                continue
            _write(sub / CAMERAS_FILENAME)

    for video in videos:
        _write(sibling_cameras_path(video))

    return written


def split_video(
    video_path,
    layout,
    output_dir=None,
    encoder="auto",
    crf=20,
    overwrite=False,
    cameras=None,
):
    """Cut a mosaic video into one clip per camera with ffmpeg.

    Each clip is the camera rectangle of the mosaic, so positions in
    that camera's tracking output (CSV or ``<camera>.nc``) apply to the
    clip without any offset. H.264 needs even dimensions, so an odd
    width or height is rounded down by one pixel.

    Parameters
    ----------
    video_path : str or Path
        Mosaic video.
    layout : CameraLayout
        Camera rectangles.
    output_dir : str or Path, optional
        Where to write ``<camera>.mp4``; default ``<video dir>/cameras_video``.
    encoder : str
        ``'auto'`` (GPU if available), ``'nvenc'`` or ``'libx264'``.
    crf : int
        Quality (lower is better; ``-cq`` for nvenc).
    overwrite : bool
        Replace existing clips.
    cameras : list of str, optional
        Subset of camera names; default all.

    Returns
    -------
    list of Path
        Written clips, in layout order.

    """
    from octron.tools._ffmpeg import h264_codec_args, resolve_encoder

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else video_path.parent / "cameras_video"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_name = resolve_encoder(encoder)
    written = []
    for cam in layout:
        if cameras is not None and cam.name not in cameras:
            continue
        out = output_dir / f"{cam.name}.mp4"
        if out.exists() and not overwrite:
            logger.warning(f"Skipping existing {out} (pass overwrite=True)")
            continue
        width = cam.width - (cam.width % 2)
        height = cam.height - (cam.height % 2)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"crop={width}:{height}:{cam.x_min}:{cam.y_min}",
            *h264_codec_args(encoder_name, crf=crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ]
        logger.info(
            f"Cutting {cam.name}: {width}x{height} at "
            f"({cam.x_min}, {cam.y_min}) -> {out}"
        )
        subprocess.run(cmd, check=True)
        written.append(out)
    return written
