"""Run one SAM predictor per camera of a mosaic video.

SAM2 encodes every frame at a fixed square size (``image_size``, 1024 for
all shipped configs). On a mosaic video (several cameras tiled into one
frame) each camera therefore gets only a fraction of that resolution and
small animals stop segmenting well. :class:`MosaicPredictor` keeps the
annotation experience unchanged (the user clicks on the mosaic, masks
live in mosaic-sized layers) but runs SAM on each camera crop at full
encoder resolution.

Design: a single model instance is shared. OCTRON's predictor classes
keep their per-video state in ``self.inference_state`` and
``self.images``, so the wrapper holds one ``(inference_state, images)``
pair per camera and swaps them in before every call. Prompts are routed
to the camera containing them, results are pasted back into mosaic
coordinates, and propagation runs the cameras in lockstep so every
yielded frame carries the union of all cameras' objects.
"""

import numpy as np
import torch
from loguru import logger

LOGIT_BACKGROUND = -10.0  # value pasted outside a camera's crop


class CameraView:
    """Lazy crop of a ``(frames, H, W, 3)`` array for one camera."""

    def __init__(self, video_data, camera):
        """Wrap ``video_data`` so indexing returns the camera crop."""
        self.video_data = video_data
        self.camera = camera
        n = video_data.shape[0]
        self.shape = (n, camera.height, camera.width, video_data.shape[3])

    def __len__(self):
        """Return the number of frames."""
        return self.shape[0]

    def __getitem__(self, idx):
        """Return the crop for one frame index or a list of indices."""
        c = self.camera
        frames = np.asarray(self.video_data[idx])
        return np.ascontiguousarray(
            frames[..., c.y_min : c.y_max, c.x_min : c.x_max, :]
        )


class _MosaicImages:
    """``predictor.images`` stand-in: prefetch every camera's store."""

    def __init__(self, owner):
        self._owner = owner

    def __getitem__(self, indices):
        """Fetch ``indices`` into every camera's image store."""
        out = None
        for name in self._owner.layout.names:
            images = self._owner._activate(name)
            out = images[indices]
        return out


class MosaicPredictor:
    """Per-camera SAM states behind one mosaic video.

    Parameters
    ----------
    predictor : SAM2_octron, SAM2HQ_octron or SAM3_octron
        A loaded predictor (weights on device). Its ``init_state``,
        ``add_new_points_or_box``, ``add_new_mask``,
        ``propagate_in_video``, ``reset_state`` and ``remove_object`` are
        reused unchanged, one state per camera.
    layout : octron.cameras.CameraLayout
        Camera rectangles in mosaic pixel coordinates.

    """

    def __init__(self, predictor, layout):
        """Wrap ``predictor`` for the cameras in ``layout``."""
        self.predictor = predictor
        self.layout = layout
        self._states = {}  # camera name -> (inference_state, images)
        self._active = None
        self.is_initialized = False
        self.last_cameras = []  # cameras touched by the last prompt
        self.frame_height = layout.frame_height
        self.frame_width = layout.frame_width

    # -- attribute delegation ---------------------------------------
    def __getattr__(self, name):
        """Delegate everything else (image_size, device, model, ...)."""
        if name in {"predictor", "layout", "_states"}:
            raise AttributeError(name)
        return getattr(self.predictor, name)

    def _activate(self, name):
        """Swap camera ``name``'s state into the predictor."""
        state, images = self._states[name]
        if self._active != name:
            self.predictor.inference_state = state
            self.predictor.images = images
            self._active = name
        return images

    def _camera_for_xy(self, x, y):
        """Camera containing pixel ``(x, y)``, else the nearest one."""
        for cam in self.layout:
            if cam.contains_point(x, y):
                return cam
        best, best_d = None, float("inf")
        for cam in self.layout:
            cx = (cam.x_min + cam.x_max) / 2
            cy = (cam.y_min + cam.y_max) / 2
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best, best_d = cam, d
        logger.warning(
            f"Prompt at ({x:.0f}, {y:.0f}) is outside every camera; "
            f"using nearest camera '{best.name}'."
        )
        return best

    def _paste(self, camera, crop_masks):
        """Paste ``(n, 1, h, w)`` camera logits into mosaic-sized logits."""
        n = crop_masks.shape[0]
        full = torch.full(
            (n, 1, self.frame_height, self.frame_width),
            LOGIT_BACKGROUND,
            dtype=crop_masks.dtype,
            device=crop_masks.device,
        )
        c = camera
        full[:, :, c.y_min : c.y_max, c.x_min : c.x_max] = crop_masks
        return full

    # -- state lifecycle --------------------------------------------
    def init_state(self, video_data, zarr_stores):
        """Create one inference state per camera.

        Parameters
        ----------
        video_data : array-like
            Mosaic frames ``(frames, H, W, 3)`` (the napari layer data).
        zarr_stores : dict
            Camera name to the camera's resized image zarr array.

        """
        assert video_data.shape[1] == self.frame_height, (
            f"Layout height {self.frame_height} != video {video_data.shape[1]}"
        )
        assert video_data.shape[2] == self.frame_width, (
            f"Layout width {self.frame_width} != video {video_data.shape[2]}"
        )
        for cam in self.layout:
            self.predictor.init_state(
                video_data=CameraView(video_data, cam),
                zarr_store=zarr_stores[cam.name],
            )
            self._states[cam.name] = (
                self.predictor.inference_state,
                self.predictor.images,
            )
            self._active = cam.name
        self.is_initialized = True
        logger.info(
            f"Mosaic SAM: {len(self.layout)} camera states "
            f"({', '.join(self.layout.names)})"
        )

    @property
    def images(self):
        """Prefetch proxy over all camera image stores."""
        return _MosaicImages(self)

    @property
    def inference_state(self):
        """Merged read-only view used by the GUI for sanity checks."""
        merged = {
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "obj_ids": [],
            "tracking_has_started": False,
        }
        for name in self.layout.names:
            state = self._states.get(name)
            if state is None:
                continue
            state = state[0]
            for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
                for obj_idx, per_frame in state.get(key, {}).items():
                    if per_frame:
                        merged[key][(name, obj_idx)] = per_frame
            for oid in state.get("obj_ids", []):
                if oid not in merged["obj_ids"]:
                    merged["obj_ids"].append(oid)
            merged["tracking_has_started"] |= bool(
                state.get("tracking_has_started", False)
            )
        return merged

    def reset_state(self):
        """Reset every camera state."""
        for name in self._states:
            self._activate(name)
            self.predictor.reset_state()

    def remove_object(self, obj_id, strict=False, need_output=True):
        """Remove ``obj_id`` from every camera that knows it."""
        found = False
        for name, (state, _) in self._states.items():
            if obj_id in state.get("obj_id_to_idx", {}):
                self._activate(name)
                self.predictor.remove_object(
                    obj_id, strict=False, need_output=need_output
                )
                found = True
        if strict and not found:
            raise RuntimeError(
                f"Cannot remove object id {obj_id}: unknown in all cameras."
            )
        return self.inference_state["obj_ids"], []

    # -- prompts ----------------------------------------------------
    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Route prompts to their cameras and paste the masks back.

        The GUI re-sends every point of the frame on each click, so the
        points are grouped by camera and each camera receives only its
        own subset. A box goes to the camera containing its centre.
        """
        groups = []  # (camera, points, labels, box)
        if box is not None:
            x0, y0, x1, y1 = (float(v) for v in box)
            cam = self._camera_for_xy((x0 + x1) / 2, (y0 + y1) / 2)
            groups.append(
                (cam, None, None, cam.shift_box([x0, y0, x1, y1]).tolist())
            )
        else:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            lbls = np.asarray(labels, dtype=np.int32).reshape(-1)
            by_cam = {}
            for pt, lbl in zip(pts, lbls, strict=True):
                cam = self._camera_for_xy(float(pt[0]), float(pt[1]))
                by_cam.setdefault(cam.name, (cam, [], []))
                by_cam[cam.name][1].append(pt)
                by_cam[cam.name][2].append(lbl)
            for cam, cam_pts, cam_lbls in by_cam.values():
                shifted = np.asarray(cam_pts, dtype=np.float32) - np.array(
                    [cam.x_min, cam.y_min], dtype=np.float32
                )
                # Prompts just outside the camera are clamped to its edge
                shifted[:, 0] = np.clip(shifted[:, 0], 0, cam.width - 1)
                shifted[:, 1] = np.clip(shifted[:, 1], 0, cam.height - 1)
                groups.append(
                    (cam, shifted, np.asarray(cam_lbls, dtype=np.int32), None)
                )

        merged = None
        touched = []
        out_frame_idx = frame_idx
        for cam, cam_pts, cam_lbls, cam_box in groups:
            self._activate(cam.name)
            out_frame, obj_ids, masks = self.predictor.add_new_points_or_box(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=cam_pts,
                labels=cam_lbls,
                clear_old_points=clear_old_points,
                normalize_coords=normalize_coords,
                box=cam_box,
            )
            if out_frame is None:
                continue
            touched.append(cam)
            row = obj_ids.index(obj_id)
            pasted = self._paste(cam, masks[row : row + 1])
            merged = (
                pasted if merged is None else torch.maximum(merged, pasted)
            )
        self.last_cameras = touched
        if merged is None:
            return None, None, None
        return out_frame_idx, [obj_id], merged

    def add_new_mask(self, frame_idx, obj_id, mask):
        """Add a mosaic-sized mask, split over the cameras it touches."""
        mask = np.asarray(mask)
        touched = []
        merged = None
        for cam in self.layout:
            crop = cam.crop(mask)
            if not np.any(crop):
                continue
            self._activate(cam.name)
            out_frame, obj_ids, masks = self.predictor.add_new_mask(
                frame_idx=frame_idx, obj_id=obj_id, mask=crop
            )
            if out_frame is None:
                continue
            touched.append(cam)
            row = obj_ids.index(obj_id)
            pasted = self._paste(cam, masks[row : row + 1])
            merged = (
                pasted if merged is None else torch.maximum(merged, pasted)
            )
        self.last_cameras = touched
        if merged is None:
            return None, None, None
        return frame_idx, [obj_id], merged

    def merge_frame_mask(self, existing, new_mask):
        """Overwrite only the touched cameras' regions of a frame mask.

        The GUI writes a whole frame into the prediction layer after a
        prompt. On a mosaic that would erase the same object's masks in
        the other cameras, so only the regions of ``last_cameras`` are
        replaced.
        """
        out = np.array(existing, copy=True)
        out[out < 0] = 0
        for cam in self.last_cameras:
            out[cam.y_min : cam.y_max, cam.x_min : cam.x_max] = new_mask[
                cam.y_min : cam.y_max, cam.x_min : cam.x_max
            ]
        return out

    # -- propagation ------------------------------------------------
    def _cameras_with_inputs(self):
        names = []
        for name, (state, _) in self._states.items():
            has_inputs = any(
                bool(v) for v in state.get("point_inputs_per_obj", {}).values()
            ) or any(
                bool(v) for v in state.get("mask_inputs_per_obj", {}).values()
            )
            if has_inputs or state.get("tracking_has_started", False):
                names.append(name)
        return names

    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        processing_order=None,
        reverse=False,
    ):
        """Propagate all cameras in lockstep; yield merged frames."""
        names = self._cameras_with_inputs()
        if not names:
            return
        if processing_order is None:
            starts = []
            for name in names:
                state = self._states[name][0]
                for od in state["output_dict_per_obj"].values():
                    starts.extend(od["cond_frame_outputs"].keys())
            if start_frame_idx is None:
                start_frame_idx = min(starts)
            num_frames = self._states[names[0]][0]["num_frames"]
            if max_frame_num_to_track is None:
                max_frame_num_to_track = num_frames
            if reverse:
                end = max(start_frame_idx - max_frame_num_to_track, 0)
                processing_order = (
                    range(start_frame_idx, end - 1, -1)
                    if start_frame_idx > 0
                    else []
                )
            else:
                end = min(
                    start_frame_idx + max_frame_num_to_track, num_frames - 1
                )
                processing_order = range(start_frame_idx, end + 1)
        processing_order = list(processing_order)

        gens = {}
        for name in names:
            self._activate(name)
            gens[name] = self.predictor.propagate_in_video(
                processing_order=processing_order, reverse=reverse
            )
        for expected_frame in processing_order:
            per_camera = []
            for name in list(gens):
                self._activate(name)
                try:
                    frame_idx, obj_ids, masks = next(gens[name])
                except StopIteration:
                    gens.pop(name)
                    continue
                if frame_idx != expected_frame:
                    logger.warning(
                        f"Camera '{name}' yielded frame {frame_idx}, "
                        f"expected {expected_frame}; skipping camera."
                    )
                    gens.pop(name)
                    continue
                per_camera.append((self.layout.get(name), obj_ids, masks))
            if not per_camera:
                return
            union = []
            for _, obj_ids, _ in per_camera:
                for oid in obj_ids:
                    if oid not in union:
                        union.append(oid)
            ref = per_camera[0][2]
            full = torch.full(
                (len(union), 1, self.frame_height, self.frame_width),
                LOGIT_BACKGROUND,
                dtype=ref.dtype,
                device=ref.device,
            )
            for cam, obj_ids, masks in per_camera:
                for i, oid in enumerate(obj_ids):
                    j = union.index(oid)
                    full[
                        j, :, cam.y_min : cam.y_max, cam.x_min : cam.x_max
                    ] = torch.maximum(
                        full[
                            j,
                            :,
                            cam.y_min : cam.y_max,
                            cam.x_min : cam.x_max,
                        ],
                        masks[i].to(full.device),
                    )
            yield expected_frame, union, full
