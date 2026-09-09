# Follow-up PR candidates

Collected while building the multicam-identity branch (2026-09-09/10).
Each item is meant to be a small, separately reviewable PR against
upstream OCTRON once the branch has been validated on the Birdpark rig.

## GUI annotation

1. **Undo the last propagation.** After "▷ 15 frames" (or the one-frame
   step) there is no way to discard what SAM just wrote. Add a button
   next to the propagation controls that clears the frames written by
   the last run: remove those frame indices from every object's mask
   zarr (set to -1 and drop them from `annotated_frames`) and from the
   per-camera SAM states (`non_cond_frame_outputs` for those frames), so
   a re-run starts from the last manual prompt. Needs the callback to
   remember `(start_frame, frame_indices)` of the last propagation, and
   must work for backward propagation too.
2. **Propagation length spinner.** `chunk_size` is hard-coded to 15
   (6 in SAM3 semantic mode) in `main.py`; the button label and the
   progress bar already read from it. Expose it as a spinbox beside the
   skip-frames spinbox, persist it in `config.yaml`
   (`propagation_frames`), and keep the semantic-mode override.
3. **Projection colours in individual mode.** `sam_layer.py` colours
   the label projection by label family; with
   `object_color_mode=individual` it should use each entry's own colour.
4. **Recolour existing objects when the colour mode changes.** Colours
   are frozen in the organizer JSON at creation; offer a "reassign
   colours" action so old projects can adopt `individual` mode.
5. **Square padding for wide cameras.** Each camera crop (and, before,
   the full frame) is stretched to the 1024x1024 SAM input. For 2.78:1
   side cameras that is a strong squeeze; pad to square before encoding
   instead and un-pad the mask.
6. **Reload after camera changes.** Saving or loading `cameras.json`
   after SAM initialised currently requires removing and re-dropping the
   video. Re-initialise the per-camera stores and states in place.

## Training

7. **Per-camera crop export for detector training.** Annotate on the
   mosaic, train on single views (horsto's preference in #87): use
   `cameras.json` to export one image + label file per camera per frame
   in `octron split`, behind a `--per-camera` flag.
8. **Identity classifier in the GUI.** `train-identity` exists only on
   the CLI; add it to the training tab and let the predict tab pass the
   identity weights (currently CLI-only via `--identity`).

## Tracking / identity

9. **Tracklet change-point split in `octron link`.** Cut a tracklet where
   confident per-frame identity votes change (prototype:
   `prototype_split_tracklets.html`), so a swap inside one tracklet does
   not get the majority identity. Options: confidence floor, minimum run
   length.
10. **Classifier embedding as BoxMOT ReID.** Export the identity
    classifier backbone to TorchScript and use it as `reid_weights`
    (prototype: `prototype_reid_from_classifier.py`). Blocked on a
    boxmot-fork patch to accept arbitrary ReID files (filename-based
    registry) and on matching crop shape / channel order.
11. **Location prior in `link`.** Optional per-camera prior (e.g. the
    nest camera is usually the female) as an additive term in the score
    matrix, replacing the implicit position cue that individual-as-class
    detectors learn.
12. **Tracker presets for erratic motion.** Document/ship a DeepOcSort
    config with short `max_age` for animals that leave the field of view
    often, so re-entries become new tracklets that `link` resolves
    instead of wrong revivals.

## Docs / upstream

13. Reply on #87 (draft was in `issue87_reply.md`, since removed) and
    propose the PR split above.
14. Move `MULTICAMERA.md` into the OCTRON-docs site once the workflow is
    accepted.
