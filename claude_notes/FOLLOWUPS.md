# Follow-ups

Open items for the multicam-identity branch, updated 2026-09-11. Each
is meant to be a small, separately reviewable change. Items already
measured and rejected are listed at the end so they are not rebuilt.

## Identity / link (next)

1. **Second refine round from `best_refine_coexist.pt`, ~10 epochs.**
   Round 1 peaked at epoch 8; see whether the right mirror
   (male 131/183) moves at all. If not, appearance has plateaued there
   and the remaining flag rate is what it is.
2. **Per-camera weakness table.** `evaluate-identity --split all`
   should print correct/matched per (camera, individual) and
   `train-identity` should print hand-labelled crop counts per
   (camera, individual) with a note where a combination is thin. The
   user then decides whether to label more; no automatic balancing.
3. **`identity_evidence` in the NetCDF export**, the tracklet's
   best-minus-runner-up score broadcast per frame, and make
   `identity_conf` the probability of the *exported* identity (today it
   is the classifier's own per-frame argmax, which can differ).
4. **Duplicate suppression in `link`.** Two tracklets co-existing with
   high box IoU over their shared frames are one animal; drop the
   weaker instead of forcing it onto the other identity. Mirrors have
   1-9 % of multi-box frames with IoU > 0.5 (nest 20 %, but that is a
   huddled pair). Also try predict `--iou-thresh 0.5`.
5. **Classifier as BoxMOT ReID (fallback only).** Export the
   classifier backbone to TorchScript; blocked on a patch to horsto's
   boxmot fork (loader keys on the weights filename, crops resized to
   256x128 BGR). Only if refine rounds stall on flips.
6. **Report the BoxMOT NumPy bug** (OcSort back-fill interpolation
   fails on NumPy >= 2.x) to horsto's fork.

## Detector

7. **Per-camera crop export for detector training** (`octron split
   --per-camera`, horsto's preference in #87) and per-camera inference:
   the only route to resolution on the small mirror birds; 4x GPU work,
   no speed gain.
8. **Fix annotation gaps** in frames 3085-3096 (unlabelled bird in
   mirrorMain, fragmented mask in mirrorRight) and retrain; largest
   expected quality gain overall.

## GUI

9. Undo the last SAM propagation (clear the frames it wrote from every
   mask zarr and the per-camera SAM states).
10. Propagation length spinner (`chunk_size` hard-coded to 15, 6 in
    SAM3 semantic mode), persisted in `config.yaml`.
11. Projection colours in `individual` colour mode; recolour existing
    objects when the mode changes.
12. Square padding for wide cameras before the 1024x1024 SAM encode.
13. Re-initialise per-camera stores in place after `cameras.json`
    changes (today: remove and re-drop the video).
14. Identity classifier in the GUI (train tab + predict tab
    `--identity`), currently CLI only.

## Docs / upstream

15. Reply on #87 with the PR split; move `MULTICAMERA.md` into the
    OCTRON-docs site once accepted.

## Measured and rejected (do not rebuild)

- Stitching tracklets by continuity in `link` (58 joins, no effect on
  flips).
- Per-camera balancing of pseudo-crops by cutting to the smallest
  group (discards real data; the problem was label noise, solved by
  co-existence frames).
- Filters on refine inputs (skip flagged, unresolved neighbour,
  per-frame disagreement): they removed the elimination-decided
  crops that fix the lone-male error.
- Non-overlapping nestCam vs mirrors on this rig (the nest is visible
  in the mirrors); feature kept for other rigs.
- Generic OSNet ReID in the tracker (flip rate 24 % -> 41 %).
- Cross-camera speed correlation as identity signal (no separation for
  a bonded pair).
- Tracklet change-point split (swap inside a tracklet): motion-only
  OcSort keeps swaps rare; revisit only if flips persist with strong
  evidence on both sides.
- imgsz 1280 for yolo26m on the 10 GB card (batch 1, mAP50-95 0.29).
