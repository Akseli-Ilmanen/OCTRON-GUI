# Findings and decisions, multicam-identity branch

What we learned on the Birdpark mosaic (two zebra finches, `bird male`
and `bird female`, four views: nestCam + three mirrors) and why the
pipeline is shaped the way it is. Dates are 2026-09-09 to 2026-09-11.
See `PIPELINE_DIAGRAM.md` for how the stages connect, `FOLLOWUPS.md`
for what is still open, `SOTA_multianiaml.md` for the literature.

## Take-aways, in order of importance

1. **Identity is decided by exclusivity far more than by appearance.**
   On all 353 annotated frames, linked identities are 86-100 % correct
   when both birds are in a camera (exclusivity + elimination decide)
   and only 37-88 % correct when one bird is alone (the classifier
   alone decides, and it calls the lone male "female" most of the time
   in the mirrors). The 15-frame test split said 1.0 everywhere; that
   was luck. **Always evaluate with `--split all`.**
2. **Self-training works only on frames the constraints decided.**
   `refine-identity` on *all* assigned frames learned the lone-male
   error (right-mirror male 46/46 -> 31/46 on hand-labelled crops,
   flip rate 24 % -> 36 %). Restricted to co-existence frames (both
   birds present in the camera, TRex "global segments") it fixed the
   nest male (1/15 -> 15/15), lifted three cameras to 0.92-1.00 on all
   annotated frames and the hand-labelled crops from 0.90 to 0.95, flip
   rate 24 % -> 22 %. Peak was at epoch 8 of 30: fine-tune ~10 epochs.
3. **Fragmentation is not what causes identity jumps.** OcSort with
   `use_byte` + IoU 0.2 cut tracklets 412 -> 321 and doubled their
   length; the flip rate did not move. Flips are the classifier
   disagreeing with itself on hard views, i.e. a training-data problem.
   Stitching by continuity was built, measured (58 joins, no effect on
   flips) and removed.
4. **Generic ReID in the tracker hurts.** DeepOcSort + OSNet: same
   tracklet count as OcSort, flip rate 24 % -> 41 %, 2.4x slower.
   Appearance matching joined the wrong bird at crossings and the
   contaminated tracklets cannot be repaired downstream. Our own
   classifier as ReID would need a boxmot-fork patch (see FOLLOWUPS)
   and is only worth it if refine rounds stall.
5. **The nest is visible in the mirrors.** Declaring nestCam
   non-overlapping with the mirrors (the `non_overlapping` feature in
   `cameras.json`) collapsed the nest male to 8/73: the annotations
   themselves show the same bird in nestCam and a mirror in dozens of
   frames, and predictions have 1354 frames with both birds in the
   nest while a mirror also sees one. The feature stays; it does not
   apply to this rig.
6. **Two `link` guards earned their keep, the rest did not.**
   `--min-evidence` (don't guess on a thin margin) and `--min-overlap`
   (a 3-frame tracker hand-over is not two animals; without it one
   4752-frame tracklet was left unassigned = 30 % of the nest camera).
   Vote aggregation (sum / majority / confident / log-likelihood) makes
   no difference; sum is fine.
7. **Detector: stay at 640 on the 10 GB card.** OCTRON's AutoBatch is
   conservative (50 % VRAM, 0.7 margin); yolo26m at 1280 got batch 1
   and mAP50-95 0.29 vs 0.55 at 640 / batch 2. Recall at 640 is capped
   by small mirror birds (median 130 px in the mosaic, 28 px at 640).
   Predict at `--conf-thresh 0.3` (bottom-left mirror bird sits at
   0.3-0.4). Validation frames 3085-3096 have annotation gaps (a bird
   unlabelled in mirrorMain, fragmented masks in mirrorRight).
8. **Speed is CPU-bound.** 27 fps prediction, GPU at 23 %; YOLO is
   ~10 ms, decode / classifier call / pandas dominate. Per-camera
   detection would be slower, not faster; its only benefit is
   resolution.
9. **Cross-camera speed correlation carries no usable signal** for a
   bonded pair (same-bird median 0.32 vs different-bird 0.07, heavy
   overlap). Dropped.

## Mental model: what each stage knows

- **BoxMOT's "ID" is a track id, not an identity.** A counter for
  "this box continues that box". No input for identities exists; ReID
  trackers compare crop embeddings to their own tracks and nothing
  else. One `update(detections, frame)` per frame: Kalman predict,
  match (IoU, optionally appearance), update / spawn / age.
  `use_byte` is a second match *inside the same call* against
  low-confidence boxes; it is not a second run.
- **`per_class` splits BoxMOT into one sub-tracker per YOLO class**
  that never sees other classes' boxes. In standard OCTRON with
  individuals as classes this means two sub-trackers can track the
  same bird at once (per-class NMS lets one bird be a `bird_male` and
  a `bird_female` box; `failure1_exclusitivity.png`). Exclusivity is
  enforced nowhere: not in YOLO, not in BoxMOT, not afterwards.
- **A detector cannot learn count rules** ("two here means none
  there"); every grid cell decides on its own. Such rules belong in
  `link`.
- **This branch:** one detector class per species (NMS works, one box
  per bird), motion-only tracker per camera, classifier per box,
  `link` per tracklet with exact exclusivity. Each stage has one job
  and the next trusts it. Going back to detector-as-ID does not fix
  flips (the detector flips per frame instead, OCTRON splits the
  track) and brings the double detections back.
- **Embeddings.** The classifier's penultimate 1280-d vector is an
  embedding; with two classes it may collapse to one axis, which is
  fine for telling *these* two birds apart and useless for new ones.
  OSNet generalises but was never shown a bird. Ranking for this rig:
  refined classifier embedding > classifier embedding > OSNet.

## Trackers shipped by OCTRON

| tracker | appearance | note |
|---|---|---|
| ByteTrack (default) | none | fastest |
| OcSort | none, observation-centric motion | our choice; `use_byte: true`, `iou_threshold: 0.2` via `--tracker-config` (`<project>/trackers/ocsort_byte.yaml`) |
| BotSort | ReID + IoU | camera-motion compensation on by default, expensive and pointless for fixed cameras |
| DeepOcSort | ReID, adaptive weight | measured: hurts here (flip 41 %) |
| HybridSort, BoostTrack | ReID + extra states | BoostTrack has no `per_class` |

BoxMOT's OcSort Kalman back-fill fails on newer NumPy ("only
0-dimensional arrays can be converted to Python scalars", 88 times in
130 frames); the observation-centric re-update is silently skipped.
Upstream bug in horsto's boxmot fork, not OCTRON.

## Decisions taken with the user

- Keep the fork mergeable: never change shipped YAMLs, tracker catalog
  or CLI defaults; workflow choices live in
  `C:\Users\aksel\Documents\octron\retrain.ps1` and `RETRAIN_COMMANDS.md`.
- Keep the species detector + classifier + `link` design; no ReID in
  the tracker; no detector-as-ID.
- `refine-identity`: co-existence frames only, no other filters, no
  per-camera balancing (uneven presence per camera is a fact about the
  animals, not a dataset defect); fine-tune from current `best.pt`.
- Exports: `confidence` (detector, movement-compatible name) and
  `identity_conf` (classifier) stay separate; no pre-combined score.
- Identity splits: detector 0.85/0.10/0.05 (OCTRON requires a test
  remainder), classifier 0.8/0.1/0.1 (test frames feed
  `evaluate-identity`).

## Referees

- `octron evaluate-identity <project> <cams> --split all`: linked
  accuracy per camera and per individual on every annotated frame.
- `python flip_rate.py <prediction folder>` (project folder): share of
  consecutive same-camera tracklets that change identity. Baseline
  24.5 %, after co-existence refine 22.4 %.
- Hand-labelled val + test crops through the classifier directly,
  per (camera, individual): where the classifier itself is weak.
