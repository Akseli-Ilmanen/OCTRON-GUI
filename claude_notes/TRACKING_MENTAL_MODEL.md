# How detection, BoxMOT and identity actually fit together

Written 2026-09-11 after a session of confusion about where identity is
decided. Read this before touching tracker settings or `link`.

## BoxMOT's "ID" is a track id, not an identity

A track id is a counter: track 1, 2, 3, ... created whenever a new
object appears. It says "this box continues that box from the last
frame". It never says who the animal is. BoxMOT has no input for
identities either: you cannot tell it "track 7 is the female", and a
ReID tracker only compares crop embeddings to its own tracks.

Per frame, one `tracker.update(detections, frame)` call does:

1. predict where every existing track should be now (Kalman filter),
2. match predictions to this frame's detections (IoU, optionally
   appearance for ReID trackers),
3. update matched tracks, start new tracks from unmatched detections,
   age unmatched tracks, drop them after `max_age`.

`use_byte` (OcSort/ByteTrack) is a second matching step *inside the
same call*, over the same frame: unmatched tracks get a second chance
against low-confidence detections (between `det_thresh` and the
predict `--conf-thresh`). Low-confidence boxes never start a track,
they only keep one alive. It is not a second run of anything.

## `per_class` splits the tracker by class

OCTRON enables `per_class` for every tracker except BoostTrack. BoxMOT
then keeps one independent sub-tracker per YOLO class id. A sub-tracker
only ever sees detections of its own class.

Consequences in **standard OCTRON with individuals as classes**
(`bird_male`, `bird_female`):

- the male sub-tracker never sees a female box, so class flips at
  crossings are rare (good),
- nothing stops both sub-trackers from tracking the *same physical
  bird* at once. YOLO's non-maximum suppression is per class, so one
  bird can yield a `bird_male` box and a `bird_female` box; each
  sub-tracker keeps its own track; no component anywhere can notice
  (this is `failure1_exclusitivity.png`).

With `per_class` off there is one class-agnostic tracker that
overwrites a track's class with the latest matched detection; OCTRON
then splits the track into label-consistent tracklets whenever the
class changes (`_track_camera_frame`, `split_track_ids`).

So in standard OCTRON, individual exclusivity ("one bird cannot be in
two places, two boxes on one bird cannot be two birds") is enforced
**nowhere**: not in YOLO (per-class NMS), not in BoxMOT (per-class
sub-trackers), not afterwards.

## Where identity is decided in the multicam-identity branch

```
YOLO (one class per species)      -> one box per bird, NMS works
BoxMOT per camera, motion-only    -> tracklets = "same box continued"
identity classifier per box       -> identity_prob_* per frame
octron link                       -> one identity per tracklet, exact
                                     exclusivity (MILP), --min-evidence,
                                     --min-overlap
```

- The detector cannot learn count rules ("two here means none there");
  every grid cell decides independently. Such rules belong in `link`.
- Identity jumping between consecutive tracklets of one bird looked
  like a *fragment* problem, but measured (2026-09-11, `flip_rate.py`
  in the project folder) it is not: OcSort with `use_byte` + IoU 0.2
  cut tracklets 412 -> 321 and doubled their length, and the flip rate
  stayed at 24 %. In 15 of 25 flip pairs both sides carry strong
  evidence for different birds: the classifier disagrees with itself
  (or the birds really swapped). So the lever is the classifier's
  training data, i.e. `refine-identity`, not the tracker. Stitching by
  continuity was built, measured (58 joins, no effect on flips) and
  removed again (Occam).
- `link` guards that earned their keep: `--min-evidence` (don't guess
  on thin margins) and `--min-overlap` (a 3-frame tracker hand-over
  overlap is not two animals; without it one 4752-frame tracklet was
  left unassigned = 30 % of the nest camera).
- `refine-identity` takes *every* assigned tracklet, no filters: the
  linked identity already integrates appearance, exclusivity and
  elimination. The male in the nest camera looks female to the
  classifier (p=0.77) and is only known to be male by elimination;
  filtering "crops that disagree with the vote" would drop exactly the
  crops that fix this. Baseline nest-camera accuracy 1.0 was partly
  exclusivity luck.
- Classifier-as-ReID inside BoxMOT remains the fallback if refine does
  not remove the flips (needs a boxmot-fork patch: loader keys on the
  weights filename and resizes crops to 256x128 BGR).
- Going back to detector-as-ID does not fix jumping (the detector flips
  per frame instead and OCTRON splits the track) and brings the double
  detections back.
- ReID inside BoxMOT decides which *track* a box joins, never which
  *class* it has. A "second BoxMOT run with known identities" cannot
  use them.

## Measured on the Birdpark mosaic (2026-09-11)

- Prediction: 27 fps, GPU at 23 %; YOLO is ~10 ms, decode/classifier
  call/pandas dominate. Per-camera detection would be slower, not
  faster. Its only benefit is resolution on small birds.
- AutoBatch is conservative (50 % VRAM, 0.7 margin): yolo26m at 1280
  -> batch 1 -> mAP50-95 0.29 vs 0.55 at 640/batch 2. Stay at 640 on
  the 10 GB card, or try 960.
- Cross-camera speed correlation does not separate same-bird from
  different-bird pairs (median 0.32 vs 0.07, heavy overlap): the pair
  moves together. Dropped.
- Duplicates (two boxes, IoU > 0.3, same camera, same frame): 20 % of
  multi-box frames in nestCam (huddled pair, real), 1-9 % in mirrors
  (likely one bird, two boxes; NMS at 0.7 lets them through).
