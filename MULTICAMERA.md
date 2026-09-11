# Multi-camera mosaics and individual identity

This page describes the *species detector + post-hoc identity* workflow
(see issues [#87](https://github.com/OCTRON-tracking/OCTRON-GUI/issues/87)
and [#91](https://github.com/OCTRON-tracking/OCTRON-GUI/issues/91)).
It lets one YOLO model serve several synchronised cameras and keeps the
identity of individuals as a separate, explicitly exclusive decision.

## 1. Build a mosaic video

OCTRON does not tile videos. Tile your synchronised recordings once with
ffmpeg, always in the same order, and treat the result as an ordinary
video:

```bash
ffmpeg -i cam0.mp4 -i cam1.mp4 -i cam2.mp4 -i cam3.mp4 \
  -filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2" \
  -c:v libx264 -crf 18 mosaic.mp4
```

## 2. Annotate labels and suffixes

Annotate the mosaic in the OCTRON GUI as usual. Use the **label** for the
species (`bird`) and the **suffix** for the individual (`1`, `2`, ...).
The detector is trained on labels only, so all suffixes of a label are
pooled into one detector class. Objects without a suffix train the
detector but are ignored by identity training.

## 3. Draw the cameras

In the project tab, **Draw** adds a `cameras` shapes layer. Draw one
rectangle per sub-camera, optionally **Name…** them, then **Save**. The
layout is stored as `<project>/<video hash>/cameras.json`:

```json
{
    "version": 1,
    "frame_width": 3840,
    "frame_height": 2160,
    "cameras": [
        {"name": "top_left", "x_min": 0, "y_min": 0, "x_max": 1920, "y_max": 1080},
        ...
    ]
}
```

Rectangles are snapped to integer pixels and clipped to the frame.
Overlaps are allowed (mirrors) and only warned about.

**Draw the cameras before you start segmenting.** SAM2 encodes every
frame at 1024x1024, so on a mosaic each camera would get a fraction of
that. With a saved `cameras.json`, OCTRON runs one SAM state per camera
(sharing one model in memory): you still click on the mosaic, but the
click is routed to its camera, SAM sees that camera's crop at full
encoder resolution, and the mask is pasted back into the mosaic-sized
layer. One object (`bird male`) can therefore have one blob per camera
in a single mask layer. Per-camera image caches are stored as
`video data <camera>.zarr` next to the video's data. If you save the
layout after SAM was initialised, reload the video. The SAM3 semantic
(text prompt) mode always runs on the full frame.

Training export is camera-aware too: polygons and identity crops are cut
per camera, so a bird visible in two views yields two polygons, two
boxes and two classifier crops rather than one shape spanning the
border. To reuse one layout for many recordings:

```bash
octron cameras cameras.json --project <project>          # into every video folder
octron cameras cameras.json --video a.mp4 --video b.mp4  # sibling <stem>_cameras.json
octron cameras cameras.json --show                       # validate and print
```

## 4. Train

```bash
octron train <project> --mode detect            # species detector on mosaic frames
octron train-identity <project>                 # identity classifier on mask crops
```

`train-identity` exports square crops of every annotated `(label,
suffix)` mask to `<project>/model_identity/training_data/` in the
ultralytics classification layout (split with the same episode-aware
block split as the detector), then trains a `yolo11n-cls` model to
`<project>/model_identity/training/weights/best.pt`.

## 5. Predict per camera

```bash
octron predict mosaic.mp4 --model <best.pt> --tracker ByteTrack \
  --cameras cameras.json --identity <project>/model_identity/training/weights/best.pt
```

Detection runs on the full mosaic frame. Each detection is assigned to
the camera containing its box centre, and every camera gets its own
BoxMOT tracker and output folder in camera-crop coordinates:

```
octron_predictions/mosaic_ByteTrack/
    cameras.json
    top_left/   predictions.zarr, bird_track_1.csv, ..., prediction_metadata.json
    top_right/  ...
```

Each camera folder loads with the existing results loader and the GUI.
Without a `--cameras` file (and no sibling `<stem>_cameras.json`), the
layout is unchanged from before.

With `--identity`, every tracked box is classified per frame and the
CSVs gain `identity`, `identity_conf` and one `identity_prob_<class>`
column per individual (class names are `<label>_<suffix>`).

## 6. Resolve exclusivity

```bash
octron link octron_predictions/mosaic_ByteTrack/*/        # per camera
octron link ... --global                                  # non-overlapping rigs
```

For every camera, `link` sums the identity probabilities of each
tracklet and solves an exact integer program (`scipy.optimize.milp`):
one identity per tracklet, and an identity may not be held by two
tracklets that overlap in time in the same camera. Cameras whose fields
of view do not overlap can be declared in `cameras.json`:

```json
"non_overlapping": [["nestCam", "*"]]
```

(`"*"` = every other camera). `link` reads the `cameras.json` next to
the folders and adds the same constraint between those camera pairs,
so a bird in the nest and a bird in a mirror at the same time are
different individuals; `--exclusive nestCam:mirrorMain` (repeatable)
does the same from the command line. Pairs not listed are assumed to
overlap. `--global` applies the constraint between every pair, which is
right only when no fields of view overlap at all. Results go to `identity_assignment.csv` and
`link_report.json` in each folder; tracklets whose best identity is
less than `--min-margin` (default 1.5) times the runner-up are flagged
for review, as are unassigned ones. With `--min-evidence N` a tracklet
whose best summed score exceeds the runner-up by fewer than `N`
confident frames is left unassigned (reason `low_evidence`) instead of
guessed; this keeps a short tracklet seen from a bad angle from getting
a confident wrong identity, and stops it competing with overlapping
tracklets. The `evidence` column in the CSV is that difference.
`--min-overlap N` (default 1) makes two tracklets exclude each other
only when they share at least `N` frames: a tracker hands one animal
from a dying track to a new one with a few frames of overlap, and
treating that as two animals can block a very long tracklet on a
3-frame artefact; 10 is a good value.

## 7. Check and improve identity

```bash
octron evaluate-identity <project> octron_predictions/mosaic_ByteTrack/*/
octron refine-identity   <project> octron_predictions/mosaic_ByteTrack/*/
```

`evaluate-identity` scores the linked tracklets against the annotated
masks of the same video, on the `test` split of the identity dataset
only (`--split val|all` to change), and writes `identity_eval.json`
per camera. It reports IDF1 (Ristani et al. 2016) for three id schemes:
raw BoxMOT track ids, the per-frame classifier argmax, and the linked
identity. IDF1 rewards consistency and is blind to a systematic swap,
so `linked_accuracy` (fraction of annotated boxes whose linked identity
is the annotated one) is the number to watch.

`refine-identity` is the self-training loop of TRex (Walter & Couzin
2021), restricted to its "global segments": only frames in which every
individual is present in the camera are used, because there `link`
decided the identities by exclusivity and elimination rather than by
the classifier's opinion of a single crop. Measured on all annotated
frames of the Birdpark mosaic, linked identities were 86-100 % correct
in such co-existence frames but only 37-88 % correct when one bird was
alone (the lone male in a mirror was mostly called female); one refine
round trained on all frames learned that mistake. Each co-existence
frame yields one crop per individual with the linked identity as
label, so the export is balanced by construction. Crops go to the
`train` split only, spread evenly over each tracklet
(`--max-per-tracklet`), never from val/test frames of the same video.
The classifier is then fine-tuned from the current `best.pt`. Repeat
predict, link, refine until `evaluate-identity` stops improving;
`--all-frames` uses every assigned frame instead, `--clear` drops
pseudo crops of earlier rounds.

## 8. Export per-camera datasets for downstream tools

```bash
octron link octron_predictions/mosaic_ByteTrack/*/ --netcdf   # link + export
octron export-nc octron_predictions/mosaic_ByteTrack             # export only
```

Writes one `<camera>.nc` per camera into the prediction folder, a
movement-style bounding-box dataset with dimensions
`(time, space, individual)`: `position`, `shape` (width, height),
`confidence`, `identity_conf`, `track_id` and `flagged`. Individuals are
the linked identities (`bird_male`, `bird_female`) on the same axis in
every file of a video, so the files can be concatenated on a `camera`
dimension downstream. Unassigned tracklets are left out and counted in
the attributes; flagged ones are kept with `flagged = 1`. Positions are
in camera-crop pixels; the camera name and its rectangle in the mosaic
are attributes (`camera`, `camera_x_min`, ...), as are the mosaic video
name and size. Needs `pip install octron[export]` (xarray, netCDF4).

To get one video file per camera that matches those coordinates with
no offset, cut the mosaic once:

```bash
octron cameras cameras.json --split-video mosaic.mp4    # -> cameras_video/<camera>.mp4
```

Odd rectangle sizes lose one pixel at the right/bottom edge (H.264 needs
even dimensions).

## Notes on tracker settings

For the species-detector + identity workflow a motion-only tracker
(ByteTrack, OcSort) is enough and is what we recommend: the identity
classifier is trained on these individuals and `link` resolves
identities with an exact exclusivity constraint, which a generic
re-identification embedding inside the tracker does not add to. Tune
the tracker to *fragment* at crossings rather than bridge them (short
`max_age`, strict matching): a fragmented but pure tracklet is stitched
by `link`, while a tracklet that swaps animals mid-way keeps one
identity for its whole length and cannot be repaired downstream. The
ReID trackers remain available for other workflows.

`per_class` keeps BoxMOT from matching a track to detections of another
class. OCTRON enables it by default for every tracker except BoostTrack
(which errors with it). With individual-as-class projects it avoids
identity flips at crossings; with the species-detector workflow above
it is irrelevant because there is only one class per species. OCTRON
always splits a BoxMOT track that changes class into separate,
label-consistent tracklets.
