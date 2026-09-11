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
tracklets that overlap in time in the same camera. `--global` extends
the constraint across cameras, which is right only when fields of view
do not overlap. Results go to `identity_assignment.csv` and
`link_report.json` in each folder; tracklets whose best identity is
less than `--min-margin` (default 1.5) times the runner-up are flagged
for review, as are unassigned ones. `YOLO_results.get_identity_assignment()`
reads the CSV back.

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
2021): every tracklet that `link` assigned without a flag becomes
pseudo-labelled training crops, provided no unresolved (unassigned or
flagged) tracklet of the same label was alive at the same time. That
coexistence rule is the negative-pair idea of idtracker.ai (Torrents et
al. 2026): two animals visible at once are different individuals, so an
identity is only certain when its competitors are accounted for. Crops
go to the `train` split only, spread evenly over each tracklet
(`--max-per-tracklet`), skipping frames where the classifier strongly
disagrees with the tracklet (`--min-frame-prob`) and skipping val/test
frames of the same video. The classifier is then fine-tuned from the
current `best.pt`. Repeat predict, link, refine until
`evaluate-identity` stops improving; `--clear` drops pseudo crops of
earlier rounds.

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
