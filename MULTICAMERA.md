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

## Notes on tracker settings

`per_class` keeps BoxMOT from matching a track to detections of another
class. OCTRON enables it by default for every tracker except BoostTrack
(which errors with it). With individual-as-class projects it avoids
identity flips at crossings; with the species-detector workflow above
it is irrelevant because there is only one class per species. OCTRON
always splits a BoxMOT track that changes class into separate,
label-consistent tracklets.
