# Multi-camera / identity: known limitations and follow-ups

Findings from the 2026-09-09 prototypes (`prototype_reid_from_classifier.py`, `prototype_split_tracklets.html` in this folder). Neither is wired into the pipeline yet.

- **A swap inside one tracklet is invisible to `link`.** `link` gives one
  identity per tracklet. If BoxMOT swaps two birds at a crossing, the
  majority identity wins and the rest is silently mislabelled. Planned
  fix: a pre-pass that cuts a tracklet where confident per-frame votes
  change (anchor frames above a floor, runs shorter than a minimum length
  ignored as flicker). Open judgement calls: uncertain frames before a
  swap are given to the earlier identity, and the minimum run length
  trades missed short swaps against false cuts.
- **BoxMOT ReID cannot load an arbitrary model.** The backbone is chosen
  from the weights *filename* and unknown names are refused, so a
  classifier-derived embedding only loads if the file is named after a
  known model (e.g. contains `osnet`); an unused OSNet is built alongside.
  Works via the TorchScript backend without touching BoxMOT, but it is a
  hack. Clean route: a patch to the boxmot fork that accepts any
  TorchScript/ONNX file plus an input-size hint.
- **BoxMOT ReID preprocessing.** Crops are resized to 256x128 and
  converted BGR to RGB. OCTRON feeds RGB frames, so every ReID model
  (OSNet included) currently sees channel-swapped, squashed crops. A
  classifier used as ReID must either compensate in its wrapper or be
  trained on the same crop shape and channel order.
- **Identity classifier and ReID use different crops.** The classifier
  is trained on square, padded crops (see `train-identity`); BoxMOT ReID
  uses tight 2:1 crops. Keep this in mind when reusing the embedding.
- **Scaling.** All identity signals here come from appearance. For
  unmarked conspecifics the classifier, the anchors and the split degrade
  together; only continuity and count constraints remain.
- **`per_class`** is irrelevant in the species-detector workflow (one
  class per species) but keep it on for individual-as-class projects.
- **Per-camera SAM (MosaicPredictor).** Implemented by swapping
  `inference_state`/`images` on one shared predictor. Memory: SAM2
  caches per-frame features per state, so N cameras cost N caches; use
  SAM2 B Plus on 4-camera rigs if VRAM is tight. Only tested with a fake
  predictor and by import, not clicked through in napari. Prompts must
  lie in one camera (a box spanning two cameras is assigned by its
  centre). SAM3 semantic mode is excluded.
