# Tracklet-to-identity assignment in multi-animal tracking

Setting: per-frame detections (bounding boxes) are already reliable. Open problem: link them into tracklets, assign tracklets to individuals, and keep identities consistent across the whole video.

## Common structure

1. **Local linking** – cheap motion matcher (IoU / centroid distance / Kalman / optical-flow shift + Hungarian) produces short, high-purity tracklets. Breaks are accepted; hard decisions are postponed.
2. **Label-free appearance signal** – *same tracklet ⇒ same animal*, *temporally overlapping tracklets ⇒ different animals*. This gives positive/negative pairs for metric learning without any manual ID labels.
3. **Global assignment** – tracklet-level evidence (motion + appearance) resolved by an optimisation that respects the constraint that coexisting tracklets cannot share an identity.

## Methods

| # | Method | Tracklet formation | Identity signal | Global assignment | Notes |
|---|--------|-------------------|-----------------|-------------------|-------|
| 1 | **maDLC stitching** (Lauer et al. 2022, *Nat Methods*) | Box/ellipse/skeleton IoU + Hungarian | Optional (shape, motion, or 2) | DAG of tracklets, edge cost = -log P(same animal) from proximity, motion, shape, appearance; **min-cost flow** with N paths | Exact solver; GUI for manual refinement. Cleanest baseline to reimplement on boxes. |
| 2 | **maDLC ReIDTransformer** (same paper) | From 1 | Transformer on pose-feature tokens, **triplet loss** with triplets sampled from tracklets (pos = same tracklet, neg = overlapping tracklet) | Feeds appearance cost into 1 | Up to ~10 % MOTA gain on fish. Trained per video; swap pose features for crop embeddings if using boxes. |
| 3 | **idtracker.ai v6** (Torrents, Costa, de Polavieja 2026, *eLife*) | "Fragments" = crops of one animal between crossings | **ResNet18 → 8-d embedding, contrastive loss** (D_pos = 1, D_neg = 10), pairs sampled from fragments, loss-aware sampler; no augmentation, no projection head | Mini-batch **k-means** (k = N), per-image prob ∝ 1/d^7, fragment consistency + coexistence constraint, residual identification | Silhouette score ≥ 0.91 as stopping criterion (< 1 % image error). Needs fragment connectivity > 0.5 and fixed known N. 99.9 % IDF1, 90–700× faster than v4. Works directly on box crops. |
| 4 | **TRex** (Walter & Couzin 2021, *eLife*) | Speed-limited motion matcher on blobs | **N-class CNN classifier** trained on "global segments" (all N visible & unambiguous); uniqueness feedback expands training set | Per-segment averaged softmax + argmax, distinct-ID constraint; motion tracker re-run with visual overrides | Simpler than 3, ~98 % IDF1. Detection tightly coupled to its own blob pipeline; the *idea* ports easily. |
| 5 | **SLEAP** (Pereira et al. 2022, *Nat Methods*) | Kalman or **flow-shift** (LK optical flow moves previous instances into current frame), OKS/IoU/centroid similarity, Hungarian/greedy, tracking window | Optional **supervised ID head** (multi-class per instance) – requires hand-labelled identities and visually distinct animals | Per-frame assignment (no error propagation) or track cleaning to N tracks | Good motion tracker; ID model only helps if animals are distinguishable and you label them. |
| 6 | **Track clustering re-ID** (2025, PMC12178989) | Any MOT output | Pretrained CNN crop embeddings, aggregated per track | **Constrained clustering** of whole tracks (cannot-link for temporally overlapping tracks), k = N | Time-agnostic → handles animals leaving and re-entering. No motion term; best as final stage after 1–3. |

## Evaluation

Use **IDF1** (Ristani et al. 2016), not MOTA: MOTA counts an ID switch once regardless of duration; IDF1 penalises persistent misidentification.

## Recommended pipeline for a small fixed group from boxes

boxes → motion matcher (1/5) → tracklets → self-supervised embedding trained on tracklet pairs (3, or 2 with crop features) → min-cost-flow stitching with motion + appearance costs (1) → optional long-gap merge by constrained clustering (6).

Off-the-shelf closest fits: `idtrackerai` v6 (gitlab.com/polavieja_lab/idtrackerai, pip), `deeplabcut.transformer_reID` + `stitch_tracklets`. Generic MOT re-ID (BoT-SORT, unsupervised SimpleReID) also applies since detection is solved.