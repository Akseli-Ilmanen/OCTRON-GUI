"""PROTOTYPE (throwaway): identity classifier embedding as BoxMOT ReID.

Question: can the embedding of a trained ``yolo classify`` identity model
be dropped into a BoxMOT ReID tracker without changing BoxMOT?

Run:  python claude_notes/prototype_reid_from_classifier.py [weights.pt]

What it does
1. Loads a YOLO classification model (default: stock yolo11n-cls, a
   stand-in for ``<project>/model_identity/training/weights/best.pt``).
2. Wraps its backbone up to the 1280-d pooled feature (the layer before
   the class head) in a module that undoes BoxMOT's ImageNet
   normalisation, traces it at BoxMOT's (256, 128) crop size and saves
   it as TorchScript.
3. Creates a BotSort tracker with that file as ``reid_weights`` using
   OCTRON's own tracker config, and pushes two frames through it.
4. Prints embedding shape and cosine similarities so you can see the
   appearance term actually comes from the classifier.

Findings are printed at the end. Nothing here is production code.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path(__file__).with_name("_proto_reid_out")


class ClassifierEmbedding(torch.nn.Module):
    """YOLO-cls backbone -> pooled feature, fed BoxMOT-normalised crops."""

    def __init__(self, yolo_cls_model):
        super().__init__()
        self.layers = yolo_cls_model.model.model[:-1]  # everything but head
        head = yolo_cls_model.model.model[-1]  # Classify: conv, pool, drop
        self.conv, self.pool = head.conv, head.pool
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # BoxMOT hands over (x/255 - mean)/std; YOLO-cls wants x/255.
        x = x * self.std + self.mean
        for layer in self.layers:
            x = layer(x)
        x = self.pool(self.conv(x))
        return x.flatten(1)


def export_torchscript(weights):
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    module = ClassifierEmbedding(yolo).eval()
    dummy = torch.zeros(1, 3, 256, 128)  # BoxMOT crop size (H, W)
    traced = torch.jit.trace(module, dummy)
    OUT_DIR.mkdir(exist_ok=True)
    # HACK: BoxMOT derives a backbone name from the file name and refuses
    # unknown names, so the file must contain a known one ("osnet").
    # The osnet backbone it builds is never used by the TorchScript
    # backend; only our traced module runs.
    out = OUT_DIR / "osnet_x0_25_identity_proto.torchscript"
    traced.save(str(out))
    with torch.no_grad():
        emb = module(dummy)
    print(f"[1] exported {out.name}: embedding dim = {emb.shape[1]}")
    return out


def make_tracker(reid_path):
    from boxmot import create_tracker

    from octron.tracking.helpers.tracker_checks import (
        load_boxmot_tracker_config,
    )

    import octron

    cfg_path = (
        Path(octron.__file__).parent / "tracking" / "configs" / "botsort.yaml"
    )
    cfg = load_boxmot_tracker_config(cfg_path)
    tracker_id = next(iter(cfg))
    params = {
        k: v["current_value"] for k, v in cfg[tracker_id]["parameters"].items()
    }
    params["nr_classes"] = 1
    params["with_reid"] = True
    tracker = create_tracker(
        tracker_type=cfg[tracker_id]["tracker_type"],
        reid_weights=reid_path,
        device="cpu",
        per_class=False,
        evolve_param_dict=params,
    )
    print(
        f"[2] BotSort created, reid backend = "
        f"{type(tracker.model.model).__name__}"
    )
    return tracker


def demo_frames():
    """Two frames with two 'animals' that swap position between frames."""
    import ultralytics
    from PIL import Image

    bus = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    img = np.asarray(Image.open(bus).convert("RGB").resize((640, 480)))
    frame0 = img.copy()
    # frame 1: mirror so the two people move; appearance stays the same
    frame1 = img[:, ::-1].copy()
    boxes0 = np.array([[20, 230, 110, 460], [200, 220, 300, 470]], float)
    boxes1 = np.array([[640 - 110, 230, 640 - 20, 460], [340, 220, 440, 470]], float)
    return frame0, frame1, boxes0, boxes1


def main():
    weights = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path.home()
        / "AppData/Local/octron/octron/models/yolo11n-cls.pt"
    )
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    reid_path = export_torchscript(weights)
    tracker = make_tracker(reid_path)

    frame0, frame1, boxes0, boxes1 = demo_frames()
    dets0 = np.hstack([boxes0, np.full((2, 1), 0.9), np.zeros((2, 1))])
    dets1 = np.hstack([boxes1, np.full((2, 1), 0.9), np.zeros((2, 1))])

    feats0 = tracker.model.get_features(boxes0, frame0)
    feats1 = tracker.model.get_features(boxes1, frame1)
    sim = feats0 @ feats1.T
    print(f"[3] feature shape per detection: {feats0.shape}")
    print("    cosine(frame0 det i, frame1 det j):")
    for i in range(2):
        print("      " + "  ".join(f"{sim[i, j]:.3f}" for j in range(2)))
    print("    (diagonal = same person mirrored, off-diagonal = other person)")

    out0 = tracker.update(dets0, frame0)
    out1 = tracker.update(dets1, frame1)
    ids0 = out0[:, 4].astype(int).tolist() if len(out0) else []
    ids1 = out1[:, 4].astype(int).tolist() if len(out1) else []
    print(f"[4] track ids frame0 = {ids0}, frame1 = {ids1}")

    print(
        "\nFindings\n"
        "- Plumbing works: TorchScript backend + filename hack, no BoxMOT "
        "changes.\n"
        "- Caveats: (a) BoxMOT resizes crops to 256x128 and swaps BGR->RGB;"
        " OCTRON feeds RGB frames, so the classifier sees channel-swapped,"
        " squashed crops unless the wrapper compensates. (b) The stock "
        "yolo11n-cls embedding is ImageNet, not identity-trained; with the "
        "real identity model the diagonal above should separate clearly.\n"
        "- Proper route: a BoxMOT patch that accepts an arbitrary "
        "TorchScript/ONNX ReID file without the name-based registry, plus "
        "an input-size/colour hint. That is a small PR to horsto's fork."
    )
    shutil.rmtree(OUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
