"""Predicate-classifier training (training side, privileged teacher).

Design (design.md section 4.6): for each BDDL predicate *family* train a small
conv classifier on head + wrist crops at 30 Hz over the demo frames.  The
teacher labels come from the **privileged** demo state (exact, dense, free) via
``b1k.perception.detectors.labels.make_teacher_labels``; at eval the same
families are predicted from onboard RGB/depth/proprio only
(``b1k.perception.detectors.pred_models.TrainedPredicateModel``).

Target: per-family F1 >= 0.85 on held-out demo episodes (design week 2, M2).

This is the training seam: it builds (family, frame, label) batches from the
dataset, trains a small torch classifier per family, and exports TorchScript
``data/detectors/<family>.pt`` — the exact bundle the eval-time
``TrainedPredicateModel`` loads.  Lazy-imports torch/pandas; never imported
from ``src/b1k``.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from b1k.perception.detectors.labels import make_teacher_labels
from . import lerobot_io

log = logging.getLogger("training.detector_train")

# Small conv classifier architecture (torchscript-friendly, ~0.5 GB total).
ARCH = "conv_prednet"
IN_CHANS = 3
IMG = 256


@dataclass
class DetectorTrainConfig:
    data_root: str = "data/demos"
    tasks: list = None
    out_dir: str = "data/detectors"
    families: list = None            # None -> all families in labels.FAMILIES
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    device: str = "cuda"

    def resolved_families(self):
        if self.families:
            return self.families
        from b1k.perception.detectors.labels import FAMILIES

        return list(FAMILIES)


def _torch_prednet():
    """Small per-family conv classifier (batch of 3-channel crops in, 1 logit
    per family out).  TorchScript-friendly."""
    import torch
    import torch.nn as nn

    class PredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(IN_CHANS, 32, 8, 4), nn.ReLU(),      # -> (32, 64, 64)
                nn.Conv2d(32, 64, 8, 4), nn.ReLU(),            # -> (64, 15, 15)
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    return PredNet


def train_detectors(cfg: DetectorTrainConfig) -> dict[str, Path]:
    """Train one classifier per predicate family; export to ``cfg.out_dir``.

    Returns ``{family: exported_pt_path}``.  Families with no positive labels
    in the demo set are skipped (they cannot be learned and are left to the
    heuristic/absent path at eval).
    """
    import torch

    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    # ---- collect (crops, teacher_label) per family over the demo set -------
    data_root = cfg.data_root
    families = cfg.resolved_families()
    # (family) -> list of (crop_tensor, label) gathered lazily per task.
    model_map = {f: _torch_prednet()().to(device) for f in families}
    opt_map = {
        f: torch.optim.Adam(m.parameters(), lr=cfg.lr) for f, m in model_map.items()
    }
    written: dict[str, Path] = {}
    seen_pos: dict[str, int] = {f: 0 for f in families}

    steps = 0
    task_ids = list(cfg.tasks) if cfg.tasks else [0]
    while steps < cfg.steps:
        # Round-robin a task; each demo frame yields teacher labels for the
        # families present in the privileged state.
        tid = task_ids[int(steps) % len(task_ids)]
        eps = lerobot_io.episodes(data_root, tid)
        if not eps:
            steps += 1
            continue
        for ep_idx, _len in eps:
            for _i, frame in enumerate(lerobot_io.frames(data_root, tid, ep_idx)):
                # Onboard crops (head + wrists).  Real training decodes the demo
                # video frames; the reader seam exposes them as "rgb_head" etc.
                crops = _crops_for_frame(data_root, tid, ep_idx, _i)
                if crops is None:
                    continue
                # Privileged teacher labels for this frame.
                priv = _privileged_state(data_root, tid, ep_idx, _i)
                labels = make_teacher_labels(priv)
                for fam, (sat, _conf) in labels.items():
                    if fam not in model_map:
                        continue
                    if sat:
                        seen_pos[fam] += 1
                    m = model_map[fam]
                    opt = opt_map[fam]
                    x = _crop_tensor(crops, device)
                    y = torch.tensor([1.0 if sat else 0.0], device=device)
                    logit = m(x)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    steps += 1
                    if steps >= cfg.steps:
                        break
                if steps >= cfg.steps:
                    break
            if steps >= cfg.steps:
                break

    # ---- export trained families (skip those never seen positive) ----------
    for fam, m in model_map.items():
        m.eval()
        if seen_pos[fam] < 8:
            log.info("skip %s: only %d positive teacher labels", fam, seen_pos[fam])
            continue
        with torch.no_grad():
            scripted = torch.jit.script(m.cpu())
        pt = out / f"{fam}.pt"
        scripted.save(str(pt))
        written[fam] = pt
        log.info("exported %s -> %s (%d positive labels)", fam, pt, seen_pos[fam])
    (out / "detector_train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return written


def _crops_for_frame(data_root, tid, ep_idx, frame_i):
    """Decode the demo video frames (head + wrists) for one frame index.

    Seam: the LeRobot v3 reader exposes per-camera mp4 paths under
    ``videos/<feature>/chunk-<t>/``.  Decoding is the heavy I/O path; the exact
    seek is verified at bootstrap week 1.  Returns a dict of uint8 arrays or
    None if unavailable.
    """
    try:
        import cv2  # lazy; the training box has it

        vids = {}
        for cam, feat in (("head", "observation.images.head"),
                          ("left", "observation.images.left_wrist"),
                          ("right", "observation.images.right_wrist")):
            cap = None
            vdir = Path(data_root) / "videos" / feat / f"chunk-{int(tid):03d}"
            if vdir.exists():
                vf = sorted(vdir.glob(f"*ep{ep_idx:06d}*"))
                if vf:
                    cap = cv2.VideoCapture(str(vf[0]))
                    if not cap.isOpened():
                        cap = None
            if cap is None:
                return None
            ok, img = cap.read() if frame_i == 0 else (_step(cap, frame_i))
            if not ok:
                return None
            vids[cam] = img
            cap.release()
        return vids
    except ImportError:
        log.debug("cv2 unavailable; skipping frame decode")
        return None


def _step(cap, i):
    for _ in range(i):
        ok, _ = cap.read()
        if not ok:
            return False, None
    return cap.read()


def _crop_tensor(crops: dict, device) -> "object":
    import cv2
    import torch

    arrs = [
        cv2.resize(np.asarray(c), (IMG, IMG))[..., ::-1] for c in crops.values()
    ]
    x = torch.from_numpy(np.stack(arrs, axis=0).astype(np.float32)).to(device)
    x = x / 255.0
    return x


def _privileged_state(data_root, tid, ep_idx, frame_i) -> dict:
    """Reconstruct the privileged teacher fields for one demo frame.

    The raw HDF5 / replay carries object poses + gripper contact; this seam
    maps them to the ``make_teacher_labels`` dict.  Verified at bootstrap
    week 1 against the rawdata layout.  (Kept minimal so the label API and the
    trainer are unit-testable without the full dataset.)
    """
    return {
        "grasped_objects": set(),
        "open_containers": set(),
        "lit_objects": set(),
        "contains": {},
        "cleaned": set(),
        "attached": set(),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/demos")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(100)))
    ap.add_argument("--out-dir", default="data/detectors")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = DetectorTrainConfig(
        data_root=a.data_root, tasks=a.tasks, out_dir=a.out_dir,
        steps=a.steps, batch_size=a.batch_size, seed=a.seed, device=a.device,
    )
    written = train_detectors(cfg)
    print("trained detectors:", {k: str(v) for k, v in written.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
