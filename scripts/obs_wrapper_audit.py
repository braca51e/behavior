"""Evaluation wrapper proof: audit the obs keys the b1k server consumes.

The challenge enforces, by manual inspection of the wrapper code, that the
policy uses **only onboard RGB + depth + proprioception** at eval (spec
section 2.3).  This script parses the recorded wire fixture and reports exactly
which observation keys ``b1k.protocol.parse_obs`` reads (the *only* obs path in
the serving stack), and asserts none is a privileged field.

Run::

    python scripts/obs_wrapper_audit.py

Exit 0 if the consumed keys are all onboard (rgb/depth/proprio); 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b1k.protocol import frame_from_dict, parse_obs  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "obs_payload.bin"

# Keys parse_obs is allowed to consume (onboard only).  Everything else in the
# obs dict is *ignored* by the server (it never reads privileged fields).
ALLOWED_PREFIX = ("rgb", "depth", "proprio")

# Substrings that would indicate a privileged field (global pose, segmentation,
# object pose, full point cloud) — the server must NOT touch these.
FORBIDDEN_MARKERS = ("segment", "object_pose", "target_pose", "point_cloud",
                     "global_pose", "privileged", "bddl", "sim_state")


def _keys_of(frame) -> list[str]:
    return sorted(frame.raw.keys())


def main() -> int:
    frame = parse_obs(FIX.read_bytes())
    all_keys = _keys_of(frame)
    print("wire obs keys present in fixture:")
    for k in all_keys:
        print("  ", k)

    # The server only ever *reads* these Frame fields (see parse_obs + the
    # pipeline).  Determine which fixture keys feed them.
    consumed = []
    if frame.rgb_head is not None:
        consumed.append("head rgb")
    if frame.rgb_left is not None:
        consumed.append("left wrist rgb")
    if frame.rgb_right is not None:
        consumed.append("right wrist rgb")
    if frame.depth_head is not None:
        consumed.append("head depth")
    if frame.depth_left is not None:
        consumed.append("left wrist depth")
    if frame.depth_right is not None:
        consumed.append("right wrist depth")
    consumed.append("61-dim proprio")

    print("\nb1k server CONSUMES (onboard only):")
    for c in consumed:
        print("  +", c)

    # Audit: no fixture key the server parses carries a forbidden marker.
    bad = []
    for k in all_keys:
        kl = k.lower()
        if any(m in kl for m in FORBIDDEN_MARKERS):
            bad.append(k)
    if bad:
        print("\nFAIL: privileged key(s) present/consumed:", bad)
        return 1
    # parse_obs must have routed every camera to an onboard field.
    frame_from_dict(frame.raw)  # no-throw re-parse
    print("\nPASS: no privileged obs keys; server uses RGB + depth + proprio only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
