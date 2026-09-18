"""Training-time pipeline (design.md section 4.6).

This package is **privileged**: it reads the LeRobot v3.0 demo dataset, uses
privileged simulator state for detector teacher labels, and (eventually) trains
the pi0.5 VLA + per-task adapters + predicate detectors.  It is *quarantined* —
nothing under ``src/b1k`` may import from here (enforced by
``tests/test_imports.py``) so no privileged path can leak into serving.

It is intentionally import-light: sub-modules load heavy deps (pandas, torch,
huggingface_hub) lazily so ``import src.training`` never fails on a CPU-only
serving box and the import boundary is testable without installing the
training stack.

Sub-modules (import as ``src.training.<name>``):

* ``download``          — per-chunk HF download (100 tasks, 3.27 TB plan).
* ``lerobot_io``        — LeRobot v3.0 frame/episode/video reader.
* ``stats``             — recompute ``meta/stats.json`` (post velocity fix).
* ``skill_annotations`` — per-episode language -> skill-segment parse.
* ``vla_finetune``      — pi0.5 skill-conditioned finetune (OpenPI fork cfg).
* ``adapters``          — per-task LoRA / task-embedding adapters.
* ``detector_train``    — predicate classifier training (privileged teacher).
"""
from __future__ import annotations

