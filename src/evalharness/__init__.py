"""Eval harness: drives the v3.9.2 evaluator over the served policy.

``run_eval`` wraps ``python -m omnigibson.eval.eval`` per task/instance;
``parallel`` fans out N server ports; ``aggregate``/``report`` recompute the
leaderboard metrics and render the report.  The harness talks to our server
only over the WebSocket — identical to how the organizers' evaluator will — so
a passing self-eval means the submission is serving-contract-correct.
"""
from .aggregate import (
    EVAL_TIMEOUT_MULTIPLIER,
    RolloutScore,
    Summary,
    aggregate,
    time_score_from_normalized,
)
from .report import render_report, summary_to_json

__all__ = [
    "EVAL_TIMEOUT_MULTIPLIER", "RolloutScore", "Summary", "aggregate",
    "time_score_from_normalized", "render_report", "summary_to_json",
]
