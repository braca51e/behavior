"""Leaderboard-format report (design.md section 4.7).

Renders an :class:`~b1k.evalharness.aggregate.Summary` to a Markdown report —
the artifact ``scripts/self_eval.sh`` writes to ``outputs/<date>/report.md``.
Includes overall Q / task_sr / time_score, a per-task table sorted by Q (worst
first, to surface the bottom-quartile adapter candidates), and a diff vs a
previous run's report for the regression gate (design week 4: no model swap
without a per-task regression check).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .aggregate import Summary


def summary_to_json(summary: Summary, out: Path) -> None:
    d = {
        "overall": summary.overall,
        "per_task": {
            "q_score": summary.per_task_q,
            "task_sr": summary.per_task_sr,
            "time_score": summary.per_task_time,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def _load_prev_q(report_or_summary_path: Path) -> dict[str, float]:
    """Extract per-task Q from a previous ``summary.json`` or ``report.md``."""
    if report_or_summary_path.suffix == ".json":
        with open(report_or_summary_path, "r", encoding="utf-8") as f:
            return json.load(f).get("per_task", {}).get("q_score", {})
    # report.md embeds a machine-readable block <!-- qjson {...} -->
    text = Path(report_or_summary_path).read_text(encoding="utf-8")
    marker = "<!-- qjson "
    end = "-->"
    i = text.find(marker)
    if i >= 0:
        j = text.find(end, i)
        if j > i:
            try:
                return json.loads(text[i + len(marker):j]).get("q_score", {})
            except Exception:
                pass
    return {}


def render_report(
    summary: Summary,
    out: Path,
    team: str = "b1k",
    previous: Path | None = None,
) -> Path:
    """Write the Markdown report + summary.json; return the report path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o = summary.overall
    tasks = sorted(summary.per_task_q, key=lambda t: summary.per_task_q[t])
    prev_q = _load_prev_q(previous) if previous else {}

    lines = []
    lines.append(f"# Self-eval report — team `{team}`")
    lines.append("")
    lines.append(f"- rollouts: **{o['num_rollouts']}** across **{o['num_tasks']}** tasks")
    lines.append(f"- **Q (mean task BDDL predicate fraction): {o['q_score']:.4f}**")
    lines.append(f"- task_sr (full success rate): {o['task_sr']:.4f}")
    lines.append(f"- time_score (efficiency tie-breaker): {o['time_score']:.4f}")
    lines.append("")
    lines.append("## Per-task (sorted worst → best)")
    lines.append("")
    lines.append("| task | Q | task_sr | time_score | ΔQ vs prev |")
    lines.append("|---|---|---|---|---|")
    for t in tasks:
        q = summary.per_task_q[t]
        sr = summary.per_task_sr.get(t, 0.0)
        ts = summary.per_task_time.get(t, 0.0)
        if t in prev_q:
            delta = q - prev_q[t]
            dstr = f"{delta:+.3f}"
        else:
            dstr = "—"
        lines.append(f"| {t} | {q:.3f} | {sr:.2f} | {ts:.3f} | {dstr} |")
    lines.append("")
    if prev_q:
        regs = [t for t in prev_q if t in summary.per_task_q and
                summary.per_task_q[t] < prev_q[t] - 0.005]
        lines.append(f"## Regressions vs previous run: {len(regs)}")
        lines.append("")
        if regs:
            for t in sorted(regs, key=lambda x: summary.per_task_q[t] - prev_q[x]):
                lines.append(f"- {t}: {prev_q[t]:.3f} -> {summary.per_task_q[t]:.3f}")
        else:
            lines.append("- none")
        lines.append("")
    lines.append("<!-- qjson " + json.dumps({"q_score": summary.per_task_q}) + " -->")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    summary_to_json(summary, out.parent / "summary.json")
    return out
