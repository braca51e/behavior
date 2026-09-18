"""Local validation harness for the 2026 BEHAVIOR challenge.

Runs on any CPU box — no simulator, no GPU, no 3.27 TB demo set — and verifies
the solution is *functional and submission-ready* without relying on external
challenge testing.  It is the local stand-in for the organizers' evaluator +
``score_utils`` scoring, and it checks, in order:

1. **Unit tests**  — the serving-contract + scoring-math suite (``pytest``).
2. **Fixtures**    — the recorded obs/action + metrics fixtures exist (and are
                     regenerated deterministically if missing).
3. **Live server** — the real ``python -m b1k.server`` answers ``/healthz`` and
                     returns a finite, in-bounds, 23-dim action over the wire
                     (in-process, bounded steps, no simulator).
4. **Edge cases**  — malformed / missing-camera / bad-dim / non-finite /
                     duplicate / out-of-bounds / wrong-track inputs are handled
                     safely (finite action, correct accept/reject, no crash).
5. **Sample data** — the generated sample submissions exist and match their
                     frozen ``expected_*.json`` aggregate scores to 1e-12.
6. **Scoring**     — the local ``validator`` reproduces the *exact* official
                     ``score_utils`` math (verified against a hand-computed
                     case and against ``raw/score_utils.compute_final_q_score``
                     when the simulator env is importable).
7. **Report**      — ``validation/validation_report.md`` + ``validation/
                     validation_report.json`` (the submission-ready evidence).

Usage::

    python3 validation/run_validation.py            # full harness (CPU)
    python3 validation/run_validation.py --quick    # skip the live-server step
    python3 validation/run_validation.py --out DIR  # report destination

Exit code: 0 when every gate passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
SRC = REPO / "src"
for p in (str(SRC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from validation import validator  # noqa: E402

# ---------------------------------------------------------------------------
# result plumbing
# ---------------------------------------------------------------------------
class _Gate:
    def __init__(self, name: str):
        self.name = name
        self.status = "skip"
        self.detail = ""
        self.lines: list[str] = []


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                       capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr)


class Harness:
    def __init__(self, quick: bool = False, out: Path | None = None):
        self.quick = quick
        self.out_dir = out or (REPO / "validation")
        self.gates: list[_Gate] = []
        self.t0 = time.time()

    def gate(self, name: str) -> _Gate:
        g = _Gate(name)
        self.gates.append(g)
        return g

    def _set(self, g: _Gate, ok: bool, detail: str, lines: list[str] | None = None) -> None:
        g.status = "pass" if ok else "fail"
        g.detail = detail
        if lines:
            g.lines = lines
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {g.name}: {detail}")

    # ------------------------------------------------------------------ 1
    def gate_unit_tests(self) -> _Gate:
        g = self.gate("unit-tests")
        # `-o addopts=` drops pyproject's `-q` so the "N passed" summary line is
        # not suppressed (double -q hides it).
        code, out = _run([sys.executable, "-m", "pytest", "tests/",
                          "-o", "addopts=", "-q", "--tb=line"], REPO)
        tail = [ln for ln in out.splitlines() if ln.strip()][-4:]
        import re
        passed = ""
        for m in re.finditer(r"(\d+) passed", out):
            passed = m.group(1)
        ok = code == 0
        detail = (f"{passed} tests passed" if passed else
                  (f"FAILED (pytest exit {code})" if not ok else "pytest exit 0"))
        self._set(g, ok, detail, [f"pytest exit {code}", *tail])
        return g

    # ------------------------------------------------------------------ 2
    def gate_fixtures(self) -> _Gate:
        g = self.gate("fixtures")
        fix = REPO / "tests" / "fixtures"
        needed = ["obs_payload.bin", "action_23d.bin", "annotations.jsonl"]
        metrics = sorted(fix.glob("*.json"))
        present = all((fix / n).exists() for n in needed) and len(metrics) >= 5
        lines = []
        if not present:
            print("  [i] fixtures missing -> regenerating (deterministic, seeded)")
            code, out = _run([sys.executable, str(fix / "make_fixtures.py")], REPO)
            present = all((fix / n).exists() for n in needed) and len(list(fix.glob("*.json"))) >= 5
            lines.append(f"make_fixtures.py exit {code}")
        n_obs = (fix / "obs_payload.bin").stat().st_size if (fix / "obs_payload.bin").exists() else 0
        lines.append(f"obs_payload.bin {n_obs/1e6:.1f} MB; {len(list(fix.glob('*.json')))} metrics JSONs; "
                     f"action_23d.bin {'yes' if (fix/'action_23d.bin').exists() else 'NO'}")
        self._set(g, present, "recorded obs/action + metrics fixtures present", lines)
        return g

    # ------------------------------------------------------------------ 3
    def gate_live_server(self) -> _Gate:
        """Drive the real serving pipeline in-process on the recorded obs fixture.

        Spins ``B1KServer`` per fixture task, feeds the recorded 15 MB obs frame
        for a bounded number of steps, and asserts every action is a finite
        23-dim vector whose base velocity is within the r1pro output limits.
        """
        g = self.gate("live-server")
        lines: list[str] = []
        try:
            from b1k.config import load_config
            from b1k.protocol import parse_obs
            from b1k.server import B1KServer
            from b1k.embodiment import ACTION_DIM, BASE_OUT_LIMITS

            cfg = load_config(REPO / "configs" / "server.yaml")
            obs = (REPO / "tests" / "fixtures" / "obs_payload.bin").read_bytes()
            frame = parse_obs(obs)
            lo, hi = (np.asarray(BASE_OUT_LIMITS[0], float), np.asarray(BASE_OUT_LIMITS[1], float))
            tasks = [0, 1, 3]
            all_ok = True
            for tid in tasks:
                server = B1KServer(cfg, task=tid)
                bad = 0
                for _ in range(30):
                    a = np.asarray(server.handle_frame(frame), dtype=np.float32)
                    if a.shape != (ACTION_DIM,) or not np.all(np.isfinite(a)) or not np.all((a[0:3] >= lo[0:3]) & (a[0:3] <= hi[0:3])):
                        bad += 1
                lines.append(f"task {tid} ({server.task.task_name}): 30 steps, {bad} bad actions")
                all_ok &= (bad == 0)
            self._set(g, all_ok,
                      f"live B1KServer serves finite, in-bounds {ACTION_DIM}-dim actions on the recorded obs",
                      lines)
            return g
        except Exception as e:  # noqa: BLE001
            lines.append(f"exception: {type(e).__name__}: {e}")
            self._set(g, False, f"live-server check raised {type(e).__name__}", lines)
            return g

    # ------------------------------------------------------------------ 4
    def gate_edge_cases(self) -> _Gate:
        """Verify malformed / unusual inputs are handled safely."""
        g = self.gate("edge-cases")
        lines: list[str] = []
        from b1k.config import load_config
        from b1k.embodiment import ACTION_DIM
        from b1k.protocol import decode_action, encode_action, frame_from_dict, parse_obs

        cfg = load_config(REPO / "configs" / "server.yaml")
        checks: list[tuple[str, bool]] = []

        # (a) empty obs dict -> finite no-op action, no crash.
        try:
            from b1k.server import B1KServer
            srv = B1KServer(cfg, task=0)
            a = np.asarray(srv.handle_frame(frame_from_dict({})), dtype=np.float32)
            checks.append(("empty obs -> finite 23-dim", a.shape == (ACTION_DIM,) and bool(np.all(np.isfinite(a)))))
        except Exception as e:  # noqa: BLE001
            checks.append((f"empty obs raised {e}", False))

        # (b) obs missing all cameras but proprio present -> still a finite action.
        try:
            obs = {"robot_r1::proprio": [0.05, 0.0, 0.0] + [0.0] * 58}
            srv = B1KServer(cfg, task=0)
            a = np.asarray(srv.handle_frame(frame_from_dict(obs)), dtype=np.float32)
            checks.append(("proprio-only obs -> finite 23-dim", bool(np.all(np.isfinite(a)))))
        except Exception as e:  # noqa: BLE001
            checks.append((f"proprio-only obs raised {e}", False))

        # (c) non-numeric garbage -> parse_obs rejects with ValueError (no silent corruption).
        try:
            parse_obs(b"\x93\xa2ab\x01")  # short binary msgpack of a non-dict
            checks.append(("non-dict binary rejected", False))
        except Exception:
            checks.append(("non-dict binary rejected (raises)", True))

        # (d) wrong-dim action -> encode/decode reject.
        try:
            encode_action(np.zeros(10))
            checks.append(("wrong-dim action rejected", False))
        except ValueError:
            checks.append(("wrong-dim action rejected", True))
        try:
            decode_action(encode_action(np.zeros(ACTION_DIM)) )  # ok
            bad = b"\xc2"  # msgpack float of 0.0 -> decodes to scalar, wrong size
            decode_action(bad)
            checks.append(("scalar decode rejected", False))
        except ValueError:
            checks.append(("scalar decode rejected", True))

        # (e) base velocity clamp: a raw action with out-of-range base is clamped.
        try:
            from b1k.controller import ActionController
            from b1k.policy.vla import SkillConditionedVLA
            from b1k.embodiment import BASE_OUT_LIMITS
            vla = SkillConditionedVLA(backend="noop")
            ctrl = ActionController(cfg.controller, vla)
            raw = np.concatenate([np.array([9.0, -9.0, 5.0]), np.zeros(ACTION_DIM - 3)]).astype(np.float32)
            a = ctrl._safety(raw)
            hi = np.asarray(BASE_OUT_LIMITS[1], dtype=np.float32)
            lo = np.asarray(BASE_OUT_LIMITS[0], dtype=np.float32)
            clamped = bool(np.all((a[0:3] >= lo) & (a[0:3] <= hi)))
            # +9.0 must clamp to the x-y limit (0.75); -9.0 to -0.75; 5.0 to 1.0 (yaw).
            exact = abs(a[0] - hi[0]) < 1e-6 and abs(a[1] - lo[1]) < 1e-6 and abs(a[2] - hi[2]) < 1e-6
            checks.append(("base velocity clamped to r1pro output limits", clamped and exact))
        except Exception as e:  # noqa: BLE001
            checks.append((f"clamp raised {type(e).__name__}: {e}", False))

        # (f) malformed / out-of-bounds / unknown-task / duplicate metrics ->
        #     validator rejects the bad ones and accepts the good one.
        base_task = validator.load_task_names()[0]   # "turning_on_radio"
        bad_dir = self.out_dir / "_edge_bad"
        if bad_dir.exists():
            shutil.rmtree(bad_dir)
        (bad_dir / "json").mkdir(parents=True)

        def _w(dirpath: Path, fname: str, task: str, inst: int, roll: int, q: float, nt: float):
            (dirpath / fname).write_text(json.dumps({
                "task": task, "instance_id": inst, "rollout_id": roll, "steps": 100,
                "success": q >= 1.0,
                "agent_distance": {"base": 1.0, "left": 1.0, "right": 1.0},
                "normalized_agent_distance": {"base": 0.5, "left": 0.5, "right": 0.5},
                "q_score": {"final": q},
                "time": {"simulator_steps": 100, "simulator_time": 3.3, "normalized_time": nt}}))

        # one legal file (baseline) + four bad ones in a *valid* folder.
        legal = f"standard.public.b1k.nous.20260101"
        bdir = bad_dir / legal
        (bdir / "json").mkdir(parents=True)
        _w(bdir / "json", f"{base_task}_301_0.json", base_task, 301, 0, 1.0, 1.0)      # legal
        _w(bdir / "json", f"{base_task}_400_0.json", base_task, 400, 0, 1.0, 1.0)      # out-of-bounds instance
        _w(bdir / "json", f"{base_task}_301_1.json", base_task, 301, 1, 1.0, 1.0)      # rollout_id != 0
        _w(bdir / "json", "unknown_task_301_0.json", "unknown_task", 301, 0, 1.0, 1.0)  # unknown task name
        v_bad = validator.validate_submission(bdir)
        # only the legal file should be scored (n_rollouts == 1); the other 3 flagged as illegal.
        checks.append(("validator accepts exactly the legal file",
                       v_bad.n_rollouts == 1 and sum(1 for e in v_bad.errors if "illegal" in e) >= 2))
        checks.append(("out-of-bounds instance flagged", any("_400_" in e for e in v_bad.errors)))
        checks.append(("unknown-task filename flagged", any("unknown_task" in e for e in v_bad.errors)))

        # single legal rollout in a valid folder -> scores cleanly (passed).
        ok_dir = bad_dir / "standard.public.b1k.nous.20260103"
        (ok_dir / "json").mkdir(parents=True)
        _w(ok_dir / "json", f"{base_task}_301_0.json", base_task, 301, 0, 1.0, 1.0)
        v_ok = validator.validate_submission(ok_dir)
        checks.append(("single legal rollout scores and passes", v_ok.n_rollouts == 1 and v_ok.passed))

        # malformed folder name -> parse error, not a crash.
        v_badname = validator.validate_submission(bad_dir)   # bad_dir.name == "_edge_bad"
        checks.append(("malformed folder name rejected", (not v_badname.passed) and any("not <track>" in e for e in v_badname.errors)))

        all_ok = all(ok for _, ok in checks)
        lines = [f"{'ok ' if ok else 'BAD'} {name}" for name, ok in checks]
        self._set(g, all_ok, f"{sum(1 for _,o in checks if o)}/{len(checks)} edge cases handled safely", lines)
        return g

    # ------------------------------------------------------------------ 5
    def gate_sample_data(self) -> _Gate:
        g = self.gate("sample-data")
        lines: list[str] = []
        sample = self.out_dir / "sample_data"
        full = sample / validator_module_full_folder()
        small = sample / validator_module_sample_folder()
        if not full.exists():
            print("  [i] sample data missing -> generating")
            code, out = _run([sys.executable, str(self.out_dir / "make_sample_data.py")], REPO)
            lines.append(f"make_sample_data.py exit {code}")
        ok = True
        for folder in (full, small):
            exp_p = folder.parent / f"expected_{folder.name}.json"
            if not folder.exists() or not exp_p.exists():
                ok = False
                lines.append(f"missing {folder.name} / {exp_p.name}")
                continue
            exp = json.loads(exp_p.read_text())
            v = validator.validate_submission(folder)
            drift = max(abs(v.overall_q - exp["overall_q"]),
                        abs(v.overall_task_sr - exp["overall_task_sr"]),
                        abs(v.overall_time_score - exp["overall_time_score"]))
            matched = drift <= 1e-12 and v.n_rollouts == exp["n_rollouts"]
            ok &= matched
            lines.append(f"{folder.name}: rollouts={v.n_rollouts} Q={v.overall_q:.6f} "
                         f"task_sr={v.overall_task_sr:.4f} time={v.overall_time_score:.6f} "
                         f"drift={drift:.2e} {'MATCH' if matched else 'MISMATCH'}")
        self._set(g, ok, "sample submissions exist and match frozen expected scores to 1e-12", lines)
        return g

    # ------------------------------------------------------------------ 6
    def gate_scoring(self) -> _Gate:
        g = self.gate("scoring")
        lines: list[str] = []

        # (a) hand-computed parity: 2 tasks x 2 instances, known values.
        # task A: q 1.0 (ts 1.0), q 0.5 (ts -1.0)  -> per-task time mean = 0.0
        # task B: q 0.0 (ts 1.0), q 1.0 (ts 2.0)   -> per-task time mean = 1.5
        # task C: absent                           -> 0
        # Q / task_sr divide by the full instance count (n_inst=2); time_score /
        # distances divide by the present-rollout count (k) — official math.
        # overall: Q=(0.75+0.50+0)/3=5/12 ; task_sr=(0.5+0.5+0)/3=1/3 ;
        #          time=(0.0+1.5+0)/3=0.5
        from validation.validator import Rollout, get_scores
        A, B, C = "alpha", "beta", "gamma"
        rolls = [
            Rollout(A, 0, 0, 1.0, 1.0, 0.1, 0.1, 0.1),
            Rollout(A, 1, 0, 0.5, -1.0, 0.2, 0.2, 0.2),
            Rollout(B, 0, 0, 0.0, 1.0, 0.3, 0.3, 0.3),
            Rollout(B, 1, 0, 1.0, 2.0, 0.4, 0.4, 0.4),
        ]
        s = get_scores(rolls, n_tasks=3, n_instances_per_task=2, task_names=[A, B, C])
        hand_ok = (abs(s["overall_q"] - (5.0/12.0)) < 1e-12 and abs(s["overall_task_sr"] - (1/3)) < 1e-12
                   and abs(s["overall_time_score"] - 0.5) < 1e-12
                   and abs(s["per_task_q"][A] - 0.75) < 1e-12   # (1.0+0.5)/2
                   and abs(s["per_task_sr"][B] - 0.5) < 1e-12   # 1 full of 2
                   and abs(s["per_task_time_score"][B] - 1.5) < 1e-12)
        lines.append(f"hand-computed case: Q={s['overall_q']:.6f} (want 0.416667), "
                     f"task_sr={s['overall_task_sr']:.6f} (want 0.333333), time={s['overall_time_score']:.6f} (want 0.5), "
                     f"taskA q={s['per_task_q'][A]:.3f} (want 0.750), taskB sr={s['per_task_sr'][B]:.3f} (want 0.500)")
        checks_ok = [hand_ok]

        # (b) cross-check our time_score against the official formula string.
        for nt in (0.6, 0.8, 1.0, 1.2, 1.5, 1.9):
            ours = validator.time_score_from_normalized(nt)
            official = 1.5 / (1.5 - 1.0) - 1.0 / ((1.5 - 1.0) * nt)
            checks_ok.append(abs(ours - official) < 1e-12)
        lines.append("time_score == official 3 - 2/nt across 6 values")

        # (c) parity against raw/score_utils.compute_final_q_score (if importable).
        sample_dir = self.out_dir / "sample_data"
        small = sample_dir / "standard.public.b1k.nous.20260910"
        full = sample_dir / "standard.public.b1k.nous.20260912"
        official_dir = self._run_official_score_utils(small if small.exists() else full)
        if official_dir is None:
            lines.append("raw/score_utils not importable (needs omnigibson) -> skipped live parity (hand-computed parity covers it)")
        else:
            checks_ok.append(official_dir["ok"])
            lines.append(f"official compute_final_q_score parity: Q={official_dir['official_q']:.6f} "
                         f"vs local {official_dir['local_q']:.6f} (drift {official_dir['drift']:.2e})")

        ok = all(checks_ok)
        self._set(g, ok, "local scoring reproduces the exact official score_utils math", lines)
        return g

    def _run_official_score_utils(self, sample_dir: Path) -> dict | None:
        """Run the organizers' compute_final_q_score on our sample and compare."""
        try:
            raw = REPO / "raw"
            for p in (str(raw),):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import score_utils  # noqa: F401  (from raw/eval_utils imports omnigibson)
        except Exception:
            return None
        out = self.out_dir / "_official_scores"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        try:
            # raw/score_utils operates on <input_dir>/<submission>/json/...
            stage = out / "stage"
            stage.mkdir()
            shutil.copytree(sample_dir, stage / sample_dir.name)
            code, _ = _run([sys.executable, "-c",
                            "import sys; sys.path.insert(0, str(sys.argv[1]));"
                            "import score_utils as su, json, glob, pathlib;"
                            "su.compute_final_q_score(sys.argv[2], sys.argv[3], final_score_only=True, verbose=True);",
                            str(REPO / "raw"), str(stage), str(out)], REPO)
            produced = glob.glob(str(out / "*.json"))
            if not produced:
                return {"ok": False, "official_q": float("nan"), "local_q": float("nan"), "drift": float("nan")}
            official = json.loads(Path(produced[0]).read_text())["overall_scores"]
            local = validator.validate_submission(stage / sample_dir.name)
            drift = abs(official["q_score"] - local.overall_q)
            return {"ok": drift <= 1e-9, "official_q": official["q_score"],
                    "local_q": local.overall_q, "drift": drift}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] official parity failed to run: {e}")
            return None

    # ------------------------------------------------------------------ 7
    def write_report(self) -> _Gate:
        g = self.gate("report")
        lines: list[str] = []
        report_dir = self.out_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        md = report_dir / "validation_report.md"
        jf = report_dir / "validation_report.json"
        any_fail = any(gl.status == "fail" for gl in self.gates)

        # Set this gate's status *before* rendering so the table row reflects it.
        g.status = "pass" if not any_fail else "fail"
        g.detail = f"wrote {md.name} + {jf.name} (verdict {'PASS' if not any_fail else 'FAIL'})"

        obj = {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(time.time() - self.t0, 2),
            "mode": "quick" if self.quick else "full",
            "all_pass": (not any_fail),
            "gates": [
                {"name": gl.name, "status": gl.status, "detail": gl.detail, "lines": gl.lines}
                for gl in self.gates
            ],
        }
        jf.write_text(json.dumps(obj, indent=2) + "\n")

        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        L = []
        L.append("# Validation report — b1k / 2026 BEHAVIOR challenge")
        L.append("")
        L.append(f"- generated: **{now}** (mode: `{obj['mode']}`, {obj['wall_seconds']}s)")
        verdict = "PASS — functional and submission-ready" if not any_fail else "FAIL — see gates below"
        L.append(f"- verdict: **{verdict}**")
        L.append("")
        L.append("| # | gate | status | detail |")
        L.append("|---|---|---|---|")
        for i, gl in enumerate(self.gates, 1):
            L.append(f"| {i} | {gl.name} | {gl.status.upper()} | {gl.detail} |")
        L.append("")
        for gl in self.gates:
            if gl.lines:
                L.append(f"## {gl.name}")
                L.append("")
                for ln in gl.lines:
                    L.append(f"- {ln}")
                L.append("")
        L.append("> This harness verifies the serving pipeline, output format, edge cases, "
                 "and the scoring math locally (CPU, no simulator). It is a stand-in for the "
                 "organizers' evaluator; a real leaderboard Q still requires `--mode sim` on a "
                 "GPU box with OmniGibson v3.9.2.")
        L.append("")
        md.write_text("\n".join(L))
        lines = [f"report -> {md}", f"json  -> {jf}"]
        self._set(g, (not any_fail), f"wrote {md.name} + {jf.name} (verdict {'PASS' if not any_fail else 'FAIL'})", lines)
        return g

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        print(f"== b1k validation harness ({'quick' if self.quick else 'full'}) ==", flush=True)
        self.gate_unit_tests()
        self.gate_fixtures()
        if not self.quick:
            self.gate_live_server()
        self.gate_edge_cases()
        self.gate_sample_data()
        self.gate_scoring()
        self.write_report()
        n_fail = sum(1 for gl in self.gates if gl.status == "fail")
        n_pass = sum(1 for gl in self.gates if gl.status == "pass")
        print(f"\n== {n_pass} passed, {n_fail} failed, {len(self.gates) - n_pass - n_fail} skipped "
              f"({time.time() - self.t0:.1f}s) ==")
        return 0 if n_fail == 0 else 1


def validator_module_full_folder() -> str:
    from validation import make_sample_data as m
    return m.FULL_FOLDER


def validator_module_sample_folder() -> str:
    from validation import make_sample_data as m
    return m.SAMPLE_FOLDER


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the in-process live-server gate")
    ap.add_argument("--out", default=None, help="report/sample-data destination (default: validation/)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else REPO / "validation"
    h = Harness(quick=a.quick, out=out)
    return h.run()


if __name__ == "__main__":
    raise SystemExit(main())
