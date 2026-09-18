"""Local validation harness for the b1k 2026 BEHAVIOR solution.

Submodules:
  * :mod:`validator`          — exact official ``score_utils`` math + submission
                                folder / file / schema validators.
  * :mod:`make_sample_data`   — deterministic sample submissions + frozen
                                expected scores.
  * :mod:`run_validation`     — the gate runner + report writer (entrypoint).

Run the whole harness::

    python3 validation/run_validation.py            # full (CPU, no simulator)
    python3 validation/run_validation.py --quick    # skip the live-server gate

Or score an existing sample / submission folder directly::

    python3 -c "import validation.validator as v; print(v.validate_submission('validation/sample_data/standard.public.b1k.nous.20260910'))"
"""
