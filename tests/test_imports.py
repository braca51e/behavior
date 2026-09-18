"""Import-boundary tests (design.md section 3 dependency rule; the CI check).

Guarantees that no privileged code path can leak into serving:
  * nothing under ``src/b1k`` imports from ``src/training`` (no teacher /
    dataset access at serve time);
  * nothing under ``src/b1k`` imports ``omnigibson`` (the simulator) or any
    other privileged/eval-only package.

This is the "no privileged leakage by construction" check from the
implementation contract — it runs with no GPU, no dataset, no simulator.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

FORBIDDEN_MODULES = ("training", "omnigibson", "src.training")


def _b1k_files() -> list[Path]:
    return sorted((SRC / "b1k").rglob("*.py"))


def _imports_of(path: Path) -> list[str]:
    """Top-level and in-function import module names in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.append(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    return mods


def test_b1k_files_exist():
    assert len(_b1k_files()) >= 12, "expected a full b1k package"


@pytest.mark.parametrize("f", _b1k_files(), ids=lambda f: f.name)
def test_b1k_no_training_or_sim_imports(f: Path):
    """No b1k module may import training/* or omnigibson (privileged)."""
    mods = _imports_of(f)
    bad = [m for m in mods if m in FORBIDDEN_MODULES]
    assert not bad, f"{f.name} imports forbidden module(s): {bad}"


def test_training_is_quarantined_package():
    """src/training is a real package (so the boundary is meaningful)."""
    assert (SRC / "training" / "__init__.py").exists()


def test_b1k_imports_clean():
    """Importing the b1k package and its submodules pulls in no privileged code."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import b1k  # noqa: F401
    import b1k.server  # noqa: F401
    import b1k.protocol  # noqa: F401
    import b1k.controller  # noqa: F401
    import b1k.planner  # noqa: F401
    import b1k.perception  # noqa: F401
    import b1k.policy  # noqa: F401
    # After importing the whole serving stack, the simulator must NOT be loaded.
    assert "omnigibson" not in sys.modules
    # The training package must not be loaded by the serving path.
    assert not any(m in sys.modules for m in sys.modules if m.startswith("training."))


def test_b1k_module_names_stable():
    """Pin the public serving modules (contract surface)."""
    names = {f.stem for f in _b1k_files()}
    for required in ("server", "protocol", "controller", "config", "embodiment"):
        assert required in names, f"missing serving module {required}"
