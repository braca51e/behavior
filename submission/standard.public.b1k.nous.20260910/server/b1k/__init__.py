"""b1k — BEHAVIOR-1K 2026 challenge solution (serving side).

A hierarchical "System 2 / System 1" embodied agent for the R1Pro robot.

Import rule (enforced by ``tests/test_imports.py`` and CI): nothing in this
package may import from ``src/training`` (privileged/teacher code) or from
``omnigibson`` (the simulator).  The package must run on a plain CPU box with
only numpy/msgpack/websockets (torch is imported lazily and only for the real
VLA path).  Everything above the VLA is deterministic and CPU-testable.
"""

__version__ = "0.1.0"

# Module dependency rule (design.md section 3):
#   server -> controller -> {planner, perception, policy}
#   planner / perception import nothing from policy (keep the brain testable on CPU)
#   training/* is never imported at eval time.
