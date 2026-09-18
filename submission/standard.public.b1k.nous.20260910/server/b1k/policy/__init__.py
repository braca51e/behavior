"""Policy: skill-conditioned VLA wrapper, norm stats, chunk blending.

Only the ``"vla"`` backend imports torch (lazily); ``"echo"``/``"noop"`` are
pure numpy so the package serves on a CPU-only box.
"""
from .chunking import blend_chunks, receding_take
from .norm import NormStats
from .vla import SkillConditionedVLA

__all__ = ["SkillConditionedVLA", "NormStats", "blend_chunks", "receding_take"]
