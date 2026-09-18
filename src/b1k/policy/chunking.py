"""Action-chunk horizon blending (design.md section 4.5, port of GR00T N1.7's
temporal ensembling to the serving layer).

When the receding-horizon controller re-queries the VLA mid-chunk, the new
chunk and the still-remaining old chunk disagree slightly at the boundary.
Blending avoids a visible "jerk" at each re-query:

    a_t = (1 - alpha_t) * old[t] + alpha_t * new[t]

with ``alpha`` ramping 0 -> 1 over ``blend_steps`` from the chunk start, so
the first few steps lean on the (already partially executed) old chunk and the
rest lean on the fresh one.  When no old chunk exists, the new one is used
verbatim.

This is deterministic and CPU-testable (``tests/test_planner.py``).
"""
from __future__ import annotations

import numpy as np


def blend_chunks(
    old: np.ndarray | None,
    new: np.ndarray,
    step_in_chunk: int,
    blend_steps: int = 4,
) -> np.ndarray:
    """Blend an old (partially consumed) action chunk with a freshly queried one.

    Parameters
    ----------
    old:
        The remaining tail of the previous chunk, shape (T, 23), or None for a
        first chunk.  Aligned so that ``old[0]`` is the action that would be
        served at ``step_in_chunk`` (i.e. the controller passes the slice of the
        cached chunk starting at the current receding offset).
    new:
        The freshly queried chunk, shape (T', 23).
    step_in_chunk:
        How many steps into the *new* chunk we currently are (0 at a fresh
        query).  Drives the alpha ramp.
    blend_steps:
        Number of steps over which alpha ramps 0 -> 1.

    Returns
    -------
    np.ndarray
        Blended chunk, shape (max(T, T'), 23); trailing rows past one of the
        inputs are taken from whichever still has values.
    """
    new = np.asarray(new, dtype=np.float32)
    if new.ndim == 1:
        new = new.reshape(1, -1)
    if old is None:
        return new.copy()

    old = np.asarray(old, dtype=np.float32)
    if old.ndim == 1:
        old = old.reshape(1, -1)

    T = max(old.shape[0], new.shape[0])
    out = np.zeros((T, new.shape[1]), dtype=np.float32)
    for t in range(T):
        if t < new.shape[0]:
            nn = new[t]
        else:
            nn = new[-1]
        if t < old.shape[0]:
            oo = old[t]
        else:
            oo = old[-1]
        alpha = min(1.0, max(0.0, (step_in_chunk + t) / float(max(1, blend_steps))))
        out[t] = (1.0 - alpha) * oo + alpha * nn
    return out


def receding_take(chunk: np.ndarray, offset: int, n: int) -> np.ndarray:
    """Return ``n`` actions from ``chunk`` starting at receding ``offset``.

    If the chunk is shorter than ``offset + n``, the last available action is
    held for the remainder (safe: it is the most recent policy output, better
    than zeroing).
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    L = chunk.shape[0]
    if offset >= L:
        return np.repeat(chunk[-1:], max(1, n), axis=0)
    take = chunk[offset : offset + n]
    if take.shape[0] < n:
        pad = np.repeat(chunk[-1:], n - take.shape[0], axis=0)
        take = np.concatenate([take, pad], axis=0)
    return take
