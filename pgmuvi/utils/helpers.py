from __future__ import annotations
import contextlib
import os
import re
import numpy as np


def ensure_dir(path: str):
    """
    Create directory `path` if missing (mkdir -p semantics).

    Parameters
    ----------
    path : str
        The directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def is_magnitude(name: str | None = None,
                 unit: str | None = None,
                 meta: dict | None = None) -> bool:
    """
    Heuristic: return True if a band/column looks like magnitudes rather than flux.
    Signals:
      - unit string contains "mag" (case-insensitive)
      - name contains common magnitude tokens: "mag", "_mag", "m_", "vmag", "gmag", etc.
      - metadata dict has unit/name hints that include "mag"
    This mirrors the monolith's intent while remaining liberal in what it accepts.
    """
    with contextlib.suppress(Exception):
        if unit and isinstance(unit, str) and "mag" in unit.lower():
            return True
    tokens = set()
    if isinstance(name, str):
        s = name.lower()
        tokens |= {s}
        tokens |= set(re.split(r'[^a-z0-9]+', s))
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, str):
                tokens.add(v.lower())

    mag_hints = {"mag", "magnitude", "_mag", "vmag", "gmag", "rmag", "imag", "zmag",
                 "ymag", "jmag", "hmag", "kmag"}
    return bool(tokens & mag_hints)


def _clean_errors(yerr: np.ndarray, jitter_floor: float) -> tuple[np.ndarray, float]:
    """
    Ensure per-point uncertainties are positive & finite.

    - Replace NaN/inf/<=0 with a robust base error (median of valid values if any;
      else `jitter_floor`).
    - Return (cleaned_errors, base_error_used).
    """
    yerr = np.asarray(yerr, dtype=float)
    valid = np.isfinite(yerr) & (yerr > 0)
    if valid.any():
        base = float(np.nanmedian(yerr[valid]))
        base = max(base, jitter_floor)
    else:
        base = jitter_floor
    yerr_eff = np.where(valid, yerr, base)
    return (yerr_eff, base)
