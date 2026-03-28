"""
Sampling quality assessment utilities for lightcurve data.

Provides metrics and validation gates to detect poorly sampled lightcurves
before GP fitting, preventing bad fits due to sparse coverage, large gaps,
or inadequate baseline.  Also provides :func:`subsample_lightcurve` to
randomly draw a size-limited subset of observations while preserving the
temporal baseline, sampling structure and gap-coverage constraints.
"""

import numpy as np


def robust_scale(y: np.ndarray, c: float = 0.6745) -> float:
    """
    Compute robust scale estimate using median absolute deviation (MAD).

    scale = MAD(y) / c

    where c = 0.6745 for equivalence to std dev under Gaussian.

    Parameters
    ----------
    y : np.ndarray
        Data values (should be finite)
    c : float, default=0.6745
        Normalization constant (0.6745 ≈ Φ^(-1)(0.75) for Gaussian)

    Returns
    -------
    float
        Robust scale estimate, or 0.0 if all values identical

    Notes
    -----
    More robust to outliers than standard deviation.
    Used to normalize measurements for scale-independent comparisons.
    """
    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        return 0.0

    median_y = np.median(y_finite)
    mad = np.median(np.abs(y_finite - median_y))
    return mad / c if mad > 0 else 0.0


def compute_sampling_metrics(
    t: np.ndarray,
    y: np.ndarray = None,
    yerr: np.ndarray = None,
) -> dict:
    """
    Compute comprehensive temporal sampling quality metrics.

    Parameters
    ----------
    t : np.ndarray
        Observation times (any units)
    y : np.ndarray, optional
        Flux values (used for additional diagnostics if provided)
    yerr : np.ndarray, optional
        Uncertainties (used for SNR metrics if provided)

    Returns
    -------
    dict
        {
            'n_points': int,
            'baseline': float,
                Total time span (max(t) - min(t))
            'max_gap': float,
                Largest gap between consecutive observations
            'max_gap_fraction': float,
                max_gap / baseline
            'median_cadence': float,
                Median time between observations
            'mean_cadence': float,
                Mean time between observations
            'cadence_std': float,
                Standard deviation of cadences
            'nyquist_period': float,
                2 * median_cadence (shortest reliably detectable period)
            'nyquist_frequency': float,
                1 / (2 * median_cadence)
            'longest_detectable_period': float,
                baseline / 2 (heuristic upper limit)
            'duty_cycle': float,
                Fraction of baseline with observations (simple estimate)
            'sampling_uniformity': float,
                1 - (std(cadences) / mean(cadences)), range [0,1]
                1 = perfectly uniform, 0 = highly irregular
        }

        If y and yerr provided, also includes:
        {
            'median_snr': float,
            'mean_snr': float,
            'fraction_snr_gt_3': float,
            'fraction_snr_gt_5': float
        }

    Examples
    --------
    >>> metrics = compute_sampling_metrics(t, y, yerr)
    >>> print(f"Nyquist period: {metrics['nyquist_period']:.2f} days")
    >>> print(f"Detectable range: {metrics['nyquist_period']:.1f}"
    ...       f" - {metrics['longest_detectable_period']:.1f} days")
    """
    t = np.asarray(t)
    if len(t) < 2:
        return {"n_points": len(t), "error": "Too few points (N < 2)"}

    # Sort times
    t_sorted = np.sort(t[np.isfinite(t)])
    n = len(t_sorted)

    if n < 2:
        return {"n_points": n, "error": "Too few finite points (N < 2)"}

    # Basic temporal metrics
    baseline = float(t_sorted[-1] - t_sorted[0])
    if baseline == 0:
        return {"n_points": n, "error": "Zero baseline (all times identical)"}

    gaps = np.diff(t_sorted)
    max_gap = float(np.max(gaps))
    median_cad = float(np.median(gaps))
    mean_cad = float(np.mean(gaps))
    std_cad = float(np.std(gaps))

    # Sampling quality indicators
    uniformity = 1.0 - (std_cad / mean_cad) if mean_cad > 0 else 0.0
    uniformity = max(0.0, min(1.0, uniformity))  # Clamp to [0,1]

    # Simple duty cycle estimate (assumes observation duration << cadence)
    duty_cycle = n * median_cad / baseline if baseline > 0 else 0.0
    duty_cycle = min(1.0, duty_cycle)  # Can't exceed 1

    metrics = {
        "n_points": n,
        "baseline": baseline,
        "max_gap": max_gap,
        "max_gap_fraction": max_gap / baseline,
        "median_cadence": median_cad,
        "mean_cadence": mean_cad,
        "cadence_std": std_cad,
        "nyquist_period": 2.0 * median_cad,
        "nyquist_frequency": 1.0 / (2.0 * median_cad) if median_cad > 0 else np.inf,
        "longest_detectable_period": baseline / 2.0,
        "duty_cycle": duty_cycle,
        "sampling_uniformity": uniformity,
    }

    # Add SNR metrics if y and yerr provided
    if y is not None and yerr is not None:
        y_arr = np.asarray(y)
        yerr_arr = np.asarray(yerr)
        valid = np.isfinite(y_arr) & np.isfinite(yerr_arr) & (yerr_arr > 0)
        y_finite = y_arr[valid]
        yerr_finite = yerr_arr[valid]

        if len(y_finite) > 0:
            snr = np.abs(y_finite) / yerr_finite
            metrics["median_snr"] = float(np.median(snr))
            metrics["mean_snr"] = float(np.mean(snr))
            metrics["fraction_snr_gt_3"] = float(np.mean(snr > 3))
            metrics["fraction_snr_gt_5"] = float(np.mean(snr > 5))
            metrics["snr_values"] = snr

    return metrics


def assess_sampling_quality(
    t: np.ndarray,
    y: np.ndarray = None,
    yerr: np.ndarray = None,
    min_points: int = 6,
    max_gap_fraction: float = 0.3,
    min_baseline_factor: float = 3.0,
    min_snr: float = 3.0,
    min_fraction_good_snr: float = 0.5,
    verbose: bool = False,
) -> tuple:
    """
    Assess whether lightcurve sampling is adequate for GP fitting.

    Applies multiple quality gates to prevent fitting poorly sampled data.

    Parameters
    ----------
    t : np.ndarray
        Observation times
    y : np.ndarray, optional
        Flux values (used for SNR assessment if provided)
    yerr : np.ndarray, optional
        Uncertainties (used for SNR assessment if provided)
    min_points : int, default=6
        Minimum number of observations required
    max_gap_fraction : float, default=0.3
        Maximum allowed gap as fraction of baseline (e.g., 0.3 = 30%)
        Large gaps cause extrapolation errors
    min_baseline_factor : float, default=3.0
        Minimum baseline / median_cadence ratio
        Ensures sufficient temporal coverage (e.g., 3 = at least 3 typical cadences)
    min_snr : float, default=3.0
        Minimum median SNR required (if y, yerr provided)
    min_fraction_good_snr : float, default=0.5
        Minimum fraction of points with SNR > min_snr (if y, yerr provided)
    verbose : bool, default=False
        Print detailed assessment report

    Returns
    -------
    passes : bool
        True if all quality gates pass
    diagnostics : dict
        {
            'metrics': dict (from compute_sampling_metrics),
            'gates': {
                'min_points': bool,
                'max_gap': bool,
                'min_baseline': bool,
                'min_snr': bool (if applicable)
            },
            'warnings': list[str],
            'recommendation': str  # 'PROCEED' or 'DO NOT FIT'
        }

    Examples
    --------
    >>> passes, diag = assess_sampling_quality(t, y, yerr, verbose=True)
    >>> if passes:
    ...     print("Safe to fit GP")
    >>> else:
    ...     print(f"Issues: {diag['warnings']}")
    """
    metrics = compute_sampling_metrics(t, y, yerr)

    if "error" in metrics:
        return False, {
            "metrics": metrics,
            "gates": {},
            "warnings": [metrics["error"]],
            "recommendation": "DO NOT FIT",
        }

    # Quality gates
    gates = {}
    warnings = []

    # Gate 1: Minimum points
    gates["min_points"] = metrics["n_points"] >= min_points
    if not gates["min_points"]:
        warnings.append(
            f"Too few points: {metrics['n_points']} < {min_points}"
        )

    # Gate 2: Maximum gap
    gates["max_gap"] = metrics["max_gap_fraction"] <= max_gap_fraction
    if not gates["max_gap"]:
        warnings.append(
            f"Large gap: {metrics['max_gap']:.2f} "
            f"({100 * metrics['max_gap_fraction']:.1f}% of baseline) "
            f"> {100 * max_gap_fraction:.0f}% threshold"
        )

    # Gate 3: Minimum baseline coverage
    median_cadence = metrics["median_cadence"]
    if median_cadence <= 0:
        # Non-positive median cadence indicates duplicate or invalid timestamps.
        baseline_factor = 0.0
        gates["min_baseline"] = False
        warnings.append(
            "Non-positive median cadence; baseline coverage cannot be reliably "
            "assessed and the baseline gate has been failed."
        )
    else:
        baseline_factor = metrics["baseline"] / median_cadence
        gates["min_baseline"] = baseline_factor >= min_baseline_factor
        if not gates["min_baseline"]:
            warnings.append(
                f"Insufficient baseline: {baseline_factor:.1f} median cadences "
                f"< {min_baseline_factor} required"
            )

    # Gate 4: SNR check (if data provided)
    if "median_snr" in metrics:
        snr_values = metrics.get("snr_values")
        if snr_values is not None:
            snr_arr = np.asarray(snr_values)
            finite_mask = np.isfinite(snr_arr)
            fraction_snr_gt_min = (
                float(np.mean(snr_arr[finite_mask] >= min_snr))
                if np.any(finite_mask)
                else 0.0
            )
        else:
            fraction_snr_gt_min = float(metrics.get("fraction_snr_gt_3", 0.0))

        gates["min_snr"] = (
            metrics["median_snr"] >= min_snr
            and fraction_snr_gt_min >= min_fraction_good_snr
        )
        if not gates["min_snr"]:
            warnings.append(
                f"Poor SNR: median={metrics['median_snr']:.1f} < {min_snr}, "
                f"fraction>={min_snr:.1f}={100 * fraction_snr_gt_min:.0f}%"
                f" < {100 * min_fraction_good_snr:.0f}%"
            )
    else:
        gates["min_snr"] = True  # Pass by default if not checkable

    passes = all(gates.values())
    recommendation = "PROCEED" if passes else "DO NOT FIT"

    diagnostics = {
        "metrics": metrics,
        "gates": gates,
        "warnings": warnings,
        "recommendation": recommendation,
    }

    if verbose:
        print("=" * 70)
        print("LIGHTCURVE SAMPLING QUALITY ASSESSMENT")
        print("=" * 70)
        print("\nTemporal Coverage:")
        print(f"  • Points: {metrics['n_points']}")
        print(f"  • Baseline: {metrics['baseline']:.2f} time units")
        print(f"  • Median cadence: {metrics['median_cadence']:.3f}")
        print(
            f"  • Max gap: {metrics['max_gap']:.2f} "
            f"({100 * metrics['max_gap_fraction']:.1f}% of baseline)"
        )
        print(f"  • Sampling uniformity: {metrics['sampling_uniformity']:.3f}")

        print("\nDetectable Period Range:")
        print(f"  • Nyquist (shortest): {metrics['nyquist_period']:.2f}")
        print(f"  • Longest: {metrics['longest_detectable_period']:.2f}")

        if "median_snr" in metrics:
            print("\nSignal Quality:")
            print(f"  • Median SNR: {metrics['median_snr']:.1f}")
            print(
                f"  • Points with SNR > 3: "
                f"{100 * metrics['fraction_snr_gt_3']:.0f}%"
            )
            print(
                f"  • Points with SNR > 5: "
                f"{100 * metrics['fraction_snr_gt_5']:.0f}%"
            )

        print("\nQuality Gates:")
        for gate, status in gates.items():
            symbol = "\u2713" if status else "\u2717"
            print(f"  {symbol} {gate}: {'PASS' if status else 'FAIL'}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  \u26a0 {w}")

        print(f"\nRecommendation: {recommendation}")
        print("=" * 70 + "\n")

    return passes, diagnostics


def subsample_lightcurve(
    t,
    max_samples=3000,
    max_gap_fraction=0.3,
    random_seed=None,
):
    """
    Return indices selecting at most *max_samples* points from *t* while
    preserving the temporal baseline and the *max_gap_fraction* constraint.

    The algorithm always keeps the earliest and latest time points so that
    the full temporal baseline is preserved.  It then fills the remaining
    budget with randomly chosen interior points.  If the resulting selection
    contains any gap larger than ``max_gap_fraction * baseline``, the
    original data point closest to the midpoint of each offending gap is
    added iteratively.  To keep the total count at or below *max_samples*,
    each added point is paired with the removal of the interior point whose
    removal creates the smallest new gap that still satisfies the constraint.
    If no such drop candidate exists, that gap is left unrepaired and repair
    continues for other gaps.  A hard iteration cap prevents infinite loops.

    Parameters
    ----------
    t : array_like
        Observation times (1-D).
    max_samples : int, default 3000
        Maximum number of points to include in the subsample.  Must be an
        integer >= 2.
    max_gap_fraction : float, default 0.3
        Maximum allowed gap between consecutive selected times as a fraction
        of the total temporal baseline.
    random_seed : int or None, optional
        Seed for the random number generator (reproducibility).

    Returns
    -------
    indices : np.ndarray of int
        Indices into *t* that form the subsample, ordered by ascending time.
        If ``len(t) <= max_samples`` the full range ``np.arange(len(t))``
        is returned unchanged.  The returned array always has length
        <= *max_samples*.

    Notes
    -----
    When filling a gap would exceed the budget, the densest interior point
    (i.e. the one whose removal creates the smallest new gap while still
    satisfying the gap constraint) is swapped out.  If no such drop candidate
    exists, the gap is left unrepaired and repair continues for other gaps.
    A hard iteration cap (``2 * max_samples + 1``) prevents infinite loops in
    pathological cases.

    Raises
    ------
    ValueError
        If *max_samples* is not an integer >= 2.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = rng.uniform(0, 100, 5000)
    >>> idx = subsample_lightcurve(t, max_samples=500, random_seed=42)
    >>> len(idx) <= 500
    True
    """
    if not isinstance(max_samples, (int, np.integer)) or max_samples < 2:
        raise ValueError(
            f"max_samples must be an integer >= 2, got {max_samples!r}"
        )

    t = np.asarray(t, dtype=float)
    n = len(t)

    if n <= max_samples:
        return np.arange(n)

    rng = np.random.default_rng(random_seed)

    sort_order = np.argsort(t)
    t_sorted = t[sort_order]

    baseline = float(t_sorted[-1] - t_sorted[0])
    if baseline == 0:
        # Degenerate case: all times identical - return first max_samples.
        return sort_order[:max_samples].copy()

    max_gap_allowed = max_gap_fraction * baseline
    
    # Boolean mask indexed by sorted position; endpoints always selected.
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[0] = True
    selected_mask[n - 1] = True

    # Fill remaining budget with randomly chosen interior sorted positions.
    interior_positions = np.arange(1, n - 1)
    target = max(0, max_samples - 2)
    chosen_pos = rng.choice(interior_positions, size=target, replace=False)
    selected_mask[chosen_pos] = True

    # Iterative gap repair with strict budget enforcement.
    # Each iteration repairs at most one gap.  The cap (2 * max_samples + 1)
    # covers at most max_samples swap cycles, preventing infinite loops in
    # pathological cases where swaps oscillate between two states.
    for _ in range(2 * max_samples + 1):
        sel_positions = np.where(selected_mask)[0]  # sorted positions
        t_sel = t_sorted[sel_positions]
        gaps = np.diff(t_sel)
        bad = np.where(gaps > max_gap_allowed)[0]
        if len(bad) == 0:
            break
            
        # Attempt repair starting from the largest offending gap.
        bad_sorted = bad[np.argsort(gaps[bad])[::-1]]
        repaired = False

        for gap_idx in bad_sorted:
            gap_idx = int(gap_idx)
            t_left = t_sel[gap_idx]
            t_right = t_sel[gap_idx + 1]
            t_mid = 0.5 * (t_left + t_right)

            # O(log N) lookup: positions in t_sorted that fall in the gap.
            lo = int(np.searchsorted(t_sorted, t_left, side="right"))
            hi = int(np.searchsorted(t_sorted, t_right, side="left"))

            if lo >= hi:
                continue

            # Unselected sorted positions strictly inside this gap.
            gap_range = np.arange(lo, hi)
            unused_in_gap = gap_range[~selected_mask[lo:hi]]

            if len(unused_in_gap) == 0:
                continue  # all points in this gap already selected

            # Best fill candidate: closest to the gap midpoint.
            best_pos = int(
                unused_in_gap[np.argmin(np.abs(t_sorted[unused_in_gap] - t_mid))]
            )

            # Budget enforcement: if at capacity, drop the interior selected
            # point whose removal creates the smallest new gap that still
            # satisfies the constraint (deterministic, from densest region).
            if np.sum(selected_mask) >= max_samples:
                drop_pos = None
                min_new_gap = float("inf")
                for k in range(1, len(sel_positions) - 1):
                    new_gap = (
                        t_sorted[sel_positions[k + 1]]
                        - t_sorted[sel_positions[k - 1]]
                    )
                    if new_gap <= max_gap_allowed and new_gap < min_new_gap:
                        min_new_gap = new_gap
                        drop_pos = int(sel_positions[k])

                if drop_pos is None:
                    # Cannot free budget without creating a new violation.
                    continue
                selected_mask[drop_pos] = False

            selected_mask[best_pos] = True
            repaired = True
            break

        if not repaired:
            break

    sel_positions = np.where(selected_mask)[0]
    return sort_order[sel_positions]
