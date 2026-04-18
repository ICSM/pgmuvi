"""
Variability detection utilities for lightcurves.

Implements a three-tier statistical testing framework to determine if a
lightcurve shows significant variability before attempting GP fitting.

References
----------
Stetson, P. B. 1996, PASP, 108, 851 (Stetson K index)
"""

import numpy as np
from scipy.special import gammaincc


def _to_numpy(arr) -> np.ndarray:
    """Convert array-like or torch.Tensor to a 1-D float64 NumPy array.

    Handles CPU and CUDA torch tensors safely via `.detach().cpu()`.

    Parameters
    ----------
    arr : array-like or torch.Tensor
        Input data.

    Returns
    -------
    np.ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If the result is not 1-D.
    """
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass
    result = np.asarray(arr, dtype=float)
    if result.ndim != 1:
        raise ValueError(
            f"Expected a 1-D array, got shape {result.shape}. "
            "Pass a 1-D array of flux values."
        )
    return result


def _validate_inputs(y: np.ndarray, yerr: np.ndarray) -> None:
    """Validate that y and yerr are matching 1-D finite arrays with positive yerr.

    Parameters
    ----------
    y : np.ndarray
        Already-converted 1-D float64 array.
    yerr : np.ndarray
        Already-converted 1-D float64 array.

    Raises
    ------
    ValueError
        If shapes mismatch, fewer than 2 points, non-finite values, or
        non-positive yerr.
    """
    if y.shape != yerr.shape:
        raise ValueError(
            f"y and yerr must have the same shape; got {y.shape} and {yerr.shape}."
        )
    if len(y) < 2:
        raise ValueError(
            f"At least 2 data points are required; got {len(y)}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(yerr)):
        raise ValueError("yerr contains non-finite values (NaN or Inf).")
    if not np.all(yerr > 0):
        raise ValueError("All yerr values must be strictly positive.")


def weighted_chi2_test(
    y, yerr
) -> tuple[float, int, float, float]:
    """
    Weighted chi-square test against a constant mean model.

    Null hypothesis H0: y_i = mu + eps_i, where eps_i ~ N(0, sig_i^2)

    Parameters
    ----------
    y : array-like or torch.Tensor
        Flux values (1-D, finite).
    yerr : array-like or torch.Tensor
        1-sigma uncertainties (1-D, finite, positive).

    Returns
    -------
    chi2 : float
        Test statistic: sum(w_i * (y_i - ybar_w)**2)
    dof : int
        Degrees of freedom (N - 1)
    ybar_w : float
        Weighted mean: sum(y_i/sig_i**2) / sum(1/sig_i**2)
    p_value : float
        Right-tail p-value: P(chi2 >= chi2_stat | dof)

    Notes
    -----
    Uses scipy.special.gammaincc for chi-square tail probability.
    This is the uniformly most powerful test for non-constant mean
    under Gaussian heteroscedastic errors with known variances.

    Raises
    ------
    ValueError
        If inputs are not 1-D, have mismatched shapes, contain non-finite
        values, contain non-positive yerr, or have fewer than 2 points.
    """
    y = _to_numpy(y)
    yerr = _to_numpy(yerr)
    _validate_inputs(y, yerr)

    weights = 1.0 / yerr**2
    ybar_w = np.sum(weights * y) / np.sum(weights)

    chi2 = np.sum(weights * (y - ybar_w) ** 2)
    dof = len(y) - 1

    # Right-tail p-value: P(chi2 >= chi2_stat | dof)
    # gammaincc(a, x) = P(chi2/2 >= x) with a = dof/2
    p_value = gammaincc(dof / 2.0, chi2 / 2.0)

    return float(chi2), int(dof), float(ybar_w), float(p_value)


def compute_fvar(y, yerr) -> float:
    """
    Compute normalized excess variance (F_var).

    F_var = np.sqrt(max(s**2 - mean_err**2, 0)) / |ybar|

    where:
      s**2 = sample variance of y
      mean_err**2 = mean(sig_i**2)

    Parameters
    ----------
    y : array-like or torch.Tensor
        Flux values (1-D).
    yerr : array-like or torch.Tensor
        1-sigma uncertainties (1-D, positive).

    Returns
    -------
    fvar : float
        Scale-free amplitude measure.
        0 = no excess variance beyond errors
        >0 = intrinsic variability detected

    Notes
    -----
    This effect size prevents keeping bands that are statistically
    significant but astrophysically tiny (common with large N).

    Raises
    ------
    ValueError
        If inputs are not 1-D, have mismatched shapes, have fewer than 2
        points, contain non-finite values, or have non-positive yerr.
    """
    y = _to_numpy(y)
    yerr = _to_numpy(yerr)
    _validate_inputs(y, yerr)

    s2 = np.var(y, ddof=1)
    mean_err2 = np.mean(yerr**2)
    ybar = np.mean(y)

    excess = s2 - mean_err2
    if excess <= 0 or ybar == 0:
        return 0.0

    return float(np.sqrt(excess) / np.abs(ybar))


def compute_stetson_k(y, yerr) -> float:
    """
    Compute Stetson K index for shape/coherence diagnostics.

    K = (1/N) * sum(|delta_i|) / np.sqrt((1/N) * sum(delta_i**2))

    where delta_i = sqrt(N/(N-1)) * (y_i - ybar) / sig_i and N is the number
    of valid points used in the statistic.

    Parameters
    ----------
    y : array-like or torch.Tensor
        Flux values (1-D).
    yerr : array-like or torch.Tensor
        1-sigma uncertainties (1-D, positive).

    Returns
    -------
    K : float
        Stetson K index.
        ~0.798 for pure Gaussian noise
        Larger values indicate more coherent/peaked residual structure.
        Returns ``np.nan`` when the statistic is not well-defined.

    Notes
    -----
    This implementation keeps the classic Stetson K form while handling
    pathological inputs robustly.

    - If finite, positive uncertainties are available, ``ybar`` is a weighted
      mean using 1/sigma^2.
    - If weighted averaging is not possible, ``ybar`` falls back to the
      unweighted mean of finite flux points.
    - ``delta_i`` uses the finite-sample factor sqrt(N/(N-1)) for N > 1.
    - For invalid/pathological cases (N < 2, invalid denominators, etc.),
      ``np.nan`` is returned instead of raising.

    References
    ----------
    Stetson, P. B. 1996, PASP, 108, 851

    The public API and return type are preserved for backward compatibility.
    """
    y = _to_numpy(y)
    yerr = _to_numpy(yerr)
    if y.shape != yerr.shape:
        return float("nan")

    finite_y = np.isfinite(y)
    finite_pos_err = np.isfinite(yerr) & (yerr > 0)
    valid_for_weighting = finite_y & finite_pos_err

    # Center the residuals with a weighted mean when possible.
    ybar = float("nan")
    if np.any(valid_for_weighting):
        weights = 1.0 / yerr[valid_for_weighting] ** 2
        wsum = np.sum(weights)
        if np.isfinite(wsum) and wsum > 0:
            ybar = np.sum(weights * y[valid_for_weighting]) / wsum

    # Fallback: use an unweighted finite mean if weighted centering failed.
    if not np.isfinite(ybar):
        finite_vals = y[finite_y]
        if finite_vals.size == 0:
            return float("nan")
        ybar = np.mean(finite_vals)
        if not np.isfinite(ybar):
            return float("nan")

    # K is defined from normalized residuals that require finite positive
    # uncertainties, so only those points contribute.
    y_valid = y[valid_for_weighting]
    yerr_valid = yerr[valid_for_weighting]
    n = y_valid.size
    if n < 2:
        return float("nan")

    delta = np.sqrt(n / (n - 1)) * (y_valid - ybar) / yerr_valid
    if not np.all(np.isfinite(delta)):
        return float("nan")

    mean_abs = np.mean(np.abs(delta))
    mean_sq = np.mean(delta**2)
    if not np.isfinite(mean_sq) or mean_sq <= 0:
        return float("nan")

    k_value = mean_abs / np.sqrt(mean_sq)
    if not np.isfinite(k_value):
        return float("nan")

    return float(k_value)


def is_variable(
    y,
    yerr,
    alpha: float = 0.01,
    fvar_min: float = 0.05,
    stetson_k_min: float = 0.95,
    min_points: int = 6,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """
    Comprehensive variability assessment using required and diagnostic tests.

    Parameters
    ----------
    y : array-like or torch.Tensor
        Flux measurements (1-D).
    yerr : array-like or torch.Tensor
        1-sigma uncertainties (1-D, positive).
    alpha : float, default=0.01
        Significance level for chi-square test
    fvar_min : float, default=0.05
        Minimum fractional excess variance (5%)
    stetson_k_min : float, default=0.95
        Diagnostic Stetson-K reference threshold used only to set
        ``tests_passed['stetson_test']`` and reporting notes.
        It is not required for the overall VARIABLE/NOT VARIABLE decision.
    min_points : int, default=6
        Minimum number of data points required
    verbose : bool, default=False
        Print diagnostic information

    Returns
    -------
    is_var : bool
        True if lightcurve passes the required variability gates
        (min_points, chi2, and F_var).
    diagnostics : dict
        {
            'n_points': int,
            'chi2': float,
            'dof': int,
            'p_value': float,
            'fvar': float,
            'stetson_k': float,
            'decision': str,  # 'VARIABLE' or reason for rejection
            'tests_passed': {
                'chi2_test': bool,
                'fvar_test': bool,
                'stetson_test': bool,
                'min_points': bool
            }
        }

    Notes
    -----
    Decision logic (required gates):
    1. N >= min_points
    2. p_value < alpha (statistically significant)
    3. F_var >= fvar_min (astrophysically significant)

    Stetson K is retained as a shape/coherence diagnostic and reported in
    diagnostics, but does not veto a lightcurve that already passes the
    required gates.

    Both numpy arrays and torch tensors (including CUDA tensors) are accepted.

    Examples
    --------
    >>> from pgmuvi.preprocess.variability import is_variable
    >>> is_var, diag = is_variable(y, yerr, verbose=True)
    >>> if is_var:
    ...     print("Proceed with GP fitting")
    ... else:
    ...     print(f"Skipping: {diag['decision']}")
    """
    # Convert once here; individual helpers also call _to_numpy but that is
    # idempotent for plain ndarray inputs.
    y = _to_numpy(y)
    yerr = _to_numpy(yerr)

    n = len(y)
    enough_points = n >= min_points

    if not enough_points:
        diagnostics = {
            "n_points": n,
            "chi2": float("nan"),
            "dof": n - 1 if n > 1 else 0,
            "p_value": float("nan"),
            "fvar": float("nan"),
            "stetson_k": float("nan"),
            "decision": f"NOT VARIABLE: insufficient data (N={n} < {min_points})",
            "tests_passed": {
                "chi2_test": False,
                "fvar_test": False,
                "stetson_test": False,
                "min_points": False,
            },
        }
        if verbose:
            print(diagnostics["decision"])
        return False, diagnostics

    chi2, dof, _ybar_w, p_value = weighted_chi2_test(y, yerr)
    fvar = compute_fvar(y, yerr)
    stetson_k = compute_stetson_k(y, yerr)

    chi2_passed = p_value < alpha
    fvar_passed = fvar >= fvar_min
    stetson_passed = stetson_k >= stetson_k_min

    tests_passed = {
        "chi2_test": chi2_passed,
        "fvar_test": fvar_passed,
        "stetson_test": stetson_passed,
        "min_points": True,
    }

    is_var = chi2_passed and fvar_passed

    diagnostic_notes = []
    if not np.isfinite(stetson_k):
        diagnostic_notes.append("stetson_k=nan")
    elif not stetson_passed:
        diagnostic_notes.append(
            f"stetson_k={stetson_k:.3f}<ref={stetson_k_min:.3f}"
        )

    if is_var:
        decision = "VARIABLE"
    else:
        reasons = []
        if not chi2_passed:
            reasons.append(f"p_value={p_value:.4f} >= alpha={alpha}")
        if not fvar_passed:
            reasons.append(f"F_var={fvar:.4f} < fvar_min={fvar_min}")
        decision = "NOT VARIABLE: " + "; ".join(reasons)
    if diagnostic_notes:
        decision += "; DIAGNOSTIC: " + ", ".join(diagnostic_notes)

    diagnostics = {
        "n_points": n,
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "fvar": fvar,
        "stetson_k": stetson_k,
        "decision": decision,
        "tests_passed": tests_passed,
    }

    if verbose:
        print(f"Variability assessment (N={n}):")
        pass_str = "PASS" if chi2_passed else "FAIL"
        print(
            f"  chi2={chi2:.2f}, dof={dof}, p-value={p_value:.4e} [{pass_str}]"
        )
        print(f"  F_var={fvar:.4f} [{'PASS' if fvar_passed else 'FAIL'}]")
        print(
            f"  Stetson K={stetson_k:.3f} "
            f"[{'PASS' if stetson_passed else 'FAIL'} | DIAGNOSTIC]"
        )
        print(f"  Decision: {decision}")

    return is_var, diagnostics
