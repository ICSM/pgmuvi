# Stack B: wavelength diagnostics / advisory / validation, and docs/CI/hygiene

Scope: origin/pr41..pr49, pr56..pr71, pr89..pr95, pr100, pr112..pr137 (wavelength), plus pr66, pr67, pr72..pr88, pr96, pr97, pr101..pr111, pr163, pr169 (docs/CI/hygiene).
Baseline: origin/main. Tip I measured against: origin/pr137-consolidate-d3-docs-and-close-validation (the last wavelength PR; pr138+ is instrument-channel calibration, which is out of scope).

## Size at a glance (origin/main..pr137, whole stack, all themes)

| bucket | +lines |
|---|---|
| new wavelength/SM-ARD modules (`pgmuvi/wavelength_*.py`, `spectral_mixture_ard*.py`) | 21,843 |
| tests for wavelength/SM-ARD/period-independent etc. | 12,440 |
| tests that only assert on docs/repo text (`tests/test_docs_*`, hygiene/contract/closeout tests) | 5,070 |
| docs rst | 5,778 (-557) |
| notebooks | 5,379 (-3,831) |
| examples (scripts) | 3,072 |
| example data (`examples/data/10131+3049.csv`, 1.1 MB) | 10,816 |
| `lightcurve.py` (all themes) | +16,619 / -3,378 |

Module sizes at pr137: wavelength_diagnostics 7,841, wavelength_validation_real_lpv 1,811, wavelength_validation_recovery 2,082, wavelength_validation_robustness_calibration 1,666, wavelength_validation_robustness 1,409, wavelength_validation 1,314, wavelength_validation_synthetic 1,182, wavelength_estimation 1,127, wavelength_results 808, wavelength_conclusions 733, spectral_mixture_ard_diagnostics 724, wavelength_hypotheses 474, spectral_mixture_ard 363, wavelength_status 309.

Overall verdict: roughly 85-90% of this code is scaffolding: JSON-safe dict plumbing, `_StringEnum` status taxonomies, result dataclasses, "advisory only / selected_model=None" boilerplate, markdown/CSV report builders, batch orchestration, and tests that grep rst files for phrases. Three things are real and should be kept:
1. Two genuine bug fixes in the existing wavelength models: the SM ARD wavelength dimension was bounded by the time axis, and the power-law/dust means were evaluated on transformed wavelengths.
2. One genuine bug fix in the uncertainty transform: `yerr` went through the affine (shifting) y-transform.
3. A small, useful set of diagnostics: per-band fixed-frequency amplitude and phase, period-free per-band flux statistics, data-driven wavelength length-scale and mean initialisation, and parameter-at-bound reporting.

There are also a few empirical findings worth keeping as documented facts.

---

## A. Feature table

LOC figures are approximate net additions summed from per-PR `git diff --stat`. Code includes Lightcurve wrapper methods.

| # | Feature | Branches | Code / tests / docs | User-facing API | What it does scientifically | Value | Recommendation |
|---|---|---|---|---|---|---|---|
| 1 | Pre-fit per-band diagnostics with period-locked amplitude and phase | pr41, pr42, pr43; corrected in pr116 | ~1,100 / ~360 / ~100 | `diagnose_wavelength_dependence_prefit`, `classify_wavelength_diagnostics`, `Lightcurve.diagnose_wavelength_dependence(period=...)` | Per band: robust flux quantiles, MAD/robust scatter, half-amplitude q05-q95, sampling-quality and variability flags. At a fixed frequency it fits a weighted linear least-squares sinusoid `y = c0 + a cos θ + b sin θ` per band, giving amplitude, phase, lag and their delta-method errors. The classifier labels the source as achromatic (amp ratio <= 1.25), wavelength-modulated (ratio >= 1.5, and power-law-like if monotonic with abs(log-log slope) >= 0.5), or possible lag (circular lag span >= 0.10 P). It maps those labels to candidate models: `2DAchromatic`, `2DWavelengthDependent`, `2DPowerLawMean`. | Useful | Port rewritten small (~250 lines). Port the pr116-corrected version: one common reference epoch for all bands and a circular lag span. pr42/43 as first written used per-band reference times, so cross-band lags were meaningless. Use amplitude uncertainties (chi² test of constant amplitude) instead of bare ratio thresholds. |
| 2 | Candidate-model comparison runner with residual and predictive diagnostics, interpretation, text report and plots | pr44, pr45, pr46, pr47; doc/example pr48; script pr49; doc rewrite pr75 ("legacy") | ~1,700 / ~530 / ~600 | `compare_wavelength_candidate_models`, `compute_wavelength_residual_diagnostics`, `interpret_wavelength_model_comparison`, `format_wavelength_diagnostics_report`, `plot_wavelength_diagnostics`, `plot_wavelength_model_comparison`, `Lightcurve.compare_wavelength_models` | Deep-copies the Lightcurve, fits each candidate model, then computes training-point residuals, standardised residuals, 1σ/2σ coverage and Gaussian NLPD, overall and per band. Ranks candidates by lowest **training-point** mean NLPD. Flags candidates with large z-RMS, poor 2σ coverage, or a per-band z-RMS outlier. | Idea useful; ranking basis wrong | Port rewritten (~250 lines). Keep the loop and the per-band residual table. Rank by exact log marginal likelihood (pr115's retained-state MLL helper) and optionally by time-blocked held-out NLPD. In-sample GP NLPD rewards overfitting. |
| 3 | Period-independent ("Level 0") wavelength-structure diagnostics | pr56, pr93 (extra quantiles) | ~450 / ~170 / ~20 | `diagnose_period_independent_wavelength_structure`, `Lightcurve.diagnose_period_independent_wavelength_structure` | Per band: quantiles, MAD, IQR scatter, trimmed stats, half-amplitudes, fractional amplitude, and noise-corrected scatter/amplitude `sqrt(max(0, s² - (k·σ_med)²))`. Across bands: max/min ratios, log-log slopes vs λ, and tolerance-based monotonicity labels (flat / increasing / decreasing / non-monotonic) for median flux, amplitude and scatter. No periodogram is used. | Useful (cheap, robust, needs no period) | Port rewritten (~120 lines). Merge with #1 into one `wavelength_summary()` table. This duplicates `wavelength_estimation._build_band_diagnostics` (#7); keep only one. |
| 4 | Parameter plan, model/kernel configs, run, score, quality, comparison report, advisory workflow, export, batch, CLI, batch tables, markdown manifest, input filter, failure artifacts, source metadata, failure fallback | pr57-pr65, pr66 (rename "fit_candidate" to "model_kernel_config"), pr68, pr69, pr70, pr71, pr89, pr91, pr92, pr94, pr95 | ~5,000 / ~3,500 / ~1,300 | 10 `*_period_independent_wavelength_*` functions plus 10 Lightcurve wrappers; `examples/run_wavelength_advisory_batch.py` (CLI) | pr57: from the Level-0 median-flux slope it picks a model order (positive slope: Dust > PowerLaw > WD; negative: PowerLaw > WD > Dust; non-monotonic: WD first) and suggests coarse init values (log-log power-law fit, quadratic fit). The rest is orchestration: build config dicts, call `fit`, wrap the outcomes. The "quality" score is the ad hoc `100 - 25·NRMSE - 5·median abs(z) - 25·outlier_frac - 0.25·log(1+χ²_red)` on training residuals. There is also a separate "viability score" (+100 fit success, +25 consensus, +2 per accepted band...). Batch adds CSV/markdown/manifest writers. | Scaffolding / noise | **Drop.** Keep the model-ordering heuristic only as a paragraph in `wavelength_models.rst`. The training-residual score is scientifically unsound and, in the D3 run, numerically broken (see C). |
| 5a | Scale-only transform for uncertainties and variances | pr114 | ~100 code in lightcurve/transform classes / ~150 tests | `Transformer.transform_uncertainty`, `transform_variance`, `Lightcurve.transform_y_uncertainty` / `transform_y_variance` | main does `self.ytransform.transform(yerr)`, which applies the **shift** as well as the scale to σ (MinMax/ZScore/RobustZScore). The fix applies scale only, and squared scale to variances. It also stops residual diagnostics from double-adding measurement noise. | **Core bug fix** | Port as-is (small; independent of the rest). Possibly owned by another theme, but it must not be lost. |
| 5b | Retained-state exact marginal likelihood | pr115 | ~250 / ~180 | internal `_piwd_compute_retained_state_marginal_likelihood` | Recomputes `ExactMarginalLogLikelihood` in train mode from the final parameters, separating the data log ML from prior terms (MAP objective) and restoring module modes. | Useful | Port as a ~40-line `Lightcurve.log_marginal_likelihood()` helper and use it for #2 ranking. |
| 5c | Ranking availability / sentinel removal; contract repairs | pr112, pr113 | ~450 / ~600 | — | Fixes bugs in #4's own ranking plumbing. | Noise | Drop with #4. |
| 5d | Period-diagnostic semantics | pr116 | ~340 / ~250 | — | Common phase reference across bands, circular lag span, configured SNR threshold in the sampling gate, and rejection of incompatible two-band frequency pairs (no midpoint consensus period). | Useful | Fold the lag parts into #1. The consensus two-band rule belongs to the consensus theme. |
| 6 | Status taxonomy, result primitives, model hypotheses, synthesised conclusions | pr117, pr118, pr119, pr120 | ~2,700 / ~1,150 / ~300 | `wavelength_status.*` (7 enums), `wavelength_results.*` (5 dataclasses), `wavelength_hypotheses.*`, `synthesize_wavelength_advisory_conclusions` | Enums such as `TechnicalOutcome`, `ScientificUsability`, `ComparisonEligibility` and `WavelengthConclusionDisposition` (remains_plausible / weakened / technically_unevaluable / ...). Static table mapping each model to a mean structure (constant/quadratic/dust/power_law) and covariance structure (separable product / joint SM ARD / achromatic). | Scaffolding | **Drop.** At most a 6-row table in the docs describing each 2D model's mean and covariance. |
| 7 | Wavelength estimation context: data-driven wavelength length-scale and wavelength-mean init/bounds; physical-wavelength evaluation of PowerLaw/Dust means | pr121, pr122, pr123; fixes in pr130, pr135 (channel vs wavelength counting) | ~1,130 module + ~450 parameter_builders/context/application + ~400 gps.py + ~330 lightcurve / ~900 tests / ~250 | `build_wavelength_estimation_context`, `build_wavelength_mean_estimation_context`, `PowerLawMean/DustMean.configure_wavelength_coordinate`, `physical_wavelength`; applied automatically in `fit` via the parameter workflow | See B.3-B.5. Also fixes a real main bug: `PowerLawMean.forward` and `DustMean.forward` evaluate λ^p on the **transformed** wavelength column. That column can be ≤0 after centring or min-max, giving NaN, or is silently clamped to 1e-6 in Dust. The fix reconstructs physical λ from the affine transform. | **Core** (mean fix); useful (init/bounds) | Port the mean physical-wavelength fix as-is (~100 lines + tests). Port the init/bounds rewritten (~250 lines) on whatever parameter-application mechanism the new stack keeps. Do not keep the context dataclasses or provenance machinery. |
| 8 | Separate temporal and wavelength ARD bounds for the full `2D` spectral-mixture kernel | pr124 | ~360 module + ~1,000 lightcurve diff (consensus merging) + ~200 builders/application / ~450 tests | `build_dimension_aware_sm_ard_estimates`; automatic in `set_default_constraints` and consensus | main applies one scalar `Interval(1/T_span, Nyquist_time)` to **both** ARD columns of `mixture_means`. The comment at main `lightcurve.py:3876` admits this. The wavelength frequency therefore cannot go near 0 (achromatic) and is bounded by time sampling. Fix: tensor bounds of shape (1,1,2), with index 0 = time and index 1 = wavelength, the same for `mixture_scales`. Consensus updates only index 0. | **Core bug fix** | Port rewritten (~200-300 lines). GPyTorch `Interval` accepts broadcastable tensor bounds. Most of the 1,000-line lightcurve diff is consensus plumbing that can shrink. |
| 9 | SM ARD scale-ceiling and bound-proximity diagnostics | pr90 (ceiling only), pr125 (general, per component × dimension × side) | ~950 / ~370 / ~150 | `diagnose_spectral_mixture_ard`; fields in advisory reports | For each SM component and ARD dimension, reports the distance of fitted `mixture_means`/`mixture_scales` to registered lower/upper bounds (fractional tolerance 5%) in model and raw coordinates. | Useful | Port rewritten as a **generic** `Lightcurve.parameters_at_bounds(tol=0.05)` (~80 lines) covering every constrained parameter, not only SM. Keep the empirical caveat (C.2). |
| 10 | Validation framework: D1 synthetic recovery, D2 robustness and calibration, D3 real-LPV | pr126, pr127, pr128, pr130, pr131, pr132, pr133, pr135, pr136, pr137 | ~9,500 / ~4,800 / ~900 | `wavelength_validation*` (6 modules), `scripts/run_representative_lpv_validation.py`, JSON manifests | Synthetic generator: GP draws (Cholesky) from QP(time) × RBF(λ) or a joint SM-ARD covariance, plus a constant/quadratic/dust/power-law mean, at V I J K W1 W2 (0.55-4.6 µm). Fit with the generating model; metrics are period relative error, λ length-scale factor error, mean-law NRMSE at the bands, and ARD boundary hits. D2 perturbs sampling, noise and band coverage across 20 seeds against truth-matched references. D3 runs 5 candidates on one real LPV. | Mostly "validation of the validation"; findings useful | **Drop the code.** Keep the findings (C) in a single docs page. Optionally add a ~150-line `make_separable_gp_2d(...)` to the existing `pgmuvi/synthetic.py` and a slow, opt-in recovery test. |
| 11 | Wavelength-constraint notebook and real dataset | pr134 | notebook 1,029; CSV 10,816 rows (1.1 MB) | `docs/source/notebooks/tutorial_wavelength_constraints.ipynb`, `examples/data/10131+3049.csv` | Tutorial on λ-scale derivation, affine round trips, constraints, ARD diagnostics and the D2 boundaries. The dataset is 17 channels / 16 wavelengths (ZTF, KELT×2, ASAS-SN, Gaia, PS1, CTIO JHK, IRTF L). 9.3k of the 10.8k rows are KELT. | Useful (data); notebook depends on dropped API | Keep the dataset, ideally thinned (for example drop most KELT rows, giving about 1.5k rows). Rewrite a short notebook on the new minimal API. |
| 12 | Wavelength docs | pr48, pr73 (advisory 330 + batch 337), pr75 (legacy), pr95, pr100 (`wavelength_models.rst`), pr101 (batch walkthrough), pr102 (interpreting results), pr106 (advisory notebook) | ~3,000 docs + ~1,100 example scripts | — | — | `wavelength_models.rst` is genuinely good user guidance (mean vs covariance, physical-wavelength requirement, model-family table, "what to try first"). Everything else documents #4/#6. | Keep `wavelength_models.rst` (trimmed, rewritten to the new API). Drop the advisory, batch, legacy and walkthrough pages and their examples. |

---

## B. The scientific core

There are four genuine ideas. Everything else wraps them.

### B.1 Period-free per-band summary (pr56; duplicated in pr121's `_build_band_diagnostics`)
For each unique wavelength λ_b, with fluxes y and errors σ:
- `med = median(y)`, `s = 1.4826·MAD(y)`, `A = (q95 - q05)/2`, `A_frac = A/abs(med)`.
- Noise-corrected values: `s_nc = sqrt(max(0, s² - σ_med²))` and `A_nc = sqrt(max(0, A² - (1.645·σ_med)²))`.
- Across bands (sorted by λ): `ratio = max/min`, log-log slope from `polyfit(log λ, log x, 1)`, and a monotonicity label with a 10%-of-span tolerance. Apply this to med, A and s.

What it is for:
- Choosing the mean family: sign of the median-flux slope gives Dust (rising), PowerLaw (falling), or WD/quadratic (non-monotonic).
- Telling the user whether the amplitude is chromatic.

This needs no period and is about 60 lines.

### B.2 Period-locked amplitude/phase per band (pr42, corrected pr116)
- Fix f from LS or consensus. Choose one global reference epoch t0, the midpoint of all times. (The original pr42 used a per-band midpoint, which was a bug.)
- Per band, solve weighted least squares `[1, cos θ, sin θ]` with `θ = 2πf(t - t0)`.
- Derived quantities: `A = hypot(a,b)`, `φ = atan2(b,a)`, `lag = φ/(2πf)`. Get σ_A and σ_φ by the delta method from `(XᵀWX)⁻¹·χ²_red`.
- Across bands: amplitude ratio and log-log slope; lag span as the **minimum circular arc** `P - max(circular gaps)` after `mod(lag, P)` (pr116).
- Classification:
  - achromatic if ratio ≤ 1.25
  - chromatic if ratio ≥ 1.5, and power-law-like if also monotonic with abs(slope) ≥ 0.5
  - lag if span/P ≥ 0.10, negligible if ≤ 0.05

Improvement for the rewrite: use σ_A, e.g. χ² of amplitudes about their weighted mean, rather than fixed ratios. Also note that a detected lag cannot be modelled by any current pgmuvi 2D kernel; the stack itself lists this as `TBD[wavelength-dependent-lags]`.

This is about 120 lines. Scientifically this is the most interpretable output for LPVs: amplitude vs λ and phase lag vs λ.

### B.3 Data-driven wavelength length-scale (pr121/122, `wavelength_estimation._lengthscale_recommendation`)
With sorted unique band wavelengths, Δ the adjacent spacings, and S the span:
- `init = sqrt(median(Δ)·S)`
- `lower = 0.25·min(Δ)`
- `upper = max(5S, 4·max(Δ), 2·init)`

These are computed in physical λ, then mapped through the x-transform to model coordinates (scale only for a length). They are intersected with any existing constraint and applied before the value (constraint-before-value ordering).

Applies to: the separable wavelength kernels of 2DWavelengthDependent, 2DDustMean, 2DPowerLawMean and 2DSeparable.

### B.4 Wavelength-mean initialisation (pr123)
All fits use per-band robust medians in **linear** flux.
- **Power law** `offset + w·λ^p`: grid-profile over p ∈ [-10,-0.05] ∪ [0.05,10] (480 values). At each p, solve linear LSQ for (offset, w). Take the minimum-MSE point. Bounds come from the set of p values with RMSE ≤ 2·best_RMSE on the same sign branch, padded. The sign of w and of p is preserved.
- **Dust** `offset + A·exp(-τ λ^-α)`: grid over α ∈ geomspace(0.1,10,48) and τ ∈ geomspace(1e-3,1e3,72), with linear LSQ for (offset, A).
- **Quadratic** (2DWavelengthDependent): `polyfit` in the model coordinate.

Bug fix bundled here: PowerLawMean and DustMean must evaluate on **physical** λ. The fix gives the mean modules `configure_wavelength_coordinate(origin, scale)` and computes `λ_phys = x·scale + origin`. Non-affine transforms are rejected.

### B.5 Independent ARD bounds for joint SM (pr124)
- `mixture_means` bounds become tensors of shape (1,1,2):
  - time column: `[1/T_span, 1/(2 Δt_min)]`
  - wavelength column: `[0.05/S_λ, 1/(2 Δλ_min)]`
- Wavelength init is 1/S_λ if any Level-0 trend is non-monotonic, else 0.1/S_λ.
- `mixture_scales`: wavelength scale bounds are `1/(2π ℓ)` for the ℓ bounds in B.3. The time scale initialises to `1/(2π T_span)` with a broad range.
- Consensus code updates only column 0.

Can the wavelength mean frequency still be pushed to ~0 (achromatic)? The lower bound is 0.05/S_λ, not 0, which is a design choice worth revisiting.

### B.6 Bound-proximity diagnostic (pr90/pr125)
For each constrained parameter element, with bounds [lo, hi] and width w:
- `near_upper = (hi - v) ≤ tol·w`, with tol = 0.05; the same rule applies for the lower bound.
- Report this per SM component and ARD dimension.

This is trivial to generalise to every parameter. Do **not** interpret a wavelength length-scale at its ceiling as "achromatic" (C.2).

### Modules that do the work vs. wrapper

| Category | Modules / functions |
|---|---|
| Does the work (≈1,500 of 21,800 lines) | `_fit_fixed_frequency_sinusoid`, `_amplitude_phase_summary`, `classify_wavelength_diagnostics`, `_period_independent_band_summary`, `_minimum_circular_span`, `compute_wavelength_residual_diagnostics` (core loop), `_piwd_compute_retained_state_marginal_likelihood` (all in `wavelength_diagnostics.py`); `wavelength_estimation.py` (`_lengthscale_recommendation`, `_power_law_mean_recommendation`, `_dust_mean_recommendation`, `_quadratic_mean_recommendation`); `spectral_mixture_ard.py` (`_coordinate_estimates`); `spectral_mixture_ard_diagnostics._boundary_status`; `wavelength_validation_synthetic` (`_separable_covariance`, `_joint_spectral_mixture_covariance`, `_sample_latent_process`); gps.py mean-coordinate fix. |
| Status enums / result dataclasses | `wavelength_status.py`, `wavelength_results.py`, `wavelength_hypotheses.py`, `wavelength_conclusions.py`, `wavelength_validation.py` |
| Report formatting / plotting | `format_*report`, `plot_*` (about 1,200 lines in wavelength_diagnostics) |
| Batch orchestration, manifests, export | `run_*_batch`, `export_*`, `_piwd_batch_*`, markdown manifest, `examples/run_wavelength_advisory_batch.py`, `wavelength_validation_real_lpv.py`, `scripts/run_representative_lpv_validation.py` |
| Validation of the validation | `wavelength_validation_recovery.py`, `_robustness.py`, `_robustness_calibration.py` (paired references, D2 policy classes, resumable atomic outputs) |

Note that the stack contains **two parallel comparison APIs**: the "legacy" period-locked `compare_wavelength_models` (pr44-47) and the "period-independent advisory workflow" (pr57-65). pr75 had to write a "legacy candidates" doc page to explain this. The replacement should have one.

---

## C. Validation results worth preserving as facts

Sources:
- `docs/source/pgmuvi.wavelength_validation_robustness_calibration.rst` (pr133)
- `docs/source/notebooks/tutorial_wavelength_constraints.ipynb` (markdown cells around line 980, pr134)
- `docs/source/howto/representative_lpv_validation.rst` (pr137, "D3 execution result")
- `docs/source/future_work.rst`
- `examples/validation/d3_representative_lpv_calibration_summary.json` (pr136)
- D1 gates (thresholds only; no executed D1 numbers are recorded): `docs/source/pgmuvi.wavelength_validation_recovery.rst`

### C.1 D1 synthetic recovery
Only thresholds are recorded, no results:
- Per run: period relative error ≤ 0.15, λ length-scale factor error ≤ 4, mean NRMSE ≤ 0.15 (0.25 for weak dependence).
- Population: ≥ 95% completion, length-scale factor median ≤ 2, pass fraction ≥ 0.80.
- Design lesson recorded: nominal cases need about 72 points per band on a **shared** time grid over 3.2 cycles. Otherwise finite-sampling phase differences mimic wavelength structure.
- Length-scale recovery is biased when the temporal kernel is misspecified (one QP component vs fundamental plus harmonic).

### C.2 D2 robustness, 20 seeds (seeds 0-19), 420 runs
- 393 completed and 27 failed. 20 of the failures were the declared-invalid 2-band × 2-point case, as expected. The other 7 were consensus-stage rejections (`insufficient_consensus_inliers` / `no_accepted_bands`), 6 of them at seed 7.
- Robust (10 perturbations): independent time grids, sparse shared grid (36/band), missing blue, red or interior band, heteroscedastic noise, 3× noise, weak and strong wavelength dependence, joint-SM sparse.
- **Uneven band counts** (72/54/42/30/24/18 per band, independent grids, dust mean): failure boundary for λ length-scale. Only 8 of 19 completed runs passed.
- **Longer sparse baseline** (36/band over 6.4 cycles, independent grids): failure boundary for **period**. Only 4 of 20 passed. A longer baseline does not compensate for low information content.
- **Large wavelength gap** (V, I, J, W2 only): degraded. Length-scale passed 9 of 20, with long-tailed factor errors.
- Joint `2D` SM: temporal **and** wavelength ARD were near their bounds in **100%** of runs while recovery stayed robust. Bound pressure is therefore *not* recovery failure, and *not* evidence for achromaticity. This is important for interpreting #9.
- The seed-7 failure exposed a consensus weakness: LS recovered the injected ~500 d period while the ACF picked a spurious ~23 d timescale. This is relevant to the consensus theme.

### C.3 D3 real LPV (IRAS/source id 10131+3049, `examples/data/10131+3049.csv`)
- 789 of 10,815 rows were used (cap 100 per channel), covering 17 channels and 16 wavelengths.
- All 5 candidates completed. All found a consensus **P = 597.37 d**, with 9 channels accepted and 8 rejected. Every fit emitted a GPyTorch small-noise NumericalWarning.
- The "ranking" was Dust > WD > Separable > PowerLaw > 2D.
- **Caveat that invalidates the ranking:** in the committed JSON, `training_nrmse_by_target_scale` is 62-1,670 while reduced χ² is 0.32-5.1 and median abs(z) is 0.06-0.24. A normalised RMSE of 60+ alongside χ²_red < 1 means the NRMSE normalisation is broken (probably a transformed vs raw scale mismatch). Scores of -1,455 to -41,655 are driven entirely by that term. Also, χ²_red ≈ 0.3-0.6 and abs(z) ≈ 0.06 on training points show the GPs interpolate the data, so in-sample residuals carry no model-selection information.
- Worth preserving as facts: P ≈ 597 d for this source, and "2D joint SM fits noticeably worse / hits ARD bounds (4 boundary hits)". Do **not** preserve the ranking.
- Largest residual bias for all models is at 3.49 µm, the IRTF L band with only 14 points.

---

## D. Docs / CI / hygiene

### Port directly (good practice, independent of dropped machinery)

| Item | Branch | Note |
|---|---|---|
| `make html-strict` (`-W --keep-going`) target | pr82 (Makefile only) | — |
| `.github/workflows/docs.yml` strict-docs job | pr83, pr86, pr111 | pr111 adds the pandoc install, needed for nbsphinx |
| `docs/source/requirements.txt` as single source (`-e .`, sphinx>=5.3, nbsphinx, ipykernel) | pr86 | — |
| Ruff workflow: pin version (0.15.21) and **remove the auto-commit-fixes bot**; sync pre-commit ruff rev; ignore E501/RUF001-3/W293 | pr111 | Good |
| Test matrix: drop prerelease 3.15, set `fail-fast: false`; tox envlist | pr111 | — |
| `.gitignore` additions: `.ruff_cache/`, `docs/build/`, `*.orig`, `*.rej`, `debug*.txt` | pr84 | Drop the `pgmuvi_current*.zip` line |
| Remove stale and silently-skipped tests | pr67 | `tests/test_acf.py`, `tests/tests.py` placeholder transforms; `tests/test_period_summary.py` rewritten to be deterministic. These files exist on main. pr67 is mostly applicable to main, but check that `test_period_summary` exercises main's API. |
| Remove `pgmuvi/test_script.py` from the package and the ruff exclude; strip machine-local paths (`c:\Users\peter\...`) from notebook outputs | commit acbf4b0 + pr163 | — |
| `.github/pull_request_template.md` | pr88 | Delete the "source snapshot" section |
| CONTRIBUTING docs-build section | pr87 | Delete the snapshot paragraph |
| `docs/source/api.rst` completion / automodule stubs | pr77 | Only for modules that survive |
| Notebook exclusion via `exclude_patterns` until refreshed | pr79 | Good pattern if notebooks are stale |
| `howto/wavelength_models.rst` | pr100 | Rewrite to the new API |
| `overview.rst`, `howto/first_workflow.rst`, `examples/first_pgmuvi_workflow.py` | pr97 | Orientation docs |
| `howto/gp_prior_sampling.rst` + refreshed `pgmuvi_mock_data_from_gp.ipynb` | pr107 | Check against the main API |

### Port selectively
- pr80/pr81, "Sphinx warning cleanup": pr80 **deletes** substantive docstring content, e.g. the full `ConsensusFitError` reason catalogue, to silence warnings. Redo it by fixing the RST instead. pr81 re-executes `PGMUVI_Lightcurve.ipynb` (2.5k/2.5k churn); just clear or re-run on the new stack.
- pr103-105 and pr108 notebook refreshes (2D consensus, preprocessing, synthetic, removal of the MCMC notebook): content is useful but tied to the consensus/parameter-workflow API. Redo after the API settles. Removing the MCMC notebook is right (`Lightcurve.mcmc` raises NotImplementedError).
- pr72 docs correctness hotfix: README and concepts fixes; check which apply to main.

### Drop (tied to dropped machinery, or tests-about-docs noise)
- pr85 `scripts/create_source_snapshot.py`: AI-review zip workflow.
- pr96 documentation roadmap, pr109 `future_work.rst` TBD-marker registry, and `scripts/audit_docs_tbd_markers.py` + tests: process scaffolding. A short "Known limitations" section suffices. Keep its content: no held-out validation, no wavelength-dependent lags, no instrument-channel calibration, no 2D PSD plot, no native magnitudes, MCMC not implemented.
- pr76 `notebook_status.rst` and triage tests.
- `docs_maintenance.rst`: fold 10 lines into CONTRIBUTING.
- All ~40 `tests/test_docs_*.py` and `test_repository_reference_*`, `test_pull_request_template.py`, `test_contributing_workflow_docs.py`, `test_docs_ci_workflow.py`, `test_docs_strict_build_contract.py`, `test_d3_*closeout*`, `test_pr136_*` (~5,000 lines). They assert that rst files contain phrases; the strict Sphinx build is the real check. pr169's reference-integrity test (doc roles, markdown links, machine-local paths) is the one that is mildly defensible. If wanted, keep it as a ~60-line check.
- pr73, pr75, pr95, pr101, pr102 advisory/batch/legacy/interpretation pages and their example scripts (`run_wavelength_advisory_batch.py`, `wavelength_advisory_batch_walkthrough.py`, `interpret_pgmuvi_outputs.py`, `validate_wavelength_model_selection_workflow.py`); pr106 advisory notebook.
- pr66 rename-only churn; pr78 (removes `.orig` files the stack itself added in pr36).

---

## E. Proposed replacement stack (this theme)

Dependencies: B2 needs the parameter-application mechanism from the parameter-workflow theme, or a simple `initialize_wavelength_hyperparameters()` that sets values and constraints directly.

| PR | Content | Approx. size |
|---|---|---|
| B0 – CI/docs hygiene | Make `html-strict`; docs.yml with pandoc; centralised `docs/source/requirements.txt`; ruff pin and removal of the auto-commit bot; pre-commit rev; test matrix 3.10-3.14 with fail-fast off; `.gitignore`; PR template; CONTRIBUTING docs section; remove `pgmuvi/test_script.py` and its exclude; strip local paths from notebook outputs. | ~150 config/docs, 0 code |
| B0b – stale tests | pr67 equivalent: drop dead/skip-only tests, make period-summary tests deterministic. | ~-280 / +25 |
| B1 – correctness fixes in existing 2D models | (a) Scale-only `transform_uncertainty`/`transform_variance`, used for `_yerr_transformed` (pr114). (b) PowerLawMean/DustMean evaluate on physical λ via the affine x-transform; reject non-affine transforms (pr123 gps part). (c) Independent (1,1,2) tensor bounds for the time and wavelength ARD columns of `mixture_means`/`mixture_scales` in `set_default_constraints`, with consensus touching only column 0 (pr124, minimal). Regression tests for each. | ~350 code / ~300 tests |
| B2 – data-driven wavelength initialisation | `wavelength_lengthscale_guess(λ)` (B.3); `power_law_mean_guess`, `dust_mean_guess`, `quadratic_mean_guess` (B.4) on per-band medians; applied during `fit` for 2DWavelengthDependent, 2DDustMean, 2DPowerLawMean, 2DSeparable and `2D`. Constraints intersected with existing ones. | ~350 code / ~250 tests |
| B3 – per-band wavelength summary | One public function `wavelength_summary(lc, period=None)` plus a `Lightcurve` wrapper and one plot. Gives the period-free stats (B.1). With a period it adds fixed-frequency amplitude/phase with a common t0, delta-method errors and circular lag span (B.2), and an uncertainty-aware classification (achromatic / chromatic / power-law-like / lag) with suggested model families. Returns a pandas-free dict or small dataclass. | ~400 code / ~250 tests |
| B4 – compare wavelength models | `compare_wavelength_models(lc, models=[...], **fit_kwargs)`: fit each on a deep copy; report exact log ML (data term, prior separated), number of hyperparameters, per-band standardised-residual mean and RMS, 2σ coverage, `parameters_at_bounds`. Rank by log ML, with an explicit caveat. Optional `holdout="time_blocks"` NLPD. No status enums, no batch. | ~350 code / ~200 tests |
| B5 – generic bound diagnostics | `Lightcurve.parameters_at_bounds(tol=0.05)` across all constrained parameters, with per-ARD-dimension labelling for SM; used by B4. Could merge into B4. | ~100 code / ~80 tests |
| B6 – docs | Rewritten `howto/wavelength_models.rst` (model families, physical-λ requirement, what to try first, the B3 → B4 workflow). A "validation notes" section recording C.1-C.3 facts (D2 boundaries; ARD bound pressure ≠ failure; P ≈ 597 d for 10131+3049, with ranking omitted). "Known limitations" list. Thinned `examples/data/10131+3049.csv` (≈1-2k rows), with a short notebook or example on it. `overview.rst` / `first_workflow.rst`. | docs only |
| B7 (optional) – synthetic 2D GP generator | `pgmuvi.synthetic.make_separable_gp_2d(...)`: QP(t) × RBF(λ) + mean law, drawn by Cholesky, for users and for one slow, opt-in recovery test. | ~150 code / ~80 tests |

Total for the theme: about 1,700 code lines, about 1,100 test lines, plus docs. That replaces roughly 30,000 lines of code and tests.

### Explicit drop list
- **Advisory workflow:** pr57-pr65, pr66 (rename), pr68-pr71, pr89, pr91, pr92, pr93 (fold 2 quantiles into B3 if wanted), pr94, pr95.
- **Ranking and contract repairs:** pr112, pr113, the pr115 plumbing (keep the MLL helper idea), and the pr116 consensus-pair part (belongs to the consensus theme).
- **Status/results/hypotheses/conclusions:** pr117-pr120 entirely: `wavelength_status.py`, `wavelength_results.py`, `wavelength_hypotheses.py`, `wavelength_conclusions.py`.
- **Estimation provenance:** pr121's context dataclasses and provenance records (keep the arithmetic); pr125 module size (replace with B5).
- **Validation framework:** pr126-pr128, pr130-pr133, pr135-pr137: all six `wavelength_validation*.py` modules, `scripts/run_representative_lpv_validation.py`, `examples/validation/*.json` (except as a source of facts for B6), and the observational-channel terminology tests.
- **Legacy period-locked API:** pr44-pr47's report/plot/interpretation layers, and the legacy period-locked comparison API as a separate path (merged into B3/B4).
- **Docs and process scaffolding:** pr48/49/73/75/76/96/101/102/106/109 docs and examples; pr85 snapshot helper; all `tests/test_docs_*` and repository-text contract tests; pr78 (moot); the pr80 docstring deletions.
