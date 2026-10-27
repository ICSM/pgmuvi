# Stack theme C: instrument-channel calibration + tutorial notebooks/examples

Baseline `origin/main`; tip `origin/pr174-add-complete-single-source-analysis-notebook` (abbrev. TIP).
All findings from `git diff/show`; no working tree changes.

---------------------------------------------------------------------------
## A. Instrument-channel calibration (pr138 -> pr165, + pr170/171 drafts)

### A.1 Size
| artefact (TIP) | lines |
|---|---|
| pgmuvi/instrument_channel_calibration.py | 9,600 |
| pgmuvi/instrument_channel_calibration_validation.py | 2,055 |
| pgmuvi/instrument_channel_calibration_validation_execution.py | 691 |
| tests (24 calibration/pairing test files) + 3 .rst + scripts/run_instrument_channel_calibration_validation.py | ~9,600 |
| examples/validation/kelt_r3_pairing_validation_{protocol,result,report}_v1.json | ~220 |

So ~22k lines in total. Per-PR deltas were 150-2,700 lines each, alternating between "define X contract" and "implement X":
pr138/139 calibration contract and implementation; pr140/141 pairing; pr142/143 orchestration;
pr144/145 coefficient uncertainty; pr146/147 predictive-uncertainty propagation;
pr148/149 scale-dependent (errors-in-variables) uncertainty; pr150/151 integrating that into fit;
pr152/153 predictive orchestration; pr154/155 instrument-specific pairing tolerance;
pr156 rule catalogue; pr158/159 catalogue discovery and local activation;
pr160 "scientifically validated catalogue" contract (2.3k); pr161 validation-protocol contract (2.7k);
pr162 executes the protocol on KELT R3 (adds the JSON results); pr164 status reconcile;
pr165 dataset-agnostic validation execution.

### A.2 What problem it solves
In the bundled light curve `examples/data/10131+3049.csv` (CIT 6 / IRAS 10131+3049), two "observational channels",
`KELT/OSN_Johnson.Cousins_R3_0` (5,706 rows) and `KELT/OSN_Johnson.Cousins_R3_1` (3,636 rows), share the same physical
wavelength 0.6561 um. The 2-D GP takes inputs (time, wavelength) and cannot tell the two channels apart. If they have different
zero-points or gains, the GP sees two inconsistent light curves at one wavelength. The module estimates an affine map
`ref = offset + scale * channel` between the channels so that both can be used.

This problem is real. The committed result `examples/validation/kelt_r3_pairing_validation_result_v1.json` gives
**scale ≈ 0.677–0.688** for R3_1 → R3_0 (offset ≈ 0.02), with held-out normalised RMSE ≈ 0.05 across 5 temporal folds.
The two KELT channels differ by about 30% in flux scale, so simply concatenating them would be wrong. That is a useful
scientific finding, and it is the strongest argument for keeping some form of this feature.

### A.3 The actual statistical model and algorithm (functions that do real work)
All in `pgmuvi/instrument_channel_calibration.py` at TIP:

1. **Detection**: `assess_instrument_channel_calibration_requirement(wavelengths, labels)` (l.1209, ~85 lines) groups
   channels by exact wavelength and returns groups with more than one channel. This is the only function used elsewhere
   (`wavelength_estimation.py`, `wavelength_validation_real_lpv.py`).
2. **Pairing**: `construct_instrument_channel_pairing` (l.4246) -> `_construct_monotonic_time_pairs` (l.4125, ~120 lines).
   It is a dynamic programme (like an edit-distance alignment) over the two time-sorted series. It finds a one-to-one,
   order-preserving matching that **maximises the number of pairs with |Δt| ≤ tol**, then **minimises the summed |Δt|**
   (ties broken deterministically). Methods: `exact_timestamp` or `nearest_within_tolerance`. There is no interpolation.
   The KELT protocol uses tol = 0.05 d.
3. **Coefficient fit**: `fit_instrument_channel_calibration` (l.8945, ~440 lines, mostly validation):
   - The model is `ref_i = a + b·ch_i + ε_i`, with b > 0 enforced.
   - It uses iteratively reweighted LS (`_weighted_affine_solution`, a 2-column lstsq) with weights
     `1/(σ_ref² + b²σ_ch²)` ("effective variance"). Each iteration applies MAD sigma-clipping
     (`_calibration_mad_sigma`, 1.4826·MAD, default 3.5σ, max_iter 8, min_pairs 3).
   - Covariance: if there are no channel errors, it uses the normal-matrix inverse via SVD (`_estimate_affine_coefficient_uncertainty`,
     l.8294), scaled by residual variance when there are no errors at all.
   - If channel-axis errors are given, it switches to
     `estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty` (l.8478, ~470 lines). This minimises the
     full Gaussian errors-in-variables marginal likelihood `½Σ[log v_i + r_i²/v_i]`, `v_i = σ_ref² + b²σ_ch²`, in
     (a, log b) with L-BFGS-B and an analytic gradient. Covariance is the inverse analytic observed Hessian. This is the
     latent-x-marginalised EIV likelihood (flat prior on the true channel flux). It is a defensible choice, but it differs
     from the York/Deming profile estimator and should be documented as such.
   - Clipping is conditioned on the inlier set; the covariance does not account for the clipping step.
4. **Apply**: `apply_instrument_channel_calibration` (l.9387) returns `a + b·flux` and errors `|b|·σ`.
   `apply_instrument_channel_calibration_with_predictive_uncertainty` (l.9433, ~170 lines) applies the delta method:
   `Var = b²σ² + Σ_aa + x²Σ_bb + 2xΣ_ab`, with an optional full covariance `J Σ Jᵀ` (`J = [1, x]`) for the shared-coefficient
   correlation between calibrated points.
5. **Validation execution**: `instrument_channel_calibration_validation_execution.py`, `_execute_fold`/`_execute_validation_dataset`
   (~150 lines of substance). It splits pairs into 5 contiguous equal-count temporal folds (`np.array_split` on pair
   midpoint time), fits on 4 folds, and on the held-out fold computes normalised RMSE, median bias and the dynamic range
   relative to the errors. Fixed acceptance gates are in the protocol JSON.

**Everything else** is contracts and bookkeeping. 47 of the 77 top-level definitions in the main module are classes
(frozen dataclasses with `to_dict/from_dict`, ~23 schema-version constants, ~12 status/disposition enums, and custom
exceptions). In total ~6,200 of 9,600 lines are dataclass/record classes. On top of that sit the plan/execute orchestration
(`define_/execute_instrument_channel_calibration_plan`, ~800), the rule catalogue with discovery/activation/compatibility
snapshots (~2,000), "scientific validation evidence" records with sha256 normalisation (~1,000), and predictive-orchestration
records (~1,000). The validation module (2,055) is again almost entirely dataclasses for
Dataset/Manifest/AcceptanceCriteria/Protocol/FoldResult/SourceResult/Result/Report.

**Estimate: about 600–700 lines of real algorithm** (pairing DP ~120, IRLS+clip ~150, covariance ~80, EIV optimiser ~200,
delta-method ~60, fold CV ~100) **in ~12.3k lines of module code (≈5%)**, plus ~9.6k lines of tests/docs that mostly assert
the contracts. This is clearly generated "define contract / implement / validate the validation" scaffolding.

### A.4 Coupling to the GP
**None.** It is not part of the likelihood, kernel or mean. It is not even a wired-in preprocessing step:
- The module docstring says it does "not ... merge channels; alter wavelengths; or integrate calibration automatically into a
  light-curve fit".
- `Lightcurve.fit(duplicate_wavelength_policy="all")` (lightcurve.py ~l.4260 and docstring ~l.11369) raises
  `NotImplementedError` "requires an applicable scientifically validated instrument-channel calibration strategy".
- Activation is blocked behind a "scientifically validated rule catalogue" that never gets populated
  (`populated_catalogue_created: false`). The KELT validation report's disposition is `inconclusive`, only because
  `minimum_independent_astrophysical_sources = 2` and there is only 1 public source.
- In practice users get `policy="first"`/`"select"`, which drop all but one channel. The `"first"`/`"select"` policies come
  from pr166/pr172, another theme.

"Propagation into GP predictions" therefore does not exist. Only propagation onto calibrated data points is implemented.

### A.5 Is it sound and worth keeping?
The core idea is sound and the KELT result shows it matters. The over-engineering is not worth keeping: the catalogue,
activation and provenance machinery, sha256 evidence records, and the "a pairing rule may only be used after a frozen
multi-source protocol passes" governance belong in a paper or notebook, not in the library.

A more principled alternative is in-model calibration: a learnable per-channel offset and scale in the likelihood/mean,
i.e. `y = a_c + b_c·f(t, λ) + ε`, with the reference channel fixed. That fits calibration jointly with the GP, propagates
uncertainty automatically, and needs no pairing or tolerance. It is a bigger change (~300–500 lines in gps.py/lightcurve.py)
and should be a later follow-up. The paired affine fit is the cheap, transparent first step and a diagnostic.

### A.6 Minimal reimplementation (≈250 lines of code + ≈250 lines of tests)
```python
# pgmuvi/calibration.py
@dataclass(frozen=True)
class ChannelCalibration:
    reference: str; channel: str; offset: float; scale: float
    cov: np.ndarray | None      # 2x2 (offset, scale)
    n_pairs: int; inliers: np.ndarray   # bool mask over pairs
    def apply(self, flux, err=None, include_coeff_uncertainty=True) -> (flux, err)

def pair_by_time(t_ref, t_ch, tol) -> tuple[np.ndarray, np.ndarray]     # DP or greedy, ~50 LOC
def fit_channel_calibration(ref, ch, ref_err=None, ch_err=None,
                            sigma_clip=3.5, max_iter=8, eiv=True) -> ChannelCalibration   # ~120 LOC
# Lightcurve integration (~80 LOC):
Lightcurve.calibrate_channels(reference: str, channels: list[str] | None = None, tol: float,
                              inplace=False) -> (Lightcurve, dict[str, ChannelCalibration])
#   -> rescales flux/err of each channel onto the reference, so fit(duplicate_wavelength_policy="all") becomes possible
```
The temporal-fold check can be a ~40-line `cross_validate_calibration(...)` helper or simply a notebook cell. Do not port
`InstrumentChannelPairingRuleCatalogue*`, discovery/activation, plan/execution/orchestration, validation-protocol
dataclasses, or the JSON evidence files.

### A.7 Overlap with `Lightcurve.merge()/concat()` and issue #130
- On main, `merge(other, band=, wavelength=, on_conflict={"raise","skip"})` and `concat(items, on_conflict=...)` refuse or
  skip same-band/same-wavelength conflicts. They do not combine them.
- #130 asks for `on_conflict="accept"` to concatenate light curves in the same band. That is exactly the "keep both
  channels at one wavelength" situation calibration is meant to make safe.
- Recommended layering:
  1. #130 `on_conflict="accept"` (tiny), keeping distinct channel labels;
  2. the duplicate-wavelength fit policy from the other theme (pr166/pr172 `first`/`select`/`all`);
  3. `calibrate_channels` as the optional step that makes `"all"` scientifically meaningful.

  `merge/concat(..., calibrate=True, reference=...)` could be sugar on top of step 3.

---------------------------------------------------------------------------
## B. Notebooks, tutorials and examples

### B.1 Already on main (merged PRs #141, #142 and earlier)
| notebook (docs/source/notebooks) | data | API | outputs stored | notes |
|---|---|---|---|---|
| PGMUVI_comparison_with_other_codes.ipynb (#141) | synthetic 1D/2D (`pgmuvi.synthetic`) | `fit`, celerite2, tinygp, GPJax | yes (17/17) | addresses #45 "comparison with other codes" |
| pgmuvi_mock_data_from_gp.ipynb (#142) | synthetic GP-prior draws | `gps.*Model`, `.sample()`, `fit` | **no** | addresses #45 item (4) "mock data from a kernel" |
| PGMUVI_QuasiPeriodic_and_Mean_Functions.ipynb | synthetic, highly uneven sampling | `1DQuasiPeriodic`, `1DPeriodicStochastic`, `1DLinearQuasiPeriodic`, `fit_LS`, `plot_acf` | yes | addresses #45 (2) and (3). Main's version pip-installs from stale branch `copilot/implement-acf-method-lightcurve` (TIP removes it) |
| pgmuvi_tutorial(_2d).ipynb, PGMUVI_Lightcurve, PGMUVI_Lomb_Scargle, PGMUVI_Gaussian_Process_fitting, tutorial_synthetic/preprocessing/model_selection, pgmuvi_tutorial_mcmc | synthetic / small CSVs | basic | mixed | the 2D tutorial on main has no outputs; the mcmc notebook is broken because `Lightcurve.mcmc` raises NotImplementedError on main |

**What main lacks: a real multiwavelength dataset (#138, #45 item (1)).**

### B.2 Added or refreshed in the stack
| branch | notebook / example | dataset | API exercised | outputs | comment |
|---|---|---|---|---|---|
| pr103–pr108 notebook-refresh programme (pr108 commit e769f8a, pr106 commit 4c02963) | refreshes tutorial_synthetic, tutorial_preprocessing (new `mixed_band_input.csv` written in-notebook), Lomb-Scargle, Lightcurve. **Deletes `pgmuvi_tutorial_mcmc.ipynb` and `tutorial_model_selection.ipynb`** | synthetic | | mostly none | removing the MCMC tutorial removes rather than answers the #45 complaint |
| **pr107-gp-prior-mock-data-workflow** | rewrites pgmuvi_mock_data_from_gp.ipynb; adds examples/gp_prior_sampling.py (292), howto/gp_prior_sampling.rst | synthetic GP prior (Matern / QP / SM) | `MaternGPModel/QuasiPeriodicGPModel/SpectralMixtureGPModel`, `model.forward(x).sample()`, **depends on `parameter_workflow.apply_parameter_estimates` / `ParameterEstimate`** (another theme) | none | docs-only PR; only real value is a cleaner "draw from prior + add noise" script |
| **pr134-add-wavelength-constraint-notebook** | tutorial_wavelength_constraints.ipynb (first version); **adds `examples/data/10131+3049.csv`** (10,816 rows, 17 channels, 16 wavelengths: ASAS-SN V, Gaia G/BP/RP, KELT R3_0/R3_1, PS1 grizy, ZTF g/r, Whitelock CTIO JHK + IRTF L) | **real, CIT 6**, file committed (~0.9 MB) | | | data provenance and citations are not recorded anywhere (no README/bib for survey credits). Needs adding before JOSS |
| **pr157-add-end-to-end-real-data-wavelength-constraint-fitting-tutorial** | rewrites the notebook; **adds `pgmuvi/wavelength_constraint_tutorial.py`** (591 → 775 lines at TIP) | CIT 6, **15 samples per channel, 30 training iterations** | `Lightcurve.fit(model="2DWavelengthDependent")` vs `"2DSeparable"` control; helpers `load_wavelength_constraint_tutorial_lightcurve`, `deterministic_time_stratified_observational_channel_indices`, `build_wavelength_constraint_position_rows`, `summarize_observational_channel_residuals` | 12/14 cells | notebook-specific helpers placed inside the package. The fit is toy-sized |
| **pr168-ensure-end-to-end-fit-and-plot-tutorial** | rewrites pgmuvi_tutorial_2d.ipynb ("End-to-end 2-D wavelength-dependent consensus fitting") | synthetic V/I/J, 120-d period, 30 iterations | `fit` + `plot` with consensus | 6/7 | fine; depends on consensus (other theme) |
| **pr174-add-complete-single-source-analysis-notebook** | tutorial_single_source_analysis.ipynb (2,862 lines JSON) + **`pgmuvi/single_source_analysis.py` (2,156)** + `period_diagnostic_comparison.py` (1,008) + `lomb_scargle_plotting.py` (348) + lightcurve.py +801 | CIT 6, 50 samples per channel, 200 iterations, two fits (parameter workflow on/off), `constraint_set="LPV"`, `2DWavelengthDependent`+RBF | per-channel LS+ACF period evidence, temporal consensus, `plot_period_diagnostic_comparison`, `get_parameter_workflow_report`, JSON report writer | **0 (no outputs stored)** | the Colab cell pins `PGMUVI_COLAB_GIT_REF = "pr174-add-complete-single-source-analysis-notebook"`, which breaks when the stack is replaced. `single_source_analysis.py` is reporting/JSON glue (summaries, `preserve_single_source_analysis_state`, report writer); `summarize_single_source_fit_explanations` alone is ~750 lines. It belongs in the notebook or in `examples/`, not in the package |
| other examples at TIP | examples/first_pgmuvi_workflow.py, interpret_pgmuvi_outputs.py, consensus_multiband_fit.py, validate_lightcurve_input.py, run_wavelength_advisory_batch.py, wavelength_advisory_batch_walkthrough.py, wavelength_model_selection_diagnostics.py, validate_wavelength_model_selection_workflow.py; tutorial_wavelength_advisory.ipynb (synthetic I/J/K/W1, `RUN_FITS=False`, no outputs) | mostly synthetic | mostly the wavelength-advisory theme | | `first_pgmuvi_workflow.py` (82) and `consensus_multiband_fit.py` are the reusable ones |
| examples/validation/ | `kelt_r3_pairing_validation_{protocol,result,report}_v1.json` (calibration); `d3_representative_lpv_{manifest,workflow,calibration_summary}.json` (wavelength theme) | | | | generated evidence records; drop from examples |

**Standalone PR #177** (`sundarjhu-multiband-consensus-notebook`, 1 commit, adds `PGMUVI_Multiband_GP_consensus_fit.ipynb`, 4.5k lines JSON, 15/17 cells with outputs):
- Data: CIT 6 `10131+3049.csv`, read from the current working directory. It is not in that branch; the stack adds it as `examples/data/`.
- API: `Lightcurve.from_csv(max_samples_per_band=150, check_sampling=True)`, `fit(model="2DWavelengthDependent", time_kernel_type="spectral_mixture", wavelength_kernel_type="rbf", mean_module="quad", constraint_set/prior_set="LPV")`, `plot_period_summary`, `generate_reproducibility_report`. It runs 1-component, 2-component and manual-harmonic experiments (~600-d period found).
- It pip-installs from `@integration-consensus-test`. Consensus does not exist on main (0 hits in main's lightcurve.py).
- It monkey-patches a private method (`lc_2d._consensus_validate_result_structure = lambda ...: None`) to work around a bug.
- It is the most scientifically readable real-data notebook: human-written interpretation, real outputs, 500–1000 iterations.

### B.3 Coverage of JOSS referee requests (#45, #138)
| request | main | stack TIP |
|---|---|---|
| (1) real multiwavelength data | ✗ | ✓ CIT 6 in tutorial_wavelength_constraints, tutorial_single_source_analysis and #177 |
| (2) quasi-periodic, uneven sampling | ✓ QP notebook (synthetic) | same notebook (only the stale install ref removed) |
| (3) non-zero mean function | ✓ QP notebook (`1DLinearQuasiPeriodic`); #177 uses `mean_module="quad"` | same |
| (4) mock data from a kernel | ✓ pgmuvi_mock_data_from_gp (#142), no outputs | rewritten by pr107, depends on parameter_workflow |
| MCMC tutorial works | ✗ (`mcmc` raises NotImplementedError) | notebook deleted |
| comparison with other codes | ✓ #141 | unchanged |

### B.4 CI execution
No notebook is executed by the test suite. `tests/test_docs_*_notebook.py` only grep notebook text and `compile()` cells. The
exception is `tests/test_docs_gp_prior_sampling.py::test_all_code_cells_execute_without_fitting_or_files`, which does `exec`
the mock-data notebook (it is skipped if gpytorch is missing, and there is no fitting).

The docs CI (`.github/workflows/docs.yml`, `make html-strict`, `docs/source/requirements.txt` installs `-e .`) uses nbsphinx
with the default `execute="auto"`. Notebooks without stored outputs are therefore re-executed at docs build:
single_source_analysis (two 200-iteration 2-D fits), mock_data, synthetic, preprocessing, wavelength_advisory. Notebooks with
outputs (wavelength_constraints, tutorial_2d, comparison, QP, #177) are rendered statically. With
`nbsphinx_allow_errors = True`, execution failures do not fail CI, so "executes in CI" gives no real guarantee.

---------------------------------------------------------------------------
## C. "[Future]" private-parquet multi-source validation (pr170 #265, pr171 #266, draft)
- pr170 (+1,377 lines) adds `instrument_channel_calibration_multisource_validation.py` (917) and
  `examples/validation/*maintainer_multisource_validation_protocol_v1.json`.
- pr171 (+2,193 lines) adds `..._multisource_execution.py` (1,092) and
  `scripts/execute_instrument_channel_multisource_validation.py` (469).
- Protocol: KELT R3_0/R3_1 only; leave-one-source-out across 5 seeded-selected independent sources; ≥100 pairs per source;
  the same RMSE/bias gates as A.3.
- **It requires a maintainer-owned private Parquet catalogue.** Its policy states: "maintainer-owned Parquet input is not
  distributed; raw astrophysical source identifiers remain private; full source-level report remains private", publishing
  only hashed source IDs and aggregate metrics.
- **pr172 already deletes all of it** (−3,625 lines), so TIP does not contain it. Issues #267/#268 track re-integration.
- Recommendation: it should **not** live in the public package. The public code would be untestable by anyone else, it
  exists only to gate a catalogue that we also recommend dropping, and the science is a one-off study. If the maintainers want
  the multi-source KELT answer, run it once in a private repo or notebook using the small public `fit_channel_calibration`
  plus a ~40-line CV loop, and report the number in docs or the paper. Close #265/#266, and repurpose #267/#268 into one issue:
  "validate KELT R3_0/R3_1 affine calibration on N sources (private analysis)".

---------------------------------------------------------------------------
## D. Recommended replacement PRs (this theme)
Ordering assumes the other themes land first: duplicate-wavelength policy (pr166/172) and consensus.

1. **Data and provenance** (~0 code lines + CSV). Add `examples/data/10131+3049.csv` (CIT 6) with
   `examples/data/README.md` giving survey sources and citations (ASAS-SN, Gaia DR3, KELT, PS1, ZTF, Whitelock et al.) and
   the column definitions. Unblocks every real-data notebook. (From pr134 commit 1d47483.)
2. **#130 `on_conflict="accept"`** for `merge/concat` (~60 code + tests). Keeps channel labels.
3. **Channel calibration core** (~250 code + ~250 tests). New `pgmuvi/calibration.py`: `pair_by_time` (port the DP from
   `_construct_monotonic_time_pairs`), `fit_channel_calibration` (IRLS + MAD clip + EIV marginal likelihood with Hessian
   covariance), `ChannelCalibration.apply` (delta-method errors). No enums, catalogue, schema versions or orchestration.
4. **Lightcurve integration** (~150 code + tests). `Lightcurve.calibrate_channels(reference, channels, tol)` returns a
   calibrated copy plus the coefficients. Allow `fit(duplicate_wavelength_policy="all")` when channels have been calibrated
   (or simply document "calibrate first, then 'all'"). Add a short how-to section showing the KELT R3 result (scale ≈ 0.68)
   with a temporal-holdout check done inline.
5. **Real-data multiwavelength notebook** (notebook only, no package module). Use #177 as the base because it has the best
   narrative and real outputs. Fix the install line (pip install `pgmuvi`/`main`), remove the private-method monkey-patch
   (fix the underlying bug in the consensus theme), and load from `examples/data/`. Optionally add one cell from the pr174 and
   pr157 notebooks (separable vs wavelength-dependent comparison; calibrated vs first-channel fit). Store outputs, and set
   `nbsphinx_execute="never"` for expensive notebooks so CI is honest. **Drop `pgmuvi/single_source_analysis.py`,
   `pgmuvi/wavelength_constraint_tutorial.py`** and the pr174/pr157 notebooks. Any genuinely reusable bit
   (`deterministic_time_stratified_observational_channel_indices`) can become a `Lightcurve.subsample(per_channel=...)` option
   in ≤50 lines. This addresses #138 and #45 (1).
6. **Mock-data notebook refresh** (small). Keep the main version (#142). If the pr107 example script is wanted, port
   `examples/gp_prior_sampling.py` without the parameter_workflow dependency, or after that theme lands. Store outputs.
7. **MCMC**: either restore a working MCMC notebook once `Lightcurve.mcmc` is reimplemented, or explicitly answer the
   referee that MCMC is out of scope. Don't silently delete it as pr108 did.
8. **pgmuvi_tutorial_2d refresh** from pr168 (synthetic consensus 2-D fit, outputs stored). Notebook only, after consensus lands.

**Drop entirely:**
- the whole `InstrumentChannelPairingRule*`/catalogue/discovery/activation/scientific-validation-evidence layer;
  the plan/execution/orchestration and predictive-orchestration records; `instrument_channel_calibration_validation*.py`;
  `scripts/run_instrument_channel_calibration_validation.py`; `examples/validation/*.json`;
- pr170/pr171 (#265/#266);
- the ~9.6k lines of contract tests;
- the "TBD[instrument-channel-calibration]" marker machinery (`scripts/audit_docs_tbd_markers.py`).

**Net effect:** about 22k lines (code + tests + docs) for this theme shrink to roughly 600 code + 500 test lines in 3–4 PRs,
plus two or three notebooks.
