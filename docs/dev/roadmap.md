# pgmuvi roadmap: PR reconciliation, reorganisation and new features

**Status: draft v1 for review (2026-10-27).** Open decisions are in section 7; please comment on the PR per decision number (e.g. "DEC-9: keep subsampling on"). When a decision is settled, its `Decision:` line is filled in here and the affected phase entries are updated. Nothing in this document is final until that happens.

Source reports (in `reports/`): `stack_A_consensus_params.md`, `stack_B_wavelength_docs.md`, `stack_C_calibration_notebooks.md`, `architecture_main.md`, `science_features.md`. These were produced by AI agents reading the branches with git, then checked by a maintainer only at the summary level; treat line counts as approximate and function names as pointers to verify, not as facts.

## 0. Where things stand

- `main`: 16.5k lines of Python; `lightcurve.py` 11.3k lines / 127 methods on one class; 1,292 tests pass in 42 s; ruff clean; numpy 2 clean.
- 104 open PRs. 5 are standalone (#67 viz, #151 separable-SM drift fix, #152 prior_set draft, #154 original giant Copilot PR, #177 CIT 6 notebook). The other 99 are one chained stack (`pr1-…` → `pr174-…`, 174 branches, ~70 of them never got a PR). Tip = +140k lines, 321 files; `pgmuvi/` grows from 16k to 80k lines (`lightcurve.py` → 26k, `instrument_channel_calibration.py` 9.6k, `wavelength_diagnostics.py` 7.8k).
- Content of the stack by theme (agent estimates of "real" vs scaffolding):
  | theme | branches | lines added (code+tests) | genuinely useful | verdict |
  |---|---|---|---|---|
  | consensus fitting, parameter-spec workflow, robustness fixes | pr1–40, 50–55, 98–99, 110, 114–115, 129, 166–167, 172–173 (+ #154, #156–162, #178–204) | ~28k | ~3k code + 3k tests | rewrite small |
  | wavelength diagnostics / advisory / validation | pr41–49, 56–65, 68–71, 89–95, 100, 112–137 | ~34k + 5k docs-grep tests | ~1.7k code + 1.1k tests + 3 bug fixes + a few empirical facts | rewrite small |
  | instrument-channel calibration | pr138–165, 170–171 | ~22k | ~600 code + 500 tests | rewrite small |
  | docs / CI / hygiene | pr66–67, 72–88, 96–97, 101–109, 111, 163, 169 | large | CI/docs config, PR template, stale-test removal, 3 rst pages | port directly |
  | notebooks | pr107, 134, 157, 168, 174, #177 | | CIT 6 dataset + one real-data notebook | rewrite |
- Contradictions inside the stack (all documented in report A §C): SM constraint caps flip pr30→33→51; transform inheritance added (pr37) then reverted for consensus (pr129); defaults flip (`learn_additional_noise` off→on, `duplicate_wavelength_policy` first→error); consensus init implemented twice (consensus vs `parameter_builders`); 3–4 overlapping sources of initial values/bounds in `_fit_core`; junk files committed (`lightcurve.py.orig` 22k lines, `debug*.txt`); several branches don't descend from their nominal parent (pr2–4, pr16, pr39, pr172).
- Standalone PRs: #151 is a real fix, mergeable, no conflicts → cherry-pick early. #152 overlaps #151 and the parameter workflow → drop, fold the `prior_set` kwarg idea into the initialisation PR. #67 → orthogonal, port into `plotting/`. #177 → best real-data narrative, but installs from a dead branch and monkey-patches a private method → rewrite on new API. #154 → superseded.
- Unopened side branches worth remembering: `feature/add-bgls` (1.1k-line torch BGLS), `fix_seed`, `2D-kernel-tests`; everything else from 2023 is stale.

## 1. Principles for the replacement stack

1. Every PR < ~800 code lines, one concern, stands alone against `main` (or against the previous PR in a short chain of ≤ 4), with tests. No "define contract / implement contract / validate contract" triplets.
2. Fixes first, then moves, then features. Mechanical moves are "no behaviour change, tests pass, aliases left behind"; behaviour changes are separate PRs with a changelog line.
3. Science goes into pure functions in small modules; `Lightcurve` becomes a façade. Nothing new is added to `lightcurve.py`.
4. Each behaviour-changing default (float64, jitter, time centering, subsampling) is an explicit maintainer decision recorded in the PR.
5. Old stack branches are archived with a tag and PRs closed with a pointer to this plan; nothing is deleted.

## 2. Phases and PR list

### Phase 0 — triage (no code, ~1 day)
- 0.1 Tag `archive/stack-2026-07` at `pr174-…` tip (and `archive/giant-pr` at `backup/giant-pr-original`); close #154, #156–162, #178–271 with a comment linking the plan; close #152; leave #151, #67, #177 open until superseded.
- 0.2 Close #265/#266; merge #267/#268 into one private-analysis issue outside the package. Close #110/#136 (numpy 2 is fine). Close #28 (done) once Phase 2.13 decides direction.
- 0.3 Put this plan in the repo (`docs/dev/roadmap.md`) and open a GitHub Project board with one card per PR below.

### Phase 1 — foundations (≈ 12 PRs, each small)
- 1.1 Hygiene & pins: delete `setup.py`, `pgmuvi/test_script.py`; move `AlfOriAAVSO_Vband.csv` to `pgmuvi/datasets/`; `[project.optional-dependencies]` mcmc/test/docs; lower bounds on torch/gpytorch/numpy/astropy; pin `arviz<1` (or port to arviz_base) since `mcmc()` is broken on fresh installs; drop seaborn; lazy-import pyro/arviz/xarray; export `Lightcurve` from `pgmuvi`; `models.py` shim emits DeprecationWarning; fix broken example import. [arch §4, stack B0]
- 1.2 CI: run on `pull_request`; matrix 3.10–3.14; CPU torch index + pip cache; ruff check-only (no auto-commit bot); pre-commit pin; strict docs build (`html-strict`, pandoc) as a separate job; PR template; CONTRIBUTING section. [stack B0, pr82/83/86/87/88/111]
- 1.3 Test harness: run existing suite under pytest (`-n auto`, `slow` marker, DeprecationWarning→error for pgmuvi); split `tests/tests.py` by topic; remove dead/skip-only tests (pr67 equivalent); seed fixture; `rng=` for bootstraps; stop tests writing PNGs to CWD. [arch §7, B0b]
- 1.4 Golden regression suite (must land before any move): seeded float64 period-recovery tests (1D sinusoid P=37 d with `1D`, `1DQuasiPeriodic`, `1DPeriodicStochastic`; 2D chromatic with `2D`, `2DSeparable`), fixed-hyperparameter snapshots (MLL, PSD on grid, `get_period_summary().as_dict()`), per-model `state_dict` key/shape snapshots, CSV/table round-trips, public-surface test. [arch §7]
- 1.5 Fix: transforms & uncertainties — `yerr` must be scale-only through the y-transform (currently mean-shifted), `transform_y` uses `xtransform`, `xtransform.min` crashes ZScore/RobustZScore in plotting, `filter_*`/band LCs drop transforms, plots never inverse-transform y, `"robust_score"` typo. [A B1–B4, pr114/pr37, arch bugs, science bugs]
- 1.6 Fix: wavelength means on physical λ (`PowerLawMean`/`DustMean` currently evaluate λ^p on transformed λ → NaN/clamp); `AchromaticGPModel` ignores `mean_module`; `wavelength_scaling='linear'` gives zero prior variance for the bluest band under minmax. [B1b, science bugs]
- 1.7 Fix: per-dimension ARD bounds for 2D SM `mixture_means`/`mixture_scales` ((1,1,2) tensors; wavelength column no longer tied to time span/Nyquist); option to pin μ_λ = 0. [B1c, pr124]
- 1.8 Fix: float64 default dtype (MJDs lose precision in float32); isolate torch default dtype in tests. [A2, pr36/110]
- 1.9 Fix: constraint integrity — `register_constraint` after init silently changes parameter values; intersect rather than overwrite; `set_default_constraints` called twice in `fit`. [A3, pr31–34, arch]
- 1.10 Fix: trainer robustness — late `NotPSDError` loses the fit (restore best state, raise jitter, halve lr), NaN loss/grad checks, `train()` KeyError without a Lightcurve, `verbose`, remove per-step `state_dict` prints in `mcmc`. [A4, pr39/pr53, arch]
- 1.11 Cherry-pick #151 (separable SM drift: MLS init gate, `mixture_means` lookup inside ProductKernel, frequency prior at MAP, wavelength outputscale cap). Rebase, keep its tests.
- 1.12 Small bug sweep: `summary()` shadows itself; `lstrip("raw_")`; `plot_results` broken; `_plot_nd`; dead methods (`append_data`, `init_hypers_from_LombScargle`, `_set_hypers_raw`, `_drop_nan_rows`, `print_periods`/`get_periods`, `trainers.Trainer`, `train_mll`/`train_variational*`, `SparseSpectralMixtureGPModel`, `CustomLinear/QuadConstantMean`); 47 commented-out blocks; 76 prints → logging (#11). [arch §6]

### Phase 2 — reorganise `main` (≈ 13 PRs; mechanical unless marked)
Target layout: `lightcurve.py` (~1.5k lines, data + façade) · `transforms.py` · `_utils.py` · `io/` · `preprocess/` · `periodogram/` (lombscargle, significance, acf) · `models/` (means, kernels, spectral_mixture, separable, alternative, registry, priors, constraints, initialization) · `inference/` (train, laplace, mcmc) · `results/` (fit_result, period_summary, psd, text) · `plotting/` · `datasets/`. Old module names become re-export shims with DeprecationWarning, removed one release later.
- 2.1 `transforms.py` (+ `Shift`/`TimeCenter` transformer; `center_time` kwarg off by default here, default decided in 3.2). [A6, pr37/40]
- 2.2 `results/`: `PeriodPeakResult`, `ACFResult`, `ComponentDiagnosticsResult`, `PeriodSummaryResult` + `to_text`/`write_*`.
- 2.3 `results/psd.py`: the 15 static PSD/peak helpers (keep `Lightcurve._x = staticmethod(...)` aliases for tests).
- 2.4 `plotting/`: `plot_period_summary` (pure, 538 lines), then `plot_psd`, `_plot_1d/_plot_2d`, `plot_acf`, mcmc plots; `save=False` default, figures closed. Port #67 as `plotting/gif.py`.
- 2.5 `io/tables.py`: `from_csv`, `from_table`, `to_table`, `to_csv`, `write_votable`, `_convert_time_to_days`. (Rewriting `from_csv` on top of `from_table` = separate behaviour PR.)
- 2.6 `preprocess/pipeline.py` from the `__init__` stages + `preprocess/combine.py` (merge/concat + `_validate_band_wavelength_mapping`). **Behaviour decision:** subsampling to 1000 points is on by default at load and silently drops data → propose default off / explicit.
- 2.7 `periodogram/`: `fit_LS` (399 lines, many return modes), `MultibandLSWithSignificance`, ACF.
- 2.8 `models/` split of `gps.py` + a registry (`get_model(name, **kw)`) replacing the 3 identical dispatch dicts, `_SM_MODELS`, and the docstring name lists.
- 2.9 **[high risk]** collapse the 12 near-duplicate SM classes into `SpectralMixtureGPModel(mean=..., ski=..., grid_size=...)` with old names as thin subclasses; assert `state_dict` keys unchanged; preserve/decide quirks (1D-only `initialize_from_data`, KISS grid defaults, `sci_kernel`).
- 2.10 `inference/train.py` + `inference/mcmc.py` (lazy pyro/arviz); extract the ~300-line MLS initial-frequency block out of `fit` into `models/initialization.py` as a pure function.
- 2.11 **[high risk, behaviour]** `ModelSpec` + `FitResult`: fitted model/likelihood/transforms/spec/history live on a `FitResult` (`torch.nn.Module`) returned by `fit()` and kept as `lc.last_fit`; the six `__FLAGS` and five model-spec attributes go away; `Lightcurve` stops subclassing `gpytorch.Module`; façade methods delegate. Only after 1.4 exists.
- 2.12 Save/load/resume (#12): `FitResult.save()/load()` = `state_dict` + JSON (model name, kwargs, transform state, band map, data hash, versions, history).
- 2.13 `predict()` API + memory-safe plotting (#149): chunked predictions, `n_pred` default ~min(2000, 20·baseline/P_min), latent prediction + analytic per-band noise, LOVE/`fast_pred_var`, inverse y-transform.

### Phase 3 — salvage the stack onto the new layout (≈ 14 PRs, all rewritten, none cherry-picked wholesale)
- 3.1 Per-band learned jitter: `BandJitterLikelihood` (σ_i² = yerr_i² + s_b²); 1D reduces to one jitter. Replaces the stack's single global `learn_additional_noise`, which the experiment shows hurts faint bands. **Default:** on. [A5 + science #2]
- 3.2 Time centering default on (after 2.1). [A6]
- 3.3 Data-driven initial values & constraints in `models/initialization.py` (no ParameterSpec/enum/report framework): median-flux mean, robust-span outputscale/weight caps, √(baseline·cadence) lengthscale, 1/baseline frequency floor, μ ∈ [2/baseline, pseudo-Nyquist], width ∈ [1/(10·baseline), μ/2], wavelength lengthscale √(median spacing·span), grid-profile init for power-law/dust means on per-band medians, noise-corrected per-band amplitude init. Absorbs #152's `prior_set` kwarg. [A7, B2, science §3]
- 3.4 `consensus.py` pure functions: per-band gating, dominant LS per band, dedupe + MAD-clipped median, min inliers, LS–ACF harmonic check (pr129 version). [A8]
- 3.5 `fit(fit_strategy="consensus", consensus=ConsensusConfig(...))`: initialise + bound SM from consensus, pr52 width cap, `ConsensusFitError`, small result dataclass, how-to doc adapted from pr99. [A9]
- 3.6 Consensus → period-kernel handoff; `TwoDSeparable` `time_kernel_type`/`wavelength_kernel_type` selectors. [A10, pr55]
- 3.7 Duplicate-wavelength channels: policies `error`/`first`/`select`; consensus over all channels. [A11, pr166/167/172]
- 3.8 `merge()/concat(on_conflict="accept")` (#130). [C]
- 3.9 `calibration.py` (~250 lines: `pair_by_time` DP matcher, `fit_channel_calibration` IRLS + MAD clipping + both-errors likelihood, `ChannelCalibration.apply` with coefficient-uncertainty propagation) + `Lightcurve.calibrate_channels()`; enables `duplicate_wavelength_policy="all"`. Catalogue/discovery/orchestration/validation layers dropped. [C]
- 3.10 `wavelength_summary(lc, period=None)`: period-free per-band stats (noise-corrected scatter/amplitude, log-log slope, monotonicity); with a period, fixed-frequency amplitude/phase/lag with a common t0 and delta-method errors (pr116 version); classification achromatic / chromatic / power-law-like / lagged; one plot. [B3]
- 3.11 `compare_models(lc, models=[...])` ranked by exact log-ML (data and prior terms separated) or held-out NLPD, with per-band standardised residuals, 2σ coverage and `parameters_at_bounds()`. Explicitly NOT the stack's training-residual ad-hoc score. [B4, B5]
- 3.12 Datasets & real-data notebook: `examples/data/10131+3049.csv` thinned (~1–2k rows) with provenance README; rewrite #177 (CIT 6 consensus fit) on the new API with stored outputs; refresh mock-data and 2D tutorials; decide MCMC notebook fate. Addresses #45 item 1 and #138. [C, B6]
- 3.13 Docs: `wavelength_models.rst`, `overview.rst`, `first_workflow.rst`; "validation notes" recording the stack's empirical facts (uneven per-band counts and sparse long baselines break recovery; ARD-at-bounds is not a failure signal; P ≈ 597.4 d for CIT 6); known-limitations list; nbsphinx must fail CI on errors. [B6]
- 3.14 (optional) multi-component consensus (#164–176 ideas, no diagnostics layer); BGLS from `feature/add-bgls` into `periodogram/bgls.py`.

### Phase 4 — new science (≈ 9 PRs)
Evidence from the toy experiment (`reports/science_features.md`, experiment section): with a single constant mean the faint band's gap σ is 30–100× its amplitude; per-band mean alone brings that to ~0.03×; amplitude modulation adds +40–60 log-ML; the chromatic SM has the best evidence and recovers 60× amplitude ratios and phase lags, but is overconfident under ML-II.
- 4.1 `BandOffsetMean` (per-band offsets, optional per-band linear trend) as the 2D default mean; DustMean/PowerLaw become static-SED + per-band residual with shrinkage. [science #1]
- 4.2 `BandModulatedKernel`: k = a(λ)a(λ')·k_t·k_λ via `Diag(a1) K Diag(a2)`; a(λ) = knots at band wavelengths lerp in log λ (default), log-polynomial for many channels; random-walk prior; unscaled k_λ over log λ; `SeparableGPModel(amplitude=...)`. **New default 2D model = BandOffsetMean + BandModulated(SM_t × RBF_logλ) + BandJitter.** Legacy `2D` keeps μ_λ = 0 constraint option. [science #3; this is the user's amplitude-modulation idea, validated]
- 4.3 2D initialisation from multiband LS (per-band amplitude/phase → weights, a(λ), φ), multi-restart over top-K peaks + yearly/harmonic aliases, L-BFGS polish after Adam in float64; dedicated red-noise term (Matérn-3/2, not RBF `add_flicker`) + lower bound on μ_q against period drift. [science §3]
- 4.4 Synthetic benchmark suite (#17): extend `synthetic.py`; grid over amplitude ratio 1–1000, lag 0–0.3 cycles, cadences (seasonal, LSST-like, NESS-like), S/N, cycle jitter; metrics: period error/alias rate, amplitude-ratio and lag error, per-band 68/95 % coverage and CRPS in gaps/faint bands, wall time; nightly job. Prototype: `experiments/amp_test3.py`.
- 4.5 `ChromaticSpectralMixtureKernel` (rank-1 CSM: per-band amplitude and phase per component, `phase='free'|'delay'`, + incoherent term) exposed as `'2DChromaticSM'`; `get_amplitude_ratios()`, `get_phase_lags()`. Flagship: optical-vs-IR amplitude ratios and lags become fitted parameters. [science #5]
- 4.6 `'2DCoregionalised'` ICM/LMC option (`IndexKernel` on band index via bucketize) for ≲10 heterogeneous bands.
- 4.7 Laplace approximation (`inference/laplace.py`): Hessian over raw params, delta-method errors on periods/lags/ratios, Laplace evidence feeding `compare_models`. [science #6]
- 4.8 MCMC rewrite: clean pyro NUTS over raw params initialised at MAP with Laplace mass matrix (fix #33, #46); optional emcee backend.
- 4.9 Time-dependent amplitude / change-point kernel (secular Mira amplitude change, obscuration events). Later.

### Phase 5 — scale & polish (≈ 5 PRs)
- 5.1 Batched independent GPs (`batch_shape=[S]`, padding/masking) + `fit_catalogue()` with checkpointing — the LSST story.
- 5.2 SKI along time only (current 2D grid is 5000×20 even for 5 bands): Σ_r D_r (W K_grid Wᵀ) D_r.
- 5.3 SVGP / VNNGP (`LMCVariationalStrategy`, `NNVariationalStrategy`); the ELBO path is currently NotImplemented.
- 5.4 SVO filter lookup (#132, `astroquery.svo_fps`) + alias table; auto-populate band/wavelength (#131); astropy `Quantity` periods in user-facing IO (#133).
- 5.5 Type hints (#16) on the public surface; docstring audit (#14).

## 3. Decisions needed from the maintainers (summary; full treatment in section 7)
1. Archive-and-close all stack PRs wholesale (recommended), or keep any open?
2. Behaviour defaults: float64 (yes); per-band jitter on (yes); time centering on (yes); subsampling-on-load off (recommended; currently silently drops data beyond 1000 points); duplicate-wavelength policy `error` (recommended).
3. `FitResult` + `Lightcurve` no longer a `gpytorch.Module` (recommended, with one release of shims) — this is the one API change users will notice.
4. pytest as runner (recommended; unittest classes run unchanged).
5. Calibration as preprocessing (recommended now) vs in-GP per-channel scale/offset (later, after 4.5).
6. Priority of 4.2 (amplitude modulation, default) vs 4.5 (chromatic SM, flagship) — recommend 4.1→4.2→4.3→4.4→4.5.
7. Where the plan lives: `docs/dev/roadmap.md` in-repo + GitHub Project board.

## 4. Rough sizes
| phase | PRs | new/changed code (lines) | notes |
|---|---|---|---|
| 1 | 12 | ~1.5k | fixes + harness |
| 2 | 13 | ~0 net (moves) + ~1k (FitResult, predict, save/load) | |
| 3 | 14 | ~4.5k code + ~4k tests | replaces ~80k lines of stack |
| 4 | 9 | ~3k | |
| 5 | 5 | ~2k | |
Total ≈ 53 PRs; package ends ~25k lines vs 80k at the stack tip.

---

## 5. Drop list in detail

How to read this: every item is still on the archived branches (`origin/pr<N>-…`; tip `origin/pr174-add-complete-single-source-analysis-notebook`). To inspect one, use `git show origin/<branch>:<path>` or `git diff origin/pr<N-1>-… origin/pr<N>-…`. "Size" is approximate net lines, code / tests. "Survives as" says what, if anything, the replacement stack keeps. Agent reports with full function lists: A = `reports/stack_A_consensus_params.md`, B = `reports/stack_B_wavelength_docs.md`, C = `reports/stack_C_calibration_notebooks.md`.

### 5.1 Dropped outright

**D1. Consensus self-validation and status machine** — pr5 (#161), #173. Size ~2.4k / ~2.0k.
Where: `lightcurve.py` at `integration-consensus-test`: `_consensus_validate_result_structure` (973 lines), `_CONSENSUS_*` constants (~430), `_consensus_set_*_status`, `_consensus_*rejection_reason*`, `_consensus_initialize_result_structure`, `_consensus_finalize_result_structure`, `_consensus_debug_checkpoint`, `_consensus_make_json_safe`; `tests/test_consensus_diagnostics.py`.
What it does: validates the module's own `lc.consensus_diagnostics` dict (~60 keys) against a hand-written schema at run time; tracks per-band status transitions.
Why drop: no science; it only checks that the code wrote the dict it just wrote.
Survives as: a frozen `ConsensusResult` dataclass with per-band `(status, reason)` strings (phase 3.5). [A §A row 5]

**D2. Fit history, configuration snapshot, provenance, reproducibility report** — giant PR #154 / pr5. Size ~2.3k / ~1.7k (`tests/test_fit_history.py` exists only in #154).
Where: `lightcurve.py`: `fit()` becomes a wrapper around `_fit_core`; `get_fit_history`, `get_fit_history_summary`, `print_fit_history`, `print_fit_history_summary`, `export_fit_history_json`, `load_fit_history_json`, `fit_configuration_to_text`, `get_last_fit_configuration`, `generate_reproducibility_report`, `clear_fit_history`, `_sanitize_fit_configuration_value` (332 lines), plot provenance annotations.
What it does: records every `fit()` call with sanitised/truncated kwargs, git SHA, package versions, RNG state.
Why drop: 2.3k lines for a feature nobody asked for in an issue; its value is covered by `FitResult.save()` (phase 2.12), which stores model name, kwargs, transform state, versions and history.
Survives as: ~80 lines, `FitResult.info = {kwargs, versions, final_loss, n_iter, timestamp}`. [A §A row 7]
**Check:** was this built for a reproducibility requirement in the NESS paper? If yes, keep the git-SHA/versions part in `FitResult.info`.

**D3. 1D-GP validation of consensus band candidates** — pr4 (#159). Size ~500 / ~165.
Where: `_consensus_validate_candidates_with_1d_gp` (333), `_consensus_prepare_gp_validation_fit_kwargs` (87); kwargs `use_gp_validation`, `gp_validation_kwargs`, `gp_frequency_tolerance_factor`.
What it does: fits a 1D GP per accepted band and rejects the band if the GP period-summary frequency is outside a tolerance of the LS candidate.
Why drop: expensive (one full GP fit per band before the real fit), circular (the GP is initialised from the same LS peak it is checking), off by default.
Survives as: nothing. Could return as an opt-in ~100-line helper if a use case appears. [A §A row 4]

**D4. `consensus_relaxed`** — pr1 (#156). Size ~20. A `fit_strategy` value that raises `NotImplementedError`. Drop. [A §A row 18]

**D5. Upload / PyPI artefact validation** — pr6 (#160). Size 118 + 97 (`.github/scripts`) / ~400 + CI YAML.
Where: `pgmuvi/upload_validation.py`, `.github/scripts/validate_uploaded_artifacts.py`, `tests/test_upload_validation.py`, `tests/test_validate_uploaded_artifacts_script.py`.
What it does: revision/sentinel checks on "uploaded files".
Why drop: appears to come from an agent treating stale chat uploads as a packaging problem; no corresponding issue; PyPI publishing was already hardened on main by #143. [A §A row 8]

**D6. Parameter-spec / estimate / application framework** — parameter stack pr1–pr29 (#178–#204) plus pr30, pr33, pr35, pr51 (which repair problems it introduced). Size ~2.2k code at tip / ~4.4k tests / ~30 docs.
Where: `pgmuvi/parameter_specs.py`, `parameter_estimates.py`, `parameter_builders.py` (952), `parameter_application.py` (594), `parameter_context.py`, `parameter_workflow.py`; `gps.py` gains `parameter_schema()` on 23 model classes (~730 lines); `Lightcurve.fit(use_parameter_workflow=True)` **on by default**, `get_parameter_workflow_summary()`, `get_parameter_workflow_report()`, `lc.parameter_workflow_result`; `tests/test_parameter_*.py`, `tests/test_spectral_mixture_parameter_schema.py`.
What it does: each model declares `ParameterSpec`s (role, domain, scale, guess strategy, constraint strategy); a builder computes initial values and bounds from a `ParameterEstimationContext`; an applicator writes them into the model in transformed space; reporting records skip reasons.
Why drop: (a) 20 of the 29 PRs change <60 code lines; (b) 7 declared `GuessStrategy` values (`LS_PERIOD`, `ACF_PERIOD`, `MAD`, …) have no implementation; (c) it is a third (fourth with pr124) layer of initial values/bounds in `_fit_core`, overriding `set_default_constraints` and the MLS init, which is why pr30/33/35/51 then had to fix caps that pinned bright data and constraints that overwrote consensus bounds; (d) consensus initialisation is implemented a second time inside `parameter_builders` (`_estimate_consensus_frequency`, `_estimate_consensus_multicomp_period`, `_estimate_frequency_fallback`).
Survives as: phase 3.3, `models/initialization.py`: `default_initial_values(lc, model)` and `default_bounds(lc, model)` implementing the five heuristics that actually exist (mean = median flux; outputscale/weights ≈ robust flux span with the pr51 widened caps; lengthscale = √(baseline·cadence); frequency floor = 1/baseline; SM means from consensus or harmonics of 1/baseline), applied through the constraint helpers of phase 1.9. ~300 / ~250. No enums, specs, collections, reports. [A §A row 9, §C items 2, 6, 7]

**D7. Wavelength advisory workflow (batch, CLI, reports)** — pr57–pr65, pr66 (rename), pr68–pr71, pr89, pr91, pr92, pr94, pr95; repairs pr112, pr113. Size ~5.0k / ~3.5k / ~1.3k docs (+ ~1k for pr112/113).
Where: `pgmuvi/wavelength_diagnostics.py` (most of its 7.8k lines): ten `*_period_independent_wavelength_*` functions (`plan_…`, `build_…_fit_candidate_configs`, `run_…`, `score_…`, `…_quality_diagnostics`, `…_comparison_report`, `…_advisory_workflow`, `export_…`, `batch_…`) plus ten `Lightcurve` wrappers; `examples/run_wavelength_advisory_batch.py` (CLI); `examples/wavelength_advisory_batch_walkthrough.py`; docs `howto/wavelength_advisory*.rst`, `batch_walkthrough.rst`.
What it does: from the Level-0 median-flux slope picks a candidate model order (positive slope → Dust > PowerLaw > WavelengthDependent; negative → PowerLaw > WD > Dust; non-monotonic → WD first), builds config dicts, fits each, scores with `100 − 25·NRMSE − 5·median|z| − 25·outlier_frac − 0.25·log(1+χ²_red)` on **training** residuals, plus a separate "viability score" (+100 fit success, +25 consensus, +2 per accepted band …), then writes CSV/markdown manifests per source and per batch.
Why drop: the score is scientifically unsound (in-sample residuals reward over-fitting; the weights are arbitrary) and was numerically broken in the real-LPV run (`training_nrmse_by_target_scale` 62–1670 while χ²_red 0.3–5). The exact marginal likelihood (pr115) is computed but not used for ranking. The batch/CLI/manifest layer is orchestration of a broken core.
Survives as: the model-ordering heuristic as one paragraph in `wavelength_models.rst`; the comparison loop itself as `compare_models()` (phase 3.11, see R3 below). [B §A row 4, §C.3]
**Check:** does any collaborator run `run_wavelength_advisory_batch.py` over a catalogue today? If so, a thin `fit_catalogue()` (phase 5.1) should replace it, not this.

**D8. Wavelength status / result / hypothesis / conclusion layers** — pr117, pr118, pr119, pr120. Size ~2.7k / ~1.15k / ~300 docs.
Where: `pgmuvi/wavelength_status.py` (7 enums: `TechnicalOutcome`, `ScientificUsability`, `ComparisonEligibility`, `WavelengthConclusionDisposition` …), `wavelength_results.py` (5 dataclasses), `wavelength_hypotheses.py` (static table model → mean structure {constant, quadratic, dust, power-law} × covariance structure {separable, joint SM-ARD, achromatic}), `wavelength_conclusions.py` (`synthesize_wavelength_advisory_conclusions`).
Why drop: typed wrappers around D7's outputs; no computation.
Survives as: a six-row table in the docs listing each 2D model's mean and covariance structure. [B §A row 6]

**D9. Wavelength validation framework** — pr126, pr127, pr128, pr130, pr131, pr132, pr133, pr135, pr136, pr137. Size ~9.5k / ~4.8k / ~900 docs.
Where: `pgmuvi/wavelength_validation.py`, `wavelength_validation_synthetic.py`, `wavelength_validation_recovery.py`, `wavelength_validation_robustness.py`, `wavelength_validation_robustness_calibration.py`, `wavelength_validation_real_lpv.py`; `scripts/run_representative_lpv_validation.py`; `examples/validation/*.json`; `tests/test_wavelength_validation_*.py` (9 files); docs `validation_*.rst`.
What it does: D1 draws synthetic 2D GPs (Cholesky from QP(t)×RBF(λ) or joint SM-ARD, plus constant/quadratic/dust/power-law mean at V I J K W1 W2) and fits with the generating model; D2 perturbs sampling/noise/band coverage over 20 seeds and calibrates pass thresholds; D3 runs five candidates on CIT 6. Almost all of the code is dataclasses, manifests and status reconciliation.
Why drop: it is a one-off experiment frozen as package code. Its results are the useful part.
Survives as: (a) the findings, recorded in docs (phase 3.13): of 420 D2 runs 393 completed; uneven per-band counts break wavelength length-scale recovery (8/19 pass); long-but-sparse baselines break period recovery (4/20); a large wavelength gap degrades recovery (9/20); for the joint 2D SM kernel the ARD parameters sat at their bounds in 100 % of runs while recovery stayed fine, so bound pressure is not a failure signal; the ACF picked a spurious ~23 d timescale against an injected ~500 d period; all D3 models found P ≈ 597.37 d for CIT 6 (ranking not trustworthy). (b) Optionally `synthetic.make_separable_gp_2d(...)` (~150 lines) plus one slow opt-in recovery test (phase 3.14 / 4.4). [B §A row 10, §C]
**Check:** should D2 be re-run on the new default model (phase 4.2) rather than just recorded? Recommended yes, as part of the phase 4.4 benchmark, since the boundaries will move.

**D10. Instrument-channel calibration scaffolding** — pr138, pr140–pr156, pr158–pr165 (everything except the algorithm). Size ~11.7k of the 12.3k lines of calibration code / ~9.6k tests+docs.
Where: `pgmuvi/instrument_channel_calibration.py` (9,600 lines; ~6,200 are frozen dataclasses/record classes with `to_dict/from_dict`, ~23 schema-version constants, ~12 status/disposition enums), the plan/execute orchestration (`define_instrument_channel_calibration_plan`, `execute_…`, ~800), the rule catalogue with discovery/activation/compatibility snapshots (`InstrumentChannelPairingRule*`, ~2,000), "scientific validation evidence" records with sha256 normalisation (~1,000), predictive-orchestration records (~1,000); `instrument_channel_calibration_validation.py` (2,055), `instrument_channel_calibration_validation_execution.py` (691); `scripts/run_instrument_channel_calibration_validation.py`; `examples/validation/*.json`; `scripts/audit_docs_tbd_markers.py`; ~9.6k lines of contract tests.
Why drop: activation is gated on a "scientifically validated rule catalogue" that is never populated (`populated_catalogue_created: false`), so at the stack tip `fit(duplicate_wavelength_policy="all")` raises `NotImplementedError`. Users only ever get `first`/`select`, which discard channels. 47 of 77 top-level definitions are dataclasses or enums.
Survives as: phase 3.9, `pgmuvi/calibration.py` (~250 lines): `pair_by_time` (port the DP from `_construct_monotonic_time_pairs`, l.4125 at tip), `fit_channel_calibration` (IRLS + MAD clipping from `fit_instrument_channel_calibration`, l.8945, plus the errors-in-both-variables likelihood with analytic-Hessian covariance from `estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty`, l.8478), `ChannelCalibration.apply` (delta method from `apply_instrument_channel_calibration_with_predictive_uncertainty`, l.9433), and the 5-fold temporal hold-out check (~100 lines). Plus `Lightcurve.calibrate_channels(reference, channels, tol)`. The KELT R3_1→R3_0 result (scale ≈ 0.68, hold-out NRMSE ≈ 0.05, `examples/validation/kelt_r3_pairing_validation_result_v1.json`) goes into the how-to. [C §A, §D]
**Check:** preprocessing calibration (fix coefficients, then fit) vs in-GP per-channel scale/offset (fit jointly, ~300–500 lines in `gps.py`/`lightcurve.py`). Plan says preprocessing now, in-GP after the chromatic SM exists.

**D11. Maintainer-private multi-source validation** — pr170 (#265), pr171 (#266), both marked "[Future]"; issues #267, #268. Size ~3.6k, already reverted by pr172.
What it does: runs the calibration validation over a private Parquet catalogue and publishes hashed source IDs.
Why drop: requires private data; cannot be tested or reproduced publicly; the package is the wrong home.
Survives as: nothing in the package; one private-analysis task outside the repo. Close #265/#266, merge #267/#268 into that task. [C §C]

**D12. Notebook glue shipped as package modules** — pr157 (#252), pr174 (#271). Size: `wavelength_constraint_tutorial.py` 775, `single_source_analysis.py` 2,156, `period_diagnostic_comparison.py` 1,008; tests ~3.6k; notebook ~3k.
What it does: `summarize_single_source_*` helpers and report builders that exist only so the notebooks can call them; `plot_period_diagnostic_comparison` overlays LS/ACF/GP-PSD period markers.
Why drop: tutorials must exercise the public API, not a private helper layer. The pr157 fits use 15 points per channel and 30 iterations, so they demonstrate nothing; the pr174 notebook stores no outputs and its Colab cell installs from the pr174 branch, which breaks when the stack is archived.
Survives as: `plot_period_diagnostic_comparison` (~200 lines) may be ported later into `plotting/`; `deterministic_time_stratified_observational_channel_indices` may become a `Lightcurve.subsample(per_channel=...)` option (≤ 50 lines). [A §A row 19, C §B]

**D13. Docs-about-docs tests** — pr73–pr109, pr136, pr169 (~40 files, ~5k lines).
Where: `tests/test_docs_*.py`, `test_repository_reference_*`, `test_pull_request_template.py`, `test_contributing_workflow_docs.py`, `test_docs_ci_workflow.py`, `test_docs_strict_build_contract.py`, `test_d3_*closeout*`, `test_pr136_*`, `test_wavelength_constraint_tutorial.py`, the notebook "tests" that only grep cell text.
Why drop: they assert that rst files contain phrases; the strict Sphinx build (phase 1.2) is the real check. nbsphinx currently has `nbsphinx_allow_errors=True`, so notebook failures never fail CI; phase 3.13 flips that.
Survives as: pr169's reference-integrity check (doc roles, markdown links, machine-local paths) as a ~60-line test. [B §D]

**D14. Process scaffolding and junk** — pr85 (`scripts/create_source_snapshot.py`, an AI-review zip workflow), pr96 (`documentation_roadmap.rst`), pr109 (`future_work.rst` TBD-marker registry + `scripts/audit_docs_tbd_markers.py` + tests), pr76 (`notebook_status.rst` + triage tests), `docs_maintenance.rst` (fold 10 lines into CONTRIBUTING), pr66 (rename-only churn), pr78 (removes `.orig` files pr36 added), pr34's `debug1-3.txt` (1,118 lines), pr35's `tests/test_constraint_integrity_regression.py.orig`, pr36's `pgmuvi/lightcurve.py.orig` (22,255 lines; present until somewhere in pr59–pr98).
Survives as: the content of `future_work.rst` as a "Known limitations" paragraph: no held-out validation; no wavelength-dependent lags (until phase 4.5); no instrument-channel calibration (until 3.9); no 2D PSD plot; no native magnitudes; MCMC not working. [B §D]

**D15. pr80 docstring deletions** — "Sphinx warning cleanup". Deleted substantive docstring content (e.g. the full `ConsensusFitError` reason catalogue) to silence warnings. Redo by fixing the RST, not by deleting text. pr81 (2.5k/2.5k churn re-executing `PGMUVI_Lightcurve.ipynb`) is replaced by re-running on the new stack. [B §D]

**D16. Standalone PR #152** — `copilot/update-prior-settings`, draft, +444. Adds `prior_set=` to `fit()`/`mcmc()` with dimension-aware prior selection and a LogNormal prior centred on the dominant LS frequency for 1D SM. Overlaps #151 (which registers the frequency prior for separable SM models) and D6. Survives as: the `prior_set` kwarg and the "MLS-centred prior for 1D SM" rule inside phase 3.3. Close the PR.

**D17. Standalone PR #154** — the original 109-commit giant Copilot PR (`copilot/add-fit-strategy-skeleton`, +16k, also `backup/giant-pr-*`). Entirely superseded by the consensus mini-stack and `integration-consensus-test`. Close; keep the backup branches under the archive tag.

### 5.2 Dropped as written, idea kept in reduced form

**R1. Multi-component consensus** — #163–#176 merged into `integration-consensus-test`. Size ~2.1k / ~2.9k.
Where: `_consensus_multicomp_fit` (812), `_consensus_cluster_component_candidates` (224), `_consensus_multicomp_reconcile_to_n_components` (286), `_consensus_build_multicomponent_frequency_consensus` (170), `_consensus_collect_band_component_candidates` (165), identity/drift diagnostics (~230), `_get_consensus_multicomp_period_summary` (137); `fit(fit_strategy="consensus_multicomp", num_mixtures=…)`.
What it does: takes the top-N LS peaks per band, clusters them across bands in log-frequency (rtol 0.1, ≥2 bands per cluster), reconciles to `num_mixtures` components, initialises each SM component; adds post-fit drift and component-identity (swap) warnings.
Why reduce: all components share one global frequency interval (the docstring calls component-specific constraints "future work"), which undercuts the point; the diagnostics are 460 lines; #176 had to fix the user's `num_mixtures` being ignored.
Survives as: phase 3.14 (optional), ~300 lines reusing the phase 3.4 helpers, with per-component bounds; no drift/identity diagnostics. [A §A row 2]
**Check:** if multi-periodic LPVs are central to the NESS analysis, promote this from optional to phase 3.6.

**R2. Consensus failure semantics** — pr7 (#162). Size ~300 / ~490.
Where: `ConsensusFitError`, `FitFailureSummary`, `lc.fit_failed`, `lc.failure_reason`, `lc.get_failure_summary()`, `_raise_if_fit_failed` guards in plot/PSD/period-summary.
Survives as: `ConsensusFitError(reason, per_band_table)` (~40 lines) raised from `fit()`; no stored failure state, no plot guards (once `fit()` returns a `FitResult`, a failed fit simply has no result). [A §A row 6]

**R3. Period-locked candidate model comparison** — pr44, pr45, pr46, pr47 (+ docs pr48, script pr49, pr75 "legacy" rewrite). Size ~1.7k / ~530 / ~600.
Where: `compare_wavelength_candidate_models`, `compute_wavelength_residual_diagnostics`, `interpret_wavelength_model_comparison`, `format_wavelength_diagnostics_report`, `plot_wavelength_diagnostics`, `plot_wavelength_model_comparison`, `Lightcurve.compare_wavelength_models`.
What it does: deep-copies the Lightcurve, fits each candidate, computes training-point residuals, standardised residuals, 1σ/2σ coverage and Gaussian NLPD overall and per band, ranks by lowest training-point mean NLPD.
Why reduce: in-sample GP NLPD rewards over-fitting; the interpretation/report/plot layers triple the size.
Survives as: phase 3.11 `compare_models(lc, models=[...], **fit_kwargs)` (~350 / ~200): same loop; reports exact log-ML (data term and prior term separated, using the pr115 `_piwd_compute_retained_state_marginal_likelihood` idea as a ~40-line `FitResult.log_marginal_likelihood()`), number of hyper-parameters, per-band standardised-residual mean/RMS, 2σ coverage, `parameters_at_bounds`; ranks by log-ML with an explicit caveat; optional `holdout="time_blocks"` NLPD. Later (phase 4.7) Laplace evidence. [B §A rows 2, 5b]

**R4. Pre-fit wavelength diagnostics and classifier** — pr41, pr42, pr43 (period-locked), pr56, pr93 (period-independent), pr116 (semantics fix). Size ~1.5k / ~530 / ~120.
Where: `diagnose_wavelength_dependence_prefit`, `classify_wavelength_diagnostics`, `Lightcurve.diagnose_wavelength_dependence(period=…)`; `diagnose_period_independent_wavelength_structure`; duplicate `wavelength_estimation._build_band_diagnostics`.
What it does: per band, robust quantiles, MAD scatter, q05–q95 half-amplitude, noise-corrected scatter/amplitude √max(0, s² − (k·σ_med)²), sampling and variability flags; across bands, max/min ratios, log-log slopes vs λ, monotonicity labels. With a period: weighted LSQ sinusoid `y = c0 + a cos θ + b sin θ` per band → amplitude, phase, lag, delta-method errors. Classifier: achromatic if amplitude ratio ≤ 1.25; wavelength-modulated if ≥ 1.5 (power-law-like if monotonic with |slope| ≥ 0.5); possible lag if circular lag span ≥ 0.10 P; maps labels to candidate models.
Why reduce: two parallel implementations of the per-band statistics; pr42/43 used a different reference epoch per band so cross-band lags were meaningless until pr116; bare ratio thresholds ignore the amplitude uncertainties the code already computes.
Survives as: phase 3.10 `wavelength_summary(lc, period=None)` (~400 / ~250): one table merging both; pr116's common-t0 lag computation; classification by χ² test of constant amplitude against the delta-method errors rather than fixed ratios; one plot. [B §A rows 1, 3, 5d, §B]

**R5. SM ARD bound-proximity diagnostics** — pr90 (ceiling only), pr125 (per component × dimension × side). Size ~950 / ~370 / ~150.
Where: `pgmuvi/spectral_mixture_ard_diagnostics.py` (724), `diagnose_spectral_mixture_ard`.
Survives as: phase 3.11 `Lightcurve.parameters_at_bounds(tol=0.05)` (~100 / ~80) over every constrained parameter, with per-ARD-dimension labels for SM. Keep the empirical caveat from D9: at-bound ARD parameters are common and do not by themselves indicate failure. [B §A row 9]

**R6. Wavelength estimation context** — pr121, pr122, pr123; fixes pr130, pr135. Size ~1.1k module + ~450 builders/context + ~400 `gps.py` + ~330 `lightcurve.py` / ~900 tests.
Where: `pgmuvi/wavelength_estimation.py` (1,156), `build_wavelength_estimation_context`, `build_wavelength_mean_estimation_context`, `PowerLawMean/DustMean.configure_wavelength_coordinate`, `physical_wavelength`.
Survives as: three things. (a) **Bug fix, phase 1.6:** `PowerLawMean.forward` and `DustMean.forward` evaluate λ^p on the transformed wavelength column, which can be ≤ 0 after centring or min-max (NaN, or silently clamped to 1e-6 in Dust); reconstruct physical λ through the affine x-transform and reject non-affine transforms (~100 lines + tests). (b) Wavelength length-scale initialisation `√(median spacing · span)` with bounds, and grid-profile initialisation of power-law/dust/quadratic means on per-band medians (phase 3.3, ~250 lines). (c) Nothing of the context dataclasses or provenance records. [B §A row 7, §B.3–B.5]

**R7. Dimension-aware 2D SM ARD bounds** — pr124. Size ~360 module + ~1,000 `lightcurve.py` (mostly consensus plumbing) + ~200 builders / ~450 tests.
Where: `pgmuvi/spectral_mixture_ard.py` (363), `build_dimension_aware_sm_ard_estimates`.
Bug on main: `set_default_constraints` applies one scalar `Interval(1/T_span, Nyquist_time)` to both ARD columns of `mixture_means` (the comment at main `lightcurve.py:3876` admits it), so the wavelength "frequency" cannot approach 0 (achromatic) and is bounded by the time sampling.
Survives as: phase 1.7 (~200–300 lines): tensor bounds of shape (1,1,2), index 0 = time, index 1 = wavelength, likewise for `mixture_scales`; consensus touches only index 0; option to pin μ_λ = 0 (science report §2 argues a cosine along wavelength is unphysical for broad bands). [B §A row 8]

**R8. Duplicate-wavelength observational channels** — pr166 (#261), pr167 (#262), pr172 (#269). Size ~850 / ~1.25k / ~200.
Where: `fit(duplicate_wavelength_policy="error"|"first"|"select"|"all", duplicate_wavelength_selection=…)`, `lc.copy_with_duplicate_wavelength_channels()`, `observational_channel_labels`, `duplicate_physical_wavelength_multiplets`.
What it does: detects bands sharing a physical wavelength (two surveys' V); resolves by policy; pr167 makes temporal consensus iterate over all channels since period estimation needs no cross-calibration; pr172 generalises from pairs to multiplets and flips the default from `first` to `error`.
Survives as: phase 3.7 (~300 / ~300): policies `error` (default), `first`, `select` in preprocessing; consensus over all channels; `all` enabled by phase 3.9 calibration. No provenance dicts. [A §A row 17]

**R9. Consensus LS–ACF cross-check** — pr3 (#158) `use_acf`, pr129 (#224) `_consensus_reconcile_ls_acf_harmonic`. Size ~450 / ~250.
Survives as: ~120-line pure function in phase 3.4 (agree / harmonic / disagree; replace an LS harmonic with the lower ACF fundamental when the ratio is an integer). Keep; LS picking P/2 is a real LPV failure mode. Note D9's finding that the ACF can lock onto a spurious short timescale, so the ACF should corroborate, not override, unless the harmonic ratio is clean. [A §A row 3]

**R10. Consensus → period-kernel handoff** — pr55; pr129 `gps.py` selectors. Size ~500 / ~220.
Survives as: phase 3.6 (~300 / ~200): consensus initialises and bounds `period_length = 1/f` for quasi-periodic separable models; `TwoDSeparable(time_kernel_type=…, wavelength_kernel_type=…)` string selectors. [A §A row 16]

**R11. Learned additional noise** — pr38, pr39 (same commit twice), pr173 (#270, default on, init 10 % of median σ²). Size ~250 / ~280.
Survives as: phase 3.1 `BandJitterLikelihood` (σ_i² = yerr_i² + s_b²), per band rather than global, because the science experiment shows one global jitter lets bright-band misfit add noise larger than a faint band's whole amplitude. 1D reduces to a single jitter. Default on. [A §A row 13; science §1(v)]

**R12. Time centering** — pr37, pr40. Size ~300 / ~220. Survives as phase 2.1 (transformer) + 3.2 (default on). The pr37 "band LCs inherit parent transforms" part is **not** kept (pr129 had to revert it for consensus because a 2D affine transform cannot apply to 1D band times). [A §A row 12, §C item 3]

**R13. Docs pages** — keep and rewrite to the new API: `howto/wavelength_models.rst` (pr100; genuinely good: mean vs covariance, physical-λ requirement, model-family table, "what to try first"), `overview.rst`, `howto/first_workflow.rst`, `examples/first_pgmuvi_workflow.py` (pr97), `howto/gp_prior_sampling.rst` (pr107), `howto/consensus_fitting.rst` (pr99, trimmed). Drop: advisory, batch, legacy, interpretation, walkthrough pages (pr73, pr75, pr95, pr101, pr102, pr106) and their example scripts. Redo rather than port: pr103–105 notebook refreshes (2D consensus, preprocessing, synthetic) and pr108's deletion of the MCMC notebook. [B §D]

**R14. CIT 6 dataset and notebooks** — pr134 (`examples/data/10131+3049.csv`, 10,816 rows, 1.1 MB, 17 channels / 16 wavelengths: ZTF, KELT×2, ASAS-SN, Gaia, PS1, CTIO JHK, IRTF L; 9.3k rows are KELT; no provenance recorded), #177 (`sundarjhu-multiband-consensus-notebook`, best narrative with real outputs, but installs from `integration-consensus-test`, monkey-patches a private method, depends on unmerged consensus code), pr168 (2D tutorial executes fit and plot).
Survives as: phase 3.12: dataset thinned to ~1–2k rows with a provenance README (survey citations, column definitions); one real-data notebook rebuilt from #177 on the public API with stored outputs; mock-data and 2D tutorials refreshed; `nbsphinx_execute="never"` for the expensive ones so CI is honest. [C §B, §D]

### 5.3 What is ported essentially as-is (for contrast)
Transform/uncertainty fixes (pr114, pr37 hunks); float64 (pr36, pr110); constraint integrity helpers `constraint_utils.py` (pr32–34, tests pr31); trainer best-state restore and non-finite checks (pr39 `1ea6d27`, pr53 `a667dcb`); physical-λ mean fix (pr123 `gps.py` part); CI/docs config (pr82, 83, 84, 86, 87, 88, 111, 163); stale-test removal (pr67); PR #151; PR #67 (into `plotting/gif.py`). Each is a small independent PR in phase 1 or 2.

## 6. Review checklist for the maintainers (superseded by section 7; kept as a tick-list)
- [ ] D2: is fit-history provenance needed for the NESS paper's reproducibility statement?
- [ ] D7: does anyone run the advisory batch CLI on a catalogue today?
- [ ] D9: re-run the robustness grid on the new default model (recommended) vs record only?
- [ ] D10: calibration as preprocessing now, in-GP later (recommended) vs in-GP first?
- [ ] R1: promote multi-component consensus from optional if multi-periodic LPVs matter?
- [ ] R11: per-band jitter default on?
- [ ] R12: time centering default on? Subsampling-on-load default off?
- [ ] R14: which surveys' terms allow redistributing the CIT 6 rows; who writes the provenance README?
- [ ] MCMC notebook: keep the referee answer open until phase 4.8 rather than accept pr108's deletion?
- [ ] Section 3 items 1, 3, 4, 6 (archive-and-close; FitResult; pytest; phase 4 order).

---

## 7. Decisions in detail

Each entry: the question, the options, what the evidence says, my recommendation, and what changes in the plan depending on the answer. Decisions are ordered by how much of the plan depends on them, not by difficulty. "Blocking" means phase 1 cannot start without it.

### 7.1 Scope and sequencing

**DEC-1. Archive and close the stack PRs wholesale? (blocking)**
Options: (a) tag the tip and the giant-PR backups, close all 99 stack PRs and #152/#154 with a comment linking the roadmap, leave the branches; (b) close only the PRs this plan drops, keep the ~20 whose content survives open "for reference" and close them one by one as replacements land; (c) try to rebase and merge a subset as-is.
Evidence: no PR in the stack stands alone (every one is diffed against its predecessor, and several do not even descend from their nominal parent: pr2–4, pr16, pr39, pr172). The per-PR content is cumulative, so (c) means merging everything below the chosen PR. (b) keeps 20 PRs open whose diffs are against branches we are not merging, which is confusing for anyone who lands on the repo and gives GitHub's "mergeable" flag no meaning.
Recommendation: (a). Also close #110/#136 (numpy 2 verified clean), #265/#266 (private data), #267/#268 (fold into one private task), #28 (already a Module; superseded by DEC-3), #152 (absorbed by 3.3), #154 (superseded). Leave #151, #67, #177 open until their replacements merge (1.11, 2.4, 3.12).
Consequence: phase 0 is one afternoon. The archive tag names (`archive/stack-2026-07`, `archive/giant-pr`) go in the roadmap so the drop-list pointers in section 5 keep working.
Decision: _open_

**DEC-2. What is the minimum for the JOSS resubmission, and does it come before the reorganisation? (blocking)**
Context: the paper is under review. The referee (#45) asked for (1) a real multi-wavelength example, (2) a quasi-periodic / uneven-sampling example, (3) a GP with a deterministic mean, (4) mock data from a kernel, and for the MCMC tutorial to work. Items 2–4 are already on main (#141, #142, the quasi-periodic notebook). Item 1 is #138, open. MCMC is broken on a fresh install (arviz 1.x) and `mcmc()` raises NotImplementedError at the stack tip.
Options: (a) "JOSS first": phase 1.1–1.5 (hygiene, pins, harness, the transform/uncertainty fixes), 1.11 (#151), then 3.12 (CIT 6 dataset + real-data notebook) on the *current* API, and an explicit answer on MCMC; resubmit; then the rest. (b) "Reorganise first": phases 1–2 then 3.12 on the new API, resubmit later. (c) Both in parallel on separate branches.
Evidence: a real-data notebook written against today's API will need rewriting after DEC-3, but it is ~1 notebook; the resubmission has already waited since 2023 (#45). The referee's MCMC comment can be answered honestly either way ("MAP + Laplace now, NUTS in a later release"), but only if Laplace (4.7) or a working `mcmc()` exists. Running the real-data fit on main also exercises the bugs in 1.5–1.7, which is exactly the check the plan wants before the moves.
Recommendation: (a), with the notebook deliberately written to the public API only (no private attributes), so the DEC-3 rewrite is a search-and-replace. Pull 4.7 (Laplace) forward to right after 2.11 so the paper can say "uncertainties from the Laplace approximation" rather than promising NUTS.
Consequence: the phase order becomes 0 → 1.1–1.5, 1.11 → 3.12 (on old API) → resubmit → rest of 1 → 2 → 3 → 4 (with 4.7 early) → 5. The paper text gets a sentence on the new default 2D model only if 4.1–4.2 land before acceptance; otherwise it stays as the current description.
Decision: _open_

**DEC-3. `FitResult`, and `Lightcurve` stops being a `gpytorch.Module`? (shapes phases 2–5)**
Options: (a) as planned: `fit()` returns a `FitResult` (a `torch.nn.Module` holding model, likelihood, transforms, `ModelSpec`, training history) and stores it as `lc.last_fit`; façade methods on `Lightcurve` delegate to it for one release with a `DeprecationWarning`; `Lightcurve` becomes a plain class. (b) Keep `Lightcurve` as the single object; just move code into helper modules and leave model state on the Lightcurve. (c) Introduce `FitResult` but keep `Lightcurve` a `gpytorch.Module`.
Evidence: on main, six name-mangled flags plus five model-spec attributes decide what `plot`, `mcmc`, `summary` and `acf` do; they are never reset when `select_bands`/`drop_bands` discards the model, so constraints are not re-applied and `__FITTED_MAP` goes stale; you cannot hold two fits of the same data (model comparison in 3.11 and the benchmark in 4.4 both need that); save/load (#12) needs a single object that owns model + transforms + spec; `train()` currently treats the Lightcurve as the model because of the Module inheritance. Six test files read `_Lightcurve__FLAGS` directly and will need the alias shims either way.
Recommendation: (a). It is the one API change users notice, so it should happen once, early (end of phase 2), behind shims, and before any new kernel is written against the old structure.
Consequence if (b): 2.11–2.12 are replaced by "clear the flags on model rebuild" (~50 lines), 3.11's `compare_models` must deep-copy the Lightcurve per candidate (as the stack did), 4.4 and 5.1 become awkward, save/load serialises the whole Lightcurve. If (c): same as (a) but `train()` keeps its special-casing. (a) is also the answer to issue #28.
Decision: _open_

**DEC-4. Test runner: pytest? (blocking for 1.3)**
Options: (a) pytest with `-n auto`, a `slow` marker, pgmuvi `DeprecationWarning` → error, existing `unittest.TestCase` classes run unchanged; (b) stay on `unittest` + tox.
Evidence: the suite is 1,292 tests in 42 s; the slow files (`test_fit_ls_return_full` 31 s) dominate wall time and would parallelise; pytest fixtures make the seeded-RNG and "no PNGs in CWD" checks one line each; the copilot-instructions file says "not pytest", which is why the stack converted a pytest file *back* to unittest (#154's last commit).
Recommendation: (a). Update `.github/copilot-instructions.md` and CONTRIBUTING in the same PR so AI-authored PRs stop fighting it.
Consequence: none on the rest of the plan; (b) just makes 1.3/1.4 slower to write.
Decision: _open_

**DEC-5. Review protocol and PR size (shapes everything)**
Context: most of the stack was AI-authored and merged into integration branches without a human reviewing every line, which is how 22k-line "validation of the validation" layers accumulated. The plan assumes ~53 PRs; someone has to read them.
Options: (a) every PR < ~800 code lines, one concern, must carry tests and (for behaviour changes) a before/after on one real light curve; one named human reviewer per PR; AI-authored PRs may not merge on AI review alone. (b) Larger PRs per phase (~5–8 total) reviewed as a whole. (c) No rule.
Evidence: the archived stack is the result of (c). (b) reproduces the giant-PR problem at smaller scale.
Recommendation: (a), written into CONTRIBUTING and the PR template (phase 1.2). Mechanical-move PRs (phase 2) can be larger if the diff is `git mv` plus imports and the golden tests pass.
Consequence: the per-phase PR counts in section 4 stand; if (b), collapse phases into ~8 PRs and accept slower review.
Decision: _open_

### 7.2 Behaviour defaults
All of these change numbers users get from existing scripts. Each is a separate PR with a changelog line and a `FutureWarning` for one release where feasible.

**DEC-6. float64 by default**
Options: (a) float64 everywhere (inputs, model, transforms); (b) keep float32 with an opt-in; (c) float64 for inputs, float32 model.
Evidence: MJDs near 6×10⁴ have ~4×10⁻³ d resolution in float32, which is 1 % of a 0.4 d period and also hurts Cholesky conditioning; the science experiment and the stack's pr36 both found float64 necessary; GPU memory is not the bottleneck at N ≲ 10⁴.
Recommendation: (a), with `dtype=` on the constructor for the rare GPU catalogue run (5.1) that wants float32.
Consequence: 1.8 lands early; golden snapshots in 1.4 are generated in float64 from the start.
Decision: _open_

**DEC-7. Per-band learned jitter on by default (replaces the stack's global `learn_additional_noise`)**
Options: (a) on, per band (σ_i² = yerr_i² + s_b², one s_b per band; 1D reduces to one); (b) on, one global s (the stack's pr173); (c) off by default.
Evidence: astronomical error bars are almost always underestimated, so some jitter is standard practice (pr38/pr173 argued this); but a single global s lets misfit in a bright band add noise larger than a faint band's entire amplitude, which is one of the two causes of #108 found in the experiment. Cost: n_b extra parameters, well-identified when each band has ≳ 10 points. Risk: with very few points per band s_b can absorb the signal; mitigate with a half-normal prior scaled to the median error.
Recommendation: (a), prior included, `likelihood="fixed"` to turn it off.
Consequence: 3.1 precedes 4.1–4.2; the benchmark (4.4) includes a "jitter absorbs signal" case.
Decision: _open_

**DEC-8. Time centering on by default**
Options: (a) on, `center_time="auto"` (midpoint of the data span); (b) off, opt-in.
Evidence: improves conditioning of linear/quadratic means and float precision; durations and frequencies unchanged; only `xdata_transformed` and any user code that reads it changes. The stack's attempt to make band light curves inherit the parent transform had to be reverted (pr129) because a 2D affine transform does not apply to 1D band times, so the per-band path must build its own.
Recommendation: (a); 2.1 implements, 3.2 flips the default.
Consequence: tests that compare `xdata_transformed` numerically are updated in 3.2.
Decision: _open_

**DEC-9. Subsampling on load: off by default**
Context: `Lightcurve.__init__` currently subsamples to `max_samples=1000` by default (500 per band for 2D after #118) and silently discards the rest; the original motivation was O(N³) exact-GP cost.
Options: (a) off by default; warn when N > ~3,000 that exact GP will be slow and point at `subsample()`/KISS; (b) keep on; (c) keep on but raise the cap.
Evidence: a user loading a 5,000-point ASAS-SN light curve today loses 80 % of it without an error; period recovery degrades with sparse sampling (D9 finding). Exact GP at N = 5,000 in float64 is ~1 s per iteration on CPU, which is tolerable; N = 20,000 is not, but that user should be told, not silently truncated.
Recommendation: (a). Keep `subsample()` as an explicit method (with the per-channel stratified option from R12 if wanted).
Consequence: 2.6; the sampling-metrics warning text is written in the same PR.
Decision: _open_

**DEC-10. Duplicate-wavelength policy default `error`**
Options: `error` (refuse to fit two channels at the same physical wavelength until the user picks or calibrates); `first` (keep the first channel silently); `select` (user-chosen).
Evidence: pr166 chose `first`, pr172 changed it to `error` after finding `first` silently dropped the better-sampled channel. Silent data loss again.
Recommendation: `error`, with the message naming the channels and the three remedies (`select`, `calibrate_channels()`, `merge(on_conflict="accept")` once 3.8 lands).
Consequence: 3.7.
Decision: _open_

**DEC-11. The default 2D model changes (phase 4.2) and what the name `"2D"` means afterwards**
Options: (a) `model="2D"` keeps meaning the joint SM-ARD kernel forever; the new default gets a new name (`"2DModulated"`) and `fit()`'s default `model=` argument switches to it with a `FutureWarning` one release ahead; (b) silently redefine `"2D"`; (c) never change the default.
Evidence: the experiment shows the joint SM-ARD kernel with a single constant mean is the worst option for faint bands (gap σ 30–100× amplitude) and puts a damped cosine along wavelength that has no physical meaning for broad bands; the modulated separable model with per-band mean and jitter is better on every metric except calibration in gaps, where it is still 2× better than main.
Recommendation: (a). Also expose `"2DAchromatic"` as the explicit null model so `compare_models()` has a baseline.
Consequence: 4.2 ships the model under its own name; the default flips one release later; the paper's description of the default model is updated only if that release precedes acceptance (see DEC-2).
Decision: _open_

### 7.3 Science choices

**DEC-12. Calibration: preprocessing now, in-GP later?**
Options: (a) `calibrate_channels()` as preprocessing: fit `ref = a + b·chan` on time-paired points, apply, propagate coefficient uncertainty into the error bars, then fit the GP; (b) per-channel scale/offset as GP hyper-parameters (a per-channel `ScaleKernel` factor and per-channel mean offset, which 4.1/4.2 almost give for free), fitted jointly; (c) both, (a) first.
Evidence: (a) is ~400 lines, already validated on KELT R3 (scale 0.68), needs time-overlapping pairs; (b) is more principled (no pairing tolerance, uses all data, uncertainty comes out of the posterior) and once `BandOffsetMean` + `BandModulatedKernel` exist the offset and scale are literally the per-band mean and amplitude parameters with the band index split by channel rather than wavelength. But (b) only identifies the scale if the channels share the variability, i.e. after the model is right, and it cannot be used to *check* a calibration independently.
Recommendation: (c): 3.9 as planned, and a note in 4.2 that channel-level (not wavelength-level) knots make the GP absorb offsets; `compare_models()` then compares "calibrated first" vs "joint" on CIT 6.
Consequence: none on ordering; 4.2 gains a `knots="band"|"channel"` switch.
Decision: _open_

**DEC-13. Phase 4 order: per-band mean → amplitude modulation → initialisation → benchmark → chromatic SM?**
Options: (a) as listed; (b) chromatic SM first (highest science value, answers the paper's "phase lags across wavelength" motivation directly); (c) initialisation and benchmark first, then both kernels.
Evidence: the experiment ranks the chromatic SM best by evidence but overconfident under ML-II (2σ coverage 0.12–0.24 without the incoherent term), with Q(2n_b−1) extra parameters that need the per-band LS amplitude/phase initialisation (4.3) to converge, and Laplace/NUTS (4.7/4.8) to be honest about uncertainty. The per-band mean + modulation (4.1–4.2) fixes most of #108 with n_b parameters and needs nothing new. The benchmark (4.4) is what decides whether the chromatic SM becomes a default or stays an option.
Recommendation: (a), but build the synthetic benchmark generator (the data half of 4.4) alongside 4.2 so each kernel PR comes with its numbers. 4.5 before 4.6 (ICM), since ICM cannot predict at unobserved wavelengths and is the fallback for ≲ 10 discrete bands only.
Consequence: if (b), 4.3 and the incoherent term must land in the same PR as the kernel, and 4.7 moves before it.
Decision: _open_

**DEC-14. Multi-component consensus: optional (3.14) or promoted?**
Evidence: R1. The stack's version shares one frequency interval across components, which is the thing that makes multi-component useful or not. Multi-periodic LPVs (e.g. long secondary periods) are real, and the NESS sample will contain them, but the single-component consensus plus a second SM component initialised from the second MLS peak (which 4.3 gives) covers the common case.
Recommendation: keep optional; revisit after 4.3 with the CIT 6 data, since 4.3's multi-restart over the top-K peaks may make a separate "consensus_multicomp" strategy unnecessary.
Consequence: none unless promoted, in which case it becomes 3.6b (~300 lines) with per-component bounds.
Decision: _open_

**DEC-15. Re-run the robustness grid (D9) on the new default, or just record its findings?**
Recommendation: record now (3.13), re-run as part of 4.4 on both the old `"2D"` and the new default, and publish the comparison in the docs. The recorded D2 boundaries (uneven per-band counts, sparse long baselines, large wavelength gaps) are precisely the regimes where amplitude modulation and per-band jitter should help, so this is the cheapest way to show the new model is better on something other than a toy.
Decision: _open_

**DEC-16. MCMC: fix, replace, or defer, and what to tell the referee**
Options: (a) rewrite `mcmc()` as a clean pyro NUTS model over raw parameters initialised at MAP with a Laplace mass matrix (4.8), restore the notebook; (b) drop NUTS, ship Laplace (4.7) and say so; (c) emcee over raw parameters as a simpler, dependency-light sampler.
Evidence: today `mcmc()` prints the whole `state_dict` every step, hard-codes SM parameter names, crashes without flux errors (#33), and breaks on arviz 1.x; the stack deleted the notebook (pr108) rather than fix it. Laplace is ~150 lines and gives the paper a defensible uncertainty statement; NUTS over 20–40 constrained hyper-parameters with an O(N³) likelihood is slow but feasible for N ≲ 500, which is the LPV regime.
Recommendation: (b) for the resubmission, (a) afterwards, with the notebook restored only when it runs in CI under a `slow` marker. Do not accept pr108's silent deletion; replace the notebook with a short page saying what is and is not available.
Consequence: 4.7 moves to right after 2.11 (see DEC-2); 4.8 stays in phase 4; the `[mcmc]` extra (1.1) pins what 4.8 will need.
Decision: _open_

### 7.4 Data, docs, process

**DEC-17. CIT 6 dataset: redistribution and provenance**
Context: pr134 added `examples/data/10131+3049.csv` (17 channels; ZTF, KELT×2, ASAS-SN, Gaia, PS1, CTIO JHK, IRTF L; 9.3k of 10.8k rows are KELT) with no provenance. The notebook in #177 uses the same source.
Decision needed: which surveys' data policies allow redistribution of extracted light curves in the repo (ZTF, ASAS-SN, Gaia, PS1 generally yes with citation; KELT and the IRTF/CTIO points may be from a paper table that needs its own citation), and who writes the README with column definitions, survey DOIs and the extraction date.
Recommendation: thin to ~1.5k rows (keep all non-KELT rows, subsample KELT), ship under `pgmuvi/datasets/` so `pgmuvi.datasets.load("CIT6")` works, with a README. If any survey's terms are unclear, keep that channel out and document how to fetch it.
Consequence: 3.12 cannot start until this is answered; everything else is unaffected.
Decision: _open_

**DEC-18. Fit-history provenance (D2): needed for the paper?**
Recommendation: no separate feature. `FitResult.info` records kwargs, package versions, git describe of pgmuvi, final loss, iteration count and timestamp (~40 lines inside 2.11); `FitResult.save()` persists it. If the NESS paper needs a reproducibility statement, that is sufficient and is the same thing the stack's 2.3k lines did.
Decision: _open_

**DEC-19. Advisory batch CLI (D7): does anyone depend on it?**
Recommendation: replace with `fit_catalogue()` (5.1) when the time comes; do not port. If a collaborator is running `run_wavelength_advisory_batch.py` on the archived branch today, pull 5.1's non-batched core (a loop over sources with checkpointing and a CSV summary, ~150 lines) forward into phase 3 as 3.15.
Decision: _open_

**DEC-20. Where the roadmap lives and how progress is tracked**
Options: (a) `docs/dev/roadmap.md` in the repo, updated by each PR that completes an item, plus a GitHub Project board with one card per PR; (b) a GitHub Discussion or wiki page; (c) outside the repo.
Recommendation: (a). The roadmap PR is itself phase 0.3 and the first small PR of the new stack. Keep section 5 (the drop list) in the repo copy so the archive pointers are discoverable.
Decision: _open_

**DEC-21. Versioning across the refactor**
Options: (a) cut `v0.x` from today's main before phase 1 ("last release of the old layout"), do phases 1–2 as `0.(x+1)` with shims, remove shims and ship the new default 2D model as `1.0`; (b) no tag now, release at the end of phase 2.
Evidence: setuptools_scm versions from tags; users installing from PyPI today get whatever the last tag was; a tag now gives them something to pin while the shims churn; JOSS archives a tagged release at acceptance.
Recommendation: (a): tag `v0.x` after 1.1 (hygiene/pins), so the pinned release at least installs cleanly; tag the JOSS release after 3.12; `1.0` after phase 4.2 lands as default.
Decision: _open_

**DEC-22. Copilot/Claude instructions file**
Context: `.github/copilot-instructions.md` currently says "not pytest", "prefer small changes over large refactors", and describes the old layout; the stack's agents followed it literally while generating 80k lines.
Recommendation: rewrite in phase 1.2 to state the new layout, the PR size rule (DEC-5), "science in pure functions outside `lightcurve.py`", "no contract/implement/validate triplets", and "every behaviour change needs a before/after on a real light curve". This is cheap and is the single most effective guard against the stack recurring.
Decision: _open_

### 7.5 Suggested outcome, as one list
If you take every recommendation above, the plan changes from v1 as follows:
1. Phase 0 adds the archive tags and the issue closures (DEC-1), a `v0.x` tag after 1.1 (DEC-21), and the roadmap PR (DEC-20).
2. A "JOSS track" is inserted: 1.1–1.5, 1.11, then 3.12 on the old public API, then resubmit (DEC-2); Laplace (4.7) moves to immediately after 2.11 (DEC-16).
3. 1.2 also rewrites the copilot instructions and adds the review rule to CONTRIBUTING (DEC-5, DEC-22).
4. Defaults: float64, per-band jitter with prior, time centering, subsampling off, duplicate policy `error` (DEC-6–10); the new 2D model ships under its own name and becomes the default one release later (DEC-11).
5. 4.2 gains `knots="band"|"channel"` (DEC-12); the benchmark generator is built alongside 4.2 and the D2 grid is re-run inside 4.4 (DEC-13, DEC-15).
6. Multi-component consensus stays optional pending 4.3 (DEC-14); provenance and the batch CLI are not ported (DEC-18, DEC-19).
7. 3.12 waits on the CIT 6 provenance answer (DEC-17).
