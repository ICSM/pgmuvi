# Architecture review of pgmuvi main (agent report, saved by orchestrator)

## Headline
- lightcurve.py 11,269 lines; Lightcurve = lines 1706-end, ~9,560 lines, 127 methods.
- ~3,500 lines pure/near-pure and movable with no behaviour change: result classes (~860), plot_period_summary (538, pure), 15 PSD/peak static helpers (~700), transforms (~200), merge validation helpers, sampling/variability wrappers.
- Hard core ~2,200 lines (set_model, priors, constraints, hypers, fit) mutating self.model, self._model_pars, six name-mangled __FLAGS.

## Structure (group | methods | lines | key methods | private state)
- Transforms (module level) | 16 | ~200 | 158-342
- IO (InputHelpers, from_csv 511 (337 lines), from_table 2230, to_table 10504, to_csv 10587, write_votable) | 10 | ~880
- Result classes | 23 | ~860 | 848-1705 (PeriodSummaryResult.to_text 326 lines)
- Container/validation (__init__ 1753 = 477 lines, properties, select_bands, drop_bands, cpu/cuda) | 20 | ~950 | writes every buffer, band, all six flags
- LS and ACF (fit_LS 4243 = 399 lines, acf 4642) | 8 | ~790
- Preprocess checks incl auto_select_model | 11 | ~600 | 5031-5629
- Model construction (set_model 2854) | 4 | ~400 | three identical dispatch dicts model_dic_1/2/alt; names appear 4x
- Priors/constraints/hypers/parameters (set_period_prior 3412=255, set_default_constraints 3806=236, set_constraint 3091=173, set_hypers 4090, get_parameters 9452) | 13 | ~1,150
- fit (5629) | 1 | 673 lines, ruff complexity 52; ~300 lines are MLS init-frequency selection that should be a pure function in initialization.py; set_default_constraints called twice (6175-6179)
- MCMC (6302) | 4 | ~350
- Period summary | 29 | ~2,280 | _get_sm_period_summary 8278, get_period_summary 8552 (clean dispatcher over 5 backends -> registry), _get_separable_2d_period_summary 7446 (temporarily swaps self.model.sci_kernel at 7528)
- plot_period_summary (8725) | 1 | 538, pure
- PSD | 3 | ~440 | 9565-10004
- Plotting | 8 | ~500 | 10004-10504; _plot_1d/_plot_2d default save=True writing PNGs to CWD; figures never closed; _plot_2d uses xtransform.min (MinMax only -> ZScore/RobustZScore crash); _plot_mcmc 10,000-pt grid x num_samples = issue #149
- Merge/concat (merge 10809=272, concat 11081=189) | 5 | ~580

__init__ ordering: NaN drop, band validation, sampling filter that silently drops bands (warning), variability gate that raises, subsampling ON BY DEFAULT (max_samples=1000) so long light curves lose data on load. Transform map typo "robust_score" (1771).

## Hidden state
- __SET_LIKELIHOOD_CALLED, __SET_MODEL_CALLED, __CONTRAINTS_SET, __PRIORS_SET, __FITTED_MAP, __FITTED_MCMC (1930-1935): never reset when select_bands/drop_bands discard the model; constraints not re-applied; __FITTED_MAP stale. Six test files read _Lightcurve__FLAGS.
- _model_pars (_make_parameter_dict 3005): hand copy of named_parameters; lstrip("raw_") strips chars (latent).
- Model spec spread over _model_str, _model_instance, _model_num_mixtures, _fit_num_mixtures_requested, _fit_num_mixtures_effective -> one ModelSpec (also what save/load needs).
- Transformer buffers set lazily on first transform(), used at 40+ sites.
- sci_kernel inconsistent across KISS classes.
- Loose results attrs: results, inference_data, mcmc_run, post, num_samples, diagnostics; cannot hold two fits.
- self.summary shadows summary() (6550) -> second call fails.
- Leaked temporaries x_fine_transformed, expanded_test_x, value_inversed.

## Environment and tests
- Scratch venv: Python 3.13.13, torch 2.14.1+cpu, gpytorch 1.15.2, linear_operator 0.6.1, numpy 2.5.4, scipy 1.18.1, astropy 8.0.2, pyro 1.9.2, arviz 1.3.0. Import of pgmuvi.lightcurve 3.0 s (1.05 s arviz).
- ruff clean. 756 missing type annotations.
- unittest discover: 1,292 tests, OK, 7 skipped, 37.7 s (42 s wall). Slowest: test_fit_ls_return_full 31.5 s, tests.py 15.3 s (189 tests), test_2d_integration 15 s.
- Gaps: only 5 files call .fit() (1-100 iters); no test checks period recovery; no mcmc tests; no serialisation tests; examples/ never executed.
- Tests use private names heavily (_xdata_raw 27x, _model_pars 25x, _Lightcurve__FLAGS in 6 files) -> every move PR needs aliases.
- Running the suite writes Lightcurve_*_fit.png into repo root.

## Dependencies and packaging
- seaborn unused. pyro/arviz/xarray only for MCMC but eagerly imported -> lazy [mcmc] extra. pandas only in test_script.py. linear_operator imported directly but transitive.
- Nothing pinned. arviz 1.x has no az.from_pyro -> mcmc() breaks at line 6451 on fresh install (after sampling). Pin arviz<1 or port to arviz_base. Add bounds for torch/gpytorch/numpy/astropy (needs LombScargleMultiband). Add [test], [docs] extras (#134).
- numpy 2 clean (only guarded np.trapz at 7670) -> close #110/#136. Bootstrap uses unseeded global np.random (multiband_ls_significance.py:48,87).
- setup.py redundant; include-package-data=false so CSV not in wheel but tests/tests.py:865 reads it; test_script.py shipped with broken imports and hard-coded home path; license text/classifier mismatch; leftover [tool.black].
- __init__ doesn't export Lightcurve. models.py shim emits no DeprecationWarning.
- CI: push only (no pull_request); matrix 3.10-3.15 incl prerelease; no CPU torch index/caching; ruff workflow --fix and auto-commits.
- examples/separable_kernels_2d.py:93 imports nonexistent QuasiPeriodicKernel; kernels.py docstring advertises it.

## Target layout
- lightcurve.py: data + facade, ~1,500 lines. transforms.py, _utils.py.
- io/ (tables.py, serialise.py); preprocess/ (+pipeline.py, combine.py); periodogram/ (lombscargle.py, significance.py, acf.py); models/ (means, kernels, spectral_mixture, alternative, separable, registry, priors, constraints, initialization); inference/ (train.py, mcmc.py); results/ (fit_result.py, period_summary.py, psd.py, text.py); plotting/ (lightcurve, period_summary, psd, acf, mcmc); datasets/.
- Old modules become re-export shims with DeprecationWarning.
- Public API: Lightcurve keeps construction/IO, data properties, select/drop/merge/concat, checks, fit_LS/acf delegations, fit() returning FitResult, plot(). Compatibility facades (get_period_summary, plot_psd, set_prior, set_constraint, print_parameters) delegate to lc.last_fit. FitResult = torch.nn.Module holding model, likelihood, transforms, ModelSpec, history, convergence; owns get_period_summary, compute_psd, plot, save/load, to_text. Free functions periodogram.lomb_scargle, models.get_model(name,...).
- Opinion: Lightcurve should stop being a gpytorch.Module (#28 was a workaround).

## PR sequence (16)
1 [low] Hygiene+pins (rm setup.py, test_script.py; CSV->data/; extras; pin arviz<1; CI on PRs w/ CPU torch; ruff check-only; shim warning)
2 [low] Regression suite + public-surface test BEFORE moves
3 [low] transforms.py (changes qualnames -> pickles)
4 [low] results/ result classes + text
5 [low] results/psd.py static helpers (keep Lightcurve._xxx = staticmethod aliases)
6 [low] plotting/period_summary.py
7 [low] io/tables.py (from_csv rewrite on from_table = separate [med] PR)
8 [low] preprocess/pipeline.py from __init__ stages + combine.py
9 [low/med] periodogram/ (fit_LS has many return modes)
10 [med] split gps.py -> models/ + registry (replaces 3 dicts, _SM_MODELS, docstring lists)
11 [high] collapse 12 SM classes into SpectralMixtureGPModel(mean=, ski=, grid_size=); old names as thin subclasses; preserve/fix quirks (only 1D variants call initialize_from_data; 1D KISS grid from data, 2D KISS default [5000,20]; sci_kernel inconsistent); assert state_dict keys unchanged
12 [med] inference/ (fix train() KeyError, logging #11, drop stubs, lazy MCMC import, remove per-step prints, extract MLS init block from fit)
13 [high] ModelSpec + FitResult; move model state off Lightcurve; replace flags (semantics change: re-fit after band selection) -- only after golden tests
14 [med] save/load (#12)
15 [low] plotting move + #149 fix (save=False, close figs, batch predictions per band, no .min, no temporaries on self)
16 [low] remove shims after one release
Risks: private-name tests; full-object pickles; notebooks reaching into lc.model.covar_module; select_bands/merge discard model -> visible under FitResult.

## Dead code
append_data, init_hypers_from_LombScargle, _set_hypers_raw (pass); _drop_nan_rows (no callers); _plot_nd; plot_results (broken .item()/value_inversed); print_periods/get_periods; trainers.Trainer, train_mll/train_variational*; SparseSpectralMixtureGPModel, CustomLinear/QuadConstantMean; 47 commented-out blocks, 76 print calls.

## Bugs for quick separate PRs
summary() overwrites itself (6550); ZScore/RobustZScore break plotting; PNGs to CWD; "robust_score" typo; mcmc hard-codes SM params + xtransform.inverse without None check; duplicate set_default_constraints in fit; lstrip("raw_"); train() KeyError without Lightcurve; arviz 1.x breaks mcmc; broken example import.

## Duplication
81 .numpy() conversions, three _to_numpy helpers (initialization.py:300, preprocess/variability.py:16, from_csv:743); 35 self.ndim branches (add lc.t, lc.wavelength, iter_bands()); 1D/2D near-copies (print_periods, _plot_1d/_plot_2d, assess_sampling_quality vs _per_band, check_variability vs _per_band); model name list 4x; 12 SM classes ~670 lines gps.py 175-846; fit and set_model repeat likelihood ladder (2930, 5820); find_peaks imported locally 5x under 3 aliases.

## Biggest units (lines, complexity)
fit 673/52; plot_period_summary 538/44; __init__ 477/40; fit_LS 399/23; from_csv 337/37; to_text 326/26; _get_sm_period_summary 274/17; merge 272/28; set_period_prior 255/18; set_default_constraints 236/19; mcmc 206/17; concat 189/18; plot_psd 180; get_period_summary 173; set_constraint 173/28; set_model 151; _get_separable_2d_period_summary 145; PeriodSummaryResult.__init__ 141; _extract_sm_params 140; select_bands 137; write_period_summary_outputs 137. Outside: trainers.train 24, synthetic.make_multi_sinusoid_chromatic_2d 15, preprocess.quality.assess_sampling_quality 15.

## Test strategy
pytest (-n auto) now; runs unittest classes unchanged; DeprecationWarning->error for pgmuvi; slow marker; split tests/tests.py (2,091 lines); default run <60 s CPU; nightly slow.
Minimal regression suite (PR 2): (1) golden period recovery: seeded float64 synthetic 1D P=37 d with 1D, 1DQuasiPeriodic, 1DPeriodicStochastic, within 2%, interval covers truth; 2D chromatic with 2D and 2DSeparable. (2) fixed-hyperparameter snapshots: MLL, PSD on fixed grid, get_period_summary().as_dict() vs JSON. (3) registry snapshots: class name, state_dict keys, param shapes per model name. (4) round-trips CSV/table 1D/2D with bands and NaNs; write_json; FitResult save/load. (5) public-surface + deprecation tests. (6) smoke: plot() all transforms no CWD writes, figures closed; mcmc(num_samples=5) behind importorskip(pyro); examples/ smoke job. Determinism fixture seeding torch/numpy/pyro; rng= for bootstraps.
