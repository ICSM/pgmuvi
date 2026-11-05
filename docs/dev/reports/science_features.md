# Science & performance feature research (agent report, saved by orchestrator)
See the orchestrator's conversation for the full text. Key points:

## Experiment on issue #108 (../experiments/amp_test3.py, scratch venv: gpytorch 1.15.2, torch 2.14.1+cpu)
Toy LPV: P=400 d, 2000 d baseline, 5 bands 0.55-4.6 um, amplitudes 3/1/0.4/0.15/0.05, mean 2x amp, 2nd harmonic, 3% noise, 600 d gap in band 3, lags to 0.15 cycles. 3 seeds + zero-lag control.
| model | mean | -logML | band3-gap sigma/amp, RMSE/amp, cov | faintest sigma/amp |
| 2D SM (current '2D') | const | +219..+1002 | 29-100, 0.7-10, 1.0 | 3-30 |
| Separable SM_t x Scale(RBF) | const | +7..+334 | 29-43 | 6-33 |
| 2D SM | per-band | -479..-500 | 0.025-0.029 | 0.03 |
| Separable | per-band | -491..-508 | 0.022-0.025 | 0.027-0.030 |
| AmpMod a(b)a(b') SM_t RBF | per-band | -547..-566 | 0.011-0.013, cov 0.46-0.66 | 0.018-0.022 |
| ICM rank1 | per-band | -517..-545 | 0.021-0.024, cov 0.7-1.0 | 0.021-0.028 |
| ICM rank2 | per-band | -533..-560 | 0.019-0.023 | 0.021-0.025 |
| Chromatic SM (rank-1 CSM) | per-band | -579..-593 | 0.005-0.006, cov 0.12-0.24 | 0.007 |
| Chromatic SM + incoherent | per-band | -570..-592 | 0.010-0.011, cov 0.39-0.47 | 0.010-0.016 |
Findings: (1) single constant mean + poor 2D init is the main cause of #108; (2) global learned extra noise is the second cause -> per-band jitter; (3) amplitude modulation k=a(l)a(l')k_t k_l gives +40-60 logML for n_b params, faint sigma 2x narrower, but overconfident in gaps; ICM diagonal gives best calibration; (4) separable/ICM/LMC/2D-SM cannot represent phase lags (cross-cov symmetric in tau); rank-1 chromatic SM (per-band amp+phase per component; PSD proven) recovered lags 0/-0.13/-0.29/-0.66/-0.93 vs true 0/0.13/0.31/0.63/0.94 and 60x amplitude ratio; overconfident under ML-II -> needs incoherent term + Laplace/NUTS; (5) current TwoDSpectralMixture puts damped cosine along wavelength (meaningless for broad bands); weights not wavelength dependent; no lags; constrain mu_lambda=0.

Bugs: wavelength_scaling='linear' gives a ∝ λ through origin (bluest band zero prior variance with minmax); AchromaticGPModel ignores mean_module; _plot_1d/_plot_2d never inverse-transform y; 2D SM has no data-driven init; data loaded float32 (lightcurve.py:745).

## Options for #108
(i) Amplitude modulation: BandModulatedKernel = D(a1) K D(a2); a(λ) knots at band wavelengths lerp in log λ (default), log-polynomial for many channels, GP/spline later, blackbody derivative as physical option (too rigid for Miras). Identifiability: unscaled k_λ, fix Σw_q=1 or reference band; prior ℓ_λ ≳ band spacing; per-band mean required. Init α_b = ½log(max(robustVar_b − mean σ²_err,b, floor)). O(N) extra.
(ii) per-band z-score/magnitudes: partial (freezes ratios, noise-inflated, needs ≥5 pts); use for init only; offer units='mag'.
(iii) ICM/LMC via IndexKernel(n_b, rank): well calibrated, n_b(r+1) params, no prediction at unobserved λ, best for ≲10 heterogeneous bands; ship as '2DCoregionalised'; band index via bucketize on wavelength keeps [t,λ] API.
(iv) Lagged multi-output SM: CSM (Ulrich+2015 NeurIPS), MOSM (Parra&Tobar 2017 arXiv:1709.01298). Recommended rank-1 chromatic SM: k=Σ_q a_q(λ)a_q(λ') exp(−2π²τ²v_q) cos(2πμ_qτ+φ_q(λ)−φ_q(λ')) [·k_λ,q]; phase free per band+harmonic or pure delay φ=2πμΔt(λ); Q(2n_b−1) params; init from per-band LS amp/phase at MLS peaks; add incoherent term; O(N³); SKI via cos/sin expansion. ~150 lines pure GPyTorch. Highest science value (optical molecular vs IR dust ratios and lags become fitted parameters).
(v) per-band noise: σ_i² = yerr_i² + s_b² as FixedNoiseGaussianLikelihood subclass (HadamardGaussianLikelihood lacks fixed part). S effort, high value.
(vi) time-dependent amplitude a(t)a(t')k / change-point: later.

Kernel sketch: class BandModulatedKernel(Kernel) with knots buffer (log band wavelengths), log_amp parameter (n_b), random-walk NormalPrior(0,2) on log_amp.diff(); amplitude(x) via bucketize+lerp; forward returns DiagLinearOperator(a1) @ K @ DiagLinearOperator(a2) (diag: a1*a2*K). Store knots in transformed units if xtransform rescales λ.
Slots in: SeparableGPModel(..., amplitude='knots'|'loglinear'|'blackbody'|None) with unscaled k_λ; BandOffsetMean default 2D mean (DustMean/PowerLaw + per-band residual optional); BandJitterLikelihood default for likelihood='learn' in 2D; ChromaticSpectralMixtureKernel(phase='free'|'delay') as '2DChromaticSM'; get_amplitude_ratios()/get_phase_lags() with Laplace errors.

## Default 2D model recommendation
BandOffsetMean + BandModulated(SM_t × unscaled RBF over log λ) + BandJitter; move to chromatic SM after benchmarking. Use log λ not minmax on linear λ. Plain separable = baseline; achromatic = null model. Constrain μ_λ=0 in legacy 2D SM.

## Feature survey
Scalability: #149 (n_pred default ~min(2000, 20·baseline/P_min), chunk, predict latent + analytic per-band noise, skip_posterior_variances for mean-only, Lightcurve.predict() with inverse y); float64 default + cholesky_jitter + NotPSD recovery (restore last good, raise jitter, halve lr); SKI along time only (current 2D grid 5000×20=1e5 even for 5 bands) Σ_r D_r(W K_grid Wᵀ)D_r O(r(N+m log m)); batched GPs batch_shape=[S] on GPU with padding (LSST catalogue story); SVGP/VNNGP via LMCVariationalStrategy/NNVariationalStrategy (ELBO currently NotImplemented); celerite/Gordon+2020 cite as related; LOVE cache in eval mode.
Inference: L-BFGS polish after Adam (torch LBFGS strong_wolfe or scipy wrapper à la botorch fit_gpytorch_mll; currently 300 AdamW lr 0.1); multi-restart from top-K MLS peaks + yearly/harmonic aliases; Laplace (Hessian over raw params, delta method to periods/lags/ratios, Laplace evidence); MCMC rewrite as clean pyro model over raw params init at MAP w/ Laplace mass matrix, or emcee, fix #33; nested sampling (dynesty/ultranest) for small N; compare_models() via Laplace/BIC/LOO (LeaveOneOutPseudoLikelihood exists).
SM specifics: data-driven 2D init (per-band LS amp/phase -> weights, a(λ), φ; width ≈1/baseline); constraints μ∈[2/baseline, pseudo-Nyquist], width ∈[1/(10·baseline), μ/2], half-normal/log-normal prior on weights; period drift fix: dedicated red-noise term (Matérn-3/2 or SHO-like, not RBF as in add_flicker), per-band linear trend, lower bound on μ_q; number of components from significant MLS peaks then prune/merge.
Means: per-band offsets default, optional per-band linear trend; DustMean/PowerLaw = static SED + per-band residuals with shrinkage; variability SED belongs in a(λ).
Usability: save/load (state_dict + JSON: model string, kwargs, transform states, band map, data hash, versions, history); FitResult (#8,#27); logging (#11); fit_catalogue() batched w/ checkpointing; SVO lookup via astroquery.svo_fps.SvoFps WavelengthPivot + alias table (#132); astropy Quantity periods (#133).
Validation (#17,#138): extend synthetic.py; grid amp ratio 1-1000, lag 0-0.3 cycles, cadences (seasonal, LSST-like, NESS-like), S/N, cycle jitter; metrics period error/alias rate, amp-ratio & lag error, per-band 68/95 coverage & CRPS in gaps/faint bands, wall time/memory.

## Top 10
1 per-band offset mean (S); 2 per-band jitter (S); 3 BandModulatedKernel + log λ + unscaled k_λ + noise-corrected init (S-M); 4 2D LS init + multi-restart + L-BFGS float64 (S-M); 5 chromatic SM = rank-1 CSM + incoherent (M); 6 Laplace + model comparison (S-M); 7 predict() + chunked plotting + y-inverse fix (S); 8 benchmark suite (M); 9 save/load + FitResult + logging (S-M); 10 batched catalogue fitting then SKI in time (M-L). Also SM constraints + red-noise term; SVO; units; Achromatic mean bug; μ_λ=0.

## References
SM arXiv:1302.4245; CSM NeurIPS 2015 paper 5966; MOSM arXiv:1709.01298 (+2202.09233); Gordon+2020 arXiv:2007.05799; He+2016 arXiv:1609.06680; Álvarez+2012 arXiv:1106.6251; Heinonen+2016 arXiv:1508.04319; KISS-GP arXiv:1503.01057; LOVE arXiv:1803.06058; SVGP arXiv:1309.6835; VNNGP arXiv:2202.01694.
