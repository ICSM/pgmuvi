# Integration notes for adding Torch-based BGLS to PGMUVI

## Files included

- `pgmuvi/bgls.py`
- `docs/BGLS_method_explained.md`
- `docs/bgls_references.bib`
- `docs/methods_section_draft.tex`
- optional patch files:
  - `lightcurve_fit_bgls.patch`
  - `init_bgls.patch`

## Public API in `pgmuvi/bgls.py`

The main public entry points are:

- `extract_lightcurve_arrays(lightcurve, ...)`
- `build_log_frequency_grid(time, ...)`
- `bgls_single_band(time, flux, flux_err=None, ...)`
- `bgls_per_band(xdata, ydata, yerr=None, ...)`
- `combine_bgls_bands(band_results, ...)`
- `fit_bgls(data, ydata=None, yerr=None, ...)`

These are the only functions that other parts of PGMUVI should call directly.

## Private helpers

All functions prefixed by `_` are private implementation details.  This keeps the integration surface narrow and reduces coupling to the rest of PGMUVI.

## Recommended minimal integration

The cleanest integration route is:

1. add `pgmuvi/bgls.py`,
2. optionally re-export it from `pgmuvi/__init__.py`,
3. optionally add a thin wrapper method `Lightcurve.fit_BGLS()` that simply forwards arrays to `pgmuvi.bgls.fit_bgls()`.

That wrapper should stay thin.  The BGLS logic should not be duplicated inside `lightcurve.py`.

## Example wrapper logic

```python
from .bgls import fit_bgls as _fit_bgls

def fit_BGLS(self, **kwargs):
    return _fit_bgls(self, **kwargs)
```

This is enough because `fit_bgls()` already knows how to extract `xdata`, `ydata`, and `yerr` from a `Lightcurve` object.

## Statistical justification

### Single-band case

The single-band code implements the weighted floating-mean sinusoid model of Zechmeister & Kürster (2009) and the Bayesian GLS interpretation of Mortier et al. (2015).

### Multiband case

The multiband code assumes a **shared frequency across bands**, in the same general spirit as multiband LS approaches such as VanderPlas & Ivezić (2015), but it does **not** implement that paper's exact multiband Fourier model.

Instead, it uses a tempered composite likelihood:

\[
\log p(f \mid D_{1:K})
=
\log p(f)
+
T \sum_k w_k \log p(D_k \mid f)
+
C.
\]

This is motivated by Lindsay (1988), Varin et al. (2011), and Ribatet et al. (2012).

### Why the calibration matters

Without calibration, the product across bands can become too sharp if evidence is duplicated across correlated bands.  The default leave-one-band-out calibration is the fastest safeguard in this module.

## Defaults chosen for speed

### Default weight method: B

Reason:
- cheaper than curvature-based option C,
- uses inverse-variance information and a cheap effective-sample-size heuristic,
- therefore fast enough to be the default.

### Default calibration: leave-one-band-out

Reason:
- scales like \(O(K)\) in the number of bands after the per-band BGLS curves exist,
- directly tests cross-band redundancy,
- much faster than repeated bootstrap recalculation.

## Performance notes

### CPU vs GPU

The implementation is written in PyTorch and will run on CPU by default.  If you pass `device=torch.device("cuda")` and the arrays live on the GPU, the trigonometric and linear-algebra steps can accelerate substantially for large frequency grids and multiple bands.

### Chunking

The single-band evaluator processes frequencies in chunks to keep memory under control.  This matters because the trigonometric arrays scale like:

\[
N_{\mathrm{freq}} \times N_{\mathrm{time}}.
\]

### When option C is acceptable

Option C is slower than option B but often still practical because it reuses already-computed per-band BGLS results.

### When the resampling calibrations become expensive

Block bootstrap and nested band/time resampling require repeated recalculation of BGLS curves and will dominate runtime for large problems.  They are best reserved for final diagnostics rather than exploratory runs.

## Comparison with LS-based workflows

To compare with LS in a statistically coherent way:

- compare the **combined BGLS posterior over frequency** with a **multiband LS periodogram**,
- do not compare posterior values directly to LS powers,
- compare peak locations, ranking, and structure instead.

## Assumptions to document in code or papers

Use wording like this:

> The per-band BGLS curves assume a weighted sinusoid-plus-offset model with Gaussian errors and a shared trial frequency across bands.  The multiband combination is performed via a tempered composite likelihood, which approximates a joint posterior over frequency but does not explicitly model cross-band covariance.

## Suggested tests after integration

1. single-band synthetic sinusoid recovery,
2. multiband shared-period recovery with different amplitudes/phases,
3. sensitivity check under omission of one band,
4. check that option B and option C give similar dominant peaks on a clean case,
5. check that `fit_BGLS()` accepts both 1D and 2D `Lightcurve` objects.

