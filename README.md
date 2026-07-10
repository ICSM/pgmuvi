# pgmuvi
Python gaussian processes for multiwavelength variability inference

[![Documentation Status](https://readthedocs.org/projects/pgmuvi/badge/?version=latest)](https://pgmuvi.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/387371146.svg)](https://zenodo.org/badge/latestdoi/387371146)


pgmuvi is based on GPyTorch and intended for us in infering the properties of astronomical sources with multiwavelength variability. It uses spectral-mixture kernels to learn an approximation of the PSD of the variability, which have been shown to be very effective for pattern discovery (see https://arxiv.org/pdf/1302.4245.pdf).

## Installation and Quickstart

pgmuvi can be installed easily with pip::

    $ pip install pgmuvi

You can also clone the latest version of pgmuvi from Github, for all the latest bugs but increased risk of features::

    $ git clone git://github.com/ICSM/pgmuvi.git

and then you can install it::

    $ cd pgmuvi
    $ pip install .


If you want to contribute to pgmuvi and develop new features, you might want an *editable* install::

    $ pip install -e .

this way you can test how things change as you go along.

## Contributing

We very much welcome contributions to pgmuvi! Please take a look at our [contributing guide](https://github.com/ICSM/pgmuvi/blob/main/CONTRIBUTING.md) for more information on how to contribute!
But don't forget to read our [code of conduct](https://github.com/ICSM/pgmuvi/blob/main/CODE_OF_CONDUCT.md) before you get started.

## Citing pgmuvi

pgmuvi is currently under review in the Journal of Open Source Software.
If you use pgmuvi in your research, please cite the paper (details will be given here when the paper is accepted!)


## Using pgmuvi

You can find full documentation for pgmuvi at [https://pgmuvi.readthedocs.io/](https://pgmuvi.readthedocs.io/). This includes a quickstart guide and a set of tutorials intended to get you up and running.

## Sampling Quality Assessment

Before fitting a GP, you can check whether your lightcurve has adequate temporal sampling:

```python
# Compute sampling metrics
metrics = lc.compute_sampling_metrics()
print(f"Nyquist period: {metrics['nyquist_period']:.2f} days")
print(f"Detectable range: {metrics['nyquist_period']:.1f}"
      f" - {metrics['longest_detectable_period']:.1f} days")

# Full quality assessment with detailed report
passes, diagnostics = lc.assess_sampling_quality(verbose=True)
if diagnostics['recommendation'] == 'PROCEED':
    lc.fit(...)
```

By default, `fit()` automatically checks sampling quality:

```python
lc.fit(model='1D', ...)  # Raises ValueError if sampling is poor
lc.fit(model='1D', ..., check_sampling=False)  # Force fitting anyway


For multiband data:
# Check each wavelength band
results = lc2d.assess_sampling_quality_per_band()
print(f"{results['summary']['n_passing']}/{results['summary']['n_bands']} bands pass")

# Compute metrics per band
band_metrics = lc2d.compute_sampling_metrics_per_band()

# Filter to well-sampled bands only
lc_good = lc2d.filter_well_sampled_bands()
lc_good.fit(model='2D', ...)
```


## Variability Detection

Before fitting a GP, you can check if your lightcurve shows significant variability
using the built-in three-tier statistical testing framework
(weighted chi-square, F_var excess variance, and Stetson K index):

```python
from pgmuvi.lightcurve import Lightcurve

lc = Lightcurve(t, y, yerr)

# Check variability
diagnostics = lc.check_variability(verbose=True)

if diagnostics['decision'] == 'VARIABLE':
    # Proceed with fitting
    lc.fit(...)
else:
    print(f"Not variable: {diagnostics['decision']}")
```

You can also enable automatic variability checking inside `fit()`:

```python
# Raises ValueError if lightcurve is not variable
lc.fit(..., check_variability=True)

# To force fitting of a non-variable source:
lc.fit(..., check_variability=False)
```

For multiband data, each band can be checked independently:

```python
lc2d = Lightcurve(xdata_2d, flux, error)


# Check each band
results = lc2d.check_variability_per_band(verbose=True)
print(f"{results['summary']['n_variable']} variable bands")

# Create a new Lightcurve with only variable bands retained
lc_var = lc2d.filter_variable_bands()
lc_var.fit(...)
```


## Wavelength-Dependence Model-Selection Diagnostics

For 2D multiwavelength light curves, ``pgmuvi`` provides a staged diagnostic
workflow for deciding which wavelength-dependent GP model families are plausible.
The first stage is pre-fit and cheap:

```python
# If the period is known from LS/ACF/consensus diagnostics, provide it here.
diag = lc2d.diagnose_wavelength_dependence(period=350.0)
print(lc2d.format_wavelength_diagnostics_report(diag))
```

The report summarizes sampling quality, variability, robust amplitude,
period-locked amplitude, phase/lag, diagnostic classification, and recommended
candidate model families.  The recommendations are conservative; they identify
plausible next fits rather than declaring a final science model.

Candidate models can then be compared through the normal ``fit()`` pathway:

```python
comparison = lc2d.compare_wavelength_models(
    diagnostic_report=diag,
    base_fit_kwargs={
        "training_iter": 500,
        "miniter": 100,
        "learn_additional_noise": True,
        "verbose": True,
    },
    residual_diagnostic_kwargs={"period": 350.0},
)
print(lc2d.format_wavelength_diagnostics_report(
    diag,
    comparison_report=comparison,
))
```

Plotting helpers return Matplotlib figures without recomputing diagnostics:

```python
figs = lc2d.plot_wavelength_diagnostics(diag, show=False)
comparison_figs = lc2d.plot_wavelength_model_comparison(comparison, show=False)
```

Use this workflow to distinguish achromatic shared variability, smooth
wavelength-dependent amplitude, power-law-like trends, dust/extinction-like
trends, possible wavelength-dependent lags, and cases where independent per-band
behavior or multi-component consensus should be considered.

### Parameter workflow initialization

Schema-enabled models can automatically initialize supported model
parameters from available light-curve diagnostics during `fit()`.

By default, the parameter workflow is enabled:
```python
    lc.fit(
        model="1DMatern",
    )
```
To disable schema-driven initialization:
```python
    lc.fit(
        model="1DMatern",
        use_parameter_workflow=False,
    )
```
After fitting, the workflow results can be inspected directly:
```python
    lc.parameter_workflow_result
```
or summarized using:
```python
    lc.get_parameter_workflow_summary()
```
The summary reports whether parameter-workflow results are available,
which parameters were successfully initialized, which were skipped,
and the corresponding parameter names.
Use `get_parameter_workflow_report()` when you need a structured list
of applied and skipped parameters with value/constraint reason metadata.
