# pgmuvi
Python gaussian processes for multiwavelength variability inference

[![Documentation Status](https://readthedocs.org/projects/pgmuvi/badge/?version=latest)](https://pgmuvi.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/387371146.svg)](https://zenodo.org/badge/latestdoi/387371146)


pgmuvi is based on GPyTorch and intended for use in inferring the properties of astronomical sources with multiwavelength variability. It uses spectral-mixture kernels to learn an approximation of the PSD of the variability, which have been shown to be very effective for pattern discovery (see https://arxiv.org/pdf/1302.4245.pdf).

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

Before fitting a GP, you can check whether your light curve has adequate temporal sampling:

```python
# Compute sampling metrics
metrics = lc.compute_sampling_metrics()
print(f"Nyquist period: {metrics['nyquist_period']:.2f} days")
print(f"Detectable range: {metrics['nyquist_period']:.1f}"
      f" - {metrics['longest_detectable_period']:.1f} days")

# Full quality assessment with detailed report
passes, diagnostics = lc.assess_sampling_quality(verbose=True)
if diagnostics['recommendation'] == 'PROCEED':
    lc.fit(model='1D')
```

Sampling gates are controlled when the `Lightcurve` is constructed, not by
passing `check_sampling` to `fit()`:

```python
from pgmuvi.lightcurve import Lightcurve

lc = Lightcurve(t, y, yerr, check_sampling=True)
# or, for CSV input:
lc = Lightcurve.from_csv("source.csv", check_sampling=True)

lc.fit(model='1D')
```

For multiband data:

```python
# Check each wavelength band
results = lc2d.assess_sampling_quality_per_band()
print(f"{results['summary']['n_passing']}/{results['summary']['n_bands']} bands pass")

# Compute metrics per band
band_metrics = lc2d.compute_sampling_metrics_per_band()

# Filter to well-sampled bands only
lc_good = lc2d.filter_well_sampled_bands()
lc_good.fit(model='2D')
```


## Variability Detection

Before fitting a GP, you can check if your light curve shows significant variability
using the built-in three-tier statistical testing framework
(weighted chi-square, F_var excess variance, and Stetson K index):

```python
from pgmuvi.lightcurve import Lightcurve

lc = Lightcurve(t, y, yerr)

diagnostics = lc.check_variability(verbose=True)

if diagnostics['decision'] == 'VARIABLE':
    lc.fit(model='1D')
else:
    print(f"Not variable: {diagnostics['decision']}")
```

Automatic variability gating is also a constructor-time option for 1-D light
curves. Do not pass `check_variability` to `fit()`; `fit()` does not perform this
gate.

```python
# Raises ValueError during construction if the light curve is not variable.
lc = Lightcurve(t, y, yerr, check_variability=True)
lc.fit(model='1D')
```

For multiband data, each band can be checked independently:

```python
lc2d = Lightcurve(xdata_2d, flux, error)

results = lc2d.check_variability_per_band(verbose=True)
print(f"{results['summary']['n_variable']} variable bands")

# Create a new Lightcurve with only variable bands retained
lc_var = lc2d.filter_variable_bands()
lc_var.fit(model='2D')
```


## Wavelength-dependence diagnostics and advisory workflows

For 2D multiwavelength light curves, `pgmuvi` includes diagnostic and advisory
workflows for assessing how variability changes with wavelength. These workflows
are advisory: they rank or compare plausible model families/configurations, but
they do **not** automatically select, install, or endorse a final science model.

> **Documentation status:** the detailed wavelength-advisory documentation is
> being updated after the PR56--PR71 workflow changes.
> **TBD[wavelength-advisory-docs]:** add the full single-source and batch user
> guide, including the JSON, Markdown, and CSV output schemas.

The older staged API remains available for pre-fit diagnostics and optional
comparison of plausible follow-up fits:

```python
# If the period is known from LS/ACF/consensus diagnostics, provide it here.
diag = lc2d.diagnose_wavelength_dependence(period=350.0)
print(lc2d.format_wavelength_diagnostics_report(diag))

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
```

This report summarizes sampling quality, variability, robust amplitude,
period-locked amplitude, phase/lag, diagnostic classification, and conservative
follow-up-fit recommendations. It can flag smooth wavelength-dependent amplitude
structure, power-law-like trends, and possible wavelength-dependent lags, but its
recommendations are not final model-selection decisions.

The newer period-independent advisory workflow evaluates explicit model/kernel
configs and is documented in:

- `docs/source/howto/wavelength_advisory.rst` for the single-source workflow;
- `docs/source/howto/wavelength_advisory_batch.rst` for the batch workflow and
  output artifact schemas.

The command-line batch wrapper is available at
`examples/run_wavelength_advisory_batch.py`.

Minimal current entry points are:

```python
# Single-source advisory workflow.
workflow = lc2d.run_period_independent_wavelength_advisory_workflow(
    include_2d_baseline=True,
    base_fit_kwargs={
        "fit_strategy": "consensus",
        "learn_additional_noise": True,
        "training_iter": 500,
        "miniter": 100,
    },
    make_text_report=True,
    make_plots=True,
)

print(workflow["advisory_only"])   # True
print(workflow["selected_model"])  # None in the current advisory workflow

# Batch advisory workflow over CSV sources.
batch = LC.run_period_independent_wavelength_advisory_workflow_batch(
    [{"source_id": "source-1", "csv_path": "source-1.csv"}],
    output_dir="wavelength_batch_outputs",
    export=True,
)
print(batch["source_results"][0]["n_model_kernel_configs"])
```

Both helpers evaluate and report model/kernel configs.  They do not install a
winning fit into the input light curve, and `selected_model` remains `None`.

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
