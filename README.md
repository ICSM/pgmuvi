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
from pgmuvi.lightcurve import Lightcurve

lc = Lightcurve(t, y, yerr)

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
```

For multiband data:

```python
lc2d = Lightcurve(xdata_2d, flux, error)

# Check each wavelength band
results = lc2d.assess_sampling_quality_per_band()
print(f"{results['summary']['n_passing']}/{results['summary']['n_bands']} bands pass")

# Compute metrics per band
band_metrics = lc2d.compute_sampling_metrics_per_band()

# Filter to well-sampled bands only
lc_good = lc2d.filter_well_sampled_bands()
lc_good.fit(model='2D', ...)
```

