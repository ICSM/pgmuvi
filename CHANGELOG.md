# Changelog

Entries are added by the PR that makes the change; roadmap item numbers refer to `docs/dev/roadmap.md`.

## Unreleased

### Changed
- `pgmuvi.Lightcurve` is exported from the top-level package (1.1).
- `pyro`, `arviz` and `xarray` are optional (`pip install "pgmuvi[mcmc]"`) and imported only when `Lightcurve.mcmc()` or its result methods run; `seaborn` is no longer a dependency (1.1).
- Lower bounds declared for `numpy`, `torch`, `gpytorch`, `astropy`, `scipy`, `matplotlib`; `arviz<1` in the `mcmc` extra because `az.from_pyro` was removed in arviz 1.x (1.1).
- The bundled Betelgeuse light curve moved to `pgmuvi/datasets/AlfOriAAVSO_Vband.csv`, reachable as `pgmuvi.datasets.get_path("AlfOriAAVSO_Vband.csv")`, and is now included in wheels (1.1).

### Deprecated
- `pgmuvi.models` emits a `DeprecationWarning`; import model classes from `pgmuvi.gps` (1.1).

### Removed
- `setup.py` (the build is `pyproject.toml` only) and the unmaintained `pgmuvi/test_script.py` (1.1).

### Fixed
- `examples/separable_kernels_2d.py` imported a kernel class that no longer exists; it now uses `make_quasi_periodic_kernel` (1.1).
