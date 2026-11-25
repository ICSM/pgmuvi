# Changelog

Entries are added by the PR that makes the change; roadmap item numbers refer to `docs/dev/roadmap.md`.

## Unreleased

### Added
- `rng=` on `MultibandLSWithSignificance.false_alarm_probability` and `fap_rng=` on `Lightcurve.fit_LS` make bootstrap and phase-scramble FAPs reproducible, including with `n_jobs>1`; one child generator is spawned per Monte Carlo sample. Without `rng` they draw from the global `np.random` state as before (1.3).

### Changed
- The test runner is pytest (`python -m pytest -n auto`); `tests/tests.py` was split into topic files; `unittest discover` still runs the `TestCase` classes but misses the plain-function tests, the per-test seeding and the warning filter; use `python -m pytest` (1.3).
- CI runs on pull requests and pushes to `main`: tests with CPU torch wheels, `ruff check` pinned to 0.16.9 (no auto-fix commits), and a strict Sphinx build (`make -C docs html-strict`) that fails on warnings; `pip install -e .[dev]` installs the same ruff. Docstrings and notebooks were tidied so that the strict build passes (1.2).
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
