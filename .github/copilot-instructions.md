# Copilot Instructions for pgmuvi

## Repository Overview

**pgmuvi** (Python Gaussian Processes for Multiwavelength Variability Inference) is a Python package
for inferring properties of astronomical sources with multiwavelength variability using Gaussian
processes. It is built on GPyTorch and uses spectral-mixture kernels to learn approximations of the
Power Spectral Density (PSD) of variability.

- **Language**: Python (>=3.10)
- **Size**: ~12,000 lines of Python code across multiple modules
- **Framework**: GPyTorch, PyTorch
- **Package Type**: Scientific computing library for astronomy
- **License**: GNU GPL v3

## Project Structure

```
pgmuvi/
├── .github/                        # GitHub workflows and configurations
│   ├── copilot-instructions.md     # These instructions
│   └── workflows/                  # CI/CD pipelines (test.yml, ruff.yml, etc.)
├── docs/                           # Sphinx documentation
├── examples/                       # Example notebooks and scripts
├── pgmuvi/                         # Main package source code
│   ├── __init__.py                 # Package initialization and version
│   ├── constraints.py              # Constraint set handling (~260 lines)
│   ├── gps.py                      # Gaussian process models (~1800 lines)
│   ├── initialization.py           # Hyperparameter initialization helpers (~340 lines)
│   ├── kernels.py                  # Custom kernel definitions (~130 lines)
│   ├── lightcurve.py               # Light curve handling (~6000 lines) — main module
│   ├── models.py                   # GP model class definitions (~30 lines)
│   ├── multiband_ls_significance.py # Multiband Lomb-Scargle with significance (~560 lines)
│   ├── priors.py                   # Prior distributions (~600 lines)
│   ├── synthetic.py                # Synthetic light curve generation (~910 lines)
│   ├── trainers.py                 # Training utilities (~220 lines)
│   ├── preprocess/                 # Preprocessing subpackage
│   │   ├── __init__.py
│   │   ├── quality.py              # Quality filtering and subsampling (~540 lines)
│   │   └── variability.py          # Variability statistics (~390 lines)
│   └── AlfOriAAVSO_Vband.csv       # Example data file
├── tests/                          # Test suite (unittest discovery via test*.py)
│   ├── tests.py                    # Core tests
│   ├── test_2d_constraints.py
│   ├── test_2d_integration.py
│   ├── test_alternative_models_integration.py
│   ├── test_best_band_init.py
│   ├── test_constraint_sets.py
│   ├── test_get_methods.py
│   ├── test_initialization.py
│   ├── test_kernels.py
│   ├── test_mls_init.py
│   ├── test_models_alternative.py
│   ├── test_period_priors.py
│   ├── test_subsampling.py
│   ├── test_synthetic.py
│   ├── test_time_units.py
│   └── test_variability.py
├── paper/                          # JOSS paper files
├── pyproject.toml                  # Project configuration and dependencies
├── setup.py                        # Minimal setup for older pip compatibility
├── tox.ini                         # Testing configuration
└── README.md                       # Main documentation
```

## Build and Development Instructions

### Installation

**Always install dependencies before any other steps:**
```bash
pip install -e .
```

### Testing

**Run tests using tox (recommended):**
```bash
tox -e py
```

Or run tests directly (unittest discovery):
```bash
python -m unittest discover -s tests -p "test*.py"
```

**Important**:
- Tests run on Python 3.10, 3.11, 3.12, 3.13, and 3.14 in CI.
- The package requires **Python >=3.10**.
- Use `PGMUVI_TEST_HIGH_PRECISION=1` to enable high-precision normalization tests in
  `test_period_priors.py`.
- New test files must match the `test*.py` glob pattern to be discovered automatically.

### Linting

**Always run Ruff before committing code:**
```bash
ruff check pgmuvi/
```

To auto-fix safe issues and see detailed output:
```bash
ruff check --fix pgmuvi/
ruff check -v --output-format=full --show-fixes pgmuvi/
```

**Ruff configuration** is in `pyproject.toml` under `[tool.ruff]`:
- **Excluded from linting**: `tests/` directory and `pgmuvi/test_script.py`
- **Line length**: 88 characters (Black-compatible; E501 is *not* ignored)
- **Enabled rule sets**: A, B, E, F, ISC, UP, RUF, W
- **Explicitly ignored**: UP004, B007, ISC001 (ISC001 conflicts with the formatter)

### Code Style

- Follow PEP 8 conventions
- Use Black formatting with **88 character line length** — lines longer than 88 chars are a lint error
- Target Python 3.10+ syntax (f-strings, walrus operator, `match` statements where appropriate)
- All code must pass Ruff linting before committing
- Docstrings must follow **numpydoc** format
- Use double quotes for strings (consistent with Black)

## Common Linting Pitfalls and How to Avoid Them

These are the most frequent Ruff violations encountered in this codebase:

### E501 — Line too long (>88 characters)

The 88-char limit is enforced strictly. Every new line of code must fit within it.

**Long function signatures** — wrap parameters:
```python
# ✗ too long
def _build_time_kernel(time_kernel_type, period, num_mixtures=4, add_flicker=False, **kwargs):

# ✓ wrap at the opening parenthesis
def _build_time_kernel(
    time_kernel_type, period, num_mixtures=4, add_flicker=False, **kwargs
):
```

**Long function calls** — break arguments across lines:
```python
# ✗ too long
ax.errorbar(self.xdata.cpu().numpy(), self.ydata.cpu().numpy(), yerr=self.yerr.cpu().numpy(), fmt="k*")

# ✓ one argument per line (or logical groups)
ax.errorbar(
    self.xdata.cpu().numpy(),
    self.ydata.cpu().numpy(),
    yerr=self.yerr.cpu().numpy(),
    fmt="k*",
)
```

**Long return expressions** — break at operators:
```python
# ✗ too long
return SMK(num_mixtures=num_mixtures, ard_num_dims=1) + ScaleKernel(RBFKernel(ard_num_dims=1))

# ✓ assign to a variable or break the expression
smk = SMK(num_mixtures=num_mixtures, ard_num_dims=1)
return smk + ScaleKernel(RBFKernel(ard_num_dims=1))
```

**Long warning/error message strings in deeply nested code** — extract to a local variable:
```python
# ✗ too long at deep indentation
warnings.warn(
    f"Only {len(_init_freqs)} MLS peak(s) found but "
    f"{num_mixtures} were requested.",
    RuntimeWarning,
    stacklevel=2,
)

# ✓ extract the message first
_msg = (
    f"Only {len(_init_freqs)} MLS peak(s) found but "
    f"{num_mixtures} were requested."
)
warnings.warn(_msg, RuntimeWarning, stacklevel=2)
```

**Long comments** — wrap at 88 chars using a continuation comment:
```python
# ✗ too long
# This Warning has to be raised after the if, so that the user-defined number of mixtures is used.

# ✓ split across two comment lines
# This warning is raised after the if-block so that the user-defined number of
# mixtures is used and they still see it if they set a value explicitly.
```

### W293 — Whitespace in blank lines

Never leave spaces or tabs on an otherwise empty line. Check with your editor's
"show whitespace" option, or rely on `ruff check --fix` to auto-remove them.

### B028 — Missing `stacklevel` in `warnings.warn()`

**Always supply an explicit `stacklevel=` argument** to every `warnings.warn()` call.
The correct level points the user at *their* code, not at library internals:

```python
# ✗ missing stacklevel — warning points at lightcurve.py internals
warnings.warn("Something happened.")

# ✓ stacklevel=2 points at the direct caller of the current function
warnings.warn("Something happened.", UserWarning, stacklevel=2)

# ✓ stacklevel=3 points one level further up (use when the current function
#   is itself a helper called by the user-facing function)
warnings.warn("Something happened.", UserWarning, stacklevel=3)
```

Match the `stacklevel` of other warnings in the same function for consistency.
In `_drop_nan_rows` and similar internal helpers called from public methods,
`stacklevel=3` is the convention used throughout this codebase.

## Design Principles

When adding or modifying code, follow these principles to keep the codebase
maintainable and readable:

### KISS — Keep It Simple
Prefer the simplest solution that correctly solves the problem. Avoid premature
optimisation or over-engineering. If a helper function, a standard-library call,
or a single well-named variable makes the intent clearer, use it.

### DRY — Don't Repeat Yourself
Extract repeated logic into a shared function or constant rather than copying it.
If you find yourself writing the same block in two places, that is a signal to
refactor into a helper. Pay attention to:
- Repeated warning/error message templates — extract to a constant or helper.
- Repeated data-access patterns (`self.xdata.cpu().numpy()`) — assign to a
  local variable at the top of the block.

### SRP — Single Responsibility
Each function or method should do one thing and do it well. Long functions in
`lightcurve.py` often mix data validation, computation, and I/O — when adding
new code, prefer adding focused helper methods over extending an existing function
that is already doing too much.

### Explicit over implicit
Use keyword arguments for optional parameters. Supply `stacklevel=`, `dtype=`,
`device=` etc. explicitly rather than relying on defaults that may change between
library versions.

## GitHub Actions Workflows

The repository has several CI workflows that run on every push:

1. **Test workflow** (`test.yml`): Runs tests via tox on Python 3.10–3.14
2. **Ruff workflow** (`ruff.yml`): Lints the `pgmuvi/` directory — **must pass**
3. **Pre-commit workflow** (`pre-commit.yml`): Runs pre-commit hooks
4. **Paper workflows**: Generate draft PDFs for the JOSS paper

**Both the test and ruff workflows must pass before any PR is merged.**

## Key Dependencies

Core dependencies (from `pyproject.toml`):
- numpy
- matplotlib
- seaborn
- torch (PyTorch)
- gpytorch (Gaussian process library)
- pyro-ppl (Probabilistic programming)
- arviz (Bayesian visualization)
- xarray (Labeled arrays)
- tqdm (Progress bars)

## Common Tasks

### Making Code Changes

1. Install package in editable mode: `pip install -e .`
2. Make your changes to files in `pgmuvi/`
3. Run linter: `ruff check pgmuvi/` — fix all errors before continuing
4. Run tests: `python -m unittest discover -s tests -p "test*.py"`
5. Confirm all GitHub Actions pass

### Adding New Features

- Add functionality to the appropriate module (see Project Structure above)
- Add tests in `tests/` with a filename matching `test_<feature>.py`
- Write numpydoc docstrings for all public functions and classes
- Run linter and full test suite
- Update documentation in `docs/` if the public API changes

### Fixing Issues

- Reproduce the issue in a test **before** fixing it
- Fix the issue with the smallest possible change
- Ensure `ruff check pgmuvi/` passes on the modified files
- Follow existing code style and patterns in the module

## Important Notes

- **Package Manager**: Uses modern `pyproject.toml` with setuptools_scm for versioning
- **Version Management**: Version is dynamically set by setuptools_scm from git tags
- **setup.py**: Minimal setup.py exists only for backward compatibility with older pip versions
- **Testing Framework**: Uses stdlib `unittest` with tox — **not pytest**
- **Documentation**: Built with Sphinx and hosted on ReadTheDocs
- **Excluding Tests from Lint**: `tests/` and `pgmuvi/test_script.py` are excluded from Ruff
- **Notebooks**: Ruff also lints Jupyter notebooks (`*.ipynb`) in addition to `.py` files

## Validation Steps

Before considering any work complete:

1. `ruff check pgmuvi/` — must produce **zero errors**
2. `python -m unittest discover -s tests -p "test*.py"` — all tests must pass
3. If adding dependencies, add them to `pyproject.toml` under `dependencies`
4. Check that changes follow existing code patterns in the module
5. Verify all docstrings are complete and follow numpydoc format
6. Confirm lines are ≤88 characters (count from column 1, including indentation)

## Tips for Working with This Repository

- `lightcurve.py` is by far the largest and most complex module (~6000 lines);
  prefer small, focused changes over large refactors
- `gps.py` (~1800 lines) defines the GP model classes built on GPyTorch
- `multiband_ls_significance.py` implements the Lomb-Scargle significance layer
  used during frequency initialisation
- The `preprocess/` subpackage provides quality filtering and subsampling helpers
- The package is designed for astronomical time series analysis, not general-purpose GP work
- Test coverage is basic — focus on not breaking existing functionality
- Documentation is important — this is intended for use by astronomers who may not be Python experts
- When in doubt about line length, use `len(line)` in a Python REPL to check

## Trust These Instructions

These instructions have been validated against the actual repository structure and workflows.
Only search for additional information if:
- These instructions are incomplete for your specific task
- You encounter errors not covered here
- The repository structure has changed significantly
