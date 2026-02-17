# Copilot Instructions for pgmuvi

## Repository Overview

**pgmuvi** (Python Gaussian Processes for Multiwavelength Variability Inference) is a Python package for inferring properties of astronomical sources with multiwavelength variability using Gaussian processes. It is built on GPyTorch and uses spectral-mixture kernels to learn approximations of the Power Spectral Density (PSD) of variability.

- **Language**: Python (3.8+, excluding 3.9.13 due to compatibility issues)
- **Size**: ~3,000 lines of Python code
- **Framework**: GPyTorch, PyTorch
- **Package Type**: Scientific computing library for astronomy
- **License**: GNU GPL v3

## Project Structure

```
pgmuvi/
├── .github/              # GitHub workflows and configurations
│   └── workflows/        # CI/CD pipelines (test.yml, ruff.yml)
├── docs/                 # Sphinx documentation
├── pgmuvi/               # Main package source code
│   ├── __init__.py      # Package initialization and version
│   ├── gps.py           # Gaussian process models (~500 lines)
│   ├── lightcurve.py    # Light curve handling (~2300 lines)
│   ├── trainers.py      # Training utilities (~200 lines)
│   └── AlfOriAAVSO_Vband.csv  # Example data file
├── tests/
│   └── tests.py         # Test suite (~140 lines)
├── paper/               # JOSS paper files
├── pyproject.toml       # Project configuration and dependencies
├── setup.py             # Minimal setup for older pip compatibility
├── tox.ini              # Testing configuration
└── README.md            # Main documentation
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

Or run tests directly:
```bash
python tests/tests.py
```

**Important**: Tests run on Python 3.10, 3.11, and 3.12 in CI. The package requires Python >=3.8 but excludes Python 3.9.13 specifically.

### Linting

**Always run Ruff before committing code:**
```bash
ruff check pgmuvi/
```

Or to see detailed output with fixes:
```bash
ruff check -v --output-format=full --show-fixes pgmuvi/
```

**Note**: Ruff configuration is in `pyproject.toml` under `[tool.ruff]`. Tests are excluded from linting. The following checks are enabled:
- A (prevent using keywords that clobber Python builtins)
- B (bugbear: security warnings)
- E (pycodestyle errors)
- F (pyflakes)
- ISC (implicit string concatenation)
- UP (alert when better syntax is available)
- RUF (Ruff-specific rules)
- W (pycodestyle warnings)

### Code Style

- Follow PEP 8 conventions
- Use Black formatting with 88 character line length
- Target Python 3.8+ syntax
- All code must pass Ruff linting
- Docstrings should follow numpydoc format

## GitHub Actions Workflows

The repository has several CI workflows that run on every push:

1. **Test workflow** (`test.yml`): Runs tests via tox on Python 3.10, 3.11, and 3.12
2. **Ruff workflow** (`ruff.yml`): Lints the `pgmuvi/` directory
3. **Paper workflows**: Generate draft PDFs for JOSS paper

**Important**: Always ensure changes pass both test and ruff workflows before finalizing.

## Key Dependencies

Core dependencies (from pyproject.toml):
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
3. Run linter: `ruff check pgmuvi/`
4. Run tests: `tox -e py` or `python tests/tests.py`
5. Ensure all GitHub Actions pass

### Adding New Features

- Add functionality to appropriate module (gps.py, lightcurve.py, or trainers.py)
- Add tests to `tests/tests.py`
- Update docstrings following numpydoc format
- Run linter and tests
- Consider updating documentation in `docs/`

### Fixing Issues

- Tests must be added or updated to cover the fix
- Ensure Ruff passes on modified code
- Follow existing code style and patterns

## Important Notes

- **Package Manager**: Uses modern `pyproject.toml` with setuptools_scm for versioning
- **Version Management**: Version is dynamically set by setuptools_scm from git tags
- **setup.py**: Minimal setup.py exists only for backward compatibility with older pip versions
- **Testing Framework**: Uses basic unittest via tox, not pytest
- **Documentation**: Built with Sphinx and hosted on ReadTheDocs
- **Excluding Tests**: Tests directory is explicitly excluded from Ruff linting

## Validation Steps

Before considering any work complete:

1. Run `ruff check pgmuvi/` - must pass with no errors
2. Run `tox -e py` or `python tests/tests.py` - all tests must pass
3. If adding dependencies, ensure they are added to `pyproject.toml` under `dependencies`
4. Check that changes follow existing code patterns in the module
5. Verify docstrings are complete and follow numpydoc format

## Tips for Working with This Repository

- The main logic is in three files: `gps.py` (GP models), `lightcurve.py` (data handling), and `trainers.py` (training utilities)
- `lightcurve.py` is the largest module with most complexity
- The package is designed for astronomical time series analysis, not general-purpose GP work
- Test coverage is basic; focus on not breaking existing functionality
- The repository follows a scientific software development approach with emphasis on reproducibility
- Documentation is important - this is intended for use by astronomers who may not be Python experts

## Trust These Instructions

These instructions have been validated against the actual repository structure and workflows. Only search for additional information if:
- These instructions are incomplete for your specific task
- You encounter errors not covered here
- The repository structure has changed significantly
