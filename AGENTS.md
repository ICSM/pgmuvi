# Working on pgmuvi

Instructions for anyone writing code in this repository, human or coding agent. This file is the single source of truth; `CLAUDE.md` and `.github/copilot-instructions.md` only point here. It replaces the old Copilot instructions, whose "not pytest" and "prefer small changes over large refactors" rules are no longer policy.

## 1. Read the roadmap before writing code

`docs/dev/roadmap.md` is the development plan. It was settled by the maintainer on 2026-11-05 and every PR is expected to map onto one of its items.

- Find the roadmap item your task corresponds to (for example `1.5`, `3.4`, `J4`). If there is none, do not start coding: open an issue describing the gap and ask whether it belongs in the plan.
- Read that item's pointers. Section 5 says what the archived stack did and why it was dropped or kept; `docs/dev/reports/` has the inventories. Treat function names and line counts in the reports as pointers to verify, not facts.
- The decisions in section 7 are settled. Do not re-open them in code: for example, per-band jitter is implemented but off by default (DEC-7), `"2D"` keeps its current meaning (DEC-11), subsampling on load is off (DEC-9). DEC-17, DEC-18 and DEC-19 are still open; say so if your work touches them.
- When a PR completes an item, tick it in the roadmap in the same PR.

## 2. Rules the archived stack broke; do not repeat them

The 99-PR stack being archived under roadmap phase 0 added 140k lines, of which about 6k were useful. These rules exist because of that.

1. **One PR per feature or fix.** Never one per phase, never a bundle. Target under ~800 lines of code; mechanical moves may be larger if the diff is `git mv` plus imports and the golden tests pass.
2. **No contract / implement / validate triplets.** Write the function and its tests in one PR. No schema, status, enum, disposition, provenance or manifest layers around results; a function returns a small dataclass or a tensor.
3. **No code that validates the module's own output.** A test checks behaviour against known inputs; a function that checks that a dict it just wrote has the keys it just wrote is deleted on review.
4. **Science goes into pure functions in small modules.** `Lightcurve` is becoming a façade. Nothing new goes into `pgmuvi/lightcurve.py` except delegation to a function that lives elsewhere.
5. **Fixes first, then moves, then features.** A mechanical move means no behaviour change, tests pass, aliases left behind. A behaviour change is its own PR with a changelog line and a before/after on a real light curve (`pgmuvi/AlfOriAAVSO_Vband.csv` today; CIT 6 once 3.12 lands).
6. **Tests test behaviour.** No tests that grep `.rst` files or notebook cells for phrases. No tests that only assert a dataclass round-trips through `to_dict`.
7. **Deprecate, do not break.** Old names become shims that emit `DeprecationWarning` for one release. Defaults change with a `FutureWarning` one release ahead.
8. **Do not revive archived branches wholesale.** Use `git show origin/<branch>:<path>` to read the old implementation, then rewrite it small. Archive tags (made in roadmap phase 0): `archive/stack-2026-07`, `archive/giant-pr`.
9. **No junk in commits.** No `.orig` files, `debug*.txt`, stray notebooks with outputs, files over 1 MB without discussion, or data without a provenance README.
10. **No new dependencies without discussion.** Heavy optional imports (`pyro`, `arviz`, `xarray`) are imported lazily inside the functions that need them.
11. **Say so when a PR is AI-assisted**, which tool was used, and what the human author checked. Review is a mix of human and AI reviewers (DEC-5).

If a rule here conflicts with something you find in the code, the rule wins and the code is the thing to fix.

## 3. Repository map

Current layout on `main` (line counts as of 2026-11-05):

| path | lines | what |
|---|---|---|
| `pgmuvi/lightcurve.py` | 11,269 | `Lightcurve`: data, preprocessing, fit, plots, summaries. One class, 127 methods. Do not add to it. |
| `pgmuvi/gps.py` | 1,768 | GP model classes (spectral mixture, separable, alternative kernels), means |
| `pgmuvi/synthetic.py` | 907 | synthetic light curve generation |
| `pgmuvi/priors.py` | 596 | priors |
| `pgmuvi/preprocess/quality.py`, `variability.py` | 589, 446 | sampling quality, subsampling, variability statistics |
| `pgmuvi/multiband_ls_significance.py` | 554 | multiband Lomb–Scargle with significance |
| `pgmuvi/initialization.py` | 342 | hyperparameter initialisation helpers |
| `pgmuvi/constraints.py` | 258 | constraint sets |
| `pgmuvi/trainers.py` | 221 | training loop |
| `pgmuvi/kernels.py` | 126 | custom kernels |
| `pgmuvi/models.py` | 27 | legacy re-exports |
| `tests/` | 27 files | `unittest` test cases; `tests/tests.py` is the old monolith |
| `docs/source/` | | Sphinx; `howto/`, `notebooks/`, one `pgmuvi.<module>.rst` per module |
| `docs/dev/` | | roadmap, reports, experiments |
| `paper/`, `paper.md`, `paper.bib` | | JOSS paper, under review (#45) |

Target layout after roadmap phase 2: `lightcurve.py` shrinks to about 1.5k lines of data handling plus façade; new subpackages `io/`, `preprocess/`, `periodogram/`, `models/`, `inference/`, `results/`, `plotting/`, `datasets/`, plus `transforms.py`. Put new code where it will live in that layout, even before the move PR for that area has happened.

## 4. Commands

Use a Python 3.10+ environment that has `torch` and `gpytorch` installed; the package does not pin them yet (roadmap 1.1). A CPU-only setup that works: `python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -e .` (`.venv` is gitignored).

```bash
pip install -e .                                           # editable install
ruff check pgmuvi/                                         # lint; must be clean (ruff config in pyproject.toml)
ruff format pgmuvi/                                        # format; pre-commit also runs this
python -m unittest discover -s tests -p "test*.py"         # full suite, 1,292 tests, 1-2 min on CPU
python -m unittest tests.test_kernels                      # one file
cd docs && make html                                       # Sphinx build
```

Notes:
- The test runner is `unittest` today. Roadmap 1.3 switches to `pytest -n auto`; until that merges, new tests are `unittest.TestCase` classes in `tests/test_<topic>.py`, which pytest also runs unchanged.
- Line length is 88 and `E501` is enforced. `tests/` is excluded from ruff.
- `docs/source/conf.py` sets `nbsphinx_allow_errors = True`, so a broken notebook does not fail the docs build. Do not rely on the build as a notebook test until 3.13 flips this.
- CI (`.github/workflows/`) runs tests on push for Python 3.10 to 3.15 and runs ruff with auto-fix. Roadmap 1.2 changes this to run on pull requests and lint check-only.

## 5. Code style

- numpydoc docstrings on every public function, class and method; type hints on new public signatures.
- Double quotes; 88 characters; keyword arguments for anything optional.
- Every `warnings.warn` has an explicit `stacklevel=` pointing at the user's code.
- Tensors: explicit `dtype=`/`device=`; float64 is the target default (DEC-6). Do not hard-code `torch.float32`.
- New code uses `logging`, not `print`.
- Prefer a small pure function with a docstring over a new method on `Lightcurve`.
- Follow the numerical conventions in `docs/dev/reports/science_features.md` for kernels and means (log-wavelength coordinate, physical wavelength for power-law means, raw-parameter Hessians for Laplace).

## 6. Known problems on `main`, and where they are fixed

Do not patch these ad hoc; implement the roadmap item instead, or reference it.

| problem | roadmap item |
|---|---|
| `yerr` mean-shifted through the y-transform; plots never inverse-transform y | 1.5 |
| `PowerLawMean`/`DustMean` evaluate λ^p on transformed wavelength | 1.6 |
| one scalar bound applied to both ARD columns of 2D SM `mixture_means` | 1.7 |
| float32 default loses MJD precision | 1.8 |
| `register_constraint` after init silently changes values | 1.9 |
| late `NotPSDError` loses the fit | 1.10 |
| separable 2D SM drifts to the maximum period | 1.11 (PR #151) |
| subsampling to 1000 points on load, silently | 2.6 |
| `plot` not memory-safe (#149) | 2.13 |
| `mcmc()` broken on arviz 1.x; prints `state_dict` every step (#33, #46) | J4 (Laplace), 4.8 |
| single constant mean and global noise cause #108 | 4.1, 4.2 |

## 7. Pull request checklist

The template at `.github/pull_request_template.md` asks for these. Fill it in; reviewers use it.

- Roadmap item named; one item per PR.
- Code under ~800 lines, one concern; nothing new in `lightcurve.py` beyond delegation.
- Tests added that exercise behaviour; `ruff check pgmuvi/` clean; suite passes locally.
- Behaviour change: changelog line plus before/after numbers or plot on a real light curve.
- Public API change: docstring, `docs/source` updated, shim for the old name.
- AI assistance disclosed, with what the human checked.
- Roadmap ticked if the item is complete.
