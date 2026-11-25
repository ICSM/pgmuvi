# Hand-off notes

Read the Pitfalls at the start of every session. Update Active work when you start or stop work on an item (AGENTS.md section 9). One entry per item; delete it when the item's PR merges. Keep the whole file short.

## Active work

### 1.4 Golden regression suite — branch `1.4/golden-suite`, PR #281 (draft)
- Done: everything in the roadmap line (`tests/test_golden_*.py`, `tests/test_public_surface.py`, snapshots in `tests/golden/`, `golden` fixture in `conftest.py`); roadmap ticked.
- Next step: reviewer pass; un-draft; maintainer squash-merge; then delete this entry.
- Do not redo: the snapshots are float32 (the data path hard-codes it); regenerate with `PGMUVI_UPDATE_GOLDEN=1` when 1.8 lands. Findings for later items, recorded as xfails/comments in `test_golden_period_recovery.py`: `fit()` does not seed `period_length` for the quasi-periodic kernels (1DQuasiPeriodic, 1DPeriodicStochastic end at ~175 d) → 3.3; `1D` misses P on 2/6 seeds at the default lr 0.1 → 3.3; `2D` drifts to ~312 d without an explicit scales guess → 1.7/1.11; `write_votable`→`from_table` scrambles x for 2D (strict xfail in `test_golden_io.py`) → 1.12/2.5; the 2D `compute_psd` normalisation differs from the 1D branch (no mirrored component, no factor 1/2; strict xfail in `test_golden_snapshots.py`) → 1.12; `1DLinearSKI` cannot be constructed (`num_dims` missing for the SKI grid; strict xfail in `test_golden_registry.py`) → 1.12.

## Pitfalls

- 2026-11-25 — `pull_request` workflows do not start while a PR conflicts with its base, and a PR chained on another branch loses its base when that branch merges. Retarget to `main` and rebase before expecting CI; no run on a push is the symptom.
- 2026-11-25 — Applying an exported patch with `git am` strips the `[1.2]`-style subject prefix; use `git am --keep-non-patch`.
- 2026-11-25 — GraphQL-backed `gh` commands (`gh pr checks`, `gh project ...`) fail from Claude Code cloud sessions. Read checks with `gh api repos/ICSM/pgmuvi/commits/<sha>/check-runs`; board edits are done from a terminal.
- 2026-11-25 — `pytest -x --durations=N` reports only the tests that ran before the first failure; take timings from a full run.
- 2026-11-25 — The auto-mode permission classifier also blocks `git reset --hard` (use `git checkout -B <branch> <ref>` on a clean tree) and the merge API call; merging is the maintainer's action in any case (AGENTS.md section 8).

- 2026-11-10 — `Build JOSS paper and tex` fails on `main` on a JOSS `fancyhdr` headheight warning that belongs to the paper (J6). Since 1.2 it only runs when `paper/**` changes, so a PR touching the paper goes red for this reason, not for its own changes.
- 2026-11-05 — `gh auth refresh` from inside a Claude Code session needs `--hostname github.com` (non-interactive). The `project` scope is required for `gh project`.
- 2026-11-05 — In Claude Code auto mode the permission classifier blocks `gh pr close`, `gh pr comment` and `gh pr ready` even when the user asked for them; `git push`, `gh pr create`, `gh api PATCH` and `gh project item-*` pass. Prepare the command and ask the user to run it with the `!` prefix.
- 2026-11-05 — `gh pr edit` silently fails on this repo (GraphQL projectCards deprecation). Use `gh api -X PATCH repos/ICSM/pgmuvi/pulls/<n> --input body.json`.
- 2026-11-05 — `gpytorch` imports `pyro` when it is installed, so "pgmuvi does not import pyro" cannot be tested via `sys.modules`; test module attributes instead (see `tests/test_package_hygiene.py`).
- 2026-11-05 — `.claude/agents/*.md` and `.claude/settings.json` are loaded at session start; a session that predates them cannot use the custom `subagent_type`s. Fall back to `general-purpose` with `model:` set.
- 2026-11-05 — The ruff formatter would reformat large pre-existing regions of `pgmuvi/lightcurve.py`; run `ruff format` only on files you created, never on `lightcurve.py` (CI does not check formatting).
