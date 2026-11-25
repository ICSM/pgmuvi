# Hand-off notes

Read the Pitfalls at the start of every session. Update Active work when you start or stop work on an item (AGENTS.md section 9). One entry per item; delete it when the item's PR merges. Keep the whole file short.

## Active work

### 1.4 Golden regression suite — branch `1.4/golden-suite`, draft PR
- Doing: seeded float64 period-recovery tests (1D sinusoid P = 37 d with `1D`, `1DQuasiPeriodic`, `1DPeriodicStochastic`; 2D chromatic with `2D`, `2DSeparable`), fixed-hyperparameter snapshots (MLL, PSD on a grid, `get_period_summary().as_dict()`), per-model `state_dict` key/shape snapshots, CSV/table round-trips, public-surface test.
- Next step: see the PR description.
- Do not redo: nothing yet.

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
