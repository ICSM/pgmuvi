# Hand-off notes

Read the Pitfalls at the start of every session. Update Active work when you start or stop work on an item (AGENTS.md section 9). One entry per item; delete it when the item's PR merges. Keep the whole file short.

## Active work

### 1.3 Test harness — branch `1.3/pytest-harness`, PR #279 (rebased on main)
- Done: everything in the roadmap line; suite 1,280 passed / 1 skipped under `pytest -n auto`; roadmap ticked.
- Next step: maintainer review; squash-merge; then delete this entry. Open question for the maintainer: raise `numpy>=1.24` to 1.25 and drop the `_spawn_generators` fallback.
- Do not redo: `tests/tests.py` is gone (its 25 re-imported `test_variability` tests now run once); `test_acf.py::test_gp_acf_smoke` stays skipped because it raises a ValueError when enabled, not for speed.

## Pitfalls

- 2026-11-10 — `Build JOSS paper and tex` fails on `main` on a JOSS `fancyhdr` headheight warning that belongs to the paper (J6). Since 1.2 it only runs when `paper/**` changes, so a PR touching the paper goes red for this reason, not for its own changes.
- 2026-11-05 — `gh auth refresh` from inside a Claude Code session needs `--hostname github.com` (non-interactive). The `project` scope is required for `gh project`.
- 2026-11-05 — In Claude Code auto mode the permission classifier blocks `gh pr close`, `gh pr comment` and `gh pr ready` even when the user asked for them; `git push`, `gh pr create`, `gh api PATCH` and `gh project item-*` pass. Prepare the command and ask the user to run it with the `!` prefix.
- 2026-11-05 — `gh pr edit` silently fails on this repo (GraphQL projectCards deprecation). Use `gh api -X PATCH repos/ICSM/pgmuvi/pulls/<n> --input body.json`.
- 2026-11-05 — `gpytorch` imports `pyro` when it is installed, so "pgmuvi does not import pyro" cannot be tested via `sys.modules`; test module attributes instead (see `tests/test_package_hygiene.py`).
- 2026-11-05 — `.claude/agents/*.md` and `.claude/settings.json` are loaded at session start; a session that predates them cannot use the custom `subagent_type`s. Fall back to `general-purpose` with `model:` set.
- 2026-11-05 — The ruff formatter would reformat large pre-existing regions of `pgmuvi/lightcurve.py`; run `ruff format` only on files you created, never on `lightcurve.py` (CI does not check formatting).
