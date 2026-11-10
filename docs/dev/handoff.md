# Hand-off notes

Read the Pitfalls at the start of every session. Update Active work when you start or stop work on an item (AGENTS.md section 9). One entry per item; delete it when the item's PR merges. Keep the whole file short.

## Active work

## Pitfalls

- 2026-11-05 — `gh auth refresh` from inside a Claude Code session needs `--hostname github.com` (non-interactive). The `project` scope is required for `gh project`.
- 2026-11-05 — In Claude Code auto mode the permission classifier blocks `gh pr close`, `gh pr comment` and `gh pr ready` even when the user asked for them; `git push`, `gh pr create`, `gh api PATCH` and `gh project item-*` pass. Prepare the command and ask the user to run it with the `!` prefix.
- 2026-11-05 — `gh pr edit` silently fails on this repo (GraphQL projectCards deprecation). Use `gh api -X PATCH repos/ICSM/pgmuvi/pulls/<n> --input body.json`.
- 2026-11-05 — `tests/tests.py` imports sibling test modules by bare name, so `python -m unittest tests.tests -k x` fails with ModuleNotFoundError. Run it via `python -m unittest discover -s tests -p "tests.py" -k x`. (Goes away with 1.3.)
- 2026-11-05 — `gpytorch` imports `pyro` when it is installed, so "pgmuvi does not import pyro" cannot be tested via `sys.modules`; test module attributes instead (see `tests/test_package_hygiene.py`).
- 2026-11-05 — `.claude/agents/*.md` and `.claude/settings.json` are loaded at session start; a session that predates them cannot use the custom `subagent_type`s. Fall back to `general-purpose` with `model:` set.
- 2026-11-05 — The ruff formatter would reformat large pre-existing regions of `pgmuvi/lightcurve.py`; run `ruff format` only on files you created, never on `lightcurve.py` (CI does not check formatting).
