# Working on pgmuvi

Instructions for anyone writing code in this repository, human or coding agent. This file is the single source of truth; `CLAUDE.md` and `.github/copilot-instructions.md` only point here. Sections 1 to 7 apply to everyone; section 8 is for working alongside other people and agents; section 9 is for whoever orchestrates agents. It replaces the old Copilot instructions, whose "not pytest" and "prefer small changes over large refactors" rules are no longer policy.

## 1. Read the roadmap before writing code

`docs/dev/roadmap.md` is the development plan. It was settled by the maintainer on 2026-11-05 and every PR is expected to map onto one of its items.

- Find the roadmap item your task corresponds to (for example `1.5`, `3.4`, `J4`). If there is none, do not start coding: open an issue describing the gap and ask whether it belongs in the plan.
- Read that item's pointers. Section 5 says what the archived stack did and why it was dropped or kept; `docs/dev/reports/` has the inventories. Treat function names and line counts in the reports as pointers to verify, not facts.
- The decisions in section 7 are settled. Do not re-open them in code: for example, per-band jitter is implemented but off by default (DEC-7), `"2D"` keeps its current meaning (DEC-11), subsampling on load is off (DEC-9). DEC-17, DEC-18 and DEC-19 are still open; say so if your work touches them.
- Claim the item before coding (section 8): check the Project board and the open PRs for one naming the item; if there is none, open a draft PR from your branch straight away. An open or draft PR means someone else is already on it.
- When a PR completes an item, mark it done in the roadmap in the same PR by appending `✓ done YYYY-MM-DD` to its line.

## 2. Rules the archived stack broke; do not repeat them

The 99-PR stack archived under roadmap phase 0 (tags `archive/stack-2026-07`, `archive/giant-pr`) added 140k lines, of which about 6k were useful. These rules exist because of that.

1. **One PR per feature or fix.** Never one per phase, never a bundle. Target under ~800 lines of code; mechanical moves may be larger if the diff is `git mv` plus imports and the golden tests pass.
2. **No contract / implement / validate triplets.** Write the function and its tests in one PR. No schema, status, enum, disposition, provenance or manifest layers around results; a function returns a small dataclass or a tensor.
3. **No code that validates the module's own output.** A test checks behaviour against known inputs; a function that checks that a dict it just wrote has the keys it just wrote is deleted on review.
4. **Science goes into pure functions in small modules.** `Lightcurve` is becoming a façade. Nothing new goes into `pgmuvi/lightcurve.py` except delegation to a function that lives elsewhere.
5. **Fixes first, then moves, then features.** A mechanical move means no behaviour change, tests pass, aliases left behind. A behaviour change is its own PR with a changelog line and a before/after on a real light curve (`pgmuvi/datasets/AlfOriAAVSO_Vband.csv` today; CIT 6 once 3.12 lands).
6. **Tests test behaviour.** No tests that grep `.rst` files or notebook cells for phrases. No tests that only assert a dataclass round-trips through `to_dict`.
7. **Deprecate, do not break.** Old names become shims that emit `DeprecationWarning` for one release. Defaults change with a `FutureWarning` one release ahead.
8. **Do not revive archived branches wholesale.** Use `git show origin/<branch>:<path>` to read the old implementation, then rewrite it small. Archive tags: `archive/stack-2026-07` (the whole stack), `archive/giant-pr` (the pre-split PR), `archive/pr<N>` for the seven closed heads the stack tag does not reach.
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
cd docs && make html-strict                                # Sphinx build, warnings are errors (CI)
```

Notes:
- The test runner is `unittest` today. Roadmap 1.3 switches to `pytest -n auto`; until that merges, new tests are `unittest.TestCase` classes in `tests/test_<topic>.py`, which pytest also runs unchanged.
- Line length is 88 and `E501` is enforced. `tests/` is excluded from ruff. `pip install -e .[dev]` gives the pinned ruff; CI pins the same version, so a newer local ruff may flag code that CI accepts, or the reverse.
- `docs/source/conf.py` sets `nbsphinx_allow_errors = True`, so a broken notebook does not fail the docs build. Do not rely on the build as a notebook test until 3.13 flips this.
- CI (`.github/workflows/`) runs on every pull request and on pushes to `main`: the test suite on Python 3.10 to 3.15 with CPU torch wheels (`test.yml`), `ruff check` pinned to the version in `.pre-commit-config.yaml` and the `dev` extra, check-only (`ruff.yml`), and a strict Sphinx build, `make -C docs html-strict`, that fails on any warning and renders notebooks from their stored outputs without executing them (`docs.yml`). The two JOSS paper workflows run only when the paper changes. Run the strict build locally before pushing a docstring or docs change: `SPHINXBUILD=$PWD/.venv/bin/sphinx-build make -C docs html-strict SPHINXOPTS="-D nbsphinx_execute=never"` (needs `pip install -e .[docs]` and pandoc).

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

## 8. Working in parallel

Several people and several agents work on this repository at once, from different time zones. Assume the other party is asleep: everything is written down in GitHub, nothing is agreed in a chat.

**Claiming an item**
- The unit of work is a roadmap item. The Project board, https://github.com/orgs/ICSM/projects/1, has one card per item. Issues and PRs are added to the board with the `Roadmap item` field set to the item number (Projects has no parent/child link; group the board by that field to see a card next to its PRs).
- Before starting, check the board and `gh pr list --state open --search "<item>"` for an open or draft PR naming the item. If one exists, the item is taken: pick another, or offer help in that PR.
- If none exists, create your branch, push it, and open a **draft PR** immediately with the item number in the title (`[1.5] Fix transform of yerr`), even before any code. Add it to the board (`gh project item-add 1 --owner ICSM --url <pr-url>`), set its `Roadmap item` field, and move the item's card to "In progress". The draft PR is the claim.
- A draft PR with no commit or comment for 7 days is stale. Ask in it whether the author is continuing; after 48 hours without reply you may take the item over by branching from their branch (never by pushing to it) and saying so in both PRs.

**Branches and files**
- One branch per item, named `<item>/<slug>` (`1.5/transform-yerr`, `j4/laplace`, `2.4/plotting-module`). Never commit to `main`.
- Before editing a file, see which other open PRs touch it: `gh pr list --state open --json number,title,files --jq '.[] | select(.files[].path == "pgmuvi/lightcurve.py") | "#\(.number) \(.title)"'`. Overlap is allowed but advisory: name the overlapping PR in your description and agree in its thread who lands first. The later PR rebases.
- `pgmuvi/lightcurve.py` is the hot spot. Keep diffs to it minimal (a one-line delegation to code in a new module conflicts far less than an edited method body), and land PRs that touch it quickly.
- Do not start an item before its prerequisite in the roadmap has merged, except as a short chain (at most 4 PRs) based on the prerequisite's branch, stated in the PR description.
- Rebase onto `main` before review and before merge (`git rebase origin/main`); never merge `main` into your branch. Force-push only your own branch, and only with `--force-with-lease`.
- Never push to someone else's branch. Suggestions go as review comments or as a PR into their branch.
- In `docs/dev/roadmap.md`, a PR edits only the line of its own item, so the tick-marks do not conflict.

**Review and merge**
- Review is a mix of human and AI reviewers. Any approving human reviewer may squash-merge once CI is green, the PR template is filled in, and the author has rebased. Agents do not merge.
- If you are blocked on a question, write it in the PR, mark the PR with a `blocked` label or "Blocked:" line in the description, and move to another item rather than waiting.

## 9. Orchestration and token economy

An orchestrator is the model running the session that plans, delegates and reviews. It is usually the most expensive model available, so its tokens are the scarce resource. Delegate by role; the Claude Code mapping is in `.claude/agents/` and the table below. Other vendors map by cost tier.

| role | Claude model | agent file | does |
|---|---|---|---|
| scout | Haiku | `.claude/agents/scout.md` | finds things: grep/glob sweeps, "where is X defined", method index of a module, summarise a report section, run a test file and report the tail, check which PRs touch a file |
| worker | Sonnet | `.claude/agents/worker.md` | implements to a spec: mechanical moves with golden tests, tests for a stated function, ports from an archive branch to a stated design, docstrings, docs pages, PR descriptions, lint fixes |
| engineer | Opus | `.claude/agents/engineer.md` | implements items needing numerical or scientific judgement from the roadmap spec; debugs failing fits; reviews correctness |
| reviewer | Opus | `.claude/agents/reviewer.md` | read-only review of a PR against this file and the roadmap item |
| architect | Fable (or Opus when Fable is unavailable) | the orchestrator itself | roadmap-level decisions, the new kernels and likelihoods of phase 4, the high-risk PRs (2.9, 2.11, 4.2, 4.5), numerical failures nobody else could resolve |

**Rules for the orchestrator**
1. Plan in ten lines before any tool call: what has to be true at the end, what you need to know, who fetches each fact.
2. Never read a whole file longer than ~300 lines. Locate with `grep -n`, then read a range of 20 to 80 lines. `pgmuvi/lightcurve.py` is never read whole by any model; its method index is `docs/dev/reports/architecture_main.md`. Read in this order: roadmap item, report pointer, grep, ranged read.
3. Delegate anything that is "find", "summarise", "run and report" to a scout; anything with a complete spec to a worker; anything needing judgement about numbers to an engineer. Keep for yourself only what the architect row lists. Do not spawn an agent for what you can do in three tool calls.
4. A brief names the roadmap item, the goal, the files or report pointers, what is already known or excluded, and the output format with a line cap. Ask for `file:line` pointers and conclusions, never file contents.
5. Verify one claim from every report with one cheap, precise check (run the one test, grep the one symbol, `git diff --stat`) before building on it. A report that confirms your own hypothesis gets a second check.
6. Run the test suite in the background and read only its last 20 lines. While iterating, run the one relevant test file. Pipe long outputs to a file in the scratchpad and grep them.
7. Batch independent delegations and tool calls in one turn.
8. Do not re-derive what the roadmap or the reports already state; cite the section.
9. When the orchestrator writes code itself, it still follows section 2, and a reviewer agent reads the diff before the PR leaves draft.
10. Record in the PR description which roles did what (for example "scout: inventory; worker: move + tests; architect: review"), so reviewers know where to look hardest.

**Concurrency and budget (soft caps)**
- At most **2 subagents run at once**. Batch independent lookups into one scout brief rather than spawning several scouts.
- When subagents work on **different roadmap items**, start them at least **30 minutes apart**, so the first has claimed its item (draft PR on the board) and its early commits are visible before the second begins, and the two do not race for `lightcurve.py`. Subagents on the same item (a scout feeding a worker) have no lag.
- **300k tokens per agent** is the soft cap, orchestrator and subagents alike. Treat the harness's context summarisation, or a context counter passing ~300k, as the signal.
  - A subagent approaching it finishes the current step, commits what passes, writes a hand-off note (below) and reports back so a fresh subagent continues. Hand off at the next natural boundary of the item: a passing test file, a completed move, a function with its tests. Do not hand off mid-edit.
  - The orchestrator approaching it tells the user plainly: how far the item is, what is committed, and that a new session would be cheaper and sharper. The user chooses; continuing is allowed. If they continue, the orchestrator writes the hand-off note first, so nothing is lost if the session dies.
- A hand-off note is written **before** the hand-off, not from memory afterwards.

**Hand-off documentation and pitfalls**
- `docs/dev/handoff.md` is the global hand-off file, committed on the branch of the item it describes (the "Active work" entry) or on `main` via the PR that learned the lesson (the "Pitfalls" entry). Every session that starts or stops work on an item updates it: branch, PR, what is done, the next step, and what the next agent should not redo.
- Every PR description carries a **"Next step"** line while the PR is open, so a hand-off is possible from the PR alone.
- Pitfalls are one line each, dated, with the fix or workaround: tooling quirks, environment traps, a test that must be run a particular way, a GitHub behaviour. Read the Pitfalls section at the start of every session; it is short on purpose. Delete an entry when the roadmap item that removes the trap merges.

**Claude Code specifics**
- `.claude/settings.json` allows read-only commands (git and gh queries, grep, ruff check, test runs) without prompts, so parallel sessions are not blocked on approvals. Personal overrides go in `.claude/settings.local.json`, which is gitignored.
- Use the agent definitions above as `subagent_type`. Give a fresh agent the item number and the pointers; it has no other context.
- `gh project` commands need the `project` scope: `gh auth refresh --hostname github.com -s project,read:project` (run in a terminal; it opens a browser device-code flow).

