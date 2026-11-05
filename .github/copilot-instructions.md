# Copilot instructions for pgmuvi

Read `AGENTS.md` at the repository root. It is the single source of truth for how to work here: the development roadmap (`docs/dev/roadmap.md`) and how PRs map onto it, the rules that keep PRs small and reviewable, the repository map, the verified commands, and the code style.

Two points from the old version of this file are no longer policy: the test runner is moving to pytest (roadmap 1.3), and large mechanical reorganisation PRs are expected in roadmap phase 2 as long as they are moves with no behaviour change. Do not add new code to `pgmuvi/lightcurve.py`; put it in a small module and delegate.
