## Roadmap item

<!-- One item per PR, e.g. "1.5" or "J4" from docs/dev/roadmap.md. If there is none, open an issue first and link it here. -->

## What changed and why

<!-- Two or three sentences. Link the issue(s). -->

## Behaviour

- [ ] No behaviour change (mechanical move, or fix whose outputs are unchanged)
- [ ] Behaviour change: changelog line added, and before/after on a real light curve is below

<!-- Before/after: numbers or a plot from `pgmuvi/datasets/AlfOriAAVSO_Vband.csv` or the CIT 6 dataset. -->

## Checklist

- [ ] Under ~800 lines of code, one concern
- [ ] Nothing new in `pgmuvi/lightcurve.py` beyond delegation to a function elsewhere
- [ ] Tests added or updated, and they test behaviour (not docs text, not dict round-trips)
- [ ] `ruff check pgmuvi/` is clean and the test suite passes locally
- [ ] numpydoc docstrings; `docs/source` updated if the public API changed; shim left for any renamed public name
- [ ] Roadmap item ticked in `docs/dev/roadmap.md` if this completes it
- [ ] AI assistance: <!-- none, or which tool, and what you checked by hand -->
