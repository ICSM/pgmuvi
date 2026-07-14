# Pull request summary

Describe the change and the motivation for it.

## Change type

- [ ] Bug fix
- [ ] Feature or API change
- [ ] Documentation-only change
- [ ] Tests or CI change
- [ ] Repository hygiene / maintenance

## Validation

Check the commands that were run locally before opening this pull request.

- [ ] Relevant unit tests were run with `PYTHONPATH=.`
- [ ] Documentation changes were checked with `cd docs && make clean && make html-strict`
- [ ] Generated artifacts were not committed (`git status --short` checked)
- [ ] `git diff --check` passed

## Documentation impact

- [ ] No documentation update was needed
- [ ] User-facing documentation was updated
- [ ] API reference or docstring coverage was updated
- [ ] Contributor or maintenance documentation was updated

## Review artifacts

For review snapshots, use the tracked-file helper instead of manually zipping the working tree:

```bash
python3 scripts/create_source_snapshot.py --output pgmuvi_current.zip
```

Confirm that review ZIPs, debug logs, Sphinx builds, caches, patch backups, and other generated artifacts are not staged.

## Notes for reviewers

Mention any expected warnings, intentionally skipped tests, compatibility concerns, or follow-up PRs.
