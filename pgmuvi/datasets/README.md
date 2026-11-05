# Bundled datasets

| file | rows | columns | source |
|---|---|---|---|
| `AlfOriAAVSO_Vband.csv` | 1,564 | `JD`, `Magnitude` | Visual-band observations of α Ori (Betelgeuse) from the AAVSO International Database (https://www.aavso.org), as downloaded for the original pgmuvi development; the download date and observer filtering were not recorded. Used by the test suite, the PR template's before/after check and the examples. |

Use `pgmuvi.datasets.get_path("AlfOriAAVSO_Vband.csv")` rather than a relative path.
Anything added here needs a row in this table with its provenance (AGENTS.md section 2, rule 9).
