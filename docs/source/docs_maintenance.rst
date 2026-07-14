Documentation maintenance
=========================

The public documentation build is expected to complete without Sphinx warnings.
Use the normal HTML build while iterating, but run the strict build before
merging documentation changes.

Strict local build
------------------

Run the strict build from a clean documentation tree::

   cd docs
   make clean
   make html-strict

The ``html-strict`` target treats warnings as errors and uses ``--keep-going``
so that one warning does not hide later warnings in the same build.  This is the
preferred final check after changing API reference pages, notebooks, toctrees,
or Sphinx configuration.

Notebook policy
---------------

Maintained public notebooks should be linked from the main tutorial toctree and
should build cleanly under Sphinx.  Stub, unfinished, unavailable, or
pending-refresh notebooks should be documented in ``notebook_status.rst`` and
excluded from Sphinx source discovery until they are refreshed.

Avoid leaving notebooks in an in-between state where they are excluded from the
public tutorial list but still discovered as orphan Sphinx source files.  That
state makes clean warning-free builds difficult to maintain.

API-reference policy
--------------------

Prefer focused API pages for modules with stable, readable docstrings.  For
large legacy modules whose implementation docstrings are not yet suitable for
full autodoc expansion, use a short manual synopsis page and link users to the
maintained high-level workflow documentation.

Continuous integration
----------------------

The strict documentation contract is enforced by
``.github/workflows/docs.yml``. The workflow installs the documentation
requirements from ``docs/source/requirements.txt`` and runs::

   cd docs
   make html-strict

This keeps the local maintenance command and the pull-request check aligned:
warnings that fail locally should also fail in CI, and warnings that are
accepted temporarily should be documented explicitly before they are allowed
back into the public docs build.

Generated artifacts and source snapshots
----------------------------------------

Local documentation builds, warning logs, patch backups, and ad-hoc review
archives should stay out of commits. The project ``.gitignore`` covers the
standard local outputs used by the documentation and PR-review workflow,
including ``docs/build/``, ``.ruff_cache/``, ``debug*.txt``, ``*.orig``,
``*.rej``, and ``pgmuvi_current*.zip``.

For a clean source snapshot, prefer a Git archive from the committed tree
instead of zipping the working directory::

   git archive --format=zip -o pgmuvi_current.zip HEAD

This avoids bundling ``.git/``, local caches, Sphinx build output, and other
untracked inspection artifacts.

Clean source snapshot helper
----------------------------

When a clean source snapshot is needed for review, prefer the helper script over
zipping the working directory::

   python scripts/create_source_snapshot.py --output pgmuvi_current.zip

The helper is a small wrapper around::

   git archive --format=zip -o pgmuvi_current.zip HEAD

The archive is produced from tracked files only at the requested Git reference,
so it excludes ``.git/``, local caches, Sphinx build output, debug logs, patch
backups, and other untracked inspection artifacts.


Documentation dependencies
--------------------------

The documentation dependency list is centralized in
``docs/source/requirements.txt``.  That file is used by local strict builds,
Read the Docs, and the GitHub Actions documentation workflow.

When a documentation dependency is added, removed, or deliberately pinned,
update ``docs/source/requirements.txt`` first.  Do not duplicate ad hoc docs
installation commands in CI workflows unless there is a narrowly documented
reason for doing so.
