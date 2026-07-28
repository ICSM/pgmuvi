Notebook status and maintenance
===============================

.. note::

   **Documentation status:** current through PR174.

   Every notebook currently shipped under ``docs/source/notebooks`` is part of
   the public Tutorials documentation.  No quarantined or pending-refresh
   notebook files remain in the repository.

Why this page exists
--------------------

The package has changed substantially: the default time transform is now
``TimeCenter``, the package uses the shared ``DEFAULT_DTYPE`` machinery, the
constraint workflow has been repaired, consensus fitting is a documented
flagship path, and the period-independent wavelength advisory workflow produces
model/kernel-config reports and batch artifacts.

Older notebooks were reviewed individually.  Maintained workflows were
refreshed and restored to the public tutorial set; misleading or unavailable
workflows were replaced or removed rather than left as executable-looking
examples.

Current public tutorial notebooks
---------------------------------

These notebooks are linked from the main Tutorials toctree:

.. list-table::
   :header-rows: 1

   * - Notebook
     - Current role
     - Maintenance note
   * - ``PGMUVI_Lightcurve.ipynb``
     - Introductory light-curve workflow.
     - Re-execute after major input-validation or default-transform changes.
   * - ``tutorial_preprocessing.ipynb``
     - Maintained preprocessing and data-quality tutorial.
     - Refreshed in PR104 with deterministic validation, sampling and variability diagnostics, reproducible subsampling, and per-band filtering.
   * - ``tutorial_synthetic.ipynb``
     - Maintained analytic synthetic-data tutorial.
     - Refreshed in PR105 with all four public generators, reproducible noise modes, chromatic trends, and explicit GP-sampling boundaries.
   * - ``tutorial_wavelength_advisory.ipynb``
     - Maintained period-independent wavelength-advisory tutorial.
     - Refreshed and renamed in PR106 to replace stale automatic-model-selection framing with diagnostics, parameter planning, model/kernel-config preparation, and an explicit no-fitting default.
   * - ``tutorial_wavelength_constraints.ipynb``
     - Maintained data-derived wavelength-constraint tutorial.
     - Added in PR134 with coordinate round trips, covariance and mean constraints, independent joint-``2D`` ARD diagnostics, reduced optional fits, and the PR133 empirical failure boundaries.
   * - ``tutorial_single_source_analysis.ipynb``
     - Maintained complete real-source analysis workflow.
     - Added in PR174 with validation, per-observational-channel period evidence, temporal consensus, wavelength constraints, a required real GP fit, predictions, plots, residual and phase diagnostics, noise provenance, and a JSON-safe structured report.
   * - ``pgmuvi_mock_data_from_gp.ipynb``
     - Maintained GP-prior mock-data tutorial.
     - Refreshed in PR107 with current parameter schemas, physical-space parameter application, reproducible quasi-periodic/Matérn/spectral-mixture prior draws, and an explicit no-fitting boundary.
   * - ``PGMUVI_Lomb_Scargle.ipynb``
     - Lomb--Scargle and period-candidate concepts.
     - Uses ``candidate`` in the period-search sense, which remains valid.
   * - ``PGMUVI_Gaussian_Process_fitting.ipynb``
     - General GP-fitting introduction.
     - Keep aligned with current dtype, transform, and parameter-workflow defaults.
   * - ``PGMUVI_comparison_with_other_codes.ipynb``
     - Comparison notebook.
     - Keep linked while it remains compatible with the supported dependency stack.
   * - ``pgmuvi_tutorial.ipynb``
     - General tutorial.
     - PR108 removes its stale pointer to an unavailable MCMC tutorial.
   * - ``PGMUVI_QuasiPeriodic_and_Mean_Functions.ipynb``
     - Quasi-periodic kernels and mean functions.
     - Retained as substantive public material; re-execute after relevant API changes.
   * - ``pgmuvi_tutorial_2d.ipynb``
     - Maintained 2-D baseline and consensus-fitting tutorial.
     - Refreshed in PR103 with deterministic data and a preparation-only default; updated in PR168 as an executed end-to-end workflow whose default path performs a required ``2DWavelengthDependent`` consensus fit, verifies fit history and period recovery, calls ``Lightcurve.plot()``, and records rendered predictions before presenting explicit follow-up configurations.

Unavailable future workflows
----------------------------

No quarantined notebook files remain.  The former
``pgmuvi_tutorial_mcmc.ipynb`` was deleted in PR108 because it presented an
executable MCMC workflow even though :meth:`pgmuvi.lightcurve.Lightcurve.mcmc`
currently raises :exc:`NotImplementedError`.

**TBD[mcmc-implementation]:** add a new MCMC tutorial only after the public MCMC
workflow is implemented, tested, and has documented convergence and posterior-
predictive diagnostics.  Do not restore the deleted notebook verbatim; it used
stale installation, parameter-setting, likelihood, and plotting patterns.

**Completed in PR134:** ``tutorial_wavelength_constraints.ipynb`` is now a
maintained public tutorial.  It derives wavelength sampling evidence and
mean/covariance intervals, demonstrates physical-to-model coordinate
round trips, keeps time and wavelength ARD dimensions independent, and
explains the empirically calibrated sparse/noisy failure boundaries.  The
notebook remains advisory and does not convert constraint pressure into
automatic model selection.

**Completed in PR174:** ``tutorial_single_source_analysis.ipynb`` is the
maintained complete single-source workflow for the bundled real source
``examples/data/10131+3049.csv``.  Its default path computes independent period
evidence for every observational channel, runs temporal consensus, applies and
verifies live wavelength constraints before optimizer training, performs a
required ``2DWavelengthDependent`` fit, generates predictions and maintained
plots, evaluates residual and phase diagnostics, distinguishes fixed measurement
variance from learned additional variance, and assembles a JSON-safe report.
The KELT shared-wavelength selection is explicit and applies only to exact-GP
training; both channels remain available to period evidence and consensus.


Notebook maintenance rules
--------------------------

When a public workflow or default changes, update this page and decide whether
an affected notebook should be refreshed, replaced, or removed.  Use structured
markers so pending work is easy to find:

.. code-block:: bash

   rg "TBD\[" docs/source/notebook_status.rst docs/source/notebooks

Before adding or retaining a notebook in the public Tutorials toctree, check
that it satisfies all of the following:

* it has explanatory Markdown, not only code cells;
* it does not contain TODO-placeholder cells;
* it runs under the current package defaults;
* it does not call unavailable APIs;
* it uses current terminology, especially ``model/kernel config`` for the
  advisory wavelength workflow and ``candidate`` only for period/consensus or
  explicitly legacy APIs;
* it has a clear companion how-to page or example script when the workflow is
  important enough to be copied by users; and
* no nonfunctional notebook is hidden only through a Sphinx exclusion pattern.
