Notebook status and maintenance
===============================

.. note::

   **Documentation status:** current through PR108.

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
     - Refreshed in PR103 with deterministic data, an explicit no-training default, failure handling, and LPV-relevant follow-up configurations.

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

**TBD[wavelength-constraint-notebook]:** after the D2 multi-seed robustness and
failure-boundary calibration is complete, add a maintained public notebook that
shows how wavelength-kernel scales and bounds are derived, transformed, applied,
and diagnosed.  It must distinguish covariance constraints from wavelength-
dependent mean constraints and explain the tested sparse/noisy failure
boundaries before it is added to the Tutorials toctree.

**D2 calibration prerequisite completed:** the canonical 20-seed,
420-run robustness calibration is now recorded in
:mod:`pgmuvi.wavelength_validation_robustness_calibration`.  The notebook
itself remains outstanding and should use those empirical robust,
degraded, failure-boundary, and expected-failure-boundary results without
turning them into automatic model-selection rules.

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
