Notebook status and triage
==========================

.. note::

   **Documentation status:** current through PR76.

   This page is a maintenance index for the notebooks shipped in the repository.
   A notebook listed in the main Tutorials toctree is intended to be usable as
   end-user documentation.  A notebook listed as quarantined, legacy, or pending
   refresh is still present in the repository, but is not advertised as a current
   tutorial until it is modernised and re-executed under the current defaults.

Why this page exists
--------------------

The package has changed substantially: the default time transform is now
``TimeCenter``, the package uses the shared ``DEFAULT_DTYPE`` machinery, the
constraint workflow has been repaired, consensus fitting is now a documented
flagship path, and the period-independent wavelength advisory workflow now
produces model/kernel-config reports and batch artifacts.

Older notebooks may still be useful as development history, but notebooks that
contain TODO cells, call unavailable APIs, or use stale terminology should not be
presented as current tutorials.

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
     - Should be re-executed after major default changes.
   * - ``PGMUVI_Lomb_Scargle.ipynb``
     - Lomb--Scargle and period-candidate concepts.
     - Uses ``candidate`` in the period-search sense, which remains valid.
   * - ``PGMUVI_Gaussian_Process_fitting.ipynb``
     - General GP fitting introduction.
     - Needs a future pass to align examples with current dtype and transform defaults.
   * - ``PGMUVI_comparison_with_other_codes.ipynb``
     - Comparison notebook.
     - Keep linked if it still runs under the current dependency stack.
   * - ``pgmuvi_tutorial.ipynb``
     - General tutorial.
     - Needs a future refresh if it bypasses newer parameter-workflow helpers.
   * - ``PGMUVI_QuasiPeriodic_and_Mean_Functions.ipynb``
     - Quasi-periodic kernels and mean functions.
     - Linked because it contains substantive material; it still needs a re-execution pass.

Quarantined or pending-refresh notebooks
----------------------------------------

These notebooks are intentionally **not** linked from the main Tutorials toctree:

.. list-table::
   :header-rows: 1

   * - Notebook
     - Status
     - Reason
     - Replacement / next action
   * - ``pgmuvi_tutorial_2d.ipynb``
     - Quarantined stub.
     - It is thin and unexecuted, but 2-D multiband fitting is central to the package.
     - **TBD[notebook-2d-consensus]:** replace with a real 2-D consensus fitting notebook.  Until then, use :doc:`howto/consensus_fitting` and ``examples/consensus_multiband_fit.py``.
   * - ``tutorial_preprocessing.ipynb``
     - Pending refresh.
     - Thin and unexecuted relative to the current preprocessing and input-checking APIs.
     - **TBD[notebook-preprocessing-refresh]:** rebuild or replace with a tested preprocessing walkthrough.
   * - ``tutorial_synthetic.ipynb``
     - Quarantined TODO skeleton.
     - Contains placeholder/TODO material rather than a finished tutorial.
     - **TBD[notebook-synthetic-refresh]:** rebuild around the current synthetic-data helpers.
   * - ``tutorial_model_selection.ipynb``
     - Quarantined legacy/stub notebook.
     - Uses old model-selection framing and TODO material.
     - **TBD[notebook-advisory-workflow]:** replace with a period-independent wavelength advisory walkthrough.
   * - ``pgmuvi_tutorial_mcmc.ipynb``
     - Quarantined unavailable workflow.
     - Calls MCMC APIs that currently raise ``NotImplementedError``.
     - **TBD[mcmc-reenable]:** restore only after the MCMC workflow is implemented and tested.
   * - ``pgmuvi_mock_data_from_gp.ipynb``
     - Pending refresh / orphan.
     - Uses older direct-hyperparameter patterns and should be reviewed against the current parameter workflow.
     - **TBD[notebook-mock-data-refresh]:** modernise or remove.

Notebook maintenance rules
--------------------------

When a new public workflow is added or a default changes, update this page and
ask whether an existing notebook should be refreshed, quarantined, or replaced.
Use structured markers so pending work is easy to find:

.. code-block:: bash

   rg "TBD\\[" docs/source/notebook_status.rst docs/source/notebooks

Before adding a notebook to the public Tutorials toctree, check that it satisfies
all of the following:

* it has real explanatory Markdown, not only code cells;
* it does not contain TODO-placeholder cells;
* it runs under the current package defaults;
* it does not call unavailable APIs such as the current MCMC placeholders;
* it uses current terminology, especially ``model/kernel config`` for the
  advisory wavelength workflow and ``candidate`` only for period/consensus or
  explicitly legacy APIs;
* it has a clear companion how-to page or example script when the workflow is
  important enough to be copied by users.
