Choosing among fitting workflows
================================

This page is a documentation map for choosing the appropriate ``pgmuvi``
fitting or diagnostic workflow.  It replaces the older habit of treating every
workflow as automatic "model selection".

.. contents:: On this page
   :local:
   :depth: 2

Current direct fitting path
---------------------------

Use :doc:`consensus_fitting` when you want to run an actual multiband GP fit.
The consensus workflow uses multiband period diagnostics to initialise and
constrain the fit, then calls the normal GP fitting machinery.  A typical direct
fit is::

   result = lc.fit(
       model="2D",
       fit_strategy="consensus",
       learn_additional_noise=True,
       training_iter=500,
       miniter=100,
   )

This path fits a model.  It can raise ``ConsensusFitError`` when the bands do
not support a coherent consensus period or the consensus stage cannot construct
a safe fit.

Current wavelength advisory path
--------------------------------

Use :doc:`wavelength_advisory` when you want to compare LPV-relevant wavelength
model families for one source without automatically choosing a winner.  Use
:doc:`wavelength_advisory_batch` for the survey-scale version.

The advisory workflow evaluates ``model/kernel config`` objects such as
``2DWavelengthDependent`` with a quasi-periodic time kernel, ``2DDustMean`` with
a quasi-periodic time kernel, ``2DPowerLawMean`` with a quasi-periodic time
kernel, and the optional ``2D`` spectral-mixture baseline.  It reports rankings
and diagnostics, but it keeps::

   advisory_only = True
   selected_model = None

That distinction is deliberate: a top-ranked advisory model is not an installed
or automatically selected production model.

Legacy candidate-based wavelength diagnostics
---------------------------------------------

The older pre-fit wavelength diagnostics and candidate-comparison workflow is
still documented in :doc:`legacy_wavelength_candidates`.  That page covers the
``diagnose_wavelength_dependence`` and ``compare_wavelength_models`` APIs.

Keep using the legacy page only when you specifically need that older API or
are maintaining code that already depends on its ``recommended_candidate_models``
and candidate-comparison reports.

One-dimensional convenience recommendation
------------------------------------------

``Lightcurve.auto_select_model()`` and the old ``examples/model_selection.py``
script are convenience recommenders.  They should not be confused with the
period-independent wavelength advisory workflow.  They do not replace direct
scientific inspection of fits, diagnostics, residuals, and period summaries.

Terminology
-----------

``candidate`` is still valid for period candidates, consensus component
candidates, and the frozen legacy wavelength-candidate API.  The current
wavelength advisory workflow instead uses ``model/kernel config`` for the
distinct model-plus-kernel fit configurations it evaluates.
