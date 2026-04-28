Glossary
========

.. glossary::
   :sorted:

   ARD
      *Automatic Relevance Determination.*  A setting that gives each input
      dimension its own length-scale parameter, allowing the model to determine
      which dimensions are most informative.

   Bayesian inference
      A statistical framework in which unknown quantities are treated as random
      variables and updated with observed data via Bayes' theorem to produce a
      *posterior* distribution.

   BIC
      *Bayesian Information Criterion.*  A penalised likelihood metric used for
      model comparison; lower BIC indicates a better balance between fit quality
      and model complexity.

   Cholesky decomposition
      A factorisation of a positive definite matrix used internally by GPyTorch
      to compute GP predictions and marginal likelihoods efficiently.

   Coherence time
      The timescale over which a quasi-periodic oscillation remains phase-coherent.
      In the spectral mixture kernel, longer coherence corresponds to narrower
      bandwidth (smaller ``mixture_scales``).

   Constraint
      A hard bound placed on a GP hyperparameter, preventing the optimiser from
      exploring values outside a specified range.  Implemented as a GPyTorch
      ``Interval`` constraint in ``pgmuvi``.

   Excess variance
   F_var
      A measure of variability amplitude equal to the excess variance beyond what
      is expected from measurement noise alone, normalised by the mean flux.
      Defined as :math:`F_\mathrm{var} = \sqrt{S^2 - \bar{\sigma^2}} / \bar{x}`.

   False alarm probability
   FAP
      The probability of observing a periodogram peak of a given height by
      chance if the data contain no periodic signal.  Used to assess the
      statistical significance of a detected period.  Available via
      :class:`~pgmuvi.multiband_ls_significance.MultibandLSWithSignificance`.

   Gaussian process
   GP
      A probability distribution over functions, fully specified by a mean
      function and a covariance (kernel) function.  Any finite collection of
      function values has a multivariate Gaussian distribution.

   GPyTorch
      An open-source Python library for Gaussian process inference built on
      PyTorch.  ``pgmuvi`` is built on top of GPyTorch.

   Hyperparameter
      A parameter of the GP kernel or likelihood (as opposed to a *function*
      value).  Examples include the frequency, bandwidth, and weight of a
      spectral mixture component, and the noise variance.

   Kernel function
   Covariance function
      A function :math:`k(\mathbf{x}, \mathbf{x}')` that defines the covariance
      between the GP function values at inputs :math:`\mathbf{x}` and
      :math:`\mathbf{x}'`.  The choice of kernel encodes prior assumptions about
      the smoothness, periodicity, and other properties of the modelled function.

   Lomb–Scargle periodogram
      A method for estimating the power spectral density of unevenly sampled
      time series data.  Used in ``pgmuvi`` to initialise spectral mixture
      kernel frequencies before optimisation.

   MAP estimation
   Maximum a posteriori estimation
      An optimisation-based approach that finds the single parameter configuration
      maximising the posterior probability.  Fast but does not provide uncertainty
      estimates on the parameters.

   MCMC
   Markov chain Monte Carlo
      A class of algorithms for sampling from a probability distribution (the
      posterior in Bayesian inference).  MCMC provides full uncertainty
      quantification but is more computationally expensive than MAP estimation.
      MCMC support (via Hamiltonian Monte Carlo) is planned for a future
      release of ``pgmuvi``; the current version supports MAP estimation only.

   Mixture component
      One term in a spectral mixture kernel, characterised by a centre frequency,
      bandwidth, and weight.  The number of components is controlled by
      ``num_mixtures``.

   Nyquist period
      The shortest variability period that can be reliably detected given the
      sampling cadence, equal to approximately twice the median inter-observation
      interval.  Signals shorter than the Nyquist period are aliased.

   Period
      The characteristic timescale of a quasi-periodic signal.  In the spectral
      mixture kernel, the period is the reciprocal of the mixture mean (centre
      frequency): :math:`P = 1 / \mu_q`.

   PSD
   Power spectral density
      A function describing how the variance of a time series is distributed
      across frequencies.  Peaks in the PSD indicate quasi-periodic variability;
      broad low-frequency power indicates correlated (red) noise.

   Prior
      A probability distribution placed on a GP hyperparameter before observing
      any data.  Priors encode domain knowledge (e.g., expected period range)
      and regularise the posterior during MCMC sampling.

   Quasi-periodic variability
      Variability that is approximately periodic but lacks exact phase coherence;
      the period drifts or the amplitude modulates over time.  Well described by
      a spectral mixture kernel with finite bandwidth.

   RBF kernel
   Squared-exponential kernel
      A smooth kernel defined by :math:`k(r) = \exp(-r^2 / 2\ell^2)` where
      :math:`\ell` is the length-scale.  Represents smooth, aperiodic variability.

   Red noise
      Stochastic variability with power concentrated at low frequencies (long
      timescales).  Common in AGN and many other astrophysical sources.

   R-hat
   :math:`\hat{R}`
      A convergence diagnostic for MCMC.  Values close to 1.0 (< 1.01 is the
      standard criterion) indicate that multiple chains have converged to the
      same distribution.

   Separable kernel
      A multi-dimensional kernel of the form
      :math:`k(\mathbf{x}, \mathbf{x}') = k_1(x_1, x_1') \cdot k_2(x_2, x_2')`.
      ``pgmuvi``'s separable 2D model family (e.g. ``"2DSeparable"``,
      ``"2DAchromatic"``) uses product kernels with a temporal component and a
      wavelength component.  The default ``model="2D"`` uses a non-separable 2D
      spectral-mixture kernel instead.

   Spectral mixture kernel
   SMK
      A kernel defined by a mixture of Gaussians in the frequency domain (see
      Wilson & Adams 2013).  Highly flexible; can represent quasi-periodic
      signals, red noise, and multiple simultaneous periodicities.

   Stetson K
      A robust index of light-curve variability, sensitive to correlated deviations
      from the mean flux.  Less sensitive to outliers than the chi-square test.

   White noise
      Stochastic variability that is uncorrelated between observations (flat PSD).
      Modelled in ``pgmuvi`` by the GP likelihood noise parameter.
