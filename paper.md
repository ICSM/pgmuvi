---
title: '`pgmuvi`: Quick and easy Gaussian Process Regression for multi-wavelength astronomical timeseries' #A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - timeseries
  - Gaussian processes
  - 
authors:
  - name: Peter Scicluna
    orcid: 0000-0002-1161-3756
    # equal-contrib: true
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Sundar Srinivasan
    orcid: 0000-0000-0000-0000
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Diego Alejandro Vasquez
    orcid: 0000-0000-0000-0000
    # corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Sara Jamal
    affiliation: 4
  - name: Stephan Waterval
    affiliation: "5, 6"
affiliations:
 - name: European Southern Observatory, Alonso de Córdova 3107, Vitacura, Santiago, Chile
   index: 1
 - name: Space Science Institute, 4750 Walnut Street, Suite 205, Boulder, CO 80301, USA
   index: 2
 - name: IRyA, Universidad Nacional Autónoma de México, Morelia, Michoacán, México
   index: 3
 - name: Max Planck Institute for Astronomy, Königstuhl 17, 69117 Heidelberg, Germany
   index: 4
 - name: New York University Abu Dhabi, PO Box 129188, Abu Dhabi, United Arab Emirates
   index: 5
 - name: Center for Astro, Particle and Planetary Physics (CAPPP), New York University Abu Dhabi, PO Box 129188, Abu Dhabi, United Arab Emirates
   index: 6
date: XX August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Time-domain observations are increasingly important in astronomy, and are often the only way to study certain objects.
The volume of time-series data is exploding as new surveys come online - for example, the Vera Rubin Observatory will produce 15 terabytes of data per night, and its Legacy Survey of Space and Time (LSST) is expected to produce five-year lightcurves for $>10^7$ sources, each consisting of 5 bands.
Historically, astronomers have worked with Fourier-based techniques such as the Lomb-Scargle periodogram or information-theoretic approaches; however, in recent years Bayesian and data-driven approaches such as Gaussian Process Regression (GPR) have gained traction.
However, the computational complexity and steep learning curve of GPR has limited its adoption.
`pgmuvi` is aims to make GPR of multi-band timeseries accessible to astronomers while having a small computational footprint.
By building on cutting-edge machine learning libraries, `pgmuvi` retains the speed and flexibility of GPR while being easy to use.
It provides easy access to GPU acceleration and Bayesian inference of the hyperparameters (e.g. the periods), and is able to scale to large datasets.


# Statement of need

Astronomical objects are in general not static, but vary in brightness over time. 
This is especially true for objects that are variable by nature, such as pulsating stars, or objects that are variable due to their orbital motion, such as eclipsing binaries. 
The study of these objects is called time-domain astronomy, and is a rapidly growing field. 
A wide range of approaches to time-series analysis have been developed, ranging from simple period-finding algorithms to more complex machine learning techniques.
Perhaps the most popular in astronomy is the Lomb-Scargle periodogram, which is a Fourier-based technique that is able to find periodic signals in unevenly sampled data.
However, the handling of unevenly sampled data is not the only challenge in time-series analysis.
<!-- The study of time-domain astronomy is often hampered by the fact that the data is not always of the same quality, or that the data is not always available in the same wavelength. 
This is especially true for data from space-based telescopes, which are often limited in their lifetime, and thus the amount of data that can be collected. -->

A particular challenge in astronomy is handling heterogeneous, multiwavelength data.
Data must often be combined from a wide variety of instruments, telecsope or surveys, and so the systematics or noise properties of different datasets can vary widely.
In addition, by combining multiple wavelengths, we can gain a better understanding of the physical processes that are driving the variability of the object.
For example, some variability mechanisms differ across wavelength only in amplitude (e.g. eclipsing binaries), while others may vary in phase (e.g. pulsating stars) or even period with wavelength (e.g. multiperiodic systems).
Thus, it is important to be able to combine data from multiple wavelengths in a way that is able to account for these differences.

Gaussian Processes (GPs) are a popular way to handle these challenges.
GPs are a flexible way to forward-model arbitrary signals, by assuming the signal is drawn from a multivariate Gaussian distribution.
By constructing a covariance function that describes the covariance between any two points in the signal, we can model the signal as a Gaussian process.
By doing so, we are freed from any assumptions about sampling, and can model the signal as a continuous function.
We are also able to model the noise in the data, and thus can account for heteroscedastic noise.
We also gain the ability to directly handle multiple wavelengths, by constructing a multi-dimensional covariance function.

However, GP regression is not without its challenges.
The most popular covariance functions are often not able to model complex signals, and thus the user must construct their own covariance function.
GPs are also computationally expensive, and thus approximations must be used to scale to large datasets.


In this paper we present a new Python package, `pgmuvi`, which is designed to perform Gaussian Process Regression (GPR) on multi-wavelength astronomical time-series data.
GPR is a machine learning technique that is able to model non-periodic signals in unevenly sampled data, and is thus well suited for the analysis of astronomical time-series data.
The package is designed to be easy to use, and to provide a quick way to perform GPR on multi-wavelength data.
The package is also designed to be flexible, and to allow the user to customize the GPR model to their needs.
`pgmuvi` exploits multiple strategies to scale regression to large datasets, making it suitable for the current era of large-scale astronomical surveys.

# Method and Features

`pgmuvi` builds on the popular GPyTorch library.
GPyTorch [@gardner2018gpytorch] is a Gaussian Process library for PyTorch, which is a popular machine learning library for Python.
By default, `pgmuvi` exploits the highly-flexible Spectral Mixture kernel [@wilson:2013] in GPyTorch, which is able to model a wide range of signals.
This kernel is particularly interesting for astronomical time-series data, as it is able to effectively model multi-periodic and quasi-periodic signals.
The spectral mixture kernel models the power spectrum of the covariance matrix as Gaussian mixture model (GMM), making it highly flexible and easy to interpret, while being able to extend to multi-dimensional input easily.
This kernel also is known for its ability to extrapolate effectively, and is thus well suited to cases where prediction is important (for example, preparing astronomical observations of variable stars).
By modelling the power spectrum in this way, `pgmuvi` effectively suppresses noise in the periodogram, and thus is able to find the dominant periods in noisy data more effectively than, for example, the Lomb-Scargle periodogram.

However, the flexibility of this kernel comes at a cost; for more than one component in the mixture, the solution space becomes highly non-convex, and thus the optimization of the kernel hyperparameters becomes difficult.
`pgmuvi` addresses this by first exploiting the Lomb-Scargle periodogram to find the dominant periods in the data, and then using these periods as initial guesses for the means of the mixture components.


Multiple options are available to accelerate inference depending on the size of the dataset.
For small datasets, the exact GPs can be used, which is able to scale to datasets of up to $\sim1000$ points.
`pgmuvi` can also exploit the Structured Kernel Interpolation (SKI) approximation [@wilson2015kernel] to scale to datasets of up to $\sim10^5$ points.
Future work will include implementing approximations for larger datasets: `pgmuvi` can in principle exploit the Sparse Variational GP (SVGP) or Variational Nearest Neighbour (VNN) approximations [@hensman2013gaussian; @wu2022variational] to scale to datasets of almost arbitrary size.
`pgmuvi` can employ GPU computing for both exact and variational GPs through a simple switch if a GPU is available.

For exact GPs and SKI, `pgmuvi` can perform maximum a posteriori (MAP) estimation of the hyperparameters, or can perform full Bayesian inference.
MAP estimation can exploit any `pytorch` optimizer, but defaults to using ADAM [@kingma2014adam].
Bayesian inference uses the `pyro` implementation of the No-U-Turn Sampler (NUTS) [@hoffman2014no], which is a Hamiltonian Monte Carlo (HMC) sampler.

To summarise, the key features of `pgmuvi` are:

- Highly flexible kernel, able to model a wide range of multiwavelength signals
- Able to exploit multiple strategies to scale to large datasets
- GPU acceleration for both exact and variational GPs
- Fully Bayesian inference using HMC
- Initialisation of kernel hyperparameters using Lomb-Scargle periodogram
- Automatic creation of diagnostic and summary plots
- Automatic reporting of kernel hyperparameters and their uncertainties, and summary of MCMC chains.



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This project was developed in part at the 2022 Astro Hack Week, hosted by the Max Planck Institute for Astronomy  and Haus der Astronomie in Heidelberg, Germany.
This work was partially supported by the Max Planck Institute for Astronomy, the European Space Agency, the Gordon and Betty Moore Foundation, the Alfred P. Sloan foundation.


# References