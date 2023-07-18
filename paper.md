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
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Sundar Srinivasan
    orcid: 0000-0000-0000-0000
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Diego Alejandro Vasquez
    orcid: 0000-0000-0000-0000
    # corresponding: true # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Sara Jamal
    affiliation: 3
  - name: Stephan Waterval
    affiliation: "4, 5"
affiliations:
 - name: European Southern Observatory, Alonso de Córdova 3107, Vitacura, Santiago, Chile
   index: 1
 - name: IRyA, Universidad Nacional Autónoma de México, Morelia, Michoacán, México
   index: 2
 - name: Max Planck Institute for Astronomy, Königstuhl 17, 69117 Heidelberg, Germany
   index: 3
 - name: New York University Abu Dhabi, PO Box 129188, Abu Dhabi, United Arab Emirates
   index: 4
 - name: Center for Astro, Particle and Planetary Physics (CAPPP), New York University Abu Dhabi, PO Box 129188, Abu Dhabi, United Arab Emirates
   index: 5
date: XX August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary



# Statement of need


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



# References