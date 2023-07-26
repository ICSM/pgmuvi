# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from datetime import date

# -- Project information -----------------------------------------------------

project = 'pgmuvi'
copyright = f'{date.today().year}, Peter Scicluna, Sundar'\
             ' Srinivasan, Stefan Waterval, Kathryn Jones, '\
             'Diego Alejandro Vasquez, Sara Jamal'
author = 'Peter Scicluna, Sundar Srinivasan, Stefan Waterval, '\
         'Kathryn Jones, Diego Alejandro Vasquez, Sara Jamal'
from importlib.metadata import version
release = version('pgmuvi')
# for example take major/minor
version = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'nbsphinx'  # 'sphinx.ext.imgmath'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['test*', "old*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


autodoc_mock_imports = ['torch', 'gpytorch', 'pyro', 'arviz']

nbsphinx_allow_errors = True

# imgmath_latex = "latex"
