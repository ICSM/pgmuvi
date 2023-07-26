.. pgmuvi documentation master file, created by
   sphinx-quickstart on Thu Jul 20 11:38:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to `pgmuvi`'s documentation!
====================================

`pgmuvi` is a package for interpreting astronomical timeseries data (although there's no reason you can't use it for other kinds of data!) using Gaussian processes. 
It is built on top of the `gpytorch <https://gpytorch.ai/>`_ package, and is designed to be easy to use and flexible. 
It is currently under active development, and we welcome contributions!

User Guide
----------

.. toctree::
   :maxdepth: 2

   notebooks/pgmuvi_tutorial
   notebooks/pgmuvi_tutorial_mcmc
   

Installation and Quickstart
---------------------------

`pgmuvi` can be installed easily with pip::

    $ pip install pgmuvi

You can also clone the latest version of `pgmuvi` from Github, for all the latest bugs but increased risk of features::

    $ git clone git://github.com/ICSM/pgmuvi.git

and then you can install it::
  
    $ cd pgmuvi
    $ pip install .


If you want to contribute to `pgmuvi` and develop new features, you might want an *editable* install::

    $ pip install -e .

this way you can test how things change as you go along.

Citing `pgmuvi`
----------------

`pgmuvi` is currently under review in the Journal of Open Source Software. 
If you use `pgmuvi` in your research, please cite the paper (details will be given here when the paper is accepted!)

API reference
-------------

.. toctree::
   :maxdepth: 2

   api


Contributing
------------

We very much welcome contributions to `pgmuvi`! Please take a look at our `github repository <https://github.com/ICSM/pgmuvi/>`_ for more information on how to contribute!



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
