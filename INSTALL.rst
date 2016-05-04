..
   Copyright (C) 2011-2012,2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Installation
============

BayesPy is a Python 3 package and it can be installed from PyPI or the latest
development version from GitHub.  The instructions below explain how to set up
the system by installing required packages, how to install BayesPy and how to
compile this documentation yourself.  However, if these instructions contain
errors or some relevant details are missing, please file a bug report at
https://github.com/bayespy/bayespy/issues.


Installing BayesPy
------------------

BayesPy can be installed easily by using Pip if the system has been properly set
up.  If you have problems with the following methods, see the following section
for some help on installing the requirements.  For instance, a bug in recent
versions of h5py and pip may require you to install some of the requirements
manually.

For users
+++++++++

First, you may want to set up a virtual environment.  Using virtual environment
is optional but recommended.  To create and activate a new virtual environment,
run (in the folder in which you want to create the environment):

.. code-block:: console

    virtualenv -p python3 --system-site-packages ENV
    source ENV/bin/activate

The latest release of BayesPy can be installed from PyPI simply as

.. code-block:: console
    
    pip install bayespy

If you want to install the latest development version of BayesPy, use GitHub
instead:

.. code-block:: console

    pip install git+https://github.com/bayespy/bayespy.git@develop

For developers
++++++++++++++

If you want to install the development version of BayesPy in such a way that you
can easily edit the package, follow these instructions.  Get the git repository:

.. code-block:: console

    git clone https://github.com/bayespy/bayespy.git
    cd bayespy

Create and activate a new virtual environment (optional but recommended):

.. code-block:: console

    virtualenv -p python3 --system-site-packages ENV
    source ENV/bin/activate

Install BayesPy in editable mode:

.. code-block:: console

    pip install -e .

Checking installation
+++++++++++++++++++++

If you have problems installing BayesPy, read the next section for more details.
It is recommended to run the unit tests in order to check that BayesPy is
working properly.  Thus, install Nose and run the unit tests:

.. code-block:: console

    pip install nose
    nosetests bayespy


Installing requirements
-----------------------

BayesPy requires Python 3.3 (or later) and the following packages:

* NumPy (>=1.8.0), 
* SciPy (>=0.13.0) 
* matplotlib (>=1.2)
* h5py

Ideally, Pip should install the necessary requirements and a manual installation
of these dependencies is not required.  However, there are several reasons why
the installation of these dependencies needs to be done manually in some cases.
Thus, this section tries to give some details on how to set up your system.  A
proper installation of the dependencies for Python 3 can be a bit tricky and you
may refer to http://www.scipy.org/install.html for more detailed instructions
about the SciPy stack.  Detailed instructions on installing recent SciPy stack
for various platforms is out of the scope of these instructions, but we provide
some general guidance here.  There are basically three ways to install the
dependencies:

  1. Install a Python distribution which includes the packages.  For Windows,
     Mac and Linux, there are several Python distributions which include all the
     necessary packages:
     http://www.scipy.org/install.html#scientific-python-distributions.  For
     instance, you may try `Anaconda <http://continuum.io/downloads>`_ or
     `Enthought <https://www.enthought.com/products/canopy/>`_.

  2. Install the packages using the system package manager.  On Linux, the
     packages might be called something like ``python-scipy`` or ``scipy``.
     However, it is possible that these system packages are not recent enough
     for BayesPy.

  3. Install the packages using Pip:

     .. code-block:: console

        pip install "distribute>=0.6.28"
        pip install "numpy>=1.8.0" "scipy>=0.13.0" "matplotlib>=1.2" h5py

     This also makes sure you have recent enough version of Distribute (required
     by Matplotlib).  However, this installation method may require that the
     system has some libraries needed for compiling (e.g., C compiler, Python
     development files, BLAS/LAPACK).  For instance, on Ubuntu (>= 12.10), you
     may install the required system libraries for each package as:

     .. code-block:: console

        sudo apt-get build-dep python3-numpy
        sudo apt-get build-dep python3-scipy    
        sudo apt-get build-dep python3-matplotlib
        sudo apt-get build-dep python-h5py

     Then installation using Pip should work.  


Compiling documentation
-----------------------

This documentation can be found at http://bayespy.org/ in HTML and PDF formats.
The documentation source files are also readable as such in reStructuredText
format in ``doc/source/`` directory.  It is possible to compile the
documentation into HTML or PDF yourself.  In order to compile the documentation,
Sphinx is required and a few extensions for it. Those can be installed as:

.. code-block:: console

    pip install "sphinx>=1.2.3" sphinxcontrib-tikz sphinxcontrib-bayesnet sphinxcontrib-bibtex "numpydoc>=0.5"

Or you can simply install BayesPy with ``doc`` extra, which will take care of
installing the required dependencies:

.. code-block:: console

    pip install bayespy[doc]

In order to visualize graphical models in HTML, you need to have ``ImageMagick``
or ``Netpbm`` installed.  The documentation can be compiled to HTML and PDF by
running the following commands in the ``doc`` directory:

.. code-block:: console

    make html
    make latexpdf

You can also run doctest to test code snippets in the documentation:

.. code-block:: console

    make doctest

or in the docstrings:

.. code-block:: console

    nosetests --with-doctest --doctest-options="+ELLIPSIS" bayespy
