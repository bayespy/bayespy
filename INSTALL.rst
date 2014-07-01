..
   Copyright (C) 2011,2012,2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public
   License. See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 3 as
   published by the Free Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.

Installation
============

BayesPy is a Python 3 package and it can be installed from PyPI or the latest
development version from GitHub.  The instructions below explain how to set up
the system by installing required packages, how to install BayesPy and how to
compile this documentation yourself.  However, if these instructions contain
errors or some relevant details are missing, please file a bug report at
https://github.com/bayespy/bayespy/issues.


Installing requirements
-----------------------

BayesPy requires Python 3.2 (or later) and the following packages:

* NumPy (>=1.8.0), 
* SciPy (>=0.13.0) 
* matplotlib (>=1.2)
* h5py

Ideally, a manual installation of these dependencies is not required and you can
skip to the next section "Installing Bayespy".  However, there are several
reasons why the installation of BayesPy as described in the next section won't
work because of your system.  Thus, this section tries to give as detailed and
robust a method of setting up your system such that the installation of BayesPy
should work.

A proper installation of the dependencies for Python 3 can be a bit tricky and
you may refer to http://www.scipy.org/install.html for more detailed
instructions about the SciPy stack.  If your system has an older version of any
of the packages (NumPy, SciPy or matplotlib) or it does not provide the packages
for Python 3, you may set up a virtual environment and install the latest
versions there.  To create and activate a new virtual environment, run

.. code-block:: console

    virtualenv -p python3 --system-site-packages ENV
    source ENV/bin/activate

If you have relevant system libraries installed (C compiler, Python development
files, BLAS/LAPACK etc.), you may be able to install the Python packages from
PyPI.  For instance, on Ubuntu (>= 12.10), you may install the required system
libraries for each package as:

.. code-block:: console

    sudo apt-get build-dep python3-numpy
    sudo apt-get build-dep python3-scipy    
    sudo apt-get build-dep python3-matplotlib
    sudo apt-get build-dep python-h5py

Then installation/upgrade from PyPI should work:

.. code-block:: console

    pip install distribute --upgrade
    pip install numpy --upgrade
    pip install scipy --upgrade
    pip install matplotlib --upgrade
    pip install h5py

Note that Matplotlib requires a quite recent version of Distribute (>=0.6.28).
If you have problems installing any of these packages, refer to the manual of
that package.


Installing BayesPy
------------------

If the system has been properly set up and the virtual environment is activated
(optional), latest release of BayesPy can be installed from PyPI simply as

.. code-block:: console
    
    pip install bayespy

If you want to install the latest development version of BayesPy, use GitHub
instead:

.. code-block:: console

    pip install https://github.com/bayespy/bayespy/archive/master.zip

It is recommended to run the unit tests in order to check that BayesPy is
working properly.  Thus, install Nose and run the unit tests:

.. code-block:: console

    pip install nose
    nosetests bayespy


Compiling documentation
-----------------------

This documentation can be found at http://bayespy.org/.  The documentation
source files are readable as such in reStructuredText format in ``doc/source/``
directory.  It is possible to compile the documentation into HTML or PDF
yourself.  In order to compile the documentation, Sphinx is required and a few
extensions for it. Those can be installed as:

.. code-block:: console

    pip install sphinx sphinxcontrib-tikz sphinxcontrib-bayesnet sphinxcontrib-bibtex

In addition, the ``numpydoc`` extension for Sphinx is required.  However, the
latest stable release (0.4) does not support Python 3, thus one needs to install
the development version:

.. code-block:: console

    pip install https://github.com/numpy/numpydoc/archive/master.zip

In order to visualize graphical models in HTML, you need to have ``pnmcrop``.
On Ubuntu, it can be installed as

.. code-block:: console

    sudo apt-get install netpbm

The documentation can be compiled to HTML and PDF by running the following
commands in the ``doc`` directory:

.. code-block:: console

    make html
    make latexpdf

You can also run doctest to test code snippets in the documentation:

.. code-block:: console

    make doctest

or in the docstrings:

.. code-block:: console

    nosetests --with-doctest bayespy
