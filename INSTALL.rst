..
   Copyright (C) 2011,2012 Jaakko Luttinen

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

.. 
    using `NumPy/SciPy <http://www.scipy.org/>`_ and
    `Matplotlib <http://matplotlib.sourceforge.net/>`_.

BayesPy is a Python 3 package and it can be installed from PyPI or the latest
development version from GitHub.  The instructions below explain how to set up
the system by installing required packages, how to install BayesPy and how to
compile this documentation yourself.  However, if these instructions contain
errors or some relevant details are missing, please file a bug report at
https://github.com/bayespy/bayespy/issues.

Installing requirements
-----------------------

BayesPy requires Python 3.2 and the following packages:

* NumPy (>=1.7.1), 
* SciPy (>=0.11) 
* matplotlib (>=1.2)
* Cython
* h5py

A proper installation of these packages can be a bit tricky and you may refer to
http://www.scipy.org/install.html for more detailed instructions about the SciPy
stack.  If your system has an older version of any of the packages (NumPy, SciPy
or matplotlib) or it does not provide the packages for Python 3.2, you may set
up a virtual environment and install the latest versions there.  To create and
activate a new virtual environment, run

.. code-block:: console

    virtualenv -p python3.2 --system-site-packages ENV
    source ENV/bin/activate

If your system is properly set up, you may be able to install them from PyPI (a
C compiler, Python development files, BLAS/LAPACK and other system files may be
required).  For instance, on Ubuntu (>= 12.10), you may install the dependencies
for each package as:

..
    sudo aptitude install build-essential python3.2-dev libatlas-base-dev gfortran

.. code-block:: console

    sudo apt-get build-dep python3-numpy
    sudo apt-get build-dep python3-scipy    
    sudo apt-get build-dep python3-matplotlib
    sudo apt-get build-dep cython
    sudo apt-get build-dep python-h5py
    # sudo aptitude install python3-tk tk-dev (for matplotlib?)

This guarantees that the required system libraries are installed.  Then
installation/upgrade from PyPI should work:

.. code-block:: console

    pip install numpy --upgrade
    pip install scipy matplotlib --upgrade

Note that this may take several minutes. You also need to install Cython and
h5py, for instance, from PyPI:

.. code-block:: console

    pip install cython h5py

If you have problems installing any of these packages, refer to the manual of
that package.

Installing BayesPy
------------------

Before proceeding, make sure you have installed Cython, h5py and the latest
versions of NumPy, Scipy and matplotlib for Python 3.2.  After the system has
been properly set up and the virtual environment is activated (if wanted),
BayesPy can be installed from PyPI simply as

.. code-block:: console
    
    pip install bayespy

If you want to install the latest development version of BayesPy, use GitHub
instead:

.. code-block:: console

    pip install https://github.com/bayespy/bayespy/archive/master.zip

If you have nose installed in the virtual environment, you can check that
BayesPy is working:

.. code-block:: console

    nosetests bayespy

Compiling documentation
-----------------------

This documentation can be found at http://bayespy.org/.  The documentation
source files are readable as such in reStructuredText format in ``doc/source/``
directory.  It is possible to compile the documentation into HTML or PDF
yourself.  However, this is not currently recommended as it may turn out to be
quite tricky because you need to have a Python 3.2 compatible modification of
numpydoc and a few other packages (e.g., sphinxcontrib-tikz and
sphinxcontrib-bayesnet).  But if you have `Sphinx <http://sphinx.pocoo.org/>`_
and all the requirements installed, the documentation can be compiled to HTML
and PDF by running the following commands in the ``doc`` folder:

.. code-block:: console

    make html
    make latexpdf

