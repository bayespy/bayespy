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

BayesPy requires NumPy (>=1.7.1), SciPy (>=0.11) and matplotlib (>=1.2).  A
proper installation of these packages can be a bit tricky and you may refer to
http://www.scipy.org/install.html for more detailed instructions.  If your
system has an older version of any of the packages (NumPy, SciPy or matplotlib),
you may set up a virtual environment and install the latest versions there.  To
create and activate a new virtual environment, run

.. code-block:: console

    virtualenv -p python3.2 --system-site-packages ENV
    source ENV/bin/activate

If your system is properly set up, you may be able to install them from PyPI (a
C compiler, Python development files and BLAS/LAPACK may be required).  For
instance, on Ubuntu (>= 12.10), you may install the dependencies as:

..
    sudo aptitude install build-essential python3.2-dev libatlas-base-dev gfortran

.. code-block:: console

    sudo apt-get build-dep python3-numpy
    sudo apt-get build-dep python3-scipy    
    sudo apt-get build-dep python3-matplotlib
    # sudo apt-get build-dep python3-cython
    # sudo aptitude install python3-tk tk-dev

This guarantees that the required system libraries are installed.  Then
installation/upgrade from PyPI should work:

.. code-block:: console

    pip install numpy --upgrade
    pip install scipy matplotlib --upgrade

Note that this may take several minutes. You also need to install Cython, for
instance, from PyPI:

.. code-block:: console

    pip install cython

Installing BayesPy
------------------

After the system has been properly set up and the virtual environment is
activated (if wanted), BayesPy can be installed from PyPI simply as

.. code-block:: console
    
    pip install bayespy

If you want to install the latest development version of BayesPy, use GitHub
instead:

.. code-block:: console

    pip install https://github.com/bayespy/bayespy/archive/master.zip


Compiling documentation
-----------------------

This documentation can be found in ``doc`` folder.  The documentation source
files are readable as such in reStructuredText format in ``doc/source/``
directory.  If you have `Sphinx <http://sphinx.pocoo.org/>`_ installed, the
documentation can be compiled to, for instance, HTML and PDF by running the
following commands in the ``doc`` folder:

.. code-block:: console

    make html
    make latexpdf

Sphinx needs to be installed for Python 3.  The documentation can be found also
at http://bayespy.org/.





It is recommended that you install BayesPy
using `virtualenv <http://www.virtualenv.org/">`_.  BayesPy requires Python 3.2
(other 3.x versions untested), so to create and activate a new virtual
environment, run

.. code-block:: console

    virtualenv -p python3.2 --system-site-packages ENV
    source ENV/bin/activate

http://www.scipy.org/install.html

Before installing BayesPy, the latest version of `NumPy <http://numpy.org>`_
needs to be pre-installed. 

`Cython <http://cython.org>`_ and the latest version of `NumPy
<http://numpy.org>`_ needs to be pre-installed, so run:

.. code-block:: console

    pip install cython
    pip install numpy --upgrade

Now, BayesPy can be installed simply as

.. code-block:: console
    
    pip install bayespy

and all dependencies (e.g., `SciPy <http://scipy.org>`_) are installed
automatically [#]_. Note that this may take several minutes.  If you want to
install the latest development version of BayesPy, use GitHub instead of PyPI:

.. code-block:: console

    pip install https://github.com/bayespy/bayespy/archive/master.zip

This documentation can be found in ``doc`` folder.  The documentation source
files are readable as such in reStructuredText format in ``doc/source/``
directory.  If you have `Sphinx <http://sphinx.pocoo.org/>`_ installed, the
documentation can be compiled to, for instance, HTML and PDF by running the
following commands in the ``doc`` folder:

.. code-block:: console

    make html
    make latexpdf

Sphinx needs to be installed for Python 3.  The documentation can be found also
at http://bayespy.org/.

.. [#] 

    If you are having problems installing NumPy/SciPy/matplotlib, some
    instructions can be found at http://www.scipy.org/Installing_SciPy and
    http://matplotlib.sourceforge.net/users/installing.html.  For matplotlib,
    tkinter backend works on Python 3 so you may want to install package
    python3-tk.

