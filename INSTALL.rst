..
   Copyright (C) 2011,2012 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public
   License. See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesPy is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.

Installation
============

BayesPy is a Python 3 package.  It depends on `SciPy
<http://www.scipy.org/>`_ and `Matplotlib
<http://matplotlib.sourceforge.net/>`_, thus make sure you have them
installed for Python 3.  Instructions can be found at
http://www.scipy.org/Installing_SciPy and
http://matplotlib.sourceforge.net/users/installing.html.  For
matplotlib, tkinter backend works on Python 3 so you may want to
install package python3-tk.

After installing the dependencies, clone the Git repository of
BayesPy:

.. code-block:: console

    git clone --recursive https://github.com/jluttine/bayespy.git
    
Install the package:

.. code-block:: console
    
    cd bayespy
    python3 setup.py install

Make sure you are using Python 3.  BayesPy is now installed and ready
for use.  Updating can be done as follows:

.. code-block:: console

   git pull
   git submodule update --init --recursive
   python3 setup.py install

This will pull the up-to-date version and re-install.

This documentation can be found in docs/ folder.  The documentation
source files are readable as such in reStructuredText format in
docs/source/ directory.  If you have `Sphinx
<http://sphinx.pocoo.org/>`_ installed, the documentation can be
compiled to, for instance, HTML or PDF using

.. code-block:: console

    cd docs
    make html
    make latexpdf

Currently, Python 2 version of Sphinx is required, but in the future
when API documentation is added, Python 3 version of Sphinx will be
required. The documentation can be found also at
http://bayespy.readthedocs.org/ in various formats.
