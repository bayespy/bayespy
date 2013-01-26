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

    git clone https://github.com/bayespy/bayespy.git
    
Install the package:

.. code-block:: console
    
    cd bayespy
    python setup.py install

Make sure you are using Python 3.  BayesPy is now installed and ready
for use.  Updating can be done as follows:

.. code-block:: console

   git pull
   python setup.py install

This will pull the up-to-date version and re-install.

This documentation can be found in doc/ folder.  The documentation
source files are readable as such in reStructuredText format in
docs/source/ directory.  If you have `Sphinx
<http://sphinx.pocoo.org/>`_ installed, the documentation can be
compiled to, for instance, HTML or PDF using

.. code-block:: console

    cd doc
    make html
    make latexpdf

Sphinx needs to be installed for Python 3.  The documentation can be
found also at http://bayespy.readthedocs.org/ in various formats.
