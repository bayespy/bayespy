
BayesPy is a Python 3 package.  It depends on `SciPy
<http://www.scipy.org/>`_ and `Matplotlib
<http://matplotlib.sourceforge.net/>`_, thus make sure you have them
installed for Python 3.  Instructions can be found at
http://www.scipy.org/Installing_SciPy and
http://matplotlib.sourceforge.net/users/installing.html.  After
installing the dependencies, clone the Git repository of BayesPy:

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
