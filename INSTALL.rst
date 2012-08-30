
BayesPy is a Python 3 package.  It depends on `SciPy
<http://www.scipy.org/>`_, thus install it if you do not have it
installed.  Instructions can be found at
http://www.scipy.org/Installing_SciPy.  After installing the
dependencies, clone the Git repository:

::

    git clone https://github.com/jluttine/bayespy.git
    
Install the package:

::
    
    cd bayespy
    python setup.py install

Make sure you are using Python 3.

This documentation can be found in docs/ folder.  The documentation
source files are readable as such in reStructuredText format in
docs/source/ directory.  If you have `Sphinx
<http://sphinx.pocoo.org/>`_ installed, the documentation can be
compiled to, for instance, HTML or PDF using

::

    cd docs
    make html
    make latexpdf

The documentation can be found also at
http://bayespy.readthedocs.org/ in various formats.
