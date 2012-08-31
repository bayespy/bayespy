
BayesPy is a Python 3 package.  It depends on `SciPy
<http://www.scipy.org/>`_ and `Matplotlib
<http://matplotlib.sourceforge.net/>`_, thus make sure you have them
installed for Python 3.  Instructions can be found at
http://www.scipy.org/Installing_SciPy and
http://matplotlib.sourceforge.net/users/installing.html.  After
installing the dependencies, clone the Git repository of BayesPy:

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
<http://sphinx.pocoo.org/>`_ installed for Python 3, the documentation
can be compiled to, for instance, HTML or PDF using

::

    cd docs
    make html
    make latexpdf

Again, make sure you are using Python 3 version of Sphinx.  The
documentation can be found also at http://bayespy.readthedocs.org/ in
various formats.
