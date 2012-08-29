Installation
============

The module depends on `SciPy <http://www.scipy.org/>`_, thus install
it if you do not have it installed.  Instructions can be found at
http://www.scipy.org/Installing_SciPy.  After installing the
dependencies, clone the Git repository:

::

    git clone https://github.com/jluttine/bayespy.git
    
Compile and install the module:

::
    
    cd bayespy
    make
    make install

This documentation can be found in docs/ folder.  The documentation
source files are readable as such in reStructuredText format.  If you
have `Sphinx <http://sphinx.pocoo.org/>`_ installed, the documentation
can be compiled to, for instance, HTML or PDF using

::

    cd docs
    make html
    make latexpdf

The documentation can be found also at
http://bayespy.readthedocs.org/ in various formats.
