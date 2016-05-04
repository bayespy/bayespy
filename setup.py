#!/usr/bin/env python

################################################################################
# Copyright (C) 2011-2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


# Read version number __version__ from the file
# See: https://packaging.python.org/en/latest/single_source_version/#single-sourcing-the-version
import os
version = {}
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, 'bayespy', 'version.py')) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

NAME         = 'bayespy'
DESCRIPTION  = 'Variational Bayesian inference tools for Python'
AUTHOR       = 'Jaakko Luttinen'
AUTHOR_EMAIL = 'jaakko.luttinen@iki.fi'
URL          = 'http://bayespy.org'
LICENSE      = 'MIT'
VERSION      = __version__
COPYRIGHT    = '2011-2016, Jaakko Luttinen and contributors'


if __name__ == "__main__":

    import os
    import sys

    python_version = int(sys.version.split('.')[0])
    if python_version < 3:
        raise RuntimeError("BayesPy requires Python 3. You are running Python "
                           "{0}.".format(python_version))

    # This is annoying: Because readthedocs.org doesn't support depending on
    # h5py, we need to remove that dependency if we are on the readthedocs
    # servers..
    ON_RTD = os.environ.get('READTHEDOCS') == 'True'
    if ON_RTD:
        # Workaround for https://github.com/rtfd/readthedocs.org/issues/2149
        install_requires = [
            'sphinx>=1.4.0'
        ]
    else:
        install_requires = [
            'numpy>=1.8.0', # 1.8 implements broadcasting in numpy.linalg
            'scipy>=0.13.0', # <0.13 have a bug in special.multigammaln
            'matplotlib>=1.2.0',
            'h5py',
        ]

    # Utility function to read the README file.
    # Used for the long_description.  It's nice, because now 1) we have a top level
    # README file and 2) it's easier to type in the README file than to put a raw
    # string in below ...
    def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

    from setuptools import setup, find_packages

    # Setup for BayesPy
    setup(
        install_requires = install_requires,
        extras_require = {
            'doc': [
                'sphinx>=1.4.0', # 1.4.0 adds imgmath extension
                'sphinxcontrib-tikz>=0.4.2',
                'sphinxcontrib-bayesnet',
                'sphinxcontrib-bibtex',
                'numpydoc>=0.5',
                'nbsphinx',
            ],
            'dev': [
                'nose',
                'nosebook',
            ]
        },
        packages         = find_packages(),
        package_data     = {
            NAME: ["tests/baseline_images/test_plot/*.png"]
        },
        name             = NAME,
        version          = VERSION,
        author           = AUTHOR,
        author_email     = AUTHOR_EMAIL,
        description      = DESCRIPTION,
        license          = LICENSE,
        url              = URL,
        long_description = read('README.rst'),
        keywords         = [
            'variational Bayes',
            'probabilistic programming',
            'Bayesian networks',
            'graphical models',
            'variational message passing'
        ],
        classifiers = [
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Information Analysis'
        ],
        entry_points = {
            'nose.plugins': [
                'warnaserror = bayespy.testing:WarnAsError',
            ]
        },
    )

