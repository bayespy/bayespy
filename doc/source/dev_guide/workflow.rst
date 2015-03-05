..
   Copyright (C) 2014 Jaakko Luttinen

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

Workflow
========

The main forum for BayesPy development is `GitHub
<https://github.com/bayespy/bayespy>`_.  Bugs and other issues can be reported
at https://github.com/bayespy/bayespy/issues.  Contributions to the code and
documentation are welcome and should be given as pull requests at
https://github.com/bayespy/bayespy/pulls.  In order to create pull requests, it
is recommended to fork the git repository, make local changes and submit these
changes as a pull request.  The style guide for writing docstrings follows the
style guide of NumPy, available at
https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt.  Detailed
instructions on development workflow can be read from NumPy guide, available at
http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html.  BayesPy
uses the following acronyms to start the commit message:

 * API: an (incompatible) API change
 * BLD: change related to building numpy
 * BUG: bug fix
 * DEMO: modification in demo code
 * DEP: deprecate something, or remove a deprecated object
 * DEV: development tool or utility
 * DOC: documentation
 * ENH: enhancement
 * MAINT: maintenance commit (refactoring, typos, etc.)
 * REV: revert an earlier commit
 * STY: style fix (whitespace, PEP8)
 * TST: addition or modification of tests
 * REL: related to releasing numpy


Making releases
---------------

 * Commit any current changes to git.

 * Edit version number in setup.py.

 * Add changes to CHANGELOG.rst.

 * Make a commit: ``git commit -am "REL: Version x.x.x"``

 * Tag the release: ``git tag x.x.x``

 * Publish in PyPI: ``python setup.py release_pypi``

 * Update the documentation web page: ``cd doc && make gh-pages``

 * Publish in mloss.org.

 * Announcements to bayespy@googlegroups.com, scipy-user@scipy.org and
   numpy-discussion@scipy.org.
