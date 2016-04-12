BayesPy - Bayesian Python
=========================

BayesPy provides tools for Bayesian inference with Python.  The user
constructs a model as a Bayesian network, observes data and runs
posterior inference.  The goal is to provide a tool which is
efficient, flexible and extendable enough for expert use but also
accessible for more casual users.

Currently, only variational Bayesian inference for
conjugate-exponential family (variational message passing) has been
implemented.  Future work includes variational approximations for
other types of distributions and possibly other approximate inference
methods such as expectation propagation, Laplace approximations,
Markov chain Monte Carlo (MCMC) and other methods. Contributions are
welcome.


Project information
-------------------

Copyright (C) 2011-2016 Jaakko Luttinen and other contributors (see below)

BayesPy including the documentation is licensed under the MIT License. See
LICENSE file for a text of the license or visit
http://opensource.org/licenses/MIT.

.. |chat| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/bayespy/bayespy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |release| image:: https://badge.fury.io/py/bayespy.svg
   :target: https://pypi.python.org/pypi/bayespy

============== =============================================
Latest release |release| 
Documentation  http://bayespy.org
Repository     https://github.com/bayespy/bayespy.git
Bug reports    https://github.com/bayespy/bayespy/issues
Author         Jaakko Luttinen jaakko.luttinen@iki.fi
Chat           |chat|
Mailing list   bayespy@googlegroups.com
============== =============================================


Continuous integration
++++++++++++++++++++++

.. |travismaster| image:: https://travis-ci.org/bayespy/bayespy.svg?branch=master
   :target: https://travis-ci.org/bayespy/bayespy/
   :align: middle
.. |travisdevelop| image:: https://travis-ci.org/bayespy/bayespy.svg?branch=develop
   :target: https://travis-ci.org/bayespy/bayespy/
   :align: middle
.. |covermaster| image:: https://coveralls.io/repos/bayespy/bayespy/badge.svg?branch=master
   :target: https://coveralls.io/r/bayespy/bayespy?branch=master
   :align: middle
.. |coverdevelop| image:: https://coveralls.io/repos/bayespy/bayespy/badge.svg?branch=develop
   :target: https://coveralls.io/r/bayespy/bayespy?branch=develop
   :align: middle
.. |docsmaster| image:: https://img.shields.io/badge/docs-master-blue.svg?style=flat
   :target: http://www.bayespy.org/en/stable/
   :align: middle
.. |docsdevelop| image:: https://img.shields.io/badge/docs-develop-blue.svg?style=flat
   :target: http://www.bayespy.org/en/latest/
   :align: middle

==================== =============== ============== =============
Branch               Test status     Test coverage  Documentation
==================== =============== ============== =============
**master (stable)**  |travismaster|  |covermaster|  |docsmaster|
**develop (latest)** |travisdevelop| |coverdevelop| |docsdevelop|
==================== =============== ============== =============
 

Similar projects
----------------

`VIBES <http://vibes.sourceforge.net/>`_
(http://vibes.sourceforge.net/) allows variational inference to be
performed automatically on a Bayesian network.  It is implemented in
Java and released under revised BSD license.

`Bayes Blocks <http://research.ics.aalto.fi/bayes/software/>`_
(http://research.ics.aalto.fi/bayes/software/) is a C++/Python
implementation of the variational building block framework.  The
framework allows easy learning of a wide variety of models using
variational Bayesian learning.  It is available as free software under
the GNU General Public License.

`Infer.NET <http://research.microsoft.com/infernet/>`_
(http://research.microsoft.com/infernet/) is a .NET framework for
machine learning.  It provides message-passing algorithms and
statistical routines for performing Bayesian inference.  It is partly
closed source and licensed for non-commercial use only.

`PyMC <https://github.com/pymc-devs/pymc>`_
(https://github.com/pymc-devs/pymc) provides MCMC methods in Python.
It is released under the Academic Free License.

`OpenBUGS <http://www.openbugs.info>`_ (http://www.openbugs.info) is a
software package for performing Bayesian inference using Gibbs
sampling.  It is released under the GNU General Public License.

`Dimple <http://dimple.probprog.org/>`_ (http://dimple.probprog.org/) provides
Gibbs sampling, belief propagation and a few other inference algorithms for
Matlab and Java.  It is released under the Apache License.

`Stan <http://mc-stan.org/>`_ (http://mc-stan.org/) provides inference using
MCMC with an interface for R and Python.  It is released under the New BSD
License.

`PBNT - Python Bayesian Network Toolbox <http://pbnt.berlios.de/>`_
(http://pbnt.berlios.de/) is Bayesian network library in Python supporting
static networks with discrete variables.  There was no information about the
license.


Contributors
------------

The list of contributors:

* Jaakko Luttinen

* Hannu Hartikainen

* Deebul Nair

* Christopher Cramer

Each file or the git log can be used for more detailed information.
