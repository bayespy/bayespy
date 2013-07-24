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

Copyright (C) 2011-2013 Jaakko Luttinen, Aalto University

BayesPy including the documentation is licensed under Version 3.0 of
the GNU General Public License. See LICENSE file for a text of the
license or visit http://www.gnu.org/copyleft/gpl.html.

* Documentation: http://bayespy.org or at ``docs`` directory

* Repository: https://github.com/bayespy/bayespy.git

* Bug reports: https://github.com/bayespy/bayespy/issues

* Mailing list: bayespy@googlegroups.com

* IRC: #bayespy @ `freenode <http://freenode.net/>`_

* Author: Jaakko Luttinen jaakko.luttinen@iki.fi

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

