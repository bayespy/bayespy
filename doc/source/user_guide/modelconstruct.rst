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


Constructing the model
======================


In BayesPy, the model is constructed by creating nodes which form a directed
network.  There are two types of nodes: stochastic and deterministic.  A
stochastic node corresponds to a random variable (or a set of random variables)
from a specific probability distribution.  A deterministic node corresponds to a
deterministic function of its parents.


Meaning of nodes
----------------

.. currentmodule:: bayespy.nodes


When constructing the network with nodes, the stochastic nodes actually define
three important aspects: 1) the prior probability distribution for the
variables; 2) the factorization of the posterior approximation; and 3) the
functional form of the posterior approximation for the variables


First, the most intuitive feature of the nodes is that they define the prior
distribution. In the previous example, ``mu`` was a stochastic
:class:`GaussianARD` node corresponding to :math:`\mu` from the normal
distribution, ``tau`` was a stochastic :class:`Gamma` node corresponding to
:math:`\tau` from the gamma distribution, and ``y`` was a stochastic
:class:`GaussianARD` node corresponding to :math:`y` from the normal
distribution with mean :math:`\mu` and precision :math:`\tau`.  If we denote the
set of all nodes by :math:`\Omega`, and by :math:`\pi_X` the set of parents of a
node :math:`X`, the model is defined as

.. math::

   p(\Omega) = \prod_{X \in \Omega} p(X|\pi_X),

where nodes correspond to the terms :math:`p(X|\pi_X)`.


Second, the nodes define the structure of the posterior approximation.  The
variational Bayesian approximation factorizes with respect to nodes, that is,
each node corresponds to an independent probability distribution in the
posterior approximation.  In the previous example, ``mu`` and ``tau`` were
separate nodes, thus the posterior approximation factorizes with respect to
them: :math:`q(\mu)q(\tau)`.  Thus, the posterior approximation can be written
as:

.. math::

   p(\tilde{\Omega}|\hat{\Omega}) \approx \prod_{X \in \tilde{\Omega}} q(X),

where :math:`\tilde{\Omega}` is the set of latent nodes and :math:`\hat{\Omega}`
is the set of observed nodes.  Sometimes one may want to avoid the factorization
between some variables.  For this purpose, there are some nodes which model
several variables jointly without factorization.  For instance,
:class:`GaussianGammaISO` is a joint node for :math:`\mu` and :math:`\tau`
variables from the normal-gamma distribution and the posterior approximation
does not factorize between :math:`\mu` and :math:`\tau`, that is, the posterior
approximation is :math:`q(\mu,\tau)`.


Last, the nodes define the functional form of the posterior approximation.
Usually, the posterior approximation has the same or similar functional form as
the prior.  For instance, :class:`Gamma` uses gamma distribution to also
approximate the posterior distribution.  Similarly, :class:`GaussianARD` uses
Gaussian distribution for the posterior.  However, the posterior approximation
of :class:`GaussianARD` uses a full covariance matrix although the prior assumes
a diagonal covariance matrix.  Thus, there can be slight differences in the
exact functional form of the posterior approximation but the rule of thumb is
that the functional form of the posterior approximation is the same as the
functional form of the prior.


Creating nodes
--------------

Creating a node is basically like writing the conditional prior distribution of
the variable in Python.  The node is constructed by giving the parent nodes,
that is, the conditioning variables as arguments.  The number of parents and
their meaning depend on the node.  For instance, a :class:`Gaussian` node is
created by giving the mean vector and the precision matrix.  These parents can
be constant numerical arrays if they are known:

.. code-block:: python3

   from bayespy.nodes import Gaussian
   X = Gaussian([2, 5], [[1.0, 0.3], [0.3, 1.0]])

or other nodes if they are unknown and given prior distributions:

.. code-block:: python3

   from bayespy.nodes import Gaussian, Wishart
   mu = Gaussian([0, 0], [[1e-6, 0], [0, 1e-6]])
   Lambda = Wishart(2, [[1, 0], [0, 1]])
   X = Gaussian(mu, Lambda)

In Bayesian framework in general, one can give quite arbitrary probability
distributions for variables.  However, one often uses distributions that are
easy to handle in practice.  Quite often this means that the parents are given
conjugate priors.  This is also one of the limitations in BayesPy: only
conjugate family prior distributions are accepted currently.  Thus, although in
principle one could give, for instance, gamma prior for the mean parameter
``mu``, only Gaussian-family distributions are accepted because of the
conjugacy.  If the parent is not of a proper type, an error is raised.


Another a bit rarely encountered limitation is that a node cannot have the same
stochastic node as several parents without intermediate stochastic nodes.  This
means that a stochastic node cannot be given to another node as a parent in
several roles.  For instance, the following would lead to an error:

.. code-block:: python3

   from bayespy.nodes import Gaussian, Dot
   X = Gaussian([0], [[1]])
   Y = Dot(X, X)

The error is raised because ``X`` is given as two parents for ``Y``.  Even if
``X`` is not given several times directly but there are some intermediate
deterministic nodes, an error is raised.  However, it is valid that a node is a
parent of another node via several paths if all except one path has intermediate
stochastic nodes.  Another way to put this is that the parents of a node should
have independent posterior approximations.  Thus, for instance, the following
construction does not raise errors:

.. code-block:: python3

   from bayespy.nodes import Gaussian, Dot
   X = Gaussian([0], [[1]])
   Z = Gaussian(X, [[1]])
   Y = Dot(X, Z)

This works because there is now an intermediate stochastic node ``Z`` on the
other path from ``X`` node to ``Y`` node.


The nodes use a few general optional keyword arguments for defining some properties.

  ``name`` :  name of the node or variable

  ``plates`` : plates of the variable (stochastic variables only)

  ``initialize`` : whether to use default initialization or not (stochastic variables only)


Defining plates
---------------

In general, the nodes take some optional parameters: ``name`` and
``plates``.  The parameter ``name`` is used to give a name for the
variable.  The parameter ``plates`` is used to define plates, that is,
a repetitive collection of nodes that are independent given the
parents. For instance, the following set of i.i.d. random variables

.. math::
   
   y_{mn} \sim \mathcal{N}(\mu, \tau),\qquad m=1,\ldots,10,
   \quad n=1,\ldots,30

would be created as

.. code-block:: python3

   from bayespy.nodes import GaussianARD
   y = GaussianARD(mu, tau, plates=(10,30))

It is also possible that the parents have plates.  The validity of the
plates between a child and a parent is checked by comparing the plates
plate-wise from the trailing plates and working the way forward.  A
plate of the child is compatible with a plate of the parent if either
of the following conditions is met:

1) The two plates have equal size
2) The parent has size 1 (or no plate)

Table below shows an example of compatible plates for a child and two
parent nodes.

+---------+-------------------------+
| node    | plates                  |
+=========+===+===+====+===+===+====+
| parent1 |   | 9 |  1 | 5 | 1 | 10 |
+---------+---+---+----+---+---+----+
| parent2 |   |   | 15 | 5 | 1 |  1 |
+---------+---+---+----+---+---+----+
| child   | 5 | 9 | 15 | 5 | 1 | 10 |
+---------+---+---+----+---+---+----+

For instance, a model

.. math::
   
   \mu_m &\sim  \mathcal{N}(0, 10^{-3}), \\
   \tau_n &\sim \mathcal{G}(10^{-3}, 10^{-3}), \\
   y_{mn} &\sim \mathcal{N}(\mu_m, \tau_n),\qquad m=1,\ldots,10,
   \quad n=1,\ldots,30

could be created as

.. code-block:: python3

   mu = bp.nodes.Normal(0, 1e-3, plates=(10,1))
   tau = bp.nodes.Gamma(1e-3, 1e-3, plates=(30,))
   y = bp.nodes.Normal(mu, tau, plates=(10,30))

Multi-dimensional nodes
+++++++++++++++++++++++

Sometimes a random variable is multi-dimensional.  For instance, a
multivariate normal distribution is a probability distribution for
vectors.  Quite often, the dimensionality can be deduced implicitly
from the parents, thus the user may not need to provide it explicitly.
However, it is important to know that the values are stored in a NumPy
array where the plates are the leading axes and the dimensions are the
trailing axes.  This becomes relevant, for instance, when providing
the data for an observed multi-dimensional node.  To make a clear
distinction between scalar and multi-dimensional distributions, there
is often a multi-dimensional counterpart of a scalar node.  For
instance, the normal distribution for scalars is provided by the node
``Normal``, but the node for the multivariate normal distribution is
``Gaussian``.  Below is a more complete table of correspondence.

============== ==================
Scalar         Multi-dimensional
============== ==================
``Normal``     ``Gaussian``
``Gamma``      ``Wishart``
``Bernoulli``  ``Categorical``
``Binomial``   ``Multinomial``
``Beta``       ``Dirichlet``
============== ==================


Deterministic and constant nodes
++++++++++++++++++++++++++++++++

In addition to the random variable nodes, there are two special types
of nodes: constant and deterministic.  Neither one has any probability
distribution associated with them.  A constant node has no parents and
the value of the node is fixed.  Constant nodes are created implicitly
as parent nodes when you give numeric values as parents when creating
a node, thus, the user is not required to create any constant nodes
explicitly.  A deterministic, on the other hand, defines some
function.  It transforms the parents to produce a new variable which
is a non-random function of the parents.  For instance, ``Dot`` node
computes the dot product of its parents.

