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


In BayesPy, the model is constructed by creating nodes which form a
network.  Roughly speaking, a node corresponds to a random variable
from a specific probability distribution.  In the example, ``mu`` was
``Normal`` node corresponding to :math:`\mu` from the normal
distribution.  However, a node can also correspond to a set of random
variables or nodes can be deterministic not corresponding to any
random variable.

When you create a node, you give its parents as parameters.  The role
and the number of the parents depend on the node.  For instance,
``Normal`` node takes two parents (mean and precision) and ``Gamma``
node takes two parents (scale and rate).

.. warning::

   Currently, it is important that the parent has the correct node
   type, because the model construction and VB inference engine are
   not yet separated.  For instance, the parents mean and precision of
   ``Normal`` node must be ``Normal`` and ``Gamma`` nodes (or other
   nodes that have similar output), respectively.  Thus, currently one
   can build only conjugate-exponential family models.

Name and plates
+++++++++++++++

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

   y = bp.nodes.Normal(mu, tau, plates=(10,30))

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

