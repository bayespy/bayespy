..
   Copyright (C) 2011,2012 Jaakko Luttinen

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



Quick user guide
================

* Construct the model (Bayesian network)

* Put the data in

* Run inference

* Examine posterior results

Simple example
--------------

Let's begin with a simple example which shows the basic steps.  In
this case, we do not use any real data but generate some toy data.
The dataset consists of ten samples from the Gaussian distribution
with mean 5 and standard deviation 10:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (0)
   :end-before: (1)

Now, given this data we would like to estimate the mean and the
standard deviation. We can construct a simple model shown below as a
directed factor graph.

.. bayesnet:: Directed factor graph of the example model.

   \node[obs]                                  (y)     {$y$} ;
   \node[latent, above left=1.5 and 0.5 of y]  (mu)    {$\mu$} ;
   \node[latent, above right=1.5 and 0.5 of y] (tau)   {$\tau$} ;
   \node[const, above=of mu, xshift=-0.5cm]    (mumu)  {$0$} ;
   \node[const, above=of mu, xshift=0.5cm]     (taumu) {$10^{-3}$} ;
   \node[const, above=of tau, xshift=-0.5cm]   (atau)  {$10^{-3}$} ;
   \node[const, above=of tau, xshift=0.5cm]    (btau)  {$10^{-3}$} ;

   \factor[above=of y] {y-f} {left:$\mathcal{N}$} {mu,tau}     {y};
   \factor[above=of mu] {} {left:$\mathcal{N}$}   {mumu,taumu} {mu};
   \factor[above=of tau] {} {left:$\mathcal{G}$}  {atau,btau}  {tau};

   \plate {} {(y)(y-f)(y-f-caption)} {10} ;

Alternatively, the model can also be defined using explicit
mathematical notation:

.. math::

   p(\mathbf{y}|\mu,\tau) &= \prod^{10}_{n=1} \mathcal{N}(y_n|\mu,\tau) \\
   p(\mu) &= \mathcal{N}(\mu|0,10^{-3}) \\
   p(\tau) &= \mathcal{G}(\tau|10^{-3},10^{-3})

Note that we parameterize the normal distribution using the mean and
the precision (i.e., the inverse of the variance).  The model can be
constructed in BayesPy as follows:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (1)
   :end-before: (2)

This is quite self-explanatory given the model definitions above.
Now, we use the generated data:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (2)
   :end-before: (3)

Next we want to estimate the posterior distribution.  In principle, we
could use different inference engines (e.g., MCMC or EP) but currently
only variational Bayesian (VB) engine is implemented.  The engine is
initialized by giving the nodes and the inference algorithm can be run
as long as wanted (20 iterations in this case):

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (3)
   :end-before: (4)

In VB, the true posterior :math:`p(\mu,\tau|\mathbf{y})` is
approximated with a factorized distribution :math:`q(\mu)q(\tau)`.
The resulting approximate posterior distributions :math:`q(\mu)` and
:math:`q(\tau)` can be examined as:

.. literalinclude:: examples/example_01.py
   :language: python3
   :start-after: (4)

.. todo::
   
   Add an example of visualizing the results.

This example was a very simple introduction to using BayesPy.  The
model can be much more complex and each phase contains more options to
give the user more control over the inference.  The following sections
give more details.


Constructing the model
----------------------

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

Providing the data
------------------

The data is provided by simply calling ``observe`` method of the node:

.. code-block:: python3

   y.observe(data)

It is important that the shape of the ``data`` array matches the shape
of the node ``y``, which is the combination of the plates and the
dimensionality.  For instance, if ``y`` is ``Wishart`` node for
:math:`3\times 3` matrices with plates ``(5,1,10)``, the actual shape
of ``y`` would be ``(5,1,10,3,3)``.  The data array must have this
shape exactly, that is, no broadcasting rules are applied.


Missing values
++++++++++++++

It is possible to mark missing values by providing a mask:

.. code-block:: python3

   y.observe(data, mask=[True, False, False, True, True,
                         False, True, True, True, False])

``True`` means that the value is observed and ``False`` means that the
value is missing.  To be more precise, the mask is applied to the
plates, *not* to the data array directly.  Unlike for the data itself,
standard NumPy broadcasting rules are applied for the mask with
respect to the plates.  So, if the variable has plates ``(5,1,10)``,
the mask could have a shape ``(1,)``, ``(10,)``, ``(5,1,1)`` or
``(5,1,10)``.


From implementational point of view, the inference algorithms ignore
the missing plates automatically if they are not needed.  Thus, the
missing values are integrated out giving more accurate approximations
and the computations may also be faster.




Performing inference
--------------------

Approximation of the posterior distribution can be divided into
several steps:

* Choosing and constructing the inference engine
* Initializing the engine
* Running the inference algorithm


Choosing the inference method
+++++++++++++++++++++++++++++

Inference methods can be found in ``bayespy.inference`` package.
Currently, only variational Bayesian approximation is implemented
(``bayespy.inference.VB``).  The inference engine is constructed by
giving the nodes of the model.

.. code-block:: python3

   from bayespy.inference import VB
   Q = VB(node1, node2, node3, node4)

Initializing the inference
++++++++++++++++++++++++++

The inference engines give some initialization to the nodes by
default.  However, the inference algorithms can be sensitive to the
initialization, thus it is sometimes necessary to have full control
over the initialization.  There may be different initialization
methods, but for VB you can, for instance, initialize in one of the
following ways:

* ``initialize_from_prior``: Use only parent nodes to update the node.
* ``initialize_from_parameters``: Use the given parameter values for
  the distribution.

A random initialization for VB has to be performed manually, because
it is not obvious what is actually wanted.  For instance, one way to
achieve it is to first update from the parents, then to draw a random
sample from that distribution and to set the values of the parameters
based on that.  For ``Normal`` node, one could draw the mean parameter
randomly and choose the precision parameter arbitrarily:

.. code-block:: python3

   x = bp.nodes.Normal(mu, tau, plates=(10,))
   x.initialize_from_prior()
   x.initialize_from_parameters(x.random(), 1)

In this case, the precision was set to one.  The default
initialization method is ``initialization_from_prior``, which is
performed when the node is created.  If the initialization uses the
values of the parents, they should be initialized before the children.

.. For VB, a random initialization could perhaps be achieved as
   follows: Draw random samples of the parent parameters but extend
   this sampling to the plates of the child node.  These values can be
   used to set the parameters of the child's distribution? No, it
   doesn't work: Consider constant parent nodes.  Then the child would
   have constant values.


Running the inference algorithm
+++++++++++++++++++++++++++++++

The approximation methods are based on iterative algorithms, which can
be run using ``update`` method.  By default, it takes one iteration
step updating all nodes once.  However, you can give as arguments the
nodes you want to update and they are updated in the given order. It
is possible to give same nodes several times, for instance:

.. code-block:: python3

   Q.update(node1, node3, node1, node4)

This would update ``node3`` and ``node4`` once, and ``node1`` twice.
In order to update several times, one can use the optional argument
``repeat``.
                
.. code-block:: python3

   Q.update(node3, node4, repeat=5)
   Q.update(node1, node2, node3, node4, repeat=10)

This first updates ``node3`` and ``node4`` five times and then all the
nodes ten times.  This might be useful, for instance, if updating some
nodes is expensive and should be done rarely or if updating some nodes
in the beginning would cause the algorithm to converge to a bad
solution.


.. warning::

   Ideally, one constructs the model and then chooses the inference
   method to be used - possibly trying several different methods.
   However, the model construction is not yet separated from the model
   construction, that is, the constructed model network is also the
   variational message passing network for VB inference.


Examining the results
---------------------

After the results have been obtained, it is important to be able to
examine the results easily.  ``show`` method prints the approximate
posterior distribution of the node.  Also, ``get_moments`` can be used
to obtain the sufficient statistics of the node.

.. todo::

   In order to examine the results more carefully, ``get_parameters``
   method should return the parameter values of the approximate
   posterior distribution.  The user may use these values for
   arbitrarily complex further analysis.


