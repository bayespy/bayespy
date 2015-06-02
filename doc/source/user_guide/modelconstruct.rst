..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


.. currentmodule:: bayespy.nodes
                
Constructing the model
======================

In BayesPy, the model is constructed by creating nodes which form a directed
network.  There are two types of nodes: stochastic and deterministic.  A
stochastic node corresponds to a random variable (or a set of random variables)
from a specific probability distribution.  A deterministic node corresponds to a
deterministic function of its parents. For a list of built-in nodes, see the
:ref:`sec-user-api`.

                
Creating nodes
--------------

                
Creating a node is basically like writing the conditional prior distribution of
the variable in Python.  The node is constructed by giving the parent nodes,
that is, the conditioning variables as arguments.  The number of parents and
their meaning depend on the node.  For instance, a :class:`Gaussian` node is
created by giving the mean vector and the precision matrix. These parents can be
constant numerical arrays if they are known:
                
>>> from bayespy.nodes import Gaussian
>>> X = Gaussian([2, 5], [[1.0, 0.3], [0.3, 1.0]])

or other nodes if they are unknown and given prior distributions:

>>> from bayespy.nodes import Gaussian, Wishart
>>> mu = Gaussian([0, 0], [[1e-6, 0],[0, 1e-6]])
>>> Lambda = Wishart(2, [[1, 0], [0, 1]])
>>> X = Gaussian(mu, Lambda)

Nodes can also be named by providing ``name`` keyword argument:

>>> X = Gaussian(mu, Lambda, name='x')

The name may be useful when referring to the node using an inference engine.

For the parent nodes, there are two main restrictions: non-constant parent nodes
must be conjugate and the parent nodes must be mutually independent in the
posterior approximation.

Conjugacy of the parents
~~~~~~~~~~~~~~~~~~~~~~~~

In Bayesian framework in general, one can give quite arbitrary probability
distributions for variables. However, one often uses distributions that are easy
to handle in practice. Quite often this means that the parents are given
conjugate priors. This is also one of the limitations in BayesPy: only conjugate
family prior distributions are accepted currently. Thus, although in principle
one could give, for instance, gamma prior for the mean parameter ``mu``, only
Gaussian-family distributions are accepted because of the conjugacy. If the
parent is not of a proper type, an error is raised. This conjugacy is checked
automatically by BayesPy and ``NoConverterError`` is raised if a parent cannot
be interpreted as being from a conjugate distribution.

Independence of the parents
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another a bit rarely encountered limitation is that the parents must be mutually
independent (in the posterior factorization). Thus, a node cannot have the same
stochastic node as several parents without intermediate stochastic nodes. For
instance, the following leads to an error:

                
>>> from bayespy.nodes import Dot
>>> Y = Dot(X, X)
Traceback (most recent call last):
    ...
ValueError: Parent nodes are not independent
                
The error is raised because ``X`` is given as two parents for ``Y``, and
obviously ``X`` is not independent of ``X`` in the posterior approximation. Even
if ``X`` is not given several times directly but there are some intermediate
deterministic nodes, an error is raised because the deterministic nodes depend
on their parents and thus the parents of ``Y`` would not be independent.
However, it is valid that a node is a parent of another node via several paths
if all the paths or all except one path has intermediate stochastic nodes. This
is valid because the intermediate stochastic nodes have independent posterior
approximations. Thus, for instance, the following construction does not raise
errors:

>>> from bayespy.nodes import Dot
>>> Z = Gaussian(X, [[1,0], [0,1]])
>>> Y = Dot(X, Z)

This works because there is now an intermediate stochastic node ``Z`` on the
other path from ``X`` node to ``Y`` node.

Effects of the nodes on inference
---------------------------------

When constructing the network with nodes, the stochastic nodes actually define
three important aspects:

1. the prior probability distribution for the variables,

2. the factorization of the posterior approximation,

3. the functional form of the posterior approximation for the variables.

Prior probability distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
First, the most intuitive feature of the nodes is that they define the prior
distribution. In the previous example, ``mu`` was a stochastic
:class:`GaussianARD` node corresponding to :math:`\mu` from the normal
distribution, ``tau`` was a stochastic :class:`Gamma` node corresponding to
:math:`\tau` from the gamma distribution, and ``y`` was a stochastic
:class:`GaussianARD` node corresponding to :math:`y` from the normal
distribution with mean :math:`\mu` and precision :math:`\tau`.  If we denote the
set of all stochastic nodes by :math:`\Omega`, and by :math:`\pi_X` the set of
parents of a node :math:`X`, the model is defined as
                
.. math::


   p(\Omega) = \prod_{X \in \Omega} p(X|\pi_X),

where nodes correspond to the terms :math:`p(X|\pi_X)`\ .

Posterior factorization
~~~~~~~~~~~~~~~~~~~~~~~

Second, the nodes define the structure of the posterior approximation.  The
variational Bayesian approximation factorizes with respect to nodes, that is,
each node corresponds to an independent probability distribution in the
posterior approximation. In the previous example, ``mu`` and ``tau`` were
separate nodes, thus the posterior approximation factorizes with respect to
them: :math:`q(\mu)q(\tau)`\ . Thus, the posterior approximation can be written
as:

.. math::


   p(\tilde{\Omega}|\hat{\Omega}) \approx \prod_{X \in \tilde{\Omega}} q(X),

                
where :math:`\tilde{\Omega}` is the set of latent stochastic nodes and
:math:`\hat{\Omega}` is the set of observed stochastic nodes.  Sometimes one may
want to avoid the factorization between some variables.  For this purpose, there
are some nodes which model several variables jointly without factorization.  For
instance, :class:`GaussianGammaISO` is a joint node for :math:`\mu` and
:math:`\tau` variables from the normal-gamma distribution and the posterior
approximation does not factorize between :math:`\mu` and :math:`\tau`, that is,
the posterior approximation is :math:`q(\mu,\tau)`.
                
Functional form of the posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                
Last, the nodes define the functional form of the posterior approximation.
Usually, the posterior approximation has the same or similar functional form as
the prior.  For instance, :class:`Gamma` uses gamma distribution to also
approximate the posterior distribution.  Similarly, :class:`GaussianARD` uses
Gaussian distribution for the posterior.  However, the posterior approximation
of :class:`GaussianARD` uses a full covariance matrix although the prior assumes
a diagonal covariance matrix.  Thus, there can be slight differences in the
exact functional form of the posterior approximation but the rule of thumb is
that the functional form of the posterior approximation is the same as or more
general than the functional form of the prior.
                
Using plate notation
--------------------

Defining plates
~~~~~~~~~~~~~~~

Stochastic nodes take the optional parameter ``plates``, which can be used to
define plates of the variable. A plate defines the number of repetitions of a
set of variables. For instance, a set of random variables
:math:`\mathbf{y}_{mn}` could be defined as

.. math::

      
   \mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.

This can also be visualized as a graphical model:

                
.. bayesnet::

    \node[latent] (y) {$\mathbf{y}_{mn}$} ;
    \node[latent, above left=1.8 and 0.4 of y] (mu) {$\boldsymbol{\mu}$} ;
    \node[latent, above right=1.8 and 0.4 of y] (Lambda) {$\mathbf{\Lambda}$} ;
    \factor[above=of y] {y-f} {left:$\mathcal{N}$} {mu,Lambda}     {y};
    \plate {m-plate} {(y)(y-f)(y-f-caption)} {$m=0,\ldots,9$} ;
    \plate {n-plate} {(m-plate)(m-plate-caption)} {$n=0,\ldots,29$} ;
                
The variable has two plates: one for the index :math:`m` and one for the
index :math:`n`\ . In BayesPy, this random variable can be constructed
as:

>>> y = Gaussian(mu, Lambda, plates=(10,30))
                
.. note:: The plates are always given as a tuple of positive integers.

Plates also define indexing for the nodes, thus you can use simple NumPy-style
slice indexing to obtain a subset of the plates:

>>> y_0 = y[0]
>>> y_0.plates
(30,)
>>> y_even = y[:,::2]
>>> y_even.plates
(10, 15)
>>> y_complex = y[:5, 10:20:5]
>>> y_complex.plates
(5, 2)

Note that this indexing is for the plates only, not for the random variable
dimensions.
                
Sharing and broadcasting plates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of having a common mean and precision matrix for all
:math:`\mathbf{y}_{mn}`\ , it is also possible to share plates with parents. For
instance, the mean could be different for each index :math:`m` and the precision
for each index :math:`n`\ :

.. math::


   \mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}_m,
   \mathbf{\Lambda}_n),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.

which has the following graphical representation:

                
.. bayesnet::

    \node[latent] (y) {$\mathbf{y}_{mn}$} ;
    \node[latent, above left=1 and 2 of y] (mu) {$\boldsymbol{\mu}_m$} ;
    \node[latent, above right=1 and 1 of y] (Lambda) {$\mathbf{\Lambda}_n$} ;
    \factor[above=of y] {y-f} {above:$\mathcal{N}$} {mu,Lambda}     {y};
    \plate {m-plate} {(mu)(y)(y-f)(y-f-caption)} {$m=0,\ldots,9$} ;
    \plate {n-plate} {(Lambda)(y)(y-f)(y-f-caption)(m-plate-caption)(m-plate.north east)} {$n=0,\ldots,29$} ;
                
This can be constructed in BayesPy, for instance, as:

>>> from bayespy.nodes import Gaussian, Wishart
>>> mu = Gaussian([0, 0], [[1e-6, 0],[0, 1e-6]], plates=(10,1))
>>> Lambda = Wishart(2, [[1, 0], [0, 1]], plates=(1,30))
>>> X = Gaussian(mu, Lambda)

There are a few things to notice here. First, the plates are defined similarly
as shapes in NumPy, that is, they use similar broadcasting rules. For instance,
the plates ``(10,1)`` and ``(1,30)`` broadcast to ``(10,30)``. In fact, one
could use plates ``(10,1)`` and ``(30,)`` to get the broadcasted plates
``(10,30)`` because broadcasting compares the plates from right to left starting
from the last axis. Second, ``X`` is not given ``plates`` keyword argument
because the default plates are the plates broadcasted from the parents and that
was what we wanted so it was not necessary to provide the keyword argument. If
we wanted, for instance, plates ``(20,10,30)`` for ``X``, then we would have
needed to provide ``plates=(20,10,30)``.

The validity of the plates between a child and its parents is checked as
follows. The plates are compared plate-wise starting from the last axis and
working the way forward. A plate of the child is compatible with a plate of the
parent if either of the following conditions is met:

1. The two plates have equal size
2. The parent has size 1 (or no plate)

Table below shows an example of compatible plates for a child node and
its two parent nodes:

                
+---------+----------------------------+
| node    | plates                     |
+=========+===+===+===+===+===+===+====+
| parent1 |   | 3 | 1 | 1 | 1 | 8 | 10 |
+---------+---+---+---+---+---+---+----+
| parent2 |   |   | 1 | 1 | 5 | 1 | 10 |
+---------+---+---+---+---+---+---+----+
| child   | 5 | 3 | 1 | 7 | 5 | 8 | 10 |
+---------+---+---+---+---+---+---+----+
                
Plates in deterministic nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that plates can be defined explicitly only for stochastic nodes.  For
deterministic nodes, the plates are defined implicitly by the plate broadcasting
rules from the parents. Deterministic nodes do not need more plates than this
because there is no randomness. The deterministic node would just have the same
value over the extra plates, but it is not necessary to do this explicitly
because the child nodes of the deterministic node can utilize broadcasting
anyway. Thus, there is no point in having extra plates in deterministic nodes,
and for this reason, deterministic nodes do not use ``plates`` keyword argument.

Plates in constants
~~~~~~~~~~~~~~~~~~~

It is useful to understand how the plates and the shape of a random variable are
connected. The shape of an array which contains all the plates of a random
variable is the concatenation of the plates and the shape of the variable. For
instance, consider a 2-dimensional Gaussian variable with plates ``(3,)``. If
you want the value of the constant mean vector and constant precision matrix to
vary between plates, they are given as ``(3,2)``-shape and ``(3,2,2)``-shape
arrays, respectively:

>>> import numpy as np
>>> mu = [ [0,0], [1,1], [2,2] ]
>>> Lambda = [ [[1.0, 0.0],
...             [0.0, 1.0]],
...            [[1.0, 0.9],
...             [0.9, 1.0]],
...            [[1.0, -0.3],
...             [-0.3, 1.0]] ]
>>> X = Gaussian(mu, Lambda)
>>> np.shape(mu)
(3, 2)
>>> np.shape(Lambda)
(3, 2, 2)
>>> X.plates
(3,)

Thus, the leading axes of an array are the plate axes and the trailing axes are
the random variable axes. In the example above, the mean vector has plates
``(3,)`` and shape ``(2,)``, and the precision matrix has plates ``(3,)`` and
shape ``(2,2)``.

Factorization of plates
~~~~~~~~~~~~~~~~~~~~~~~

It is important to undestand the independency structure the plates induce for
the model. First, the repetitions defined by a plate are independent a priori
given the parents. Second, the repetitions are independent in the posterior
approximation, that is, the posterior approximation factorizes with respect to
plates. Thus, the plates also have an effect on the independence structure of
the posterior approximation, not only prior. If dependencies between a set of
variables need to be handled, that set must be handled as a some kind of
multi-dimensional variable.

.. _sec-irregular-plates:

Irregular plates
~~~~~~~~~~~~~~~~

The handling of plates is not always as simple as described above. There are
cases in which the plates of the parents do not map directly to the plates of
the child node. The user API should mention such irregularities.

For instance, the parents of a mixture distribution have a plate which contains
the different parameters for each cluster, but the variable from the mixture
distribution does not have that plate:

>>> from bayespy.nodes import Gaussian, Wishart, Categorical, Mixture
>>> mu = Gaussian([[0], [0], [0]], [ [[1]], [[1]], [[1]] ])
>>> Lambda = Wishart(1, [ [[1]], [[1]], [[1]]])
>>> Z = Categorical([1/3, 1/3, 1/3], plates=(100,))
>>> X = Mixture(Z, Gaussian, mu, Lambda)
>>> mu.plates
(3,)
>>> Lambda.plates
(3,)
>>> Z.plates
(100,)
>>> X.plates
(100,)
                
The plates ``(3,)`` and ``(100,)`` should not broadcast according to the rules
mentioned above. However, when validating the plates, :class:`Mixture` removes
the plate which corresponds to the clusters in ``mu`` and ``Lambda``. Thus,
``X`` has plates which are the result of broadcasting plates ``()`` and
``(100,)`` which equals ``(100,)``.
                
Also, sometimes the plates of the parents may be mapped to the variable
axes. For instance, an automatic relevance determination (ARD) prior for a
Gaussian variable is constructed by giving the diagonal elements of the
precision matrix (or tensor). The Gaussian variable itself can be a scalar, a
vector, a matrix or a tensor. A set of five :math:`4 \times 3`
-dimensional Gaussian matrices with ARD prior is constructed as:

>>> from bayespy.nodes import GaussianARD, Gamma
>>> tau = Gamma(1, 1, plates=(5,4,3))
>>> X = GaussianARD(0, tau, shape=(4,3))
>>> tau.plates
(5, 4, 3)
>>> X.plates
(5,)

Note how the last two plate axes of ``tau`` are mapped to the variable axes of
``X`` with shape ``(4,3)`` and the plates of ``X`` are obtained by taking the
remaining leading plate axes of ``tau``.




Example model: Principal component analysis
-------------------------------------------

Now, we'll construct a bit more complex model which will be used in the
following sections.  The model is a probabilistic version of principal component
analysis (PCA):

.. math::

    \mathbf{Y} = \mathbf{C}\mathbf{X}^T + \mathrm{noise}

where :math:`\mathbf{Y}` is :math:`M\times N` data matrix, :math:`\mathbf{C}` is
:math:`M\times D` loading matrix, :math:`\mathbf{X}` is :math:`N\times D` state
matrix, and noise is isotropic Gaussian.  The dimensionality :math:`D` is
usually assumed to be much smaller than :math:`M` and :math:`N`.


A probabilistic formulation can be written as:

.. math::

   p(\mathbf{Y}) &= \prod^{M-1}_{m=0} \prod^{N-1}_{n=0} \mathcal{N}(y_{mn} |
   \mathbf{c}_m^T \mathbf{x}_n, \tau)
   \\
   p(\mathbf{X}) &= \prod^{N-1}_{n=0} \prod^{D-1}_{d=0} \mathcal{N}(x_{nd} |
   0, 1)
   \\
   p(\mathbf{C}) &= \prod^{M-1}_{m=0} \prod^{D-1}_{d=0} \mathcal{N}(c_{md} |
   0, \alpha_d)
   \\
   p(\boldsymbol{\alpha}) &= \prod^{D-1}_{d=0} \mathcal{G} (\alpha_d | 10^{-3},
   10^{-3})
   \\
   p(\tau) &= \mathcal{G} (\tau | 10^{-3}, 10^{-3})

where we have given automatic relevance determination (ARD) prior for
:math:`\mathbf{C}`.  This can be visualized as a graphical model:

.. bayesnet::

    \node[latent] (y) {$\mathbf{y}_{mn}$} ;
    \node[det, above=of y] (dot) {dot} ;
    \node[latent, right=2 of dot] (tau) {$\tau$} ;
    \node[latent, above left=1 and 2 of dot] (C) {$c_{md}$} ;
    \node[latent, above=of C] (alpha) {$\alpha_d$} ;
    \node[latent, above right=1 and 1 of dot] (X) {$x_{nd}$} ;

    \factor[above=of y] {y-f} {left:$\mathcal{N}$} {dot,tau} {y};
    \factor[above=of C] {C-f} {left:$\mathcal{N}$} {alpha} {C};
    \factor[above=of X] {X-f} {above:$\mathcal{N}$} {} {X};
    \factor[above=of alpha] {alpha-f} {above:$\mathcal{G}$} {} {alpha};
    \factor[above=of tau] {tau-f} {above:$\mathcal{G}$} {} {tau};
    \edge {C,X} {dot};

    \tikzstyle{plate caption} += [below left=0pt and 0pt of #1.north east] ;
    \plate {d-plate} {(X)(X-f)(X-f-caption)(C)(C-f)(C-f-caption)(alpha)(alpha-f)(alpha-f-caption)} {$d=0,\ldots,2$} ;
    \tikzstyle{plate caption} += [below left=5pt and 0pt of #1.south east] ;
    \plate {m-plate} {(y)(y-f)(y-f-caption)(C)(C-f)(C-f-caption)(d-plate.south west)} {$m=0,\ldots,9$} ;
    \plate {n-plate} {(y)(y-f)(y-f-caption)(X)(X-f)(X-f-caption)(m-plate-caption)(m-plate.north east)(d-plate.south east)} {$n=0,\ldots,99$} ;

Now, let us construct this model in BayesPy.  First, we'll define the
dimensionality of the latent space in our model:

>>> D = 3

Then the prior for the latent states :math:`\mathbf{X}`:

>>> X = GaussianARD(0, 1,
...                 shape=(D,),
...                 plates=(1,100),
...                 name='X')

Note that the shape of ``X`` is ``(D,)``, although the latent dimensions are
marked with a plate in the graphical model and they are conditionally
independent in the prior.  However, we want to (and need to) model the posterior
dependency of the latent dimensions, thus we cannot factorize them, which would
happen if we used ``plates=(1,100,D)`` and ``shape=()``.  The first plate axis
with size 1 is given just for clarity.

The prior for the ARD parameters :math:`\boldsymbol{\alpha}` of the loading
matrix:

>>> alpha = Gamma(1e-3, 1e-3,
...               plates=(D,),
...               name='alpha')

The prior for the loading matrix :math:`\mathbf{C}`:

>>> C = GaussianARD(0, alpha,
...                 shape=(D,),
...                 plates=(10,1),
...                 name='C')

Again, note that the shape is the same as for ``X`` for the same reason.  Also,
the plates of ``alpha``, ``(D,)``, are mapped to the full shape of the node
``C``, ``(10,1,D)``, using standard broadcasting rules.

The dot product is just a deterministic node:

>>> F = Dot(C, X)

However, note that ``Dot`` requires that the input Gaussian nodes have the same
shape and that this shape has exactly one axis, that is, the variables are
vectors.  This the reason why we used shape ``(D,)`` for ``X`` and ``C`` but
from a bit different perspective.  The node computes the inner product of
:math:`D`-dimensional vectors resulting in plates ``(10,100)`` broadcasted from
the plates ``(1,100)`` and ``(10,1)``:

>>> F.plates
(10, 100)

The prior for the observation noise :math:`\tau`:

>>> tau = Gamma(1e-3, 1e-3, name='tau')

Finally, the observations are conditionally independent Gaussian scalars:

>>> Y = GaussianARD(F, tau, name='Y')

Now we have defined our model and the next step is to observe some data and to
perform inference.
