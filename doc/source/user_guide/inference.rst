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

.. testsetup::

    # This is the PCA model from the previous section
    import numpy as np
    np.random.seed(1)
    from bayespy.nodes import GaussianARD, Gamma, Dot
    D = 3
    X = GaussianARD(0, 1,
                    shape=(D,),
                    plates=(1,100),
                    name='X')
    alpha = Gamma(1e-3, 1e-3,
                  plates=(D,),
                  name='alpha')
    C = GaussianARD(0, alpha,
                    shape=(D,),
                    plates=(10,1),
                    name='C')
    F = Dot(C, X)
    tau = Gamma(1e-3, 1e-3)
    Y = GaussianARD(F, tau)
                
                
Performing inference
====================

Approximation of the posterior distribution can be divided into several steps:

-  Observe some nodes

-  Choose the inference engine

-  Initialize the posterior approximation

-  Run the inference algorithm

In order to illustrate these steps, we'll be using the PCA model constructed in
the previous section.

Observing nodes
---------------

First, let us generate some toy data:

>>> c = np.random.randn(10, 2)
>>> x = np.random.randn(2, 100)
>>> data = np.dot(c, x) + 0.1*np.random.randn()

The data is provided by simply calling ``observe`` method of a stochastic node:

>>> Y.observe(data)
                
It is important that the shape of the ``data`` array matches the plates and
shape of the node ``Y``.  For instance, if ``Y`` was :class:`Wishart` node for
:math:`3\times 3` matrices with plates ``(5,1,10)``, the full shape of ``Y``
would be ``(5,1,10,3,3)``.  The ``data`` array should have this shape exactly,
that is, no broadcasting rules are applied.
                
Missing values
++++++++++++++

It is possible to mark missing values by providing a mask which is a boolean
array:

.. code:: python

>>> Y.observe(data, mask=[[True], [False], [False], [True], [True],
...                       [False], [True], [True], [True], [False]])

``True`` means that the value is observed and ``False`` means that the value is
missing.  The shape of the above mask is ``(10,1)``, which broadcasts to the
plates of Y, ``(10,100)``.  Thus, the above mask means that the second, third,
sixth and tenth rows of the :math:`10\times 100` data matrix are missing. 

The mask is applied to the *plates*, not to the data array directly.  This means
that it is not possible to observe a random variable partially, each repetition
defined by the plates is either fully observed or fully missing.  Thus, the mask
is applied to the plates.  It is often possible to circumvent this seemingly
tight restriction by adding an observable child node which factorizes more.

The shape of the mask is broadcasted to plates using standard NumPy broadcasting
rules. So, if the variable has plates ``(5,1,10)``, the mask could have a shape
``()``, ``(1,)``, ``(1,1)``, ``(1,1,1)``, ``(10,)``, ``(1,10)``, ``(1,1,10)``,
``(5,1,1)`` or ``(5,1,10)``.  In order to speed up the inference, missing values
are automatically integrated out if they are not needed as latent variables to
child nodes.  This leads to faster convergence and more accurate approximations.

Choosing the inference method
-----------------------------

                
Inference methods can be found in :mod:`bayespy.inference` package.  Currently,
only variational Bayesian approximation is implemented
(:class:`bayespy.inference.VB`).  The inference engine is constructed by giving
the stochastic nodes of the model.
                
>>> from bayespy.inference import VB
>>> Q = VB(Y, C, X, alpha, tau)

There is no need to give any deterministic nodes.  Currently, the inference
engine does not automatically search for stochastic parents and children, thus
it is important that all stochastic nodes of the model are given.  This should
be made more robust in future versions.

A node of the model can be obtained by using the name of the node as a key:

>>> Q['X']
<bayespy.inference.vmp.nodes.gaussian.GaussianARD object at 0x...>

Note that the returned object is the same as the node object itself:

>>> Q['X'] is X
True

Thus, one may use the object ``X`` when it is available.  However, if the model
and the inference engine are constructed in another function or module, the node
object may not be available directly and this feature becomes useful.


Initializing the posterior approximation
----------------------------------------

The inference engines give some initialization to the stochastic nodes by
default.  However, the inference algorithms can be sensitive to the
initialization, thus it is sometimes necessary to have better control over the
initialization.  For VB, the following initialization methods are available:

- ``initialize_from_prior``: Use the current states of the parent nodes to
  update the node. This is the default initialization.

- ``initialize_from_parameters``: Use the given parameter values for the
  distribution.

- ``initialize_from_value``: Use the given value for the variable.

- ``initialize_from_random``: Draw a random value for the variable.  The random
  sample is drawn from the current state of the node's distribution.

Note that ``initialize_from_value`` and ``initialize_from_random`` initialize
the distribution with a value of the variable instead of parameters of the
distribution.  Thus, the distribution is actually a delta distribution with a
peak on the value after the initialization.  This state of the distribution does
not have proper natural parameter values nor normalization, thus the VB lower
bound terms are ``np.nan`` for this initial state.

These initialization methods can be used to perform even a bit more complex
initializations.  For instance, a Gaussian distribution could be initialized
with a random mean and variance 0.1.  In our PCA model, this can be obtained by

>>> C.initialize_from_parameters(np.random.randn(10, 1, D), 10)

Note that the shape of the random mean is the sum of the plates ``(10, 1)`` and
the variable shape ``(D,)``.  In addition, instead of variance,
:class:`GaussianARD` uses precision as the second parameter.

By default, nodes are initialized with the method ``initialize_from_prior``.
The method is not very time consuming but if for any reason you want to avoid
that default initialization computation, you can provide ``initialize=False``
when creating the stochastic node.  However, the node does not have a proper
state in that case, which leads to errors in VB learning unless the distribution
is initialized using the above methods.




Running the inference algorithm
-------------------------------

The approximation methods are based on iterative algorithms, which can
be run using ``update`` method. By default, it takes one iteration step
updating all nodes once:

>>> Q.update()
Iteration 1: loglike=-9.423766e+02 (... seconds)

The order in which the nodes are updated is the same as the order in which the
nodes were given when creating ``Q``.  If you want to change the order or update
only some of the nodes, you can give as arguments the nodes you want to update
and they are updated in the given order:

>>> Q.update(C, X)
Iteration 2: loglike=-9.406813e+02 (... seconds)

It is also possible to give the same node several times:

>>> Q.update(C, X, C, tau)
Iteration 3: loglike=-9.406672e+02 (... seconds)

Note that each call to ``update`` is counted as one iteration step although not
variables are necessarily updated.  Instead of doing one iteration step,
``repeat`` keyword argument can be used to perform several iteration steps:

>>> Q.update(repeat=10)
Iteration 4: loglike=-9.395617e+02 (... seconds)
Iteration 5: loglike=-9.386064e+02 (... seconds)
Iteration 6: loglike=-9.381864e+02 (... seconds)
Iteration 7: loglike=-9.379849e+02 (... seconds)
Iteration 8: loglike=-9.378842e+02 (... seconds)
Iteration 9: loglike=-9.378328e+02 (... seconds)
Iteration 10: loglike=-9.378063e+02 (... seconds)
Iteration 11: loglike=-9.377926e+02 (... seconds)
Iteration 12: loglike=-9.377855e+02 (... seconds)
Iteration 13: loglike=-9.377818e+02 (... seconds)

The VB algorithm stops if it converges, that is, the change in the lower bound
is below some threshold.

>>> Q.update(repeat=100)
Iteration 14: loglike=-9.377799e+02 (... seconds)
Iteration 15: loglike=-9.377789e+02 (... seconds)
Iteration 16: loglike=-9.377784e+02 (... seconds)
Converged.

Now it did not perform 100 more iterations but only three because the algorithm
converged.

Instead of using ``update`` method of the inference engine ``VB``, it is
possible to use the ``update`` methods of the nodes directly as

>>> C.update()

or

>>> Q['C'].update()

However, this is not recommended, because the ``update`` method of the inference
engine ``VB`` is a wrapper which, in addition to calling the nodes' ``update``
methods, checks for convergence and does a few other useful minor things.  But
if for any reason these direct update methods are needed, they can be used.


Parameter expansion
+++++++++++++++++++

Sometimes the VB algorithm converges very slowly.  This may happen when the
variables are strongly coupled in the true posterior but factorized in the
approximate posterior.  One solution to this problem is to use parameter
expansion.  The idea is to add an auxiliary variable which parameterizes the
posterior approximation of several variables.  Then optimizing this auxiliary
variable actually optimizes several posterior approximations jointly leading to
faster convergence.

The parameter expansion is model specific.  In BayesPy, only state-space models
can utilize the parameter expansion currently.  These models have contain a
variable which is a dot product of two variables (plus some noise):

.. math::

    y = \mathbf{c}^T\mathbf{x} + \mathrm{noise}

We can add an auxiliary variable which rotates the variables :math:`\mathbf{c}`
and :math:`\mathbf{x}` so that the dot product is unaffected:

.. math::

    y &= \mathbf{c}^T\mathbf{x} + \mathrm{noise}
    = \mathbf{c}^T \mathbf{R} \mathbf{R}^{-1}\mathbf{x} + \mathrm{noise}
    = (\mathbf{R}^T\mathbf{c})^T(\mathbf{R}^{-1}\mathbf{x}) + \mathrm{noise}

Now, applying this rotation to the posterior approximations
:math:`q(\mathbf{c})` and :math:`q(\mathbf{x})`, and optimizing the VB lower
bound with respect to the rotation leads to parameterized joint optimization of
:math:`\mathbf{c}` and :math:`\mathbf{x}`.

The parameter expansion is used in BayesPy as ..
