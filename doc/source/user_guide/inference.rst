..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


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
    tau = Gamma(1e-3, 1e-3, name='tau')
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
>>> data = np.dot(c, x) + 0.1*np.random.randn(10, 100)

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

>>> X.initialize_from_parameters(np.random.randn(1, 100, D), 10)

Note that the shape of the random mean is the sum of the plates ``(1, 100)`` and
the variable shape ``(D,)``.  In addition, instead of variance,
:class:`GaussianARD` uses precision as the second parameter, thus we initialized
the variance to :math:`\frac{1}{10}`.  This random initialization is important
in our PCA model because the default initialization gives ``C`` and ``X`` zero
mean.  If the mean of the other variable was zero when the other is updated, the
other variable gets zero mean too.  This would lead to an update algorithm where
both means remain zeros and effectively no latent space is found.  Thus, it is
important to give non-zero random initialization for ``X`` if ``C`` is updated
before ``X`` the first time.  It is typical that at least some nodes need be
initialized with some randomness.

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
Iteration 1: loglike=-9.305259e+02 (... seconds)

The ``loglike`` tells the VB lower bound.  The order in which the nodes are
updated is the same as the order in which the nodes were given when creating
``Q``.  If you want to change the order or update only some of the nodes, you
can give as arguments the nodes you want to update and they are updated in the
given order:

>>> Q.update(C, X)
Iteration 2: loglike=-8.818976e+02 (... seconds)

It is also possible to give the same node several times:

>>> Q.update(C, X, C, tau)
Iteration 3: loglike=-8.071222e+02 (... seconds)

Note that each call to ``update`` is counted as one iteration step although not
variables are necessarily updated.  Instead of doing one iteration step,
``repeat`` keyword argument can be used to perform several iteration steps:

>>> Q.update(repeat=10)
Iteration 4: loglike=-7.167588e+02 (... seconds)
Iteration 5: loglike=-6.827873e+02 (... seconds)
Iteration 6: loglike=-6.259477e+02 (... seconds)
Iteration 7: loglike=-4.725400e+02 (... seconds)
Iteration 8: loglike=-3.270816e+02 (... seconds)
Iteration 9: loglike=-2.208865e+02 (... seconds)
Iteration 10: loglike=-1.658761e+02 (... seconds)
Iteration 11: loglike=-1.469468e+02 (... seconds)
Iteration 12: loglike=-1.420311e+02 (... seconds)
Iteration 13: loglike=-1.405139e+02 (... seconds)

The VB algorithm stops automatically if it converges, that is, the relative
change in the lower bound is below some threshold:

>>> Q.update(repeat=1000)
Iteration 14: loglike=-1.396481e+02 (... seconds)
...
Iteration 488: loglike=-1.224106e+02 (... seconds)
Converged at iteration 488.

Now the algorithm stopped before taking 1000 iteration steps because it
converged.  The relative tolerance can be adjusted by providing ``tol`` keyword
argument to the ``update`` method:

>>> Q.update(repeat=10000, tol=1e-6)
Iteration 489: loglike=-1.224094e+02 (... seconds)
...
Iteration 847: loglike=-1.222506e+02 (... seconds)
Converged at iteration 847.

Making the tolerance smaller, may improve the result but it may also
significantly increase the iteration steps until convergence.

Instead of using ``update`` method of the inference engine ``VB``, it is
possible to use the ``update`` methods of the nodes directly as

>>> C.update()

or

>>> Q['C'].update()

However, this is not recommended, because the ``update`` method of the inference
engine ``VB`` is a wrapper which, in addition to calling the nodes' ``update``
methods, checks for convergence and does a few other useful minor things.  But
if for any reason these direct update methods are needed, they can be used.

.. _sec-parameter-expansion:

Parameter expansion
+++++++++++++++++++

Sometimes the VB algorithm converges very slowly.  This may happen when the
variables are strongly coupled in the true posterior but factorized in the
approximate posterior.  This coupling leads to zigzagging of the variational
parameters which progresses slowly.  One solution to this problem is to use
parameter expansion.  The idea is to add an auxiliary variable which
parameterizes the posterior approximation of several variables.  Then optimizing
this auxiliary variable actually optimizes several posterior approximations
jointly leading to faster convergence.

The parameter expansion is model specific.  Currently in BayesPy, only
state-space models have built-in parameter expansions available.  These
state-space models contain a variable which is a dot product of two variables
(plus some noise):

.. math::

    y = \mathbf{c}^T\mathbf{x} + \mathrm{noise}

The parameter expansion can be motivated by noticing that we can add an
auxiliary variable which rotates the variables :math:`\mathbf{c}` and
:math:`\mathbf{x}` so that the dot product is unaffected:

.. math::

    y &= \mathbf{c}^T\mathbf{x} + \mathrm{noise}
    = \mathbf{c}^T \mathbf{R} \mathbf{R}^{-1}\mathbf{x} + \mathrm{noise}
    = (\mathbf{R}^T\mathbf{c})^T(\mathbf{R}^{-1}\mathbf{x}) + \mathrm{noise}

Now, applying this rotation to the posterior approximations
:math:`q(\mathbf{c})` and :math:`q(\mathbf{x})`, and optimizing the VB lower
bound with respect to the rotation leads to parameterized joint optimization of
:math:`\mathbf{c}` and :math:`\mathbf{x}`.

The available parameter expansion methods are in module ``transformations``:

>>> from bayespy.inference.vmp import transformations

First, you create the rotation transformations for the two variables:

>>> rotX = transformations.RotateGaussianARD(X)
>>> rotC = transformations.RotateGaussianARD(C, alpha)

.. currentmodule:: bayespy.inference.vmp.transformations

Here, the rotation for ``C`` provides the ARD parameters ``alpha`` so they are
updated simultaneously.  In addition to :class:`RotateGaussianARD`, there are a
few other built-in rotations defined, for instance, :class:`RotateGaussian` and
:class:`RotateGaussianMarkovChain`.  It is extremely important that the model
satisfies the assumptions made by the rotation class and the user is mostly
responsible for this.  The optimizer for the rotations is constructed by giving
the two rotations and the dimensionality of the rotated space:

.. currentmodule:: bayespy.nodes

>>> R = transformations.RotationOptimizer(rotC, rotX, D)

Now, calling ``rotate`` method will find optimal rotation and update the
relevant nodes (``X``, ``C`` and ``alpha``) accordingly:

>>> R.rotate()

Let us see how our iteration would have gone if we had used this parameter
expansion.  First, let us re-initialize our nodes and VB algorithm:

>>> alpha.initialize_from_prior()
>>> C.initialize_from_prior()
>>> X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
>>> tau.initialize_from_prior()
>>> Q = VB(Y, C, X, alpha, tau)

Then, the rotation is set to run after each iteration step:

>>> Q.callback = R.rotate

Now the iteration converges to the relative tolerance :math:`10^{-6}` much
faster:

>>> Q.update(repeat=1000, tol=1e-6)
Iteration 1: loglike=-9.363500e+02 (... seconds)
...
Iteration 18: loglike=-1.221354e+02 (... seconds)
Converged at iteration 18.

The convergence took 18 iterations with rotations and 488 or 847 iterations
without the parameter expansion.  In addition, the lower bound is improved
slightly.  One can compare the number of iteration steps in this case because
the cost per iteration step with or without parameter expansion is approximately
the same.  Sometimes the parameter expansion can have the drawback that it
converges to a bad local optimum.  Usually, this can be solved by updating the
nodes near the observations a few times before starting to update the
hyperparameters and to use parameter expansion.  In any case, the parameter
expansion is practically necessary when using state-space models in order to
converge to a proper solution in a reasonable time.


