..
   Copyright (C) 2014-2015 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


.. testsetup::

    import numpy as np
    np.random.seed(1)
    # This is the PCA model from the previous sections
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
    Y = GaussianARD(F, tau, name='Y')
    c = np.random.randn(10, 2)
    x = np.random.randn(2, 100)
    data = np.dot(c, x) + 0.1*np.random.randn(10, 100)
    Y.observe(data)
    from bayespy.inference import VB
    import bayespy.plot as bpplt
    Q = VB(Y, C, X, alpha, tau)
    X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
    from bayespy.inference.vmp import transformations
    rotX = transformations.RotateGaussianARD(X)
    rotC = transformations.RotateGaussianARD(C, alpha)
    R = transformations.RotationOptimizer(rotC, rotX, D)
    Q = VB(Y, C, X, alpha, tau)
    Q.callback = R.rotate
    Q.update(repeat=1000, tol=1e-6, verbose=False)
    import warnings
    warnings.simplefilter('error', UserWarning)

Advanced topics
===============

This section contains brief information on how to implement some advanced
methods in BayesPy.  These methods include Riemannian conjugate gradient
methods, pattern search, simulated annealing, collapsed variational inference
and stochastic variational inference.  In order to use these methods properly,
the user should understand them to some extent.  They are also considered
experimental, thus you may encounter bugs or unimplemented features.  In any
case, these methods may provide huge performance improvements easily compared to
the standard VB-EM algorithm.


Gradient-based optimization
---------------------------

Variational Bayesian learning basically means that the parameters of the
approximate posterior distributions are optimized to maximize the lower bound of
the marginal log likelihood :cite:`Honkela:2010`.  This optimization can be done
by using gradient-based optimization methods.  In order to improve the
gradient-based methods, it is recommended to take into account the information
geometry by using the Riemannian (a.k.a. natural) gradient.  In fact, the
standard VB-EM algorithm is equivalent to a gradient ascent method which uses
the Riemannian gradient and step length 1.  Thus, it is natural to try to
improve this method by using non-linear conjugate gradient methods instead of
gradient ascent.  These optimization methods are especially useful when the
VB-EM update equations are not available but one has to use fixed form
approximation.  But it is possible that the Riemannian conjugate gradient method
improve performance even when the VB-EM update equations are available.


.. currentmodule:: bayespy.inference

The optimization algorithm in :func:`VB.optimize` has a simple interface.
Instead of using the default Riemannian geometry, one can use the Euclidean
geometry by giving :code:`riemannian=False`.  It is also possible to choose the
optimization method from gradient ascent (:code:`method='gradient'`) or
conjugate gradient methods (only :code:`method='fletcher-reeves'` implemented at
the moment).  For instance, we could optimize nodes ``C`` and ``X`` jointly
using Euclidean gradient ascent as:

>>> Q = VB(Y, C, X, alpha, tau)
>>> Q.optimize(C, X, riemannian=False, method='gradient', maxiter=5)
Iteration ...

Note that this is very inefficient way of updating those nodes (bad geometry and
not using conjugate gradients).  Thus, one should understand the idea of these
optimization methods, otherwise one may do something extremely inefficient.
Most likely this method can be found useful in combination with the advanced
tricks in the following sections.


.. note::

   The Euclidean gradient has not been implemented for all nodes yet.  The
   Euclidean gradient is required by the Euclidean geometry based optimization
   but also by the conjugate gradient methods in the Riemannian geometry.  Thus,
   the Riemannian conjugate gradient may not yet work for all models.


It is possible to construct custom optimization algorithms with the tools
provided by :class:`VB`.  For instance, :func:`VB.get_parameters` and
:func:`VB.set_parameters` can be used to handle the parameters of nodes.
:func:`VB.get_gradients` is used for computing the gradients of nodes.  The
parameter and gradient objects are not numerical arrays but more complex nested
lists not meant to be accessed by the user.  Thus, for simple arithmetics with
the parameter and gradient objects, use functions :func:`VB.add` and
:func:`VB.dot`.  Finally, :func:`VB.compute_lowerbound` and
:func:`VB.has_converged` can be used to monitor the lower bound.


Collapsed inference
-------------------

The optimization method can be used efficiently in such a way that some of the
variables are collapsed, that is, marginalized out :cite:`Hensman:2012`.  The
collapsed variables must be conditionally independent given the observations and
all other variables.  Probably, one also wants that the size of the marginalized
variables is large and the size of the optimized variables is small.  For
instance, in our PCA example, we could optimize as follows:

>>> Q.optimize(C, tau, maxiter=10, collapsed=[X, alpha])
Iteration ...

The collapsed variables are given as a list.  This optimization does basically
the following: It first computes the gradients for ``C`` and ``tau`` and takes
an update step using the desired optimization method.  Then, it updates the
collapsed variables by using the standard VB-EM update equations.  These two
steps are taken in turns.  Effectively, this corresponds to collapsing the
variables ``X`` and ``alpha`` in a particular way.  The point of this method is
that the number of parameters in the optimization reduces significantly and the
collapsed variables are updated optimally.  For more details, see
:cite:`Hensman:2012`.

It is possible to use this method in such a way, that the collapsed variables
are not conditionally independent given the observations and all other
variables.  However, in that case, the method does not anymore correspond to
collapsing the variables but just using VB-EM updates after gradient-based
updates.  The method does not check for conditional independence, so the user is
free to do this.

.. note::

   Although the Riemannian conjugate gradient method has not yet been
   implemented for all nodes, it may be possible to collapse those nodes and
   optimize the other nodes for which the Euclidean gradient is already
   implemented.


Pattern search
--------------

The pattern search method estimates the direction in which the approximate
posterior distributions are updating and performs a line search in that
direction :cite:`Honkela:2003`.  The search direction is based on the difference
in the VB parameters on successive updates (or several updates).  The idea is
that the VB-EM algorithm may be slow because it just zigzags and this can be
fixed by moving to the direction in which the VB-EM is slowly moving.

BayesPy offers a simple built-in pattern search method
:func:`VB.pattern_search`.  The method updates the nodes twice, measures the
difference in the parameters and performs a line search with a small number of
function evaluations:

>>> Q.pattern_search(C, X)
Iteration ...

Similarly to the collapsed optimization, it is possible to collapse some of the
variables in the pattern search.  The same rules of conditional independence
apply as above.  The collapsed variables are given as list:

>>> Q.pattern_search(C, tau, collapsed=[X, alpha])
Iteration ...

Also, a maximum number of iterations can be set by using ``maxiter`` keyword
argument.  It is not always obvious whether a pattern search will improve the
rate of convergence or not but if it seems that the convergence is slow because
of zigzagging, it may be worth a try.  Note that the computational cost of the
pattern search is quite high, thus it is not recommended to perform it after
every VB-EM update but every now and then, for instance, after every 10
iterations.  In addition, it is possible to write a more customized VB learning
algorithm which uses pattern searches by using the different methods of
:class:`VB` discussed above.


Deterministic annealing
-----------------------

The standard VB-EM algorithm converges to a local optimum which can often be
inferior to the global optimum and many other local optima.  Deterministic
annealing aims at finding a better local optimum, hopefully even the global
optimum :cite:`Katahira:2008`.  It does this by increasing the weight on the
entropy of the posterior approximation in the VB lower bound.  Effectively, the
annealed lower bound becomes closer to a uniform function instead of the
original multimodal lower bound.  The weight on the entropy is recovered slowly
and the optimization is much more robust to initialization.

In BayesPy, the annealing can be set by using :func:`VB.set_annealing`.  The
given annealing should be in range :math:`(0,1]` but this is not validated in
case the user wants to do something experimental.  If annealing is set to 1, the
original VB lower bound is recovered.  Annealing with 0 would lead to an
improper uniform distribution, thus it will lead to errors.  The entropy term is
weighted by the inverse of this annealing term.  An alternative view is that the
model probability density functions are raised to the power of the annealing
term.

Typically, the annealing is used in such a way that the annealing is small at
the beginning and increased after every convergence of the VB algorithm until
value 1 is reached.  After the annealing value is increased, the algorithm
continues from where it had just converged.  The annealing can be used for
instance as:

>>> beta = 0.1
>>> while beta < 1.0:
...     beta = min(beta*1.5, 1.0)
...     Q.set_annealing(beta)
...     Q.update(repeat=100, tol=1e-4)
Iteration ...

Here, the ``tol`` keyword argument is used to adjust the threshold for
convergence.  In this case, it is a bit larger than by default so the algorithm
does not need to converge perfectly but a rougher convergence is sufficient for
the next iteration with a new annealing value.


Stochastic variational inference
--------------------------------

In stochastic variational inference :cite:`Hoffman:2013`, the idea is to use
mini-batches of large datasets to compute noisy gradients and learn the VB
distributions by using stochastic gradient ascent.  In order for it to be
useful, the model must be such that it can be divided into "intermediate" and
"global" variables.  The number of intermediate variables increases with the
data but the number of global variables remains fixed.  The global variables are
learnt in the stochastic optimization.

By denoting the data as :math:`Y=[Y_1, \ldots, Y_N]`, the intermediate variables
as :math:`Z=[Z_1, \ldots, Z_N]` and the global variables as :math:`\theta`, the
model needs to have the following structure:

.. math::

   p(Y, Z, \theta) &= p(\theta) \prod^N_{n=1} p(Y_n|Z_n,\theta) p(Z_n|\theta)

The algorithm consists of three steps which are iterated: 1) a random mini-batch
of the data is selected, 2) the corresponding intermediate variables are updated
by using normal VB update equations, and 3) the global variables are updated
with (stochastic) gradient ascent as if there was as many replications of the
mini-batch as needed to recover the original dataset size.

The learning rate for the gradient ascent must satisfy:

.. math::

   \sum^\infty_{i=1} \alpha_i = \infty \qquad \text{and} \qquad
   \sum^\infty_{i=1} \alpha^2 < \infty,

where :math:`i` is the iteration number.  An example of a valid learning
parameter is :math:`\alpha_i = (\delta + i)^{-\gamma}`, where :math:`\delta \geq
0` is a delay and :math:`\gamma\in (0.5, 1]` is a forgetting rate.

Stochastic variational inference is relatively easy to use in BayesPy.  The idea
is that the user creates a model for the size of a mini-batch and specifies a
multiplier for those plate axes that are replicated.  For the PCA example, the
mini-batch model can be costructed as follows.  We decide to use ``X`` as an
intermediate variable and the other variables are global.  The global variables
``alpha``, ``C`` and ``tau`` are constructed identically as before.  The
intermediate variable ``X`` is constructed as:

>>> X = GaussianARD(0, 1,
...                 shape=(D,),
...                 plates=(1,5),
...                 plates_multiplier=(1,20),
...                 name='X')

Note that the plates are ``(1,5)`` whereas they are ``(1,100)`` in the full
model.  Thus, we need to provide a plates multiplier ``(1,20)`` to define how
the plates are replicated to get the full dataset.  These multipliers do not
need to be integers, in this case the latter plate axis is multiplied by
:math:`100/5=20`.  The remaining variables are defined as before:

>>> F = Dot(C, X)
>>> Y = GaussianARD(F, tau, name='Y')

Note that the plates of ``Y`` and ``F`` also correspond to the size of the
mini-batch and they also deduce the plate multipliers from their parents, thus
we do not need to specify the multiplier here explicitly (although it is ok to
do so).

Let us construct the inference engine for the new mini-batch model:

>>> Q = VB(Y, C, X, alpha, tau)

Use random initialization for ``C`` to break the symmetry in ``C`` and ``X``:

>>> C.initialize_from_random()

Then, stochastic variational inference algorithm could look as follows:

>>> Q.ignore_bound_checks = True
>>> for n in range(200):
...     subset = np.random.choice(100, 5)
...     Y.observe(data[:,subset])
...     Q.update(X)
...     learning_rate = (n + 2.0) ** (-0.7)
...     Q.gradient_step(C, alpha, tau, scale=learning_rate)
Iteration ...

First, we ignore the bound checks because they are noisy.  Then, the loop
consists of three parts: 1) Draw a random mini-batch of the data (5 samples from
100).  2) Update the intermediate variable ``X``.  3) Update global variables
with gradient ascent using a proper learning rate.


Black-box variational inference
-------------------------------

NOT YET IMPLEMENTED.
