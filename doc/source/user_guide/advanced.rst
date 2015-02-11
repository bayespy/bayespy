..
   Copyright (C) 2014-2015 Jaakko Luttinen

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



Advanced topics
===============

This section contains brief information on how to implement some advanced
methods in BayesPy.  These methods include Riemannian conjugate gradient
methods, pattern search, simulated annealing, collapsed variational inference,
stochastic variational inference and black box variational inference.  In order
to use these methods properly, the user should understand them to some extent.
They are also considered experimental, thus you may encounter bugs or
unimplemented features.  In any case, these methods may provide huge performance
improvements easily compared to the standard VB-EM algorithm.


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

Typically, the annealing is used in such a way that the annealing is very small
at the beginning (e.g., 0.01) and increased after every convergence of the VB
algorithm until value 1 is reached.  After the annealing value is increased, the
algorithm continues from where it had just converged.  The annealing can be used
for instance as:

>>> beta = 0.01
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


Black-box variational inference
-------------------------------


