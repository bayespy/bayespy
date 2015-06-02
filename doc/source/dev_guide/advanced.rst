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
    Q = None


Advanced topics
===============

The VB lower bound and its gradients
------------------------------------

The VB lower bound:

.. math::

   \mathcal{L} &= \underbrace{\langle \log p(X,Z) \rangle}_{\equiv \mathcal{L}_p}
   - \underbrace{\langle \log q(Z) \rangle}_{\equiv \mathcal{L}_q}


The child nodes pass the gradient to the parent node so the parent node can
optimize its parameters.  In general, :math:`\mathcal{L}_p` can be almost
arbitrarily complex function of :math:`Z`:

.. math::

   \mathcal{L}_p = \langle \log p(X,Z) \rangle.


The gradient is

.. math::

   \frac{\partial}{\partial \xi} \mathcal{L}_p
   &= \frac{\partial}{\partial \xi} \langle \log p(X,Z) \rangle
   \\
   &=  \langle \log p(X,Z) \frac{\partial}{\partial \xi} \log q(Z) \rangle

which can be computed, for instance, by sampling from :math:`q(Z)`.  Note that
:math:`\xi` can represent, for instance, the expectation parameters
:math:`\bar{u}` of :math:`q(Z)` in order to obtain the Riemannian gradient for
an exponential family :math:`q(Z)`.


Often, :math:`\mathcal{L}_p` has a simpler form (or it can be further lower
bounded by a simpler form).  If :math:`\mathcal{L}_p` can be written as a
function of :math:`\bar{u}` as

.. math::

   \mathcal{L}_p = \bar{u}^T \psi + \mathrm{const},

the gradient with respect to the moments is

.. math::

   \frac{\partial}{\partial \bar{u}} \mathcal{L}_p = \psi.
   
It can be that :math:`\psi` can be computed exactly by using moments of other
nodes or it needs to be approximated by using sampling from the distribution of
other nodes.


To summarize, the gradient message can be a numerical gradient, an approximate
stochastic gradient (by sampling other nodes) or a function which can be used to
compute an approximate stochastic gradient by sampling the node itself (and
possibly other nodes).


Riemannian gradient
-------------------

In principle, the VB lower bound can be maximized with respect to any
parameterization of the approximate distribution.  However, normal gradient can
perform badly, because it doesn't take into account the geometry of the space of
probability distributions.  This can be fixed by using Riemannian (i.e.,
natural) gradient.  In general, the Riemannian gradient is defined as

.. math::

   \tilde{\nabla}_\xi \mathcal{L} = G^{-1} \nabla_\xi \mathcal{L}

where

.. math::

   [G]_{ij} = \left\langle \frac{\partial \log q(Z)}{\partial \xi_i}
   \frac{\partial \log q(Z)}{\partial \xi_j} \right\rangle = - \left\langle
   \frac{\partial^2 \log q(Z)}{\partial \xi_i \partial \xi_j} \right\rangle.

For exponential family distributions, the Riemannian gradient with respect to
the natural parameters :math:`\phi` can be computed easily by taking the
gradient with respect to the moments :math:`\bar{u}`:

.. math::

   \tilde{\nabla}_\phi = G^{-1} \nabla_\phi \mathcal{L} = \nabla_{\bar{u}}
   \mathcal{L}.

Note that :math:`G` depends only on the approximate distribution :math:`q(Z)`.
Thus, the model itself does not need to be in the exponential family but only
the approximation, in order to use this property.  The Riemannian gradient of
:math:`\mathcal{L}_q` for exponential family distributions :math:`q(Z)` is

.. math::

   \tilde{\nabla}_\phi \mathcal{L}_q = \nabla_{\bar{u}} \mathcal{L}_q =
   \nabla_{\bar{u}} [ \bar{u}^T \phi + \langle f(Z)
   \rangle + g(\phi) ] = \phi.

Thus, the Riemannian gradient is

.. math::

   \tilde{\nabla}_\phi \mathcal{L} = \nabla_{\bar{u}}
   \mathcal{L}_p - \phi.

.. todo::

   Should f(Z) be taken into account? It cancels out if prior and q are in the
   same family. But if they are not, it doesn't cancel out. Does it affect the
   gradient?

Nonlinear conjugate gradient methods :cite:`Hensman:2012`:

* Fletcher-Reeves:

.. math::

   \beta_n = \frac { \langle \tilde{g}_n, \tilde{g}_n \rangle_n } { \langle
   \tilde{g}_{n-1}, \tilde{g}_{n-1} \rangle_{n-1} } = \frac { \langle g_n,
   \tilde{g}_n \rangle } { \langle g_{n-1}, \tilde{g}_{n-1} \rangle }

* Polak-Ribiere:

.. math::

   \beta_n = \frac { \langle \tilde{g}_n, \tilde{g}_n - \tilde{g}_{n-1}
   \rangle_n } { \langle \tilde{g}_{n-1}, \tilde{g}_{n-1} \rangle_{n-1} } =
   \frac { \langle g_n, \tilde{g}_n - \tilde{g}_{n-1} \rangle } {
   \langle g_{n-1}, \tilde{g}_{n-1} \rangle }

* Hestenes-Stiefel:

.. math::

   \beta_n = - \frac { \langle \tilde{g}_n, \tilde{g}_n - \tilde{g}_{n-1}
   \rangle_n } { \langle \tilde{g}_{n-1}, \tilde{g}_{n-1} \rangle_{n-1} } = -
   \frac { \langle g_n, \tilde{g}_n - \tilde{g}_{n-1} \rangle } { \langle
   g_{n-1}, \tilde{g}_{n-1} \rangle }

where :math:`\langle \rangle_i` denotes the inner product in the Riemannian
geometry, :math:`\langle \rangle` denotes the inner product in the Euclidean
space, :math:`\tilde{g}` denotes the Riemannian gradient and :math:`g` denotes
the gradient, and the following property has been used:

.. math::

   \langle \tilde{g}_n, \tilde{x} \rangle_n = \tilde{g}_n^T G_n \tilde{x} = g^T
   G^{-1}_n G_n \tilde{x} = g^T \tilde{x} = \langle g, \tilde{x} \rangle

TODO
----

 * simulated annealing

 * Riemannian (conjugate) gradient

 * black box variational inference

 * stochastic variational inference

 * pattern search

 * fast inference

 * parameter expansion
