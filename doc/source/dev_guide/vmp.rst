..
   Copyright (C) 2012 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Variational message passing
===========================

This section briefly describes the variational message passing (VMP) framework,
which is currently the only implemented inference engine in BayesPy.  The
variational Bayesian (VB) inference engine in BayesPy assumes that the posterior
approximation factorizes with respect to nodes and plates.  VMP is based on
updating one node at a time (the plates in one node can be updated
simultaneously) and iteratively updating all nodes in turns until convergence.

Standard update equation
------------------------

The general update equation for the factorized approximation of node
:math:`\boldsymbol{\theta}` is the following:

.. math::
   :label: vmp_general_update

   \log q(\boldsymbol{\theta}) 
   &= 
   \langle 
     \log p\left( \boldsymbol{\theta} |
                  \mathrm{pa}(\boldsymbol{\theta}) \right)
   \rangle 
   + \sum_{\mathbf{x} \in \mathrm{ch}(\boldsymbol{\theta})} 
     \langle \log p(\mathbf{x}|\mathrm{pa}(\mathbf{x})) \rangle
   + \mathrm{const},

where :math:`\mathrm{pa}(\boldsymbol{\theta})` and
:math:`\mathrm{ch}(\boldsymbol{\theta})` are the set of parents and children of
:math:`\boldsymbol{\theta}`, respectively.  Thus, the posterior approximation of
a node is updated by taking a sum of the expectations of all log densities in
which the node variable appears.  The expectations are over the approximate
distribution of all other variables than :math:`\boldsymbol{\theta}`.  Actually,
not all the variables are needed, because the non-constant part depends only on
the Markov blanket of :math:`\boldsymbol{\theta}`.  This leads to a local
optimization scheme, which uses messages from neighbouring nodes.

The messages are simple for conjugate exponential family models.  An exponential
family distribution has the following log probability density function:

.. math::
   :label: likelihood

   \log p(\mathbf{x}|\mathbf{\Theta}) 
   &= 
   \mathbf{u}_{\mathbf{x}}(\mathbf{x})^{\mathrm{T}}
   \boldsymbol{\phi}_{\mathbf{x}}(\mathbf{\Theta})
   + g_{\mathbf{x}}(\mathbf{\Theta})
   + f_{\mathbf{x}}(\mathbf{x}),

where :math:`\mathbf{\Theta}=\{\boldsymbol{\theta}_j\}` is the set of parents,
:math:`\mathbf{u}` is the sufficient statistic vector, :math:`\boldsymbol{\phi}`
is the natural parameter vector, :math:`g` is the negative log normalizer, and
:math:`f` is the log base function.  Note that the log density is linear with
respect to the terms that are functions of :math:`\mathbf{x}`:
:math:`\mathbf{u}` and :math:`f`.  If a parent has a conjugate prior,
:eq:`likelihood` is also linear with respect to the parent's sufficient
statistic vector.  Thus, :eq:`likelihood` can be re-organized with respect to a
parent :math:`\boldsymbol{\theta}_j` as

.. math::

   \log p(\mathbf{x}|\mathbf{\Theta}) 
   &= 
   \mathbf{u}_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)^{\mathrm{T}}
   \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j}
   (\mathbf{x}, \{\boldsymbol{\theta}_k\}_{k\neq j})
   + \mathrm{const},

where :math:`\mathbf{u}_{\boldsymbol{\theta}_j}` is the sufficient statistic
vector of :math:`\boldsymbol{\theta}_j` and the constant part is constant with
respect to :math:`\boldsymbol{\theta}_j`.  Thus, the update equation
:eq:`vmp_general_update` for :math:`\boldsymbol{\theta}_j` can be written as

.. math::

   \log q(\boldsymbol{\theta}_j) 
   &=
   \mathbf{u}_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)^{\mathrm{T}}
     \langle \boldsymbol{\phi}_{\boldsymbol{\theta}_j} \rangle
   + f_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)
   + 
   \mathbf{u}_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)^{\mathrm{T}}
   \sum_{\mathbf{x} \in \mathrm{ch}(\boldsymbol{\theta}_j)}
         \langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} \rangle
   + \mathrm{const},
   \\
   &=
   \mathbf{u}_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)^{\mathrm{T}}
   \left(
     \langle \boldsymbol{\phi}_{\boldsymbol{\theta}_j} \rangle
     + \sum_{\mathbf{x} \in \mathrm{ch}(\boldsymbol{\theta}_j)}
         \langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} \rangle
   \right)
   + f_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)
   + \mathrm{const},

where the summation is over all the child nodes of
:math:`\boldsymbol{\theta}_j`.  Because of the conjugacy,
:math:`\langle\boldsymbol{\phi}_{\boldsymbol{\theta}_j}\rangle` depends
(multi)linearly on the parents' sufficient statistic vector.  Similarly,
:math:`\langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j}
\rangle` depends (multi)linearly on the expectations of the children's and
co-parents' sufficient statistics.  This gives the following update equation for
the natural parameter vector of the posterior approximation
:math:`q(\boldsymbol{\phi}_j)`:

.. math::
   :label: update_phi

   \tilde{\boldsymbol{\phi}}_j &= \langle \boldsymbol{\phi}_{\boldsymbol{\theta}_j} \rangle
     + \sum_{\mathbf{x} \in \mathrm{ch}(\boldsymbol{\theta}_j)} \langle
         \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} \rangle.

Variational messages
--------------------

The update equation :eq:`update_phi` leads to a message passing scheme: the term
:math:`\langle \boldsymbol{\phi}_{\boldsymbol{\theta}_j} \rangle` is a function
of the parents' sufficient statistic vector and the term :math:`\langle
\boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} \rangle` can be
interpreted as a message from the child node :math:`\mathbf{x}`.  Thus, the
message from the child node :math:`\mathbf{x}` to the parent node
:math:`\boldsymbol{\theta}` is

.. math::

   \mathbf{m}_{\mathbf{x}\rightarrow\boldsymbol{\theta}}
   &\equiv
   \langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}} \rangle,

which can be computed as a function of the sufficient statistic vector of the
co-parent nodes of :math:`\boldsymbol{\theta}` and the sufficient statistic
vector of the child node :math:`\mathbf{x}`.  The message from the parent node
:math:`\boldsymbol{\theta}` to the child node :math:`\mathbf{x}` is simply the
expectation of the sufficient statistic vector:

.. math::

   \mathbf{m}_{\mathbf{\boldsymbol{\theta}}\rightarrow\mathbf{x}}
   &\equiv
   \langle \mathbf{u}_{\boldsymbol{\theta}} \rangle.

In order to compute the expectation of the sufficient statistic vector we need
to write :math:`q(\boldsymbol{\theta})` as

.. math::

   \log q(\boldsymbol{\theta}) &= 
   \mathbf{u}(\boldsymbol{\theta})^{\mathrm{T}}
   \tilde{\boldsymbol{\phi}}
   + \tilde{g}(\tilde{\boldsymbol{\phi}})
   + f(\boldsymbol{\theta}),

where :math:`\tilde{\boldsymbol{\phi}}` is the natural
parameter vector of :math:`q(\boldsymbol{\theta})`.  Now, the expectation of the
sufficient statistic vector is defined as

.. math::
   :label: moments

   \langle \mathbf{u}_{\boldsymbol{\theta}} \rangle 
   &= - \frac{\partial \tilde{g}}{\partial
   \tilde{\boldsymbol{\phi}}_{\boldsymbol{\theta}}} 
   (\tilde{\boldsymbol{\phi}}_{\boldsymbol{\theta}}).

We call this expectation of the sufficient statistic vector as the moments
vector.




Lower bound
-----------


Computing the VB lower bound is not necessary in order to find the posterior
approximation, although it is extremely useful in monitoring convergence and
possible bugs.  The VB lower bound can be written as


.. math::

   \mathcal{L} = \langle \log p(\mathbf{Y}, \mathbf{X}) \rangle - \langle \log
   q(\mathbf{X}) \rangle,

where :math:`\mathbf{Y}` is the set of all observed variables and
:math:`\mathbf{X}` is the set of all latent variables.  It can also be written as

.. math::

   \mathcal{L} = \sum_{\mathbf{y} \in \mathbf{Y}} \langle \log p(\mathbf{y} |
   \mathrm{pa}(\mathbf{y})) \rangle
   + \sum_{\mathbf{x} \in \mathbf{X}} \left[ \langle \log p(\mathbf{x} |
     \mathrm{pa}(\mathbf{x})) \rangle - \langle \log q(\mathbf{x}) \right],

which shows that observed and latent variables contribute differently to the
lower bound.  These contributions have simple forms for exponential family
nodes.  Observed exponential family nodes contribute to the lower bound as
follows:

.. math::

   \langle \log p(\mathbf{y}|\mathrm{pa}(\mathbf{y})) \rangle &=
   \mathbf{u}(\mathbf{y})^T \langle \boldsymbol{\phi} \rangle
   + \langle g \rangle + f(\mathbf{x}),

where :math:`\mathbf{y}` is the observed data.  On the other hand, latent
exponential family nodes contribute to the lower bound as follows:

.. math::

   \langle \log p(\mathbf{x}|\boldsymbol{\theta}) \rangle
   - \langle \log q(\mathbf{x}) \rangle &= \langle \mathbf{u} \rangle^T (\langle
   \boldsymbol{\phi} \rangle - \tilde{\boldsymbol{\phi}} )
   + \langle g \rangle - \tilde{g}.

If a node is partially observed and partially unobserved, these formulas are
applied plate-wise appropriately.
   
.. _sec-vmp-terms:

Terms
-----

To summarize, implementing VMP requires one to write for each stochastic
exponential family node:

    :math:`\langle \boldsymbol{\phi} \rangle` : the expectation of the prior
    natural parameter vector

        Computed as a function of the messages from parents.

    :math:`\tilde{\boldsymbol{\phi}}` : natural parameter vector of the
    posterior approximation

        Computed as a sum of :math:`\langle \boldsymbol{\phi} \rangle` and the
        messages from children.

    :math:`\langle \mathbf{u} \rangle` : the posterior moments vector

        Computed as a function of :math:`\tilde{\boldsymbol{\phi}}` as defined
        in :eq:`moments`.

    :math:`\mathbf{u}(\mathbf{x})` : the moments vector for given data

        Computed as a function of of the observed data :math:`\mathbf{x}`.

    :math:`\langle g \rangle` : the expectation of the negative log normalizer
    of the prior

        Computed as a function of parent moments.

    :math:`\tilde{g}` : the negative log normalizer of the posterior
    approximation

        Computed as a function of :math:`\tilde{\boldsymbol{\phi}}`.

    :math:`f(\mathbf{x})` : the log base measure for given data

        Computed as a function of the observed data :math:`\mathbf{x}`.

    :math:`\langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}}
    \rangle` : the message to parent :math:`\boldsymbol{\theta}`

        Computed as a function of the moments of this node and the other
        parents.


Deterministic nodes require only the following terms:


    :math:`\langle \mathbf{u} \rangle` : the posterior moments vector

        Computed as a function of the messages from the parents.

    :math:`\mathbf{m}` : the message to a parent

        Computed as a function of the messages from the other parents and all
        children.



