Variational message passing
===========================

The general update equation for factorized approximation:

.. math::

   \log q(X_j) 
   &= 
   \langle \log p(X_j|\mathrm{pa}_j) \rangle 
   + \sum_{k\in \mathrm{ch}_j} \langle \log p(X_k|\mathrm{pa}_k) \rangle
   + \mathrm{const},

where the expectations are over the approximate distribution of all
other variables than :math:`X_j`, that is, :math:`\prod_{i\neq j}
q(X_i)`.  Actually, not all the variables are needed, because the
non-constant part uses only the Markov blanket of :math:`X_j`.  Thus,
the optimization can be done locally using messages from neighbouring
nodes.


The messages are simple for conjugate-exponential models.
Exponential-family distributions have the following form:

.. math::

   \log p(\mathbf{x}|\mathbf{\Theta}) 
   &= 
   \mathbf{u}(\mathbf{x})^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{\Theta}) 
   + g(\mathbf{\Theta}) 
   + f(\mathbf{x})

.. include:: vmp/vmp_normal.rst

.. include:: vmp/vmp_gaussian.rst

.. include:: vmp/vmp_gamma.rst

.. include:: vmp/vmp_wishart.rst




