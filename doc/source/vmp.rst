..
   Copyright (C) 2012 Jaakko Luttinen

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

Variational message passing
===========================

The general update equation for factorized approximation:

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
:math:`\mathrm{ch}(\boldsymbol{\theta})` are the set of parents and
children of :math:`\boldsymbol{\theta}`, respectively.  The
expectations are over the approximate distribution of all other
variables than :math:`\boldsymbol{\theta}`.  Actually, not all the
variables are needed, because the non-constant part uses only the
Markov blanket of :math:`\boldsymbol{\theta}`.  Thus, the optimization
can be done locally using messages from neighbouring nodes.


The messages are simple for conjugate-exponential models.
Exponential-family distributions have the form

.. math::
   :label: likelihood

   \log p(\mathbf{x}|\mathbf{\Theta}) 
   &= 
   \mathbf{u}_{\mathbf{x}}(\mathbf{x})^{\mathrm{T}}
   \boldsymbol{\phi}_{\mathbf{x}}(\mathbf{\Theta})
   + g_{\mathbf{x}}(\mathbf{\Theta})
   + f_{\mathbf{x}}(\mathbf{x}),

where :math:`\mathbf{\Theta}=\{\boldsymbol{\theta}_j\}` is the set of
parents.  If a parent has a conjugate prior, :eq:`likelihood` is
linear with respect to the parent's natural statistics.  Thus,
:eq:`likelihood` can be re-organized with respect to
:math:`\boldsymbol{\theta}_j` as

.. math::

   \log p(\mathbf{x}|\mathbf{\Theta}) 
   &= 
   \mathbf{u}_{\boldsymbol{\theta}_j}(\boldsymbol{\theta}_j)^{\mathrm{T}}
   \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j}
   (\mathbf{x}, \{\boldsymbol{\theta}_k\}_{k\neq j})
   + \mathrm{const},

where :math:`\mathbf{u}_{\boldsymbol{\theta}_j}` is the natural
statistics of :math:`\boldsymbol{\theta}_j`.  Thus, the update
equation :eq:`vmp_general_update` can be given as

.. math::

   \log q(\boldsymbol{\theta}_j) 
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
:math:`\langle\boldsymbol{\phi}_{\boldsymbol{\theta}_j}\rangle`
depends (multi)linearly on the expectations of the parents' natural
statistics.  Similarly, :math:`\langle
\boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j}
\rangle` depends (multi)linearly on the expectations of the children's
and co-parents' natural statistics.

The required expectations can be computed locally by using messages
from the parents and the children.  The message from a parent node
:math:`\boldsymbol{\theta}_j` to a child node :math:`\mathbf{x}` is

.. math::

   \mathbf{m}_{\mathbf{\boldsymbol{\theta}_j}\rightarrow\mathbf{x}}
   &=
   \langle \mathbf{u}_{\boldsymbol{\theta}_j} \rangle
   =
   \tilde{\mathbf{u}}_{\boldsymbol{\theta}_j} 
   (\tilde{\boldsymbol{\phi}}_{\mathbf{\boldsymbol{\theta}_j}}),

and the message from a child node :math:`\mathbf{x}` to a parent node
:math:`\boldsymbol{\theta}_j` is

.. math::
   \mathbf{m}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j}
   &=
   \langle \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} \rangle
   =
   \boldsymbol{\phi}_{\mathbf{x}\rightarrow\boldsymbol{\theta}_j} 
   \left( \langle \mathbf{u}_{\mathbf{x}} \rangle, 
     \{ \mathbf{m}_{\theta_k\rightarrow\mathbf{x}} \}_
     %\{ \langle \mathbf{u}_{\theta_k} \rangle \}_
     {k \neq j}
     %\in \mathrm{cp}({\boldsymbol{\theta}_j}, \mathbf{x})} 
     \right).

Using the messages, the natural parameters of
:math:`q(\boldsymbol{\theta})` can be updated as

.. math::
   
   \tilde{\boldsymbol{\phi}}_{\boldsymbol{\theta}} 
   &=
   \boldsymbol{\phi}_{\boldsymbol{\theta}}
   \left( 
     \{ \mathbf{m}_{\mathbf{z}\rightarrow\boldsymbol{\theta}} \}_
     {\mathbf{z} \in \mathrm{pa}(\boldsymbol{\theta})}
   \right)
   + \sum_{\mathbf{x} \in \mathrm{ch}(\boldsymbol{\theta})}
     \mathbf{m}_{\mathbf{x}\rightarrow\boldsymbol{\theta}}.

.. include:: vmp/vmp_normal.rst

.. include:: vmp/vmp_gaussian.rst

.. include:: vmp/vmp_gamma.rst

.. include:: vmp/vmp_wishart.rst

.. include:: vmp/vmp_normal_gamma.rst

.. include:: vmp/vmp_gaussian_wishart.rst

.. include:: vmp/vmp_gaussian_gamma.rst

.. include:: vmp/vmp_mixture.rst

