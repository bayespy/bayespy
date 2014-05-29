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


Nodes
*****


Stochastic nodes
================

.. currentmodule:: bayespy.nodes

Nodes for Gaussian variables:

.. autosummary::
   :toctree: generated/

   Gaussian
   GaussianARD

Nodes for precision and scale variables:

.. autosummary::
   :toctree: generated/

   Gamma
   Wishart
   Exponential

Nodes for modelling Gaussian and precision variables jointly (useful as prior
for Gaussian nodes):

.. autosummary::
   :toctree: generated/

   GaussianGammaISO
   GaussianGammaARD
   GaussianWishart

Nodes for discrete count variables:

.. autosummary::
   :toctree: generated/

   Bernoulli
   Binomial
   Categorical
   Multinomial
   Poisson

Nodes for probabilities:

.. autosummary::
   :toctree: generated/

   Beta
   Dirichlet

Nodes for dynamic variables:

.. autosummary::
   :toctree: generated/

   CategoricalMarkovChain
   GaussianMarkovChain
   SwitchingGaussianMarkovChain
   VaryingGaussianMarkovChain

Other stochastic nodes:

.. autosummary::
   :toctree: generated/

   Mixture



Deterministic nodes
===================

.. autosummary::
   :toctree: generated/

   Dot
   SumMultiply
   Gate



Base nodes (for developers)
===========================

.. currentmodule:: bayespy.inference.vmp.nodes

These nodes should be interesting only for developers.

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   node.Node
   stochastic.Stochastic
   deterministic.Deterministic

.. autosummary::
   :toctree: generated/

   constant.Constant

