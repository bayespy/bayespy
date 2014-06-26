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


Developer nodes
===============

.. currentmodule:: bayespy.inference.vmp.nodes

The following base classes are useful if writing new nodes:

.. autosummary::
   :toctree: generated/

   node.Node
   stochastic.Stochastic
   expfamily.ExponentialFamily
   deterministic.Deterministic

The following nodes are examples of special nodes that remain hidden for the
user although they are often implicitly used:

.. autosummary::
   :toctree: generated/

   constant.Constant
   gaussian.GaussianToGaussianGammaISO
   gaussian.GaussianGammaISOToGaussianGammaARD
   gaussian.GaussianGammaARDToGaussianWishart
   gaussian.WrapToGaussianGammaISO
   gaussian.WrapToGaussianGammaARD
   gaussian.WrapToGaussianWishart
