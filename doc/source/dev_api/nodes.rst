..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


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
   gaussian.GaussianToGaussianGamma
   gaussian.GaussianGammaToGaussianWishart
   gaussian.WrapToGaussianGamma
   gaussian.WrapToGaussianWishart
