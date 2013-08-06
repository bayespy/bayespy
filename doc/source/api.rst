..
   Copyright (C) 2011,2012 Jaakko Luttinen

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

API
***

Stochastic nodes in VMP
=======================

.. currentmodule:: bayespy.nodes

For testing, I can tell that :class:`Gaussian` is used for modelling Gaussian
variables.

.. autosummary::
   :toctree: generated/

   Normal
   Gaussian
   Gamma
   Wishart
   Dirichlet

.. autosummary::
   :toctree: generated/

   normal
   Categorical
   Mixture
   GaussianMarkovChain

.. autosummary::
   :toctree: generated/

   Normal.lowerbound

.. autoclass:: Normal

Deterministic
=============

.. autosummary::
   :toctree: generated/

   Dot

.. currentmodule:: bayespy.inference.vmp

.. autosummary::
   :toctree: generated/

   nodes


Utility functions
=================

.. currentmodule:: bayespy.utils.utils

.. 
   autosummary::
   :toctree: generated/

   kalman_filter
   rts_smoother

.. currentmodule:: bayespy

.. 
   autosummary::
   :toctree: generated/

   demos
   inference
   utils.utils
