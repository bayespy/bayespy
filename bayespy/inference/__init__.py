################################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Package for Bayesian inference engines

Inference engines
-----------------

.. autosummary::
   :toctree: generated/

   VB

Parameter expansions
--------------------

.. autosummary::
   :toctree: generated/

   vmp.transformations.RotationOptimizer
   vmp.transformations.RotateGaussian
   vmp.transformations.RotateGaussianARD
   vmp.transformations.RotateGaussianMarkovChain
   vmp.transformations.RotateSwitchingMarkovChain
   vmp.transformations.RotateVaryingMarkovChain
   vmp.transformations.RotateMultiple
"""

from .vmp.vmp import VB
