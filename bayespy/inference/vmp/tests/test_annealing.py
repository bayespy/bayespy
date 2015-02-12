######################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

"""
Unit tests for `vmp` module.
"""

import numpy as np

from scipy import special

from numpy import testing

from bayespy.nodes import (Gaussian, 
                           GaussianARD,
                           GaussianGammaISO,
                           Gamma,
                           Wishart)

from ..vmp import VB

from bayespy.utils import misc
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.misc import TestCase

class TestVB(TestCase):

    def test_annealing(self):

        X = GaussianARD(3, 4)
        X.initialize_from_parameters(-1, 6)

        Q = VB(X)
        Q.set_annealing(0.1)

        #
        # Check that the gradient is correct
        #

        # Initial parameters 
        phi0 = X.phi
        # Gradient
        rg = X.get_riemannian_gradient()
        g = X.get_gradient(rg)
        # Numerical gradient of the first parameter
        eps = 1e-6
        p0 = X.get_parameters()
        l0 = Q.compute_lowerbound(ignore_masked=False)
        g_num = [(), ()]
        e = eps
        p1 = p0[0] + e
        X.set_parameters([p1, p0[1]])
        l1 = Q.compute_lowerbound(ignore_masked=False)
        g_num[0] = (l1 - l0) / eps
        # Numerical gradient of the second parameter
        p1 = p0[1] + e
        X.set_parameters([p0[0], p1])
        l1 = Q.compute_lowerbound(ignore_masked=False)
        g_num[1] = (l1 - l0) / (eps)
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])

        #
        # Gradient should be zero after updating
        #

        X.update()
        # Initial parameters 
        phi0 = X.phi
        # Numerical gradient of the first parameter
        eps = 1e-8
        p0 = X.get_parameters()
        l0 = Q.compute_lowerbound(ignore_masked=False)
        g_num = [(), ()]
        e = eps
        p1 = p0[0] + e
        X.set_parameters([p1, p0[1]])
        l1 = Q.compute_lowerbound(ignore_masked=False)
        g_num[0] = (l1 - l0) / eps
        # Numerical gradient of the second parameter
        p1 = p0[1] + e
        X.set_parameters([p0[0], p1])
        l1 = Q.compute_lowerbound(ignore_masked=False)
        g_num[1] = (l1 - l0) / (eps)
        # Check
        self.assertAllClose(0,
                            g_num[0],
                            atol=1e-5)
        self.assertAllClose(0,
                            g_num[1],
                            atol=1e-5)

        # Not at the optimum
        X.initialize_from_parameters(-1, 6)
        # Initial parameters 
        phi0 = X.phi
        # Gradient
        g = X.get_riemannian_gradient()
        # Parameters after VB-EM update
        X.update()
        phi1 = X.phi
        # Check
        self.assertAllClose(g[0],
                            phi1[0] - phi0[0])
        self.assertAllClose(g[1],
                            phi1[1] - phi0[1])
        

        pass
