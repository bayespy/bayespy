################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `gamma` module.
"""

import numpy as np

from scipy import special

from numpy import testing

from .. import gaussian
from bayespy.nodes import (Gaussian, 
                           GaussianARD,
                           GaussianGamma,
                           Gamma,
                           Wishart)

from ...vmp import VB

from bayespy.utils import misc
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestGammaGradient(TestCase):
    """Numerically check Riemannian gradient of several nodes.
    
    Using VB-EM update equations will take a unit length step to the
    Riemannian gradient direction.  Thus, the change caused by a VB-EM
    update and the Riemannian gradient should be equal.
    """

    def test_riemannian_gradient(self):
        """Test Riemannian gradient of a Gamma node."""

        #
        # Without observations
        #
        
        # Construct model
        a = np.random.rand()
        b = np.random.rand()
        tau = Gamma(a, b)
        # Random initialization
        tau.initialize_from_parameters(np.random.rand(),
                                       np.random.rand())
        # Initial parameters 
        phi0 = tau.phi
        # Gradient
        g = tau.get_riemannian_gradient()
        # Parameters after VB-EM update
        tau.update()
        phi1 = tau.phi
        # Check
        self.assertAllClose(g[0],
                            phi1[0] - phi0[0])
        self.assertAllClose(g[1],
                            phi1[1] - phi0[1])

        #
        # With observations
        #
        
        # Construct model
        a = np.random.rand()
        b = np.random.rand()
        tau = Gamma(a, b)
        mu = np.random.randn()
        Y = GaussianARD(mu, tau)
        Y.observe(np.random.randn())
        # Random initialization
        tau.initialize_from_parameters(np.random.rand(),
                                       np.random.rand())
        # Initial parameters 
        phi0 = tau.phi
        # Gradient
        g = tau.get_riemannian_gradient()
        # Parameters after VB-EM update
        tau.update()
        phi1 = tau.phi
        # Check
        self.assertAllClose(g[0],
                            phi1[0] - phi0[0])
        self.assertAllClose(g[1],
                            phi1[1] - phi0[1])

        pass
        


    def test_gradient(self):
        """Test standard gradient of a Gamma node."""
        D = 3

        np.random.seed(42)

        #
        # Without observations
        #
        
        # Construct model
        a = np.random.rand(D)
        b = np.random.rand(D)
        tau = Gamma(a, b)
        Q = VB(tau)
        # Random initialization
        tau.initialize_from_parameters(np.random.rand(D),
                                       np.random.rand(D))
        # Initial parameters 
        phi0 = tau.phi
        # Gradient
        rg = tau.get_riemannian_gradient()
        g = tau.get_gradient(rg)
        # Numerical gradient
        eps = 1e-8
        p0 = tau.get_parameters()
        l0 = Q.compute_lowerbound(ignore_masked=False)
        g_num = [np.zeros(D), np.zeros(D)]
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[0] + e
            tau.set_parameters([p1, p0[1]])
            l1 = Q.compute_lowerbound(ignore_masked=False)
            g_num[0][i] = (l1 - l0) / eps
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[1] + e
            tau.set_parameters([p0[0], p1])
            l1 = Q.compute_lowerbound(ignore_masked=False)
            g_num[1][i] = (l1 - l0) / eps
                
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])

        #
        # With observations
        #
        
        # Construct model
        a = np.random.rand(D)
        b = np.random.rand(D)
        tau = Gamma(a, b)
        mu = np.random.randn(D)
        Y = GaussianARD(mu, tau)
        Y.observe(np.random.randn(D))
        Q = VB(Y, tau)
        # Random initialization
        tau.initialize_from_parameters(np.random.rand(D),
                                       np.random.rand(D))
        # Initial parameters 
        phi0 = tau.phi
        # Gradient
        rg = tau.get_riemannian_gradient()
        g = tau.get_gradient(rg)
        # Numerical gradient
        eps = 1e-8
        p0 = tau.get_parameters()
        l0 = Q.compute_lowerbound(ignore_masked=False)
        g_num = [np.zeros(D), np.zeros(D)]
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[0] + e
            tau.set_parameters([p1, p0[1]])
            l1 = Q.compute_lowerbound(ignore_masked=False)
            g_num[0][i] = (l1 - l0) / eps
        for i in range(D):
            e = np.zeros(D)
            e[i] = eps
            p1 = p0[1] + e
            tau.set_parameters([p0[0], p1])
            l1 = Q.compute_lowerbound(ignore_masked=False)
            g_num[1][i] = (l1 - l0) / eps
                
        # Check
        self.assertAllClose(g[0],
                            g_num[0])
        self.assertAllClose(g[1],
                            g_num[1])

        pass
        

