######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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
Unit tests for `gaussian_wishart` module.
"""

import warnings

import numpy as np
from scipy import special

from bayespy.nodes import (GaussianGammaISO,
                           GaussianARD,
                           Gaussian,
                           Wishart,
                           Gamma)

#from bayespy.inference.vmp.nodes.gaussian import GaussianMoments

from bayespy.utils import (random,
                           linalg,
                           misc)

from bayespy.utils.misc import TestCase


class TestGaussianGammaISO(TestCase):
    """
    Unit tests for GaussianGammaISO node.
    """
    

    def test_init(self):
        """
        Test the creation of GaussianGammaISO node
        """

        # Simple construction
        X_alpha = GaussianGammaISO([1,2,3], np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Plates
        X_alpha = GaussianGammaISO([1,2,3], np.identity(3), 2, 10, plates=(4,))
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Plates in mu
        X_alpha = GaussianGammaISO(np.ones((4,3)), np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in Lambda
        X_alpha = GaussianGammaISO(np.ones(3), np.ones((4,3,3))*np.identity(3), 2, 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in a
        X_alpha = GaussianGammaISO(np.ones(3), np.identity(3), np.ones(4), 10)
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        # Plates in Lambda
        X_alpha = GaussianGammaISO(np.ones(3), np.identity(3), 2, np.ones(4))
        self.assertEqual(X_alpha.plates, (4,))
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # Inconsistent plates
        self.assertRaises(ValueError,
                          GaussianGammaISO,
                          np.ones((4,3)),
                          np.identity(3), 
                          2,
                          10,
                          plates=())
        
        # Inconsistent plates
        self.assertRaises(ValueError,
                          GaussianGammaISO,
                          np.ones((4,3)),
                          np.identity(3), 
                          2,
                          10,
                          plates=(5,))

        # Unknown parameters
        mu = Gaussian(np.zeros(3), np.identity(3))
        Lambda = Wishart(10, np.identity(3))
        b = Gamma(1, 1)
        X_alpha = GaussianGammaISO(mu, Lambda, 2, b)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))

        # mu is Gaussian-gamma
        mu_tau = GaussianGammaISO(np.ones(3), np.identity(3), 5, 5)
        X_alpha = GaussianGammaISO(mu_tau, np.identity(3), 5, 5)
        self.assertEqual(X_alpha.plates, ())
        self.assertEqual(X_alpha.dims, ( (3,), (3,3), (), () ))
        
        pass
        

    def test_message_to_child(self):
        """
        Test the message to child of GaussianGammaISO node.
        """

        # Simple test
        mu = np.array([1,2,3])
        Lambda = np.identity(3)
        a = 2
        b = 10
        X_alpha = GaussianGammaISO(mu, Lambda, a, b)
        u = X_alpha._message_to_child()
        self.assertEqual(len(u), 4)
        tau = np.array(a/b)
        self.assertAllClose(u[0],
                            tau[...,None] * mu)
        self.assertAllClose(u[1],
                            (linalg.inv(Lambda) 
                             + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2],
                            tau)
        self.assertAllClose(u[3],
                            -np.log(b) + special.psi(a))

        # Test with unknown parents
        mu = Gaussian(np.arange(3), 10*np.identity(3))
        Lambda = Wishart(10, np.identity(3))
        a = 2
        b = Gamma(3, 15)
        X_alpha = GaussianGammaISO(mu, Lambda, a, b)
        u = X_alpha._message_to_child()
        (mu, mumu) = mu._message_to_child()
        Cov_mu = mumu - linalg.outer(mu, mu)
        (Lambda, _) = Lambda._message_to_child()
        (b, _) = b._message_to_child()
        (tau, logtau) = Gamma(a, b + 0.5*np.sum(Lambda*Cov_mu))._message_to_child()
        self.assertAllClose(u[0],
                            tau[...,None] * mu)
        self.assertAllClose(u[1],
                            (linalg.inv(Lambda)
                             + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2],
                            tau)
        self.assertAllClose(u[3],
                            logtau)

        # Test with plates
        mu = Gaussian(np.reshape(np.arange(3*4), (4,3)),
                      10*np.identity(3),
                      plates=(4,))
        Lambda = Wishart(10, np.identity(3))
        a = 2
        b = Gamma(3, 15)
        X_alpha = GaussianGammaISO(mu, Lambda, a, b, plates=(4,))
        u = X_alpha._message_to_child()
        (mu, mumu) = mu._message_to_child()
        Cov_mu = mumu - linalg.outer(mu, mu)
        (Lambda, _) = Lambda._message_to_child()
        (b, _) = b._message_to_child()
        (tau, logtau) = Gamma(a, 
                              b + 0.5*np.sum(Lambda*Cov_mu, 
                                             axis=(-1,-2)))._message_to_child()
        self.assertAllClose(u[0] * np.ones((4,1)),
                            np.ones((4,1)) * tau[...,None] * mu)
        self.assertAllClose(u[1] * np.ones((4,1,1)),
                            np.ones((4,1,1)) * (linalg.inv(Lambda)
                                                + tau[...,None,None] * linalg.outer(mu, mu)))
        self.assertAllClose(u[2] * np.ones(4),
                            np.ones(4) * tau)
        self.assertAllClose(u[3] * np.ones(4),
                            np.ones(4) * logtau)
        
        pass


    def test_mask_to_parent(self):
        """
        Test the mask handling in GaussianGammaISO node
        """

        pass
