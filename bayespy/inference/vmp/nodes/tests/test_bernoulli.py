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
Unit tests for `bernoulli` module.
"""

import numpy as np
import scipy

from bayespy.nodes import (Bernoulli,
                           Beta,
                           Mixture)

from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestBernoulli(TestCase):
    """
    Unit tests for Bernoulli node
    """

    
    def test_init(self):
        """
        Test the creation of Bernoulli nodes.
        """

        # Some simple initializations
        X = Bernoulli(0.5)
        X = Bernoulli(Beta([2,3]))

        # Check that plates are correct
        X = Bernoulli(0.7, plates=(4,3))
        self.assertEqual(X.plates,
                         (4,3))
        X = Bernoulli(0.7*np.ones((4,3)))
        self.assertEqual(X.plates,
                         (4,3))
        X = Bernoulli(Beta([4,3], plates=(4,3)))
        self.assertEqual(X.plates,
                         (4,3))
        
        # Invalid probability
        self.assertRaises(ValueError,
                          Bernoulli,
                          -0.5)
        self.assertRaises(ValueError,
                          Bernoulli,
                          1.5)

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Bernoulli,
                          0.5*np.ones(4),
                          plates=(3,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Bernoulli,
                          0.5*np.ones(4),
                          plates=(1,))

        pass

    
    def test_moments(self):
        """
        Test the moments of Bernoulli nodes.
        """

        # Simple test
        X = Bernoulli(0.7)
        u = X._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0],
                            0.7)

        # Test plates in p
        p = np.random.rand(3)
        X = Bernoulli(p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p)
        
        # Test with beta prior
        P = Beta([7, 3])
        logp = P._message_to_child()[0]
        p0 = np.exp(logp[0]) / (np.exp(logp[0]) + np.exp(logp[1]))
        X = Bernoulli(P)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p0)

        # Test with broadcasted plates
        P = Beta([7, 3], plates=(10,))
        X = Bernoulli(P)
        u = X._message_to_child()
        self.assertAllClose(u[0] * np.ones(X.get_shape(0)),
                            p0*np.ones(10))

        pass


    def test_mixture(self):
        """
        Test mixture of Bernoulli
        """
        P = Mixture([2,0,0], Bernoulli, [0.1, 0.2, 0.3])
        u = P._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0], [0.3, 0.1, 0.1])
        pass


    def test_observed(self):
        """
        Test observation of Bernoulli node
        """
        Z = Bernoulli(0.3)
        Z.observe(2 < 3)
        pass


    def test_random(self):
        """
        Test random sampling in Bernoulli node
        """
        p = [1.0, 0.0]
        with np.errstate(divide='ignore'):
            Z = Bernoulli(p, plates=(3,2)).random()
        self.assertArrayEqual(Z, np.ones((3,2))*p)
