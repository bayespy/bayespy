################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


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
