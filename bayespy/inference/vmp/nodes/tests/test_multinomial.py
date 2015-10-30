################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `multinomial` module.
"""

import numpy as np
import scipy

from bayespy.nodes import (Multinomial,
                           Dirichlet,
                           Mixture)

from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestMultinomial(TestCase):
    """
    Unit tests for Multinomial node
    """

    
    def test_init(self):
        """
        Test the creation of multinomial nodes.
        """

        # Some simple initializations
        X = Multinomial(10, [0.1, 0.3, 0.6])
        X = Multinomial(10, Dirichlet([5,4,3]))

        # Check that plates are correct
        X = Multinomial(10, [0.1, 0.3, 0.6], plates=(3,4))
        self.assertEqual(X.plates,
                         (3,4))
        X = Multinomial(10, 0.25*np.ones((2,3,4)))
        self.assertEqual(X.plates,
                         (2,3))
        n = 10 * np.ones((3,4), dtype=np.int)
        X = Multinomial(n, [0.1, 0.3, 0.6])
        self.assertEqual(X.plates,
                         (3,4))
        X = Multinomial(n, Dirichlet([2,1,9], plates=(3,4)))
        self.assertEqual(X.plates,
                         (3,4))
        

        # Probabilities not a vector
        self.assertRaises(ValueError,
                          Multinomial,
                          10,
                          0.5)

        # Invalid probability
        self.assertRaises(ValueError,
                          Multinomial,
                          10,
                          [-0.5, 1.5])
        self.assertRaises(ValueError,
                          Multinomial,
                          10,
                          [0.5, 1.5])

        # Invalid number of trials
        self.assertRaises(ValueError,
                          Multinomial,
                          -1,
                          [0.5, 0.5])
        self.assertRaises(ValueError,
                          Multinomial,
                          8.5,
                          [0.5, 0.5])

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Multinomial,
                          10,
                          0.25*np.ones((2,4)),
                          plates=(3,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Multinomial,
                          10,
                          0.25*np.ones((2,4)),
                          plates=(1,))

        pass

    
    def test_moments(self):
        """
        Test the moments of multinomial nodes.
        """

        # Simple test
        X = Multinomial(1, [0.7,0.2,0.1])
        u = X._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0],
                            [0.7,0.2,0.1])

        # Test n
        X = Multinomial(10, [0.7,0.2,0.1])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [7,2,1])

        # Test plates in p
        n = np.random.randint(1, 10)
        p = np.random.dirichlet([1,1], size=3)
        X = Multinomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n)
        
        # Test plates in n
        n = np.random.randint(1, 10, size=(3,))
        p = np.random.dirichlet([1,1,1,1])
        X = Multinomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n[:,None])

        # Test plates in p and n
        n = np.random.randint(1, 10, size=(4,1))
        p = np.random.dirichlet([1,1], size=3)
        X = Multinomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n[...,None])

        # Test with Dirichlet prior
        P = Dirichlet([7, 3])
        logp = P._message_to_child()[0]
        p0 = np.exp(logp[0]) / (np.exp(logp[0]) + np.exp(logp[1]))
        p1 = np.exp(logp[1]) / (np.exp(logp[0]) + np.exp(logp[1]))
        X = Multinomial(1, P)
        u = X._message_to_child()
        p = np.array([p0, p1])
        self.assertAllClose(u[0],
                            p)

        # Test with broadcasted plates
        P = Dirichlet([7, 3], plates=(10,))
        X = Multinomial(5, P)
        u = X._message_to_child()
        self.assertAllClose(u[0] * np.ones(X.get_shape(0)),
                            5*p*np.ones((10,1)))

        pass


    def test_lower_bound(self):
        """
        Test lower bound for multinomial node.
        """

        # Test for a bug found in multinomial
        X = Multinomial(10, [0.3, 0.5, 0.2])
        l = X.lower_bound_contribution()
        self.assertAllClose(l, 0.0)
        
        pass

    
    def test_mixture(self):
        """
        Test multinomial mixture
        """

        p0 = [0.1, 0.5, 0.2, 0.2]
        p1 = [0.5, 0.1, 0.1, 0.3]
        p2 = [0.3, 0.2, 0.1, 0.4]
        X = Mixture(2, Multinomial, 10, [p0, p1, p2])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            10*np.array(p2))

        pass

