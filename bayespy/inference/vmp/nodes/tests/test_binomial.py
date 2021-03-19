################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `binomial` module.
"""

import numpy as np
import scipy

from bayespy.nodes import (Binomial,
                           Beta,
                           Mixture)

from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestBinomial(TestCase):
    """
    Unit tests for Binomial node
    """


    def test_init(self):
        """
        Test the creation of binomial nodes.
        """

        # Some simple initializations
        X = Binomial(10, 0.5)
        X = Binomial(10, Beta([2,3]))

        # Check that plates are correct
        X = Binomial(10, 0.7, plates=(4,3))
        self.assertEqual(X.plates,
                         (4,3))
        X = Binomial(10, 0.7*np.ones((4,3)))
        self.assertEqual(X.plates,
                         (4,3))
        n = np.ones((4,3), dtype=np.int)
        X = Binomial(n, 0.7)
        self.assertEqual(X.plates,
                         (4,3))
        X = Binomial(10, Beta([4,3], plates=(4,3)))
        self.assertEqual(X.plates,
                         (4,3))

        # Invalid probability
        self.assertRaises(ValueError,
                          Binomial,
                          10,
                          -0.5)
        self.assertRaises(ValueError,
                          Binomial,
                          10,
                          1.5)

        # Invalid number of trials
        self.assertRaises(ValueError,
                          Binomial,
                          -1,
                          0.5)
        self.assertRaises(ValueError,
                          Binomial,
                          8.5,
                          0.5)

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Binomial,
                          10,
                          0.5*np.ones(4),
                          plates=(3,))

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Binomial,
                          10,
                          0.5*np.ones(4),
                          plates=(1,))

        pass


    def test_moments(self):
        """
        Test the moments of binomial nodes.
        """

        # Simple test
        X = Binomial(1, 0.7)
        u = X._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0],
                            0.7)

        # Test n
        X = Binomial(10, 0.7)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            10*0.7)

        # Test plates in p
        n = np.random.randint(1, 10)
        p = np.random.rand(3)
        X = Binomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n)

        # Test plates in n
        n = np.random.randint(1, 10, size=(3,))
        p = np.random.rand()
        X = Binomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n)

        # Test plates in p and n
        n = np.random.randint(1, 10, size=(4,1))
        p = np.random.rand(3)
        X = Binomial(n, p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p*n)

        # Test with beta prior
        P = Beta([7, 3])
        logp = P._message_to_child()[0]
        p0 = np.exp(logp[0]) / (np.exp(logp[0]) + np.exp(logp[1]))
        X = Binomial(1, P)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p0)

        # Test with broadcasted plates
        P = Beta([7, 3], plates=(10,))
        X = Binomial(5, P)
        u = X._message_to_child()
        self.assertAllClose(u[0] * np.ones(X.get_shape(0)),
                            5*p0*np.ones(10))

        pass


    def test_mixture(self):
        """
        Test binomial mixture
        """

        X = Mixture(2, Binomial, 10, [0.1, 0.2, 0.3, 0.4])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            3.0)

        pass


    def test_observed(self):
        """
        Test observation of Bernoulli node
        """
        Z = Binomial(10, 0.3)
        Z.observe(10)
        u = Z._message_to_child()
        self.assertAllClose(u[0],
                            10)

        Z = Binomial(10, 0.9)
        Z.observe(2)
        u = Z._message_to_child()
        self.assertAllClose(u[0],
                            2)
        pass

    def test_random(self):
        """
        Test random sampling in Binomial node
        """
        N = [ [5], [50] ]
        p = [1.0, 0.0]
        with np.errstate(divide='ignore'):
            Z = Binomial(N, p, plates=(3,2,2)).random()
        self.assertArrayEqual(Z, np.ones((3,2,2))*N*p)

    def test_mixture_with_count_array(self):
        """
        Test binomial mixture with varying number of trials
        """

        p0 = 0.6
        p1 = 0.9
        p2 = 0.3
        counts = [[10], [5], [3]]
        X = Mixture(2, Binomial, counts, [p0, p1, p2])
        u = X._message_to_child()
        self.assertAllClose(
            u[0],
            np.array(counts)[:, 0] * np.array(p2)
        )

        # Multi-mixture and count array
        # Shape(p) = (2, 1, 3) + ()
        p = [
            [[0.6, 0.9, 0.8]],
            [[0.1, 0.2, 0.3]],
        ]
        # Shape(Z1) = (1, 3) + (2,) -> () + (2,)
        Z1 = 1
        # Shape(Z2) = (1,) + (3,) -> () + (3,)
        Z2 = 2
        # Shape(counts) = (5, 1)
        counts = [[10], [5], [3], [2], [4]]
        # Shape(X) = (5,) + ()
        X = Mixture(
            Z1,
            Mixture,
            Z2,
            Binomial,
            counts,
            p,
            # NOTE: We mix over axes -3 and -1. But as we first mix over the
            # default (-1), then the next mixing happens over -2 (because one
            # axis was already dropped).
            cluster_plate=-2,
        )
        self.assertAllClose(
            X._message_to_child()[0],
            np.array(counts)[:, 0] * np.array(p)[Z1, :, Z2]
        )

        # Can't have non-singleton axis in counts over the mixed axis
        p0 = 0.6
        p1 = 0.9
        p2 = 0.3
        counts = [10, 5, 3]
        self.assertRaises(
            ValueError,
            Mixture,
            2,
            Binomial,
            counts,
            [p0, p1, p2],
        )

        return
