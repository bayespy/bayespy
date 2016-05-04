################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `categorical` module.
"""

import warnings

import numpy as np
import scipy

from bayespy.nodes import (Categorical,
                           Dirichlet,
                           Mixture,
                           Gamma)

from bayespy.utils import random
from bayespy.utils import misc

from bayespy.utils.misc import TestCase


class TestCategorical(TestCase):
    """
    Unit tests for Categorical node
    """

    
    def test_init(self):
        """
        Test the creation of categorical nodes.
        """

        # Some simple initializations
        X = Categorical([0.1, 0.3, 0.6])
        X = Categorical(Dirichlet([5,4,3]))

        # Check that plates are correct
        X = Categorical([0.1, 0.3, 0.6], plates=(3,4))
        self.assertEqual(X.plates,
                         (3,4))
        X = Categorical(0.25*np.ones((2,3,4)))
        self.assertEqual(X.plates,
                         (2,3))
        X = Categorical(Dirichlet([2,1,9], plates=(3,4)))
        self.assertEqual(X.plates,
                         (3,4))
        

        # Probabilities not a vector
        self.assertRaises(ValueError,
                          Categorical,
                          0.5)

        # Invalid probability
        self.assertRaises(ValueError,
                          Categorical,
                          [-0.5, 1.5],
                          n=10)
        self.assertRaises(ValueError,
                          Categorical,
                          [0.5, 1.5],
                          n=10)

        # Inconsistent plates
        self.assertRaises(ValueError,
                          Categorical,
                          0.25*np.ones((2,4)),
                          plates=(3,),
                          n=10)

        # Explicit plates too small
        self.assertRaises(ValueError,
                          Categorical,
                          0.25*np.ones((2,4)),
                          plates=(1,),
                          n=10)

        pass

    
    def test_moments(self):
        """
        Test the moments of categorical nodes.
        """

        # Simple test
        X = Categorical([0.7,0.2,0.1])
        u = X._message_to_child()
        self.assertEqual(len(u), 1)
        self.assertAllClose(u[0],
                            [0.7,0.2,0.1])

        # Test plates in p
        p = np.random.dirichlet([1,1], size=3)
        X = Categorical(p)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            p)
        
        # Test with Dirichlet prior
        P = Dirichlet([7, 3])
        logp = P._message_to_child()[0]
        p0 = np.exp(logp[0]) / (np.exp(logp[0]) + np.exp(logp[1]))
        p1 = np.exp(logp[1]) / (np.exp(logp[0]) + np.exp(logp[1]))
        X = Categorical(P)
        u = X._message_to_child()
        p = np.array([p0, p1])
        self.assertAllClose(u[0],
                            p)

        # Test with broadcasted plates
        P = Dirichlet([7, 3], plates=(10,))
        X = Categorical(P)
        u = X._message_to_child()
        self.assertAllClose(u[0] * np.ones(X.get_shape(0)),
                            p*np.ones((10,1)))

        pass


    def test_observed(self):
        """
        Test observed categorical nodes
        """

        # Single observation
        X = Categorical([0.7,0.2,0.1])
        X.observe(2)
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [0,0,1])

        # One plate axis
        X = Categorical([0.7,0.2,0.1], plates=(2,))
        X.observe([2,1])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [[0,0,1],
                             [0,1,0]])

        # Several plate axes
        X = Categorical([0.7,0.1,0.1,0.1], plates=(2,3,))
        X.observe([[2,1,1],
                   [0,2,3]])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [ [[0,0,1,0],
                               [0,1,0,0],
                               [0,1,0,0]],
                              [[1,0,0,0],
                               [0,0,1,0],
                               [0,0,0,1]] ])

        # Check invalid observations
        X = Categorical([0.7,0.2,0.1])
        self.assertRaises(ValueError,
                          X.observe,
                          -1)
        self.assertRaises(ValueError,
                          X.observe,
                          3)
        self.assertRaises(ValueError,
                          X.observe,
                          1.5)

        pass

    
    def test_constant(self):
        """
        Test constant categorical nodes
        """

        # Basic test
        Y = Mixture(2, Gamma, [1, 2, 3], [1, 1, 1])
        u = Y._message_to_child()
        self.assertAllClose(u[0],
                            3/1)

        # Test with one plate axis
        alpha = [[1, 2, 3],
                 [4, 5, 6]]
        Y = Mixture([2, 1], Gamma, alpha, 1)
        u = Y._message_to_child()
        self.assertAllClose(u[0],
                            [3, 5])

        # Test with two plate axes
        alpha = [ [[1, 2, 3],
                   [4, 5, 6]],
                  [[7, 8, 9],
                   [10, 11, 12]] ]
        Y = Mixture([[2, 1], [0, 2]], Gamma, alpha, 1)
        u = Y._message_to_child()
        self.assertAllClose(u[0],
                            [[3, 5],
                             [7, 12]])

        pass


    def test_initialization(self):
        """
        Test initialization of categorical nodes
        """

        # Test initialization from random
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            Z = Categorical([[0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
            Z.initialize_from_random()
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [[0, 1, 0],
                                 [0, 0, 1]])

        pass


    def test_gradient(self):
        """
        Check the Euclidean gradient of the categorical node
        """

        Z = Categorical([[0.3, 0.5, 0.2], [0.1, 0.6, 0.3]])
        Y = Mixture(Z, Gamma, [2, 3, 4], [5, 6, 7])
        Y.observe([4.2, 0.2])
        def f(x):
            Z.set_parameters([np.reshape(x, Z.get_shape(0))])
            return Z.lower_bound_contribution() + Y.lower_bound_contribution()
        def df(x):
            Z.set_parameters([np.reshape(x, Z.get_shape(0))])
            g = Z.get_riemannian_gradient()
            return Z.get_gradient(g)[0]
        x0 = np.ravel(np.log([[2, 3, 7], [0.1, 3, 1]]))
        self.assertAllClose(
            misc.gradient(f, x0),
            np.ravel(df(x0))
        )

        pass
