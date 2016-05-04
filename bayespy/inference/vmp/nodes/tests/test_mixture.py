################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for mixture module.
"""

import warnings

import numpy as np

from bayespy.nodes import (GaussianARD,
                           Gamma,
                           Mixture,
                           Categorical,
                           Bernoulli,
                           Multinomial,
                           Beta,
                           Gate,
                           Dirichlet)

from bayespy.utils import random
from bayespy.utils import linalg

from bayespy.utils.misc import TestCase


class TestMixture(TestCase):

    def test_init(self):
        """
        Test the creation of Mixture node
        """

        # Do not accept non-negative cluster plates
        z = Categorical(np.random.dirichlet([1,1]))
        self.assertRaises(ValueError,
                          Mixture,
                          z,
                          GaussianARD, 
                          GaussianARD(0, 1, plates=(2,)),
                          Gamma(1, 1, plates=(2,)),
                          cluster_plate=0)
        
        # Try constructing a mixture without any of the parents having the
        # cluster plate axis
        z = Categorical(np.random.dirichlet([1,1]))
        self.assertRaises(ValueError,
                          Mixture,
                          z,
                          GaussianARD, 
                          GaussianARD(0, 1, plates=()),
                          Gamma(1, 1, plates=()))
        

    def test_message_to_child(self):
        """
        Test the message to child of Mixture node.
        """

        K = 3

        #
        # Estimate moments from parents only
        #

        # Simple case
        mu = GaussianARD([0,2,4], 1,
                         ndim=0,
                         plates=(K,))
        alpha = Gamma(1, 1,
                      plates=(K,))
        z = Categorical(np.ones(K)/K)
        X = Mixture(z, GaussianARD, mu, alpha)
        self.assertEqual(X.plates, ())
        self.assertEqual(X.dims, ( (), () ))
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            2)
        self.assertAllClose(u[1],
                            2**2+1)

        # Broadcasting the moments on the cluster axis
        mu = GaussianARD(2, 1,
                         ndim=0,
                         plates=(K,))
        alpha = Gamma(1, 1,
                      plates=(K,))
        z = Categorical(np.ones(K)/K)
        X = Mixture(z, GaussianARD, mu, alpha)
        self.assertEqual(X.plates, ())
        self.assertEqual(X.dims, ( (), () ))
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            2)
        self.assertAllClose(u[1],
                            2**2+1)

        #
        # Estimate moments with observed children
        #
        
        pass

    def test_message_to_parent(self):
        """
        Test the message to parents of Mixture node.
        """

        K = 3

        # Broadcasting the moments on the cluster axis
        Mu = GaussianARD(2, 1,
                         ndim=0,
                         plates=(K,))
        (mu, mumu) = Mu._message_to_child()
        Alpha = Gamma(3, 1,
                      plates=(K,))
        (alpha, logalpha) = Alpha._message_to_child()
        z = Categorical(np.ones(K)/K)
        X = Mixture(z, GaussianARD, Mu, Alpha)
        tau = 4
        Y = GaussianARD(X, tau)
        y = 5
        Y.observe(y)
        (x, xx) = X._message_to_child()
        m = z._message_from_children()
        self.assertAllClose(m[0] * np.ones(K),
                            random.gaussian_logpdf(xx*alpha,
                                                   x*alpha*mu,
                                                   mumu*alpha,
                                                   logalpha,
                                                   0)
                            * np.ones(K))
        m = Mu._message_from_children()
        self.assertAllClose(m[0],
                            1/K * (alpha*x) * np.ones(3))
        self.assertAllClose(m[1],
                            -0.5 * 1/K * alpha * np.ones(3))

        # Some parameters do not have cluster plate axis
        Mu = GaussianARD(2, 1,
                         ndim=0,
                         plates=(K,))
        (mu, mumu) = Mu._message_to_child()
        Alpha = Gamma(3, 1) # Note: no cluster plate axis!
        (alpha, logalpha) = Alpha._message_to_child()
        z = Categorical(np.ones(K)/K)
        X = Mixture(z, GaussianARD, Mu, Alpha)
        tau = 4
        Y = GaussianARD(X, tau)
        y = 5
        Y.observe(y)
        (x, xx) = X._message_to_child()
        m = z._message_from_children()
        self.assertAllClose(m[0] * np.ones(K),
                            random.gaussian_logpdf(xx*alpha,
                                                   x*alpha*mu,
                                                   mumu*alpha,
                                                   logalpha,
                                                   0)
                            * np.ones(K))
                                                   
        m = Mu._message_from_children()
        self.assertAllClose(m[0],
                            1/K * (alpha*x) * np.ones(3))
        self.assertAllClose(m[1],
                            -0.5 * 1/K * alpha * np.ones(3))

        # Cluster assignments do not have as many plate axes as parameters.
        M = 2
        Mu = GaussianARD(2, 1,
                         ndim=0,
                         plates=(K,M))
        (mu, mumu) = Mu._message_to_child()
        Alpha = Gamma(3, 1,
                      plates=(K,M))
        (alpha, logalpha) = Alpha._message_to_child()
        z = Categorical(np.ones(K)/K)
        X = Mixture(z, GaussianARD, Mu, Alpha, cluster_plate=-2)
        tau = 4
        Y = GaussianARD(X, tau)
        y = 5 * np.ones(M)
        Y.observe(y)
        (x, xx) = X._message_to_child()
        m = z._message_from_children()
        self.assertAllClose(m[0]*np.ones(K),
                            np.sum(random.gaussian_logpdf(xx*alpha,
                                                          x*alpha*mu,
                                                          mumu*alpha,
                                                          logalpha,
                                                          0) *
                                   np.ones((K,M)),
                                   axis=-1))
                                                   
        m = Mu._message_from_children()
        self.assertAllClose(m[0] * np.ones((K,M)),
                            1/K * (alpha*x) * np.ones((K,M)))
        self.assertAllClose(m[1] * np.ones((K,M)),
                            -0.5 * 1/K * alpha * np.ones((K,M)))
        

        # Mixed distribution broadcasts g
        # This tests for a found bug. The bug caused an error.
        Z = Categorical([0.3, 0.5, 0.2])
        X = Mixture(Z, Categorical, [[0.2,0.8], [0.1,0.9], [0.3,0.7]])
        m = Z._message_from_children()

        #
        # Test nested mixtures
        #
        t1 = [1, 1, 0, 3, 3]
        t2 = [2]
        p = Dirichlet([1, 1], plates=(4, 3))
        X = Mixture(t1, Mixture, t2, Categorical, p)
        X.observe([1, 1, 0, 0, 0])
        p.update()
        self.assertAllClose(
            p.phi[0],
            [
                [[1, 1], [1, 1], [2, 1]],
                [[1, 1], [1, 1], [1, 3]],
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [3, 1]],
            ]
        )

        # Test sample plates in nested mixtures
        t1 = Categorical([0.3, 0.7], plates=(5,))
        t2 = [[1], [1], [0], [3], [3]]
        t3 = 2
        p = Dirichlet([1, 1], plates=(2, 4, 3))
        X = Mixture(t1, Mixture, t2, Mixture, t3, Categorical, p)
        X.observe([1, 1, 0, 0, 0])
        p.update()
        self.assertAllClose(
            p.phi[0],
            [
                [
                    [[1, 1], [1, 1], [1.3, 1]],
                    [[1, 1], [1, 1], [1, 1.6]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [1, 1], [1.6, 1]],
                ],
                [
                    [[1, 1], [1, 1], [1.7, 1]],
                    [[1, 1], [1, 1], [1, 2.4]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1], [1, 1], [2.4, 1]],
                ]
            ]
        )

        # Check that Gate and nested Mixture are equal
        t1 = Categorical([0.3, 0.7], plates=(5,))
        t2 = Categorical([0.1, 0.3, 0.6], plates=(5, 1))
        p = Dirichlet([1, 2, 3, 4], plates=(2, 3))
        X = Mixture(t1, Mixture, t2, Categorical, p)
        X.observe([3, 3, 1, 2, 2])
        t1_msg = t1._message_from_children()
        t2_msg = t2._message_from_children()
        p_msg = p._message_from_children()
        t1 = Categorical([0.3, 0.7], plates=(5,))
        t2 = Categorical([0.1, 0.3, 0.6], plates=(5, 1))
        p = Dirichlet([1, 2, 3, 4], plates=(2, 3))
        X = Categorical(Gate(t1, Gate(t2, p)))
        X.observe([3, 3, 1, 2, 2])
        t1_msg2 = t1._message_from_children()
        t2_msg2 = t2._message_from_children()
        p_msg2 = p._message_from_children()
        self.assertAllClose(t1_msg[0], t1_msg2[0])
        self.assertAllClose(t2_msg[0], t2_msg2[0])
        self.assertAllClose(p_msg[0], p_msg2[0])

        pass


    def test_lowerbound(self):
        """
        Test log likelihood lower bound for Mixture node
        """

        # Mixed distribution broadcasts g
        # This tests for a found bug. The bug caused an error.
        Z = Categorical([0.3, 0.5, 0.2])
        X = Mixture(Z, Categorical, [[0.2,0.8], [0.1,0.9], [0.3,0.7]])
        X.lower_bound_contribution()

        pass

    def test_mask_to_parent(self):
        """
        Test the mask handling in Mixture node
        """

        K = 3
        Z = Categorical(np.ones(K)/K,
                        plates=(4,5,1))
        Mu = GaussianARD(0, 1,
                         shape=(2,),
                         plates=(4,K,5))
        Alpha = Gamma(1, 1,
                      plates=(4,K,5,2))
        X = Mixture(Z, GaussianARD, Mu, Alpha, cluster_plate=-3)
        Y = GaussianARD(X, 1, ndim=1)
        mask = np.reshape((np.mod(np.arange(4*5), 2) == 0),
                          (4,5))
        Y.observe(np.ones((4,5,2)),
                  mask=mask)
        self.assertArrayEqual(Z.mask,
                              mask[:,:,None])
        self.assertArrayEqual(Mu.mask,
                              mask[:,None,:])
        self.assertArrayEqual(Alpha.mask,
                              mask[:,None,:,None])

        pass


    def test_nans(self):
        """
        Test multinomial mixture
        """

        # The probabilities p1 cause problems
        p0 = [0.1, 0.9]
        p1 = [1.0-1e-50, 1e-50]
        Z = Categorical([1-1e-10, 1e-10])
        X = Mixture(Z, Multinomial, 10, [p0, p1])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [1, 9])

        p0 = [0.1, 0.9]
        p1 = [1.0-1e-10, 1e-10]
        Z = Categorical([1-1e-50, 1e-50])
        X = Mixture(Z, Multinomial, 10, [p0, p1])
        u = X._message_to_child()
        self.assertAllClose(u[0],
                            [1, 9])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            p0 = [0.1, 0.9]
            p1 = [1.0, 0.0]
            X = Mixture(0, Multinomial, 10, [p0, p1])
            u = X._message_to_child()
            self.assertAllClose(u[0], [1, 9])

        pass


    def test_random(self):
        """
        Test random sampling of mixture node
        """

        o = 1e-20
        X = Mixture([1, 0, 2], Categorical, [ [o, o, o, 1],
                                              [o, o, 1, o],
                                              [1, o, o, o] ])
        x = X.random()
        self.assertAllClose(x, [2, 3, 0])

        pass

