################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for bayespy.inference.vmp.nodes.categorical_markov_chain module.
"""

import warnings

import numpy as np
from bayespy.utils import misc

from bayespy.inference.vmp.nodes import CategoricalMarkovChain, \
                                        Dirichlet


class TestCategoricalMarkovChain(misc.TestCase):

    def test_init(self):
        """
        Test the creation of CategoricalMarkovChain object
        """

        #
        # Plates
        #
        
        p0 = np.random.dirichlet([1, 1])
        P = np.random.dirichlet([1, 1], size=(3,2))
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")
        
        p0 = np.random.dirichlet([1, 1], size=(4,))
        P = np.random.dirichlet([1, 1], size=(3,2))
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((4,), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")
    
        p0 = np.random.dirichlet([1, 1])
        P = np.random.dirichlet([1, 1], size=(4,3,2))
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((4,), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")

        # Test some overflow bugs
        p0 = np.array([0.5, 0.5])
        P = Dirichlet(1e-3*np.ones(2),
                      plates=(2,))
        Z = CategoricalMarkovChain(p0, P,
                                   states=2000)
        u = Z._message_to_child()
        self.assertTrue(np.all(~np.isnan(u[0])), 
                        msg="Nans in moments")
        self.assertTrue(np.all(~np.isnan(u[1])), 
                        msg="Nans in moments")
        
        pass
    
    def test_message_to_child(self):
        """
        Test the message of CategoricalMarkovChain to child
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Deterministic oscillator
            p0 = np.array([1.0, 0.0])
            P = np.array(3*[[[0.0, 1.0],
                             [1.0, 0.0]]])
            Z = CategoricalMarkovChain(p0, P)
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [1.0, 0])
            self.assertAllClose(u[1],
                                [ [[0.0, 1.0],
                                   [0.0, 0.0]],
                                  [[0.0, 0.0],
                                   [1.0, 0.0]],
                                  [[0.0, 1.0],
                                   [0.0, 0.0]] ])

            # Maximum randomness
            p0 = np.array([0.5, 0.5])
            P = np.array(3*[[[0.5, 0.5],
                             [0.5, 0.5]]])
            Z = CategoricalMarkovChain(p0, P)
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [0.5, 0.5])
            self.assertAllClose(u[1],
                                [ [[0.25, 0.25],
                                   [0.25, 0.25]],
                                  [[0.25, 0.25],
                                   [0.25, 0.25]],
                                  [[0.25, 0.25],
                                   [0.25, 0.25]] ])

            # Random init, deterministic dynamics
            p0 = np.array([0.5, 0.5])
            P = np.array(3*[[[0, 1],
                             [1, 0]]])
            Z = CategoricalMarkovChain(p0, P)
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [0.5, 0.5])
            self.assertAllClose(u[1],
                                [ [[0.0, 0.5],
                                   [0.5, 0.0]],
                                  [[0.0, 0.5],
                                   [0.5, 0.0]],
                                  [[0.0, 0.5],
                                   [0.5, 0.0]] ])

            # Test plates
            p0 = np.array([ [1.0, 0.0],
                            [0.5, 0.5] ])
            P = np.array([ [ [[0.0, 1.0],
                              [1.0, 0.0]] ],
                           [ [[0.5, 0.5],
                              [0.5, 0.5]] ] ])
            Z = CategoricalMarkovChain(p0, P)
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [[1.0, 0.0],
                                 [0.5, 0.5]])
            self.assertAllClose(u[1],
                                [ [ [[0.0, 1.0],
                                     [0.0, 0.0]] ],
                                  [ [[0.25, 0.25],
                                     [0.25, 0.25]] ] ])

            # Test broadcasted state transition probabilities
            p0 = np.array([1.0, 0.0])
            P = Dirichlet([1e-10, 1e10],
                          plates=(3,2))
            Z = CategoricalMarkovChain(p0, P)
            u = Z._message_to_child()
            self.assertAllClose(u[0],
                                [1.0, 0])
            self.assertAllClose(u[1],
                                [ [[0.0, 1.0],
                                   [0.0, 0.0]],
                                  [[0.0, 0.0],
                                   [0.0, 1.0]],
                                  [[0.0, 0.0],
                                   [0.0, 1.0]] ])


        pass


    def test_random(self):
        """
        Test random sampling of categorical Markov chain
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Simple random sample
            Z = CategoricalMarkovChain([1, 0], [[0, 1],
                                                [1, 0]],
                                       states=3)
            z = Z.random()
            self.assertAllClose(z, [0, 1, 0])

            # Draw sample with plates
            p0 = [ [1,0], [0,1] ]
            P = [ [ [[0,1],
                     [1,0]] ],
                  [ [[1,0],
                     [0,1]] ] ]
            Z = CategoricalMarkovChain(p0, P,
                                       states=3)
            z = Z.random()
            self.assertAllClose(z, [[0, 1, 0], [1, 1, 1]])

            # Draw sample with plates, parameters broadcasted
            Z = CategoricalMarkovChain([1, 0], [[0, 1],
                                                [1, 0]],
                                       states=3,
                                       plates=(3,4))
            z = Z.random()
            self.assertAllClose(z, np.ones((3,4,1))*[0, 1, 0])

            # Draw sample with time-varying transition matrix
            p0 = [1, 0]
            P = [ [[0,1],
                   [1,0]],
                  [[1,0],
                   [0,1]],
                  [[1, 0],
                   [1, 0]] ]
            Z = CategoricalMarkovChain(p0, P,
                                       states=4)
            z = Z.random()
            self.assertAllClose(z, [0, 1, 1, 0])
        
        pass
