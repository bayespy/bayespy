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
Unit tests for bayespy.inference.vmp.nodes.categorical_markov_chain module.
"""

import numpy as np
from bayespy.utils import utils

from bayespy.inference.vmp.nodes import CategoricalMarkovChain, \
                                        Dirichlet

class TestCategoricalMarkovChain(utils.TestCase):

    def test_init(self):
        """
        Test the creation of CategoricalMarkovChain object
        """

        #
        # Plates
        #
        
        p0 = np.random.rand(2)
        P = np.random.rand(3, 2, 2)
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")
        
        p0 = np.random.rand(4, 2)
        P = np.random.rand(3, 2, 2)
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((4,), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")
    
        p0 = np.random.rand(2)
        P = np.random.rand(4, 3, 2, 2)
        Z = CategoricalMarkovChain(p0, P)
        self.assertEqual((4,), Z.plates, msg="Incorrect plates")
        self.assertEqual(((2,),(3,2,2)), Z.dims, msg="Incorrect dimensions")

        pass
    
    def test_message_to_child(self):
        """
        Test the message of CategoricalMarkovChain to child
        """

        np.seterr(divide='ignore')

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

        # Unnormalized probabilities
        p0 = np.array([2, 2])
        P = np.array([ [[4, 4],
                        [4, 4]],
                       [[8, 8],
                        [8, 8]],
                       [[20, 20],
                        [20, 20]] ])
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
