################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `gate` module.
"""

import numpy as np

from bayespy.nodes import (Gate,
                           GaussianARD,
                           Gamma,
                           Categorical,
                           Bernoulli,
                           Multinomial)

from bayespy.inference.vmp.nodes.gaussian import GaussianMoments

from bayespy.utils import random
from bayespy.utils import linalg

from bayespy.utils.misc import TestCase


class TestGate(TestCase):
    """
    Unit tests for Gate node.
    """
    

    def test_init(self):
        """
        Test the creation of Gate node
        """

        # Gating scalar node
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, shape=(), plates=(3,))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, ())
        self.assertEqual(Y.dims, ( (), () ))

        # Gating non-scalar node
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, shape=(2,), plates=(3,))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, ())
        self.assertEqual(Y.dims, ( (2,), (2,2) ))

        # Plates from Z
        Z = Categorical(np.ones(3)/3, plates=(4,))
        X = GaussianARD(0, 1, shape=(2,), plates=(3,))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, (4,))
        self.assertEqual(Y.dims, ( (2,), (2,2) ))

        # Plates from X
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, shape=(2,), plates=(4,3))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, (4,))
        self.assertEqual(Y.dims, ( (2,), (2,2) ))

        # Plates from Z and X
        Z = Categorical(np.ones(3)/3, plates=(5,))
        X = GaussianARD(0, 1, shape=(2,), plates=(4,1,3))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, (4,5))
        self.assertEqual(Y.dims, ( (2,), (2,2) ))

        # Gating non-default plate
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, shape=(), plates=(3,4))
        Y = Gate(Z, X, gated_plate=-2)
        self.assertEqual(Y.plates, (4,))
        self.assertEqual(Y.dims, ( (), () ))

        # Fixed gating
        Z = 2
        X = GaussianARD(0, 1, shape=(2,), plates=(3,))
        Y = Gate(Z, X)
        self.assertEqual(Y.plates, ())
        self.assertEqual(Y.dims, ( (2,), (2,2) ))

        # Fixed X
        Z = Categorical(np.ones(3)/3)
        X = [1, 2, 3]
        Y = Gate(Z, X, moments=GaussianMoments(()))
        self.assertEqual(Y.plates, ())
        self.assertEqual(Y.dims, ( (), () ))

        # Do not accept non-negative cluster plates
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, plates=(3,))
        self.assertRaises(ValueError,
                          Gate,
                          Z,
                          X,
                          gated_plate=0)
        
        # None of the parents have the cluster plate axis
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1)
        self.assertRaises(ValueError,
                          Gate,
                          Z,
                          X)

        # Inconsistent cluster plate
        Z = Categorical(np.ones(3)/3)
        X = GaussianARD(0, 1, plates=(2,))
        self.assertRaises(ValueError,
                          Gate,
                          Z,
                          X)

        pass
        

    def test_message_to_child(self):
        """
        Test the message to child of Gate node.
        """

        # Gating scalar node
        Z = 2
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertEqual(len(u), 2)
        self.assertAllClose(u[0], 3)
        self.assertAllClose(u[1], 3**2+1)

        # Fixed X
        Z = 2
        X = [1, 2, 3]
        Y = Gate(Z, X, moments=GaussianMoments(()))
        u = Y._message_to_child()
        self.assertEqual(len(u), 2)
        self.assertAllClose(u[0], 3)
        self.assertAllClose(u[1], 3**2)

        # Uncertain gating
        Z = Categorical([0.2,0.3,0.5])
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertAllClose(u[0], 0.2*1 + 0.3*2 + 0.5*3)
        self.assertAllClose(u[1], 0.2*2 + 0.3*5 + 0.5*10)

        # Plates in Z
        Z = [2, 0]
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertAllClose(u[0], [3, 1])
        self.assertAllClose(u[1], [10, 2])

        # Plates in X
        Z = 2
        X = GaussianARD([1,2,3], 1, shape=(), plates=(4,3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertAllClose(np.ones(4)*u[0], np.ones(4)*3)
        self.assertAllClose(np.ones(4)*u[1], np.ones(4)*10)

        # Gating non-default plate
        Z = 2
        X = GaussianARD([[1],[2],[3]], 1, shape=(), plates=(3,4))
        Y = Gate(Z, X, gated_plate=-2)
        u = Y._message_to_child()
        self.assertAllClose(np.ones(4)*u[0], np.ones(4)*3)
        self.assertAllClose(np.ones(4)*u[1], np.ones(4)*10)

        # Gating non-scalar node
        Z = 2
        X = GaussianARD([1*np.ones(4),
                         2*np.ones(4),
                         3*np.ones(4)],
                        1,
                        shape=(4,), plates=(3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertAllClose(u[0], 3*np.ones(4))
        self.assertAllClose(u[1], 9*np.ones((4,4)) + 1*np.identity(4))
        
        # Broadcasting the moments on the cluster axis
        Z = 2
        X = GaussianARD(1, 1, shape=(), plates=(3,))
        Y = Gate(Z, X)
        u = Y._message_to_child()
        self.assertEqual(len(u), 2)
        self.assertAllClose(u[0], 1)
        self.assertAllClose(u[1], 1**2+1)

        pass


    def test_message_to_parent(self):
        """
        Test the message to parents of Gate node.
        """

        # Unobserved and broadcasting
        Z = 2
        X = GaussianARD(0, 1, shape=(), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        m = F._message_to_parent(0)
        self.assertEqual(len(m), 1)
        self.assertAllClose(m[0], 0*np.ones(3))
        m = F._message_to_parent(1)
        self.assertEqual(len(m), 2)
        self.assertAllClose(m[0]*np.ones(3), [0, 0, 0])
        self.assertAllClose(m[1]*np.ones(3), [0, 0, 0])
        
        # Gating scalar node
        Z = 2
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe(10)
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*2, 10*2-0.5*5, 10*3-0.5*10])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [0, 0, 10])
        self.assertAllClose(m[1], [0, 0, -0.5])
        
        # Fixed X
        Z = 2
        X = [1,2,3]
        F = Gate(Z, X, moments=GaussianMoments(()))
        Y = GaussianARD(F, 1)
        Y.observe(10)
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*1, 10*2-0.5*4, 10*3-0.5*9])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [0, 0, 10])
        self.assertAllClose(m[1], [0, 0, -0.5])

        # Uncertain gating
        Z = Categorical([0.2, 0.3, 0.5])
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe(10)
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*2, 10*2-0.5*5, 10*3-0.5*10])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [0.2*10, 0.3*10, 0.5*10])
        self.assertAllClose(m[1], [-0.5*0.2, -0.5*0.3, -0.5*0.5])

        # Plates in Z
        Z = [2, 0]
        X = GaussianARD([1,2,3], 1, shape=(), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe([10, 20])
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [[10*1-0.5*2, 10*2-0.5*5, 10*3-0.5*10],
                                   [20*1-0.5*2, 20*2-0.5*5, 20*3-0.5*10]])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [20, 0, 10])
        self.assertAllClose(m[1], [-0.5, 0, -0.5])

        # Plates in X
        Z = 2
        X = GaussianARD([[1,2,3], [4,5,6]], 1, shape=(), plates=(2,3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe([10, 20])
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*2 + 20*4-0.5*17,
                                   10*2-0.5*5 + 20*5-0.5*26,
                                   10*3-0.5*10 + 20*6-0.5*37])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [[0, 0, 10],
                                   [0, 0, 20]])
        self.assertAllClose(m[1]*np.ones((2,3)), [[0, 0, -0.5],
                                                  [0, 0, -0.5]])

        # Gating non-default plate
        Z = 2
        X = GaussianARD([[1],[2],[3]], 1, shape=(), plates=(3,1))
        F = Gate(Z, X, gated_plate=-2)
        Y = GaussianARD(F, 1)
        Y.observe([10])
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*2, 10*2-0.5*5, 10*3-0.5*10])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [[0], [0], [10]])
        self.assertAllClose(m[1], [[0], [0], [-0.5]])

        # Gating non-scalar node
        Z = 2
        X = GaussianARD([[1,4],[2,5],[3,6]], 1, shape=(2,), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe([10,20])
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*1-0.5*2 + 20*4-0.5*17,
                                   10*2-0.5*5 + 20*5-0.5*26,
                                   10*3-0.5*10 + 20*6-0.5*37])
        m = F._message_to_parent(1)
        I = np.identity(2)
        self.assertAllClose(m[0], [[0,0], [0,0], [10,20]])
        self.assertAllClose(m[1], [0*I, 0*I, -0.5*I])
        
        # Broadcasting the moments on the cluster axis
        Z = 2
        X = GaussianARD(2, 1, shape=(), plates=(3,))
        F = Gate(Z, X)
        Y = GaussianARD(F, 1)
        Y.observe(10)
        m = F._message_to_parent(0)
        self.assertAllClose(m[0], [10*2-0.5*5, 10*2-0.5*5, 10*2-0.5*5])
        m = F._message_to_parent(1)
        self.assertAllClose(m[0], [0, 0, 10])
        self.assertAllClose(m[1], [0, 0, -0.5])

        pass


    def test_mask_to_parent(self):
        """
        Test the mask handling in Gate node
        """

        pass
