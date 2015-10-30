################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `concatenate` module.
"""

import warnings

import numpy as np

from bayespy.nodes import (Concatenate,
                           GaussianARD,
                           Gamma)

from bayespy.utils import random

from bayespy.utils.misc import TestCase


class TestConcatenate(TestCase):
    """
    Unit tests for Concatenate node.
    """
    

    def test_init(self):
        """
        Test the creation of Concatenate node
        """

        # One parent only
        X = GaussianARD(0, 1, plates=(3,), shape=())
        Y = Concatenate(X)
        self.assertEqual(Y.plates, (3,))
        self.assertEqual(Y.dims, ( (), () ))

        X = GaussianARD(0, 1, plates=(3,), shape=(2,4))
        Y = Concatenate(X)
        self.assertEqual(Y.plates, (3,))
        self.assertEqual(Y.dims, ( (2,4), (2,4,2,4) ))

        # Two parents
        X1 = GaussianARD(0, 1, plates=(2,), shape=())
        X2 = GaussianARD(0, 1, plates=(3,), shape=())
        Y = Concatenate(X1, X2)
        self.assertEqual(Y.plates, (5,))
        self.assertEqual(Y.dims, ( (), () ))

        # Two parents with shapes
        X1 = GaussianARD(0, 1, plates=(2,), shape=(4,6))
        X2 = GaussianARD(0, 1, plates=(3,), shape=(4,6))
        Y = Concatenate(X1, X2)
        self.assertEqual(Y.plates, (5,))
        self.assertEqual(Y.dims, ( (4,6), (4,6,4,6) ))

        # Two parents with non-default axis
        X1 = GaussianARD(0, 1, plates=(2,4), shape=())
        X2 = GaussianARD(0, 1, plates=(3,4), shape=())
        Y = Concatenate(X1, X2, axis=-2)
        self.assertEqual(Y.plates, (5,4))
        self.assertEqual(Y.dims, ( (), () ))

        # Three parents
        X1 = GaussianARD(0, 1, plates=(2,), shape=())
        X2 = GaussianARD(0, 1, plates=(3,), shape=())
        X3 = GaussianARD(0, 1, plates=(4,), shape=())
        Y = Concatenate(X1, X2, X3)
        self.assertEqual(Y.plates, (9,))
        self.assertEqual(Y.dims, ( (), () ))

        # Constant parent
        X1 = [7.2, 3.5]
        X2 = GaussianARD(0, 1, plates=(3,), shape=())
        Y = Concatenate(X1, X2)
        self.assertEqual(Y.plates, (5,))
        self.assertEqual(Y.dims, ( (), () ))

        # Different moments
        X1 = GaussianARD(0, 1, plates=(3,))
        X2 = Gamma(1, 1, plates=(4,))
        self.assertRaises(ValueError,
                          Concatenate,
                          X1,
                          X2)

        # Incompatible shapes
        X1 = GaussianARD(0, 1, plates=(3,), shape=(2,))
        X2 = GaussianARD(0, 1, plates=(2,), shape=())
        self.assertRaises(ValueError,
                          Concatenate,
                          X1,
                          X2)
        
        # Incompatible plates
        X1 = GaussianARD(0, 1, plates=(4,3), shape=())
        X2 = GaussianARD(0, 1, plates=(5,2,), shape=())
        self.assertRaises(ValueError,
                          Concatenate,
                          X1,
                          X2)
        
        pass
        

    def test_message_to_child(self):
        """
        Test the message to child of Concatenate node.
        """

        # Two parents without shapes
        X1 = GaussianARD(0, 1, plates=(2,), shape=())
        X2 = GaussianARD(0, 1, plates=(3,), shape=())
        Y = Concatenate(X1, X2)
        u1 = X1.get_moments()
        u2 = X2.get_moments()
        u = Y.get_moments()
        self.assertAllClose((u[0]*np.ones((5,)))[:2],
                            u1[0]*np.ones((2,)))
        self.assertAllClose((u[1]*np.ones((5,)))[:2],
                            u1[1]*np.ones((2,)))
        self.assertAllClose((u[0]*np.ones((5,)))[2:],
                            u2[0]*np.ones((3,)))
        self.assertAllClose((u[1]*np.ones((5,)))[2:],
                            u2[1]*np.ones((3,)))

        # Two parents with shapes
        X1 = GaussianARD(0, 1, plates=(2,), shape=(4,))
        X2 = GaussianARD(0, 1, plates=(3,), shape=(4,))
        Y = Concatenate(X1, X2)
        u1 = X1.get_moments()
        u2 = X2.get_moments()
        u = Y.get_moments()
        self.assertAllClose((u[0]*np.ones((5,4)))[:2],
                            u1[0]*np.ones((2,4)))
        self.assertAllClose((u[1]*np.ones((5,4,4)))[:2],
                            u1[1]*np.ones((2,4,4)))
        self.assertAllClose((u[0]*np.ones((5,4)))[2:],
                            u2[0]*np.ones((3,4)))
        self.assertAllClose((u[1]*np.ones((5,4,4)))[2:],
                            u2[1]*np.ones((3,4,4)))

        # Test with non-constant axis
        X1 = GaussianARD(0, 1, plates=(2,4), shape=())
        X2 = GaussianARD(0, 1, plates=(3,4), shape=())
        Y = Concatenate(X1, X2, axis=-2)
        u1 = X1.get_moments()
        u2 = X2.get_moments()
        u = Y.get_moments()
        self.assertAllClose((u[0]*np.ones((5,4)))[:2],
                            u1[0]*np.ones((2,4)))
        self.assertAllClose((u[1]*np.ones((5,4)))[:2],
                            u1[1]*np.ones((2,4)))
        self.assertAllClose((u[0]*np.ones((5,4)))[2:],
                            u2[0]*np.ones((3,4)))
        self.assertAllClose((u[1]*np.ones((5,4)))[2:],
                            u2[1]*np.ones((3,4)))

        # Test with constant parent
        X1 = np.random.randn(2, 4)
        X2 = GaussianARD(0, 1, plates=(3,), shape=(4,))
        Y = Concatenate(X1, X2)
        u1 = Y.parents[0].get_moments()
        u2 = X2.get_moments()
        u = Y.get_moments()
        self.assertAllClose((u[0]*np.ones((5,4)))[:2],
                            u1[0]*np.ones((2,4)))
        self.assertAllClose((u[1]*np.ones((5,4,4)))[:2],
                            u1[1]*np.ones((2,4,4)))
        self.assertAllClose((u[0]*np.ones((5,4)))[2:],
                            u2[0]*np.ones((3,4)))
        self.assertAllClose((u[1]*np.ones((5,4,4)))[2:],
                            u2[1]*np.ones((3,4,4)))


        pass


    def test_message_to_parent(self):
        """
        Test the message to parents of Concatenate node.
        """

        # Two parents without shapes
        X1 = GaussianARD(0, 1, plates=(2,), shape=())
        X2 = GaussianARD(0, 1, plates=(3,), shape=())
        Z = Concatenate(X1, X2)
        Y = GaussianARD(Z, 1)
        Y.observe(np.random.randn(*Y.get_shape(0)))
        m1 = X1._message_from_children()
        m2 = X2._message_from_children()
        m = Z._message_from_children()
        self.assertAllClose((m[0]*np.ones((5,)))[:2],
                            m1[0]*np.ones((2,)))
        self.assertAllClose((m[1]*np.ones((5,)))[:2],
                            m1[1]*np.ones((2,)))
        self.assertAllClose((m[0]*np.ones((5,)))[2:],
                            m2[0]*np.ones((3,)))
        self.assertAllClose((m[1]*np.ones((5,)))[2:],
                            m2[1]*np.ones((3,)))

        # Two parents with shapes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)

            X1 = GaussianARD(0, 1, plates=(2,), shape=(4,6))
            X2 = GaussianARD(0, 1, plates=(3,), shape=(4,6))
            Z = Concatenate(X1, X2)
            Y = GaussianARD(Z, 1)
            Y.observe(np.random.randn(*Y.get_shape(0)))
            m1 = X1._message_from_children()
            m2 = X2._message_from_children()
            m = Z._message_from_children()
            self.assertAllClose((m[0]*np.ones((5,4,6)))[:2],
                                m1[0]*np.ones((2,4,6)))
            self.assertAllClose((m[1]*np.ones((5,4,6,4,6)))[:2],
                                m1[1]*np.ones((2,4,6,4,6)))
            self.assertAllClose((m[0]*np.ones((5,4,6)))[2:],
                                m2[0]*np.ones((3,4,6)))
            self.assertAllClose((m[1]*np.ones((5,4,6,4,6)))[2:],
                                m2[1]*np.ones((3,4,6,4,6)))

            # Two parents with non-default concatenation axis
            X1 = GaussianARD(0, 1, plates=(2,4), shape=())
            X2 = GaussianARD(0, 1, plates=(3,4), shape=())
            Z = Concatenate(X1, X2, axis=-2)
            Y = GaussianARD(Z, 1)
            Y.observe(np.random.randn(*Y.get_shape(0)))
            m1 = X1._message_from_children()
            m2 = X2._message_from_children()
            m = Z._message_from_children()
            self.assertAllClose((m[0]*np.ones((5,4)))[:2],
                                m1[0]*np.ones((2,4)))
            self.assertAllClose((m[1]*np.ones((5,4)))[:2],
                                m1[1]*np.ones((2,4)))
            self.assertAllClose((m[0]*np.ones((5,4)))[2:],
                                m2[0]*np.ones((3,4)))
            self.assertAllClose((m[1]*np.ones((5,4)))[2:],
                                m2[1]*np.ones((3,4)))

            # Constant parent
            X1 = np.random.randn(2,4,6)
            X2 = GaussianARD(0, 1, plates=(3,), shape=(4,6))
            Z = Concatenate(X1, X2)
            Y = GaussianARD(Z, 1)
            Y.observe(np.random.randn(*Y.get_shape(0)))
            m1 = Z._message_to_parent(0)
            m2 = X2._message_from_children()
            m = Z._message_from_children()
            self.assertAllClose((m[0]*np.ones((5,4,6)))[:2],
                                m1[0]*np.ones((2,4,6)))
            self.assertAllClose((m[1]*np.ones((5,4,6,4,6)))[:2],
                                m1[1]*np.ones((2,4,6,4,6)))
            self.assertAllClose((m[0]*np.ones((5,4,6)))[2:],
                                m2[0]*np.ones((3,4,6)))
            self.assertAllClose((m[1]*np.ones((5,4,6,4,6)))[2:],
                                m2[1]*np.ones((3,4,6,4,6)))

        pass


    def test_mask_to_parent(self):
        """
        Test the mask handling in Concatenate node
        """

        pass
