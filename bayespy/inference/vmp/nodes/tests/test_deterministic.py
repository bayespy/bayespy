################################################################################
# Copyright (C) 2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
Unit tests for `deterministic` module.
"""

import unittest

import numpy as np
import scipy

from numpy import testing

from ..node import Node, Moments
from ..deterministic import tile
from ..stochastic import Stochastic


class TestTile(unittest.TestCase):

    def check_message_to_children(self, tiles, u_parent, u_tiled,
                                  dims=None, plates=None):
        # Set up the dummy model
        class Dummy(Stochastic):
            _parent_moments = ()
            _moments = Moments()
            def __init__(self, u, dims=dims, plates=plates):
                super().__init__(dims=dims, plates=plates, initialize=False)
                self.u = u

        X = Dummy(u_parent, dims=dims, plates=plates)
        Y = tile(X, tiles)

        u_Y = Y._compute_moments(u_parent)

        for (x,y) in zip(u_Y, u_tiled):
            self.assertEqual(np.shape(x), np.shape(y),
                             msg="Incorrect shape.")
            testing.assert_allclose(x, y,
                                    err_msg="Incorrect moments.")


    def test_message_to_children(self):
        """
        Test the moments of Tile node.
        """
        # Define th check function
        check_message_to_children = self.check_message_to_children
        # Check scalar (and broadcasting)
        check_message_to_children(2,
                                  (5,),
                                  (5,), 
                                  dims=[()],
                                  plates=()),
        # Check 1-D
        check_message_to_children(2,
                                  ([1,2],),
                                  ([1,2,1,2],),
                                  dims=[()],
                                  plates=(2,))
        # Check N-D
        check_message_to_children(2,
                                  ([[1,2],
                                    [3,4],
                                    [5,6]],),
                                  ([[1,2,1,2],
                                    [3,4,3,4],
                                    [5,6,5,6]],),
                                  dims=[()],
                                  plates=(3,2))
        # Check not-last plate
        check_message_to_children([2,1],
                                  ([[1,2],
                                    [3,4]],),
                                  ([[1,2],
                                    [3,4],
                                    [1,2],
                                    [3,4]],),
                                  dims=[()],
                                  plates=(2,2))
        # Check several plates
        check_message_to_children([2,3],
                                  ([[1,2],
                                    [3,4]],),
                                  ([[1,2,1,2,1,2],
                                    [3,4,3,4,3,4],
                                    [1,2,1,2,1,2],
                                    [3,4,3,4,3,4]],),
                                  dims=[()],
                                  plates=(2,2))
        # Check non-zero dimensional variables
        check_message_to_children(2,
                                  ([[1,2],
                                    [3,4]],),
                                  ([[1,2],
                                    [3,4],
                                    [1,2],
                                    [3,4]],),
                                  dims=[(2,)],
                                  plates=(2,))
        # Check several moments
        check_message_to_children(2,
                                  ([[1,2],
                                    [3,4]],
                                   [1,2]),
                                  ([[1,2],
                                    [3,4],
                                    [1,2],
                                    [3,4]],
                                   [1,2,1,2]),
                                  dims=[(2,),()],
                                  plates=(2,))
        # Check broadcasting of tiled plate
        check_message_to_children(2,
                                  ([[1,],
                                    [2,]],),
                                  ([[1,],
                                    [2,]],),
                                  dims=[()],
                                  plates=(2,2))
        # Check broadcasting of non-tiled plate
        check_message_to_children(2,
                                  ([[1,2]],),
                                  ([[1,2,1,2]],),
                                  dims=[()],
                                  plates=(2,2))
        # Check broadcasting of leading plates that are not in parent
        check_message_to_children([2,1],
                                  ([1,2],),
                                  ([1,2],),
                                  dims=[()],
                                  plates=(2,))
        
        
    def check_message_to_parent(self, tiles, m_children, m_true,
                                dims=None, plates_parent=None,
                                plates_children=None):
        # Set up the dummy model
        class Dummy(Stochastic):
            _parent_moments = ()
            _moments = Moments()
        X = Dummy(dims=dims, plates=plates_parent, initialize=False)
        Y = tile(X, tiles)

        m = Y._compute_message_to_parent(0, m_children, None)

        for (x,y) in zip(m, m_true):
            self.assertEqual(np.shape(x), np.shape(y),
                             msg="Incorrect shape.")
            testing.assert_allclose(x, y,
                                    err_msg="Incorrect message.")


    def test_message_to_parent(self):
        """
        Test the parent message of Tile node.
        """
        # Define th check function
        check = self.check_message_to_parent
        # Check scalar
        check(2,
              ([5,5],), 
              (10,),
              dims=[()],
              plates_parent=(),
              plates_children=(2,)),
        # Check 1-D
        check(2,
              ([1,2,3,4],),
              ([4,6],),
              dims=[()],
              plates_parent=(2,),
              plates_children=(4,))
        # Check N-D
        check(2,
              ([[1,2,7,8],
                [3,4,9,0],
                [5,6,1,2]],),
              ([[8,10],
                [12,4],
                [6,8]],),
              dims=[()],
              plates_parent=(3,2),
              plates_children=(3,4))
        # Check not-last plate
        check([2,1],
              ([[1,2],
                [3,4],
                [5,6],
                [7,8]],),
              ([[6,8],
                [10,12]],),
              dims=[()],
              plates_parent=(2,2),
              plates_children=(4,2))
        # Check several plates
        check([2,3],
              ([[1,2,1,2,1,2],
                [3,4,3,4,3,4],
                [1,2,1,2,1,2],
                [3,4,3,4,3,4]],),
              ([[6,12],
                [18,24]],),
              dims=[()],
              plates_parent=(2,2),
              plates_children=(4,6))
        # Check broadcasting if message has unit axis for tiled plate
        check(2,
              ([[1,],
                [2,],
                [3,]],),
              ([[2,],
                [4,],
                [6,]],),
              dims=[()],
              plates_parent=(3,2),
              plates_children=(3,4))
        # Check broadcasting if message has unit axis for non-tiled plate
        check(2,
              ([[1,2,3,4]],),
              ([[4,6]],),
              dims=[()],
              plates_parent=(3,2),
              plates_children=(3,4))
        # Check non-zero dimensional variables
        check(2,
              ([[1,2],
                [3,4],
                [5,6],
                [7,8]],),
              ([[6,8],
                [10,12]],),
              dims=[(2,)],
              plates_parent=(2,),
              plates_children=(4,))
        # Check several moments
        check(2,
              ([[1,2],
                [3,4],
                [5,6],
                [7,8]],
               [1,2,3,4]),
              ([[6,8],
                [10,12]],
               [4,6]),
              dims=[(2,),()],
              plates_parent=(2,),
              plates_children=(4,))
        
    def check_mask_to_parent(self, tiles, mask_child, mask_true,
                             plates_parent=None,
                             plates_children=None):
        # Set up the dummy model
        class Dummy(Stochastic):
            _moments = Moments()
            _parent_moments = ()
        X = Dummy(dims=[()], plates=plates_parent, initialize=False)
        Y = tile(X, tiles)

        mask = Y._compute_weights_to_parent(0, mask_child) != 0

        self.assertEqual(np.shape(mask), np.shape(mask_true),
                         msg="Incorrect shape.")
        testing.assert_equal(mask, mask_true,
                             err_msg="Incorrect mask.")


    def test_mask_to_parent(self):
        """
        Test the mask message to parent of Tile node.
        """
        # Define th check function
        check = self.check_mask_to_parent
        # Check scalar parent
        check(2,
              [True,False], 
              True,
              plates_parent=(),
              plates_children=(2,))
        check(2,
              [False,False], 
              False,
              plates_parent=(),
              plates_children=(2,))
        # Check 1-D
        check(2,
              [True,False,False,False],
              [True,False],
              plates_parent=(2,),
              plates_children=(4,))
        # Check N-D
        check(2,
              [[True,False,True,False],
               [False,True,False,False]],
              [[True,False],
               [False,True]],
              plates_parent=(2,2),
              plates_children=(2,4))
        # Check not-last plate
        check([2,1],
              [[True,False],
               [False,True],
               [True,False],
               [False,False]],
              [[True,False],
               [False,True]],
              plates_parent=(2,2),
              plates_children=(4,2))
        # Check several plates
        check([2,3],
              [[False,False,True,False,False,False],
               [False,False,False,False,False,False],
               [False,False,False,False,False,False],
               [False,True,False,False,False,False]],
              [[True,False],
               [False,True]],
              plates_parent=(2,2),
              plates_children=(4,6))
        # Check broadcasting if message has unit axis for tiled plate
        check(2,
              [[True,],
               [False,],
               [True,]],
              [[True,],
               [False,],
               [True,]],
              plates_parent=(3,2),
              plates_children=(3,4))
        # Check broadcasting if message has unit axis for non-tiled plate
        check(2,
              [[False,False,False,True]],
              [[False,True]],
              plates_parent=(3,2),
              plates_children=(3,4))
        
        
        

