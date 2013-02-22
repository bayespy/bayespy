######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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
Unit tests for `dot` module.
"""

import unittest


import numpy as np
import scipy

from numpy import testing

from ..node import Node

from ...vmp import VB

from bayespy import utils

class TestNode(unittest.TestCase):

    def check_message_to_parent(self, plates_child, plates_message,
                                plates_mask, plates_parent, dims=(2,)):

        # Dummy message
        msg = np.random.randn(*(plates_message+dims))
        # Mask with every other True and every other False
        mask = np.mod(np.arange(np.prod(plates_mask)).reshape(plates_mask),
                      2) == 0

        # Set up the dummy model
        class Dummy(Node):
            def _get_message_and_mask_to_parent(self, index):
                return ([msg], mask)
        parent = Dummy(dims=[dims], plates=plates_parent)
        child = Dummy(parent, dims=[dims], plates=plates_child)

        m = child._message_to_parent(0)[0] * np.ones(plates_parent+dims)

        # Brute-force computation of the message without too much checking
        m_true = msg * mask[...,np.newaxis] * np.ones(plates_child+dims)
        for ind in range(len(plates_child)):
            axis = -ind - 2
            if ind >= len(plates_parent):
                m_true = np.sum(m_true, axis=axis, keepdims=False)
            elif plates_parent[-ind-1] == 1:
                m_true = np.sum(m_true, axis=axis, keepdims=True)

        testing.assert_allclose(m, m_true,
                                err_msg="Incorrect message.")

    def test_message_to_parent(self):
        """
        Test plate handling in _message_to_parent.
        """

        # Test empty plates with scalar messages
        self.check_message_to_parent((),
                                     (),
                                     (),
                                     (),
                                     dims=())
        # Test singular plates
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,3,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,3,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,1,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,3,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,1,4),
                                     (2,3,4))
        self.check_message_to_parent((2,3,4),
                                     (2,3,4),
                                     (2,1,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,3,4),
                                     (2,1,4))
        self.check_message_to_parent((2,3,4),
                                     (2,1,4),
                                     (2,1,4),
                                     (2,1,4))
        # Test missing plates
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (4,3),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (4,3),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (  3,),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (4,3),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (  3,),
                                     (4,3))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (4,3),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (4,3),
                                     (  3,),
                                     (  3,))
        self.check_message_to_parent((4,3),
                                     (  3,),
                                     (  3,),
                                     (  3,))
        # A complex test
        self.check_message_to_parent((7,6,5,4,3),
                                     (  6,1,4,3),
                                     (1,1,5,4,1),
                                     (  6,5,1,3))
        # Test errors for inconsistent shapes
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (1,3,),
                          (3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (3,),
                          (1,3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (1,3,),
                          (1,3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (3,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (3,),
                          (4,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (4,),
                          (3,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (3,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (3,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (3,),
                          (4,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (4,),
                          (3,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (3,),
                          (4,),
                          (1,))
        self.assertRaises(ValueError, 
                          self.check_message_to_parent,
                          (1,),
                          (4,),
                          (4,),
                          (1,))


        

