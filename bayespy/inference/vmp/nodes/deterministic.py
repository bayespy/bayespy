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

import numpy as np

from bayespy.utils import utils

from .node import Node

class Deterministic(Node):
    """
    Base class for nodes that are deterministic.

    Sub-classes must implement:
    1. For implementing the deterministic function:
       _compute_moments(self, *u)
    2. One of the following options:
       a) Simple static methods:
          _compute_message_to_parent(index, m, *u)
          _compute_mask_to_parent(index, mask)
       b) More control with:
          _compute_message_and_mask_to_parent(self, index, m, *u)

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_mask_to_parent(index, mask)
       _plates_to_parent(self, index)
       _plates_from_parent(self, index)
    
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, plates=None, **kwargs)

    def get_moments(self):
        u_parents = [parent._message_to_child() for parent in self.parents]
        return self._compute_moments(*u_parents)

    def _compute_message_and_mask_to_parent(self, index, m_children, *u_parents):
        # The following methods should be implemented by sub-classes.
        m = self._compute_message_to_parent(index, m_children, *u_parents)
        mask = self._compute_mask_to_parent(index, self.mask)
        return (m, mask)

    def _get_message_and_mask_to_parent(self, index):
        u_parents = self._message_from_parents(exclude=index)
        m_children = self._message_from_children()
        return self._compute_message_and_mask_to_parent(index,
                                                        m_children,
                                                        *u_parents)
        
    
    def _compute_moments(self, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()
