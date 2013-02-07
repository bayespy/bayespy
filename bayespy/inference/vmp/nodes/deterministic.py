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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, plates=None, **kwargs)

    ## def _get_message_mask(self):
    ##     # No need to perform any masking because the messages from children have
    ##     # already been masked.
    ##     return True
    
    ## def get_mask(self):
    ##     # Combine the masks from children
    ##     mask = False
    ##     for (child, index) in self.children:
    ##         mask = np.logical_or(mask, child._mask_to_parent(index))
    ##     return mask

    def get_moments(self):
        u_parents = [parent._message_to_child() for parent in self.parents]
        return self._compute_moments(*u_parents)

    def _get_message_to_parent(self, index):
        u_parents = self._message_from_parents(exclude=index)
        m_children = self._message_from_children()
        return self._compute_message_to_parent(index, m_children, *u_parents)
        
    
    def _compute_moments(self, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _compute_message_to_parent(self, index, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()
