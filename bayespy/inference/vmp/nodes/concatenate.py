######################################################################
# Copyright (C) 2015 Jaakko Luttinen
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

from bayespy.utils import misc

from .deterministic import Deterministic
from .node import Moments

class Concatenate(Deterministic):
    """
    Concatenate similar nodes along a plate axis.

    Nodes must be of same type and dimensionality. Also, plates must be
    identical except for the plate axis along which the concatenation is
    performed.

    See also
    --------
    numpy.concatenate
    """


    def __init__(self, *nodes, axis=-1, **kwargs):
        if axis >= 0:
            raise ValueError("Currently, only negative axis indeces "
                             "are allowed.")
        self._axis = axis
        parent_moments = None
        for node in nodes:
            try:
                parent_moments = node._moments
            except:
                pass
            else:
                break
        # All parents must have same moments
        self._parent_moments = (parent_moments,) * len(nodes)
        self._moments = parent_moments
        # Convert nodes
        try:
            nodes = [self._ensure_moments(node, self._parent_moments[0])
                     for node in nodes]
        except Moments.NoConverterError:
            raise ValueError("Parents have different moments")
        # Dimensionality of the node
        dims = tuple([dim for dim in nodes[0].dims])
        for node in nodes:
            if node.dims != dims:
                raise ValueError("Parents have different dimensionalities")
        super().__init__(*nodes, dims=dims, **kwargs)
        # Compute start indices for each parent on the concatenated plate axis
        self._indices = np.zeros(len(nodes)+1, dtype=np.int)
        self._indices[1:] = np.cumsum([int(parent.plates[axis])
                                       for parent in self.parents])
        self._lengths = [parent.plates[axis] for parent in self.parents]
        return


    def _compute_plates_to_parent(self, index, plates):
        plates = list(plates)
        plates[self._axis] = self.parents[index].plates[self._axis]
        return tuple(plates)


    def _compute_plates_from_parent(self, index, plates):
        plates = list(plates)
        plates[self._axis] = 0
        for parent in self.parents:
            plates[self._axis] += parent.plates[self._axis]
        return tuple(plates)


    def _plates_multiplier_from_parent(self, index):
        multipliers = [parent.plates_multiplier for parent in self.parents]
        for m in multipliers:
            if np.any(np.array(m) != 1):
                raise ValueError("Concatenation node does not support plate "
                                 "multipliers.")
        return ()


    def _compute_mask_to_parent(self, index, mask):
        axis = self._axis
        indices = self._indices[index:(index+1)]
        if np.ndim(mask) >= abs(axis) and np.shape(mask)[axis] > 1:
            # Take the middle one of the returned three arrays
            return np.split(mask, indices, axis=axis)[1]
        else:
            return mask


    def _compute_message_to_parent(self, index, m, *u_parents):
        msg = []
        indices = self._indices[index:(index+2)]
        for i in range(len(m)):
            # Fix plate axis to array axis
            axis = self._axis - len(self.dims[i])
            # Find the slice from the message
            if np.ndim(m[i]) >= abs(axis) and np.shape(m[i])[axis] > 1:
                mi = np.split(m[i], indices, axis=axis)[1]
            else:
                mi = m[i]
            msg.append(mi)
        return msg


    def _compute_moments(self, *u_parents):
        # TODO/FIXME: Unfortunately, np.concatenate doesn't support
        # broadcasting but moment messages may use broadcasting.
        #
        # WORKAROUND: Broadcast the arrays explcitly to have same shape
        # except for the concatenated axis.
        u = []
        for i in range(len(self.dims)):
            # Fix plate axis to array axis
            axis = self._axis - len(self.dims[i])
            # Find broadcasted shape
            ui_parents = [u_parent[i] for u_parent in u_parents]
            shapes = [list(np.shape(uip)) for uip in ui_parents]
            for i in range(len(shapes)):
                if len(shapes[i]) >= abs(axis):
                    shapes[i][axis] = 1
            ## shapes = [np.shape(uip[:axis]) + (1,) + np.shape(uip[(axis+1)])
            ##           if np.ndim(uip) >= abs(self._axis) else
            ##           np.shape(uip)
            ##           for uip in ui_parents]
            bc_shape = misc.broadcasted_shape(*shapes)
            # Concatenated axis must be broadcasted explicitly
            bc_shapes = [misc.broadcasted_shape(bc_shape,
                                                (length,) + (1,)*(abs(axis)-1))
                         for length in self._lengths]
            # Broadcast explicitly
            ui_parents = [uip * np.ones(shape)
                          for (uip, shape) in zip(ui_parents, bc_shapes)]
            # Concatenate
            ui = np.concatenate(ui_parents, axis=axis)
            u.append(ui)

        return u
