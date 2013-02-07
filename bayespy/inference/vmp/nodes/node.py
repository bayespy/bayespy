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


"""
This module contains a sketch of a new implementation of the framework.
"""

class Node():
    """
    Base class for all nodes.

    mask
    dims
    plates
    parents
    children
    name
    """
    
    def __init__(self, *parents, dims=None, plates=None, name=""):

        if dims is None:
            raise Exception("You need to specify the dimensionality of the "
                            "distribution for class %s"
                            % str(self.__class__))

        self.dims = dims
        self.name = name

        # Parents
        self.parents = parents
        # Inform parent nodes
        for (index,parent) in enumerate(self.parents):
            if parent:
                parent._add_child(self, index)

        # Check plates
        parent_plates = [self._plates_from_parent(index) 
                         for index in range(len(self.parents))]
        if plates is None:
            # By default, use the minimum number of plates determined
            # from the parent nodes
            try:
                self.plates = utils.broadcasted_shape(*parent_plates)
            except ValueError:
                raise ValueError("The plates of the parents do not broadcast.")
        else:
            # Use custom plates
            self.plates = plates
            # TODO/FIXME: Check that these plates are consistent with parents.
            # This is not a good test yet.. You need to check that the
            # parent_plates are a subset of plates.
            try:
                plates_broadcasted = utils.broadcasted_shape(plates, *parent_plates)
            except ValueError:
                raise ValueError("The given plates and the plates of the "
                                 "parents do not broadcast.")
            

        # By default, ignore all plates
        self.mask = False

        # Children
        self.children = list()

    @staticmethod
    def _compute_dims_from_parents(*parents):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError("Not implemented for %s" % self.__class__)

    @staticmethod
    def _compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    def _plates_to_parent(self, index):
        # Sub-classes may want to overwrite this if they manipulate plates
        return self.plates

    def _plates_from_parent(self, index):
        # Sub-classes may want to overwrite this if they manipulate plates
        return self.parents[index].plates

    def get_shape(self, ind):
        return self.plates + self.dims[ind]

    def _add_child(self, child, index):
        """
        Add a child node.

        Parameters
        ----------
        child : node
        index : int
           The parent index of this node for the child node.  
           The child node recognizes its parents by their index 
           number.
        """
        self.children.append((child, index))

    def get_mask(self):
        return self.mask
    
    ## def _get_message_mask(self):
    ##     return self.mask
    
    def _set_mask(self, mask):
        # Sub-classes may overwrite this method if they have some other masks to
        # be combined (for instance, observation mask)
        self.mask = mask
    
    def _update_mask(self):
        # Combine masks from children
        mask = False
        for (child, index) in self.children:
            mask = np.logical_or(mask, child._mask_to_parent(index))
        # Set the mask of this node
        self._set_mask(mask)
        if not utils.is_shape_subset(np.shape(self.mask), self.plates):

            raise ValueError("The mask of the node %s has updated "
                             "incorrectly. The plates in the mask %s are not a "
                             "subset of the plates of the node %s."
                             % (self.name,
                                np.shape(self.mask),
                                self.plates))
        
        # Tell parents to update their masks
        for parent in self.parents:
            parent._update_mask()

    ## @staticmethod
    ## def _compute_mask_to_parent(index, mask):
    ##     # Sub-classes may want to overwrite this if they do something to plates.
    ##     return mask

    @staticmethod
    def _compute_mask_to_parent(index, mask):
        # Sub-classes may want to overwrite this method if they do something to
        # plates
        return mask

    def _mask_to_parent(self, index):
        mask = self._compute_mask_to_parent(index, self.mask)

        # Check the shape of the mask
        plates_to_parent = self._plates_to_parent(index)
        if not utils.is_shape_subset(np.shape(mask), plates_to_parent):
            print(self._plates_to_parent(index))
            raise ValueError("In node %s, the mask being sent to "
                             "parent[%d] (%s) has invalid shape: The shape of "
                             "the mask %s is not a sub-shape of the plates of "
                             "the node with respect to the parent %s. It could "
                             "be that this node (%s) is manipulating plates "
                             "but has not overwritten the method "
                             "_compute_mask_to_parent."
                             % (self.name,
                                index,
                                self.parents[index].name,
                                np.shape(mask),
                                plates_to_parent,
                                self.__class__.__name__))

        ## if not utils.is_shape_subset(parent_plates, np.shape(mask)):
        ##     print(self._plates_to_parent(index))
        ##     raise ValueError("The mask of the node %s being sent to parent[%d] "
        ##                      "(%s) has invalid shape: The plates of the "
        ##                      "parent %s is not a subset of the shape of the "
        ##                      "mask %s. It could be that this node (%s) is "
        ##                      "manipulating plates but has not overwritten the "
        ##                      "method _compute_mask_to_parent."
        ##                      % (self.name,
        ##                         index,
        ##                         self.parents[index].name,
        ##                         parent_plates,
        ##                         np.shape(mask),
        ##                         self.__class__.__name__))

        # "Sum" (i.e., logical or) over the plates that have unit length in 
        # the parent node.
        parent_plates = self.parents[index].plates
        s = utils.axes_to_collapse(np.shape(mask), parent_plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = utils.squeeze_to_dim(mask, len(parent_plates))
        return mask
    #return self._compute_mask_to_parent(index, self.get_mask())

    def _message_to_child(self):
        return self.get_moments()
    
    def _message_to_parent(self, index):
        # Compute the message, check plates, apply mask and sum over some plates
        if index >= len(self.parents):
            raise ValueError("Parent index larger than the number of parents")

        # Compute the message
        # TODO/FIXME: If several deterministic nodes as a chain, get_mask will
        # be expensive..
        m = self._get_message_to_parent(index)
        #my_mask = self.mask #self._get_message_mask()
        my_mask = self._compute_mask_to_parent(index, self.mask)

        # The parent we're sending the message to
        parent = self.parents[index]

        # Compact the message to a proper shape
        for i in range(len(m)):

            # Empty messages are given as None. We can ignore those.
            if m[i] is not None:

                # Ignorations (add extra axes to broadcast properly).
                # This sends zero messages to parent from such
                # variables we are ignoring in this node. This is
                # useful for handling missing data.

                # Apply the mask to the message
                # Sum the dimensions of the message matrix to match
                # the dimensionality of the parents natural
                # parameterization (as there may be less plates for
                # parents)

                # Extend the plate mask to include unit axes for variable
                # dimensions.
                shape_mask = np.shape(my_mask) + (1,) * len(parent.dims[i])
                my_mask2 = np.reshape(my_mask, shape_mask)

                # Apply the mask to the message
                m[i] = np.where(my_mask2, m[i], 0)

                shape_m = np.shape(m[i])
                # If some dimensions of the message matrix were
                # singleton although the corresponding dimension
                # of the parents natural parameterization is not,
                # multiply the message (i.e., broadcasting)

                # Plates in the message
                dim_parent = len(parent.dims[i])
                if dim_parent > 0:
                    plates_m = shape_m[:-dim_parent]
                else:
                    plates_m = shape_m

                # Compute the multiplier (multiply by the number of
                # plates for which both the message and parent have
                # single plates)
                plates_self = self._plates_to_parent(index)
                try:
                    ## print('node.msg', self.name, plates_self, 
                    ##                           plates_m,
                    ##                           parent.plates)
                    r = self._plate_multiplier(plates_self, 
                                               plates_m,
                                               parent.plates)
                except ValueError:
                    raise ValueError("The plates of the message and "
                                     "parent[%d] node (%s) are not a "
                                     "broadcastable subset of the plates "
                                     "of this node (%s).  The message has "
                                     "shape %s, meaning plates %s. This "
                                     "node has plates %s with respect to "
                                     "the parent[%d], which has plates %s."
                                     % (index,
                                        parent.name,
                                        self.name,
                                        np.shape(m[i]), 
                                        plates_m, 
                                        plates_self, 
                                        index,
                                        parent.plates))

                shape_parent = parent.get_shape(i)

                s = utils.axes_to_collapse(shape_m, shape_parent)
                m[i] = np.sum(m[i], axis=s, keepdims=True)

                m[i] = utils.squeeze_to_dim(m[i], len(shape_parent))
                m[i] *= r

        return m

    def _message_from_children(self):
        msg = [np.array(0.0) for i in range(len(self.dims))]
        for (child,index) in self.children:
            m = child._message_to_parent(index)
            for i in range(len(self.dims)):
                if m[i] is not None:
                    # Check broadcasting shapes
                    sh = utils.broadcasted_shape(self.get_shape(i), np.shape(m[i]))
                    try:
                        # Try exploiting broadcasting rules
                        msg[i] += m[i]
                    except ValueError:
                        msg[i] = msg[i] + m[i]

        return msg

    def _message_from_parents(self, exclude=None):
        return [parent._message_to_child() 
                if ind != exclude else
                None
                for (ind,parent) in enumerate(self.parents)]

    def get_moments(self):
        raise NotImplementedError()


    @staticmethod
    def _plate_multiplier(plates, *args):
        # Check broadcasting of the shapes
        for arg in args:
            utils.broadcasted_shape(plates, arg)

        # Check that each arg-plates are a subset of plates?
        for arg in args:
            if not utils.is_shape_subset(arg, plates):
                raise ValueError("The shapes in args are not a sub-shape of "
                                 "plates.")
            
        r = 1
        for j in range(-len(plates),0):
            mult = True
            for arg in args:
                if not (-j > len(arg) or arg[j] == 1):
                    mult = False
            if mult:
                r *= plates[j]
        return r

