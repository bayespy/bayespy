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

def message_sum_multiply(plates_parent, dims_parent, *arrays):
    """
    Compute message to parent and sum over plates.

    Divide by the plate multiplier.
    """
    # The shape of the full message
    shapes = [np.shape(array) for array in arrays]
    shape_full = utils.broadcasted_shape(*shapes)
    # Find axes that should be summed
    shape_parent = plates_parent + dims_parent
    sum_axes = utils.axes_to_collapse(shape_full, shape_parent)
    # Compute the multiplier for cancelling the
    # plate-multiplier.  Because we are summing over the
    # dimensions already in this function (for efficiency), we
    # need to cancel the effect of the plate-multiplier
    # applied in the message_to_parent function.
    r = 1
    for j in sum_axes:
        if j >= 0 and j < len(plates_parent):
            r *= shape_full[j]
        elif j < 0 and j < -len(dims_parent):
            r *= shape_full[j]
    # Compute the sum-product
    m = utils.sum_multiply(*arrays,
                           axis=sum_axes,
                           sumaxis=True,
                           keepdims=True) / r
    # Remove extra axes
    m = utils.squeeze_to_dim(m, len(shape_parent))
    return m

class Node():
    """
    Base class for all nodes.

    mask
    dims
    plates
    parents
    children
    name

    Sub-classes must implement:
    1. For computing the message to children:
       get_moments(self):
    2. For computing the message to parents:
       _get_message_and_mask_to_parent(self, index)

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_mask_to_parent(index, mask)
       _plates_to_parent(self, index)
       _plates_from_parent(self, index)
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
            # Check that the parent_plates are a subset of plates.
            for p in parent_plates:
                if not utils.is_shape_subset(p, plates):
                    raise ValueError("The plates of the parents are not "
                                     "subsets of the given plates.")
                                                 

        # By default, ignore all plates
        self.mask = np.array(False)

        # Children
        self.children = list()

    ## @staticmethod
    ## def _compute_dims_from_parents(*parents):
    ##     """ Compute the dimensions of phi and u. """
    ##     raise NotImplementedError("Not implemented for %s" % self.__class__)

    ## @staticmethod
    ## def _compute_dims_from_values(x):
    ##     """ Compute the dimensions of phi and u. """
    ##     raise NotImplementedError()

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
        mask = np.array(False)
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

    # TODO: Rename to _compute_message_mask_to_parent(index, mask)
    @staticmethod
    def _compute_mask_to_parent(index, mask):
        """
        Compute the mask used for messages sent to parent[index].

        The mask tells which plates in the messages are active. This method is
        used for obtaining the mask which is used to set plates in the messages
        to parent to zero.
        
        Sub-classes may want to overwrite this method if they do something to
        plates so that the mask is somehow altered. 
        """
        return mask

    def _mask_to_parent(self, index):
        """
        Get the mask with respect to parent[index].

        The mask tells which plate connections are active. The mask is "summed"
        (logical or) and reshaped into the plate shape of the parent. Thus, it
        can't be used for masking messages, because some plates have been summed
        already. This method is used for propagating the mask to parents.
        """
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
    
    #def _get_message_and_mask_to_parent(self, index):
    #    raise NotImplementedError()

    def _message_to_parent(self, index):

        # Compute the message, check plates, apply mask and sum over some plates
        if index >= len(self.parents):
            raise ValueError("Parent index larger than the number of parents")

        # Compute the message and mask
        (m, mask) = self._get_message_and_mask_to_parent(index)

        # Plates in the mask
        plates_mask = np.shape(mask)

        # The parent we're sending the message to
        parent = self.parents[index]

        # Compact the message to a proper shape
        for i in range(len(m)):

            # Empty messages are given as None. We can ignore those.
            if m[i] is not None:

                # Plates in the message
                shape_m = np.shape(m[i])
                dim_parent = len(parent.dims[i])
                if dim_parent > 0:
                    plates_m = shape_m[:-dim_parent]
                else:
                    plates_m = shape_m

                # Compute the multiplier (multiply by the number of plates for
                # which the message, the mask and the parent have single
                # plates).  Such a plate is meant to be broadcasted but because
                # the parent has singular plate axis, it won't broadcast (and
                # sum over it), so we need to multiply it.
                plates_self = self._plates_to_parent(index)
                try:
                    r = self._plate_multiplier(plates_self, 
                                               plates_m,
                                               plates_mask,
                                               parent.plates)
                except ValueError:
                    raise ValueError("The plates of the message, the mask and "
                                     "parent[%d] node (%s) are not a "
                                     "broadcastable subset of the plates of "
                                     "this node (%s).  The message has shape "
                                     "%s, meaning plates %s. The mask has "
                                     "plates %s. This node has plates %s with "
                                     "respect to the parent[%d], which has "
                                     "plates %s."
                                     % (index,
                                        parent.name,
                                        self.name,
                                        np.shape(m[i]), 
                                        plates_m, 
                                        plates_mask,
                                        plates_self,
                                        index, 
                                        parent.plates))

                # Add variable axes to the mask
                shape_mask = np.shape(mask) + (1,) * len(parent.dims[i])
                mask_i = np.reshape(mask, shape_mask)

                # Sum over plates that are not in the message nor in the parent
                shape_parent = parent.get_shape(i)
                shape_msg = utils.broadcasted_shape(shape_m, shape_parent)
                axes_mask = utils.axes_to_collapse(shape_mask, shape_msg)
                mask_i = np.sum(mask_i, axis=axes_mask, keepdims=True)

                # Compute the masked message and sum over the plates that the
                # parent does not have.
                axes_msg = utils.axes_to_collapse(shape_msg, shape_parent)
                m[i] = utils.sum_multiply(mask_i, m[i], r, 
                                          axis=axes_msg, 
                                          keepdims=True)

                # Remove leading singular plates if the parent does not have
                # those plate axes.
                m[i] = utils.squeeze_to_dim(m[i], len(shape_parent))

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

    def move_plates(self, from_plate, to_plate):
        return _MovePlate(self, 
                          from_plate,
                          to_plate,
                          name=self.name + ".move_plates")

    def add_plate_axis(self, to_plate):
        return AddPlateAxis(to_plate)(self,
                                      name=self.name+".add_plate_axis")


from .deterministic import Deterministic

def AddPlateAxis(to_plate):
    
    if to_plate >= 0:
        raise Exception("Give negative value for axis index to_plate.")

    class _AddPlateAxis(Deterministic):

        def __init__(self, X, **kwargs):

            nonlocal to_plate

            N = len(X.plates) + 1

            # Check the parameters
            if to_plate >= 0 or to_plate < -N:
                raise ValueError("Invalid plate position to add.")

            # Use positive indexing only
            ## if to_plate < 0:
            ##     to_plate += N
            # Use negative indexing only
            if to_plate >= 0:
                to_plate -= N
                #self.to_plate = to_plate

            super().__init__(X, 
                             dims=X.dims,
                             **kwargs)

        def _plates_to_parent(self, index):
            plates = list(self.plates)
            plates.pop(to_plate)
            return tuple(plates)
        #return self.plates[:to_plate] + self.plates[(to_plate+1):]

        def _plates_from_parent(self, index):
            plates = list(self.parents[index].plates)
            plates.insert(len(plates)-to_plate+1, 1)
            return tuple(plates)
        #raise Exception("IMPLEMENT THIS")

        @staticmethod
        def _compute_mask_to_parent(index, mask):
            # Ouch, how can you compute this in a static function?

            # Maybe this does not have to be a static funtion?

            # Maybe Mixture node requires this and _plates_to/from_parent to be
            # static as well..

            # Remove the added mask plate
            #diff = len(self.plates) - np.ndim(mask)
            #mask = utils.add_leading_axes(mask, diff)
            if abs(to_plate) <= np.ndim(mask):
                sh_mask = list(np.shape(mask))
                sh_mask.pop(to_plate)
                mask = np.reshape(mask, sh_mask)
            return mask
        #raise Exception("IMPLEMENT THIS")


        def _compute_message_and_mask_to_parent(self, index, m, *u_parents):
            """
            Compute the message to a parent node.
            """

            # Get the message from children
            #(m, mask) = self.message_from_children()

            # Remove the added message plate
            for i in range(len(m)):
                # Make sure the message has all the axes
                #diff = len(self.plates) + len(self.dims[i]) - np.ndim(m[i])
                #m[i] = utils.add_leading_axes(m[i], diff)
                # Remove the axis
                if np.ndim(m[i]) >= abs(to_plate) + len(self.dims[i]):
                    axis = to_plate - len(self.dims[i])
                    #ndims = np.ndim(m[i]) - len(self.dims[i]) + 1
                    sh_m = list(np.shape(m[i]))
                    sh_m.pop(axis)
                    m[i] = np.reshape(m[i], sh_m)

            mask = self._compute_mask_to_parent(index, self.mask)

            return (m, mask)

        def _compute_moments(self, u):
            """
            Get the moments with an added plate axis.
            """

            # Get parents' moments
            #u = self.parents[0].message_to_child()

            # Move a plate axis
            u = list(u)
            for i in range(len(u)):
                # Make sure the moments have all the axes
                #diff = len(self.plates) + len(self.dims[i]) - np.ndim(u[i]) - 1
                #u[i] = utils.add_leading_axes(u[i], diff)
                
                # The location of the new axis/plate:
                axis = np.ndim(u[i]) - abs(to_plate) - len(self.dims[i]) + 1
                if axis > 0:
                    # Add one axes to the correct position
                    sh_u = list(np.shape(u[i]))
                    sh_u.insert(axis, 1)
                    u[i] = np.reshape(u[i], sh_u)

            return u

    return _AddPlateAxis
        

## class _MovePlate(Node):
##     """
##     Move a plate to a given position.

##     NOTE: This has NOT been tested yet..
##     """

##     def __init__(self, X, from_plate, to_plate, plates=None, **kwargs):

##         if plates is not None:
##             raise ValueError("Do not specify plates.")

##         plates = X.plates

##         # Check the parameters
##         if from_plate >= len(plates) or from_plate < -len(plates):
##             raise ValueError("Invalid plate to move from.")
##         if to_plate >= len(plates) or to_plate < -len(plates):
##             raise ValueError("Invalid plate to move to.")

##         # Use positive indexing only
##         if from_plate < 0:
##             from_plate = len(plates) + from_plate
##         if to_plate < 0:
##             to_plate = len(plates) + to_plate + 1

##         # Move the plate
##         plates = list(plates)
##         plates.insert(to_plate, plates.pop(from_plate))

##         super().__init__(X, 
##                          plates=tuple(plates),
##                          dims=X.dims,
##                          **kwargs)

##     def get_moments(self):
##         """
##         Get the moments with moved plates.
##         """

##         # Get parents' moments
##         u = self.parents[0].message_to_child()

##         # Move a plate axis
##         u = list(u)
##         for i in range(len(u)):
##             u[i] = utils.moveaxis(u[i], self.from_plate, self.to_plate)

##         return tuple(u)

##     def get_message(self, index, u_parents):
##         """
##         Compute the message to a parent node.
##         """

##         # Get the message from children
##         (m, mask) = self.message_from_children()

##         # Move message plates
##         for i in range(len(m)):
##             diff = len(self.plates) + len(self.dims[i]) - self.np.ndim(m[i])
##             m[i] = utils.add_leading_axes(m[i], diff)
##             m[i] = utils.moveaxis(m[i], self.to_plate, self.from_plate)

##         # Move mask plates
##         mask = utils.add_leading_axes(mask, len(self.plates) - np.ndim(mask))
##         mask = utils.moveaxis(mask, self.to_plate, self.from_plate)

##         return (m, mask)

    
class NodeConstant(Node):
    def __init__(self, u, **kwargs):
        self.u = u
        Node.__init__(self, **kwargs)

    def message_to_child(self, gradient=False):
        if gradient:
            return (self.u, [])
        else:
            return self.u


class NodeConstantScalar(NodeConstant):
    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        return ([x], 0)

    def __init__(self, a, **kwargs):
        NodeConstant.__init__(self,
                              [a],
                              plates=np.shape(a),
                              dims=[()],
                              **kwargs)

    def start_optimization(self):
        # FIXME: Set the plate sizes appropriately!!
        x0 = self.u[0]
        #self.gradient = np.zeros(np.shape(x0))
        def transform(x):
            # E.g., for positive scalars you could have exp here.
            self.gradient = np.zeros(np.shape(x0))
            self.u[0] = x
        def gradient():
            # This would need to apply the gradient of the
            # transformation to the computed gradient
            return self.gradient
            
        return (x0, transform, gradient)

    def add_to_gradient(self, d):
        self.gradient += d

    def message_to_child(self, gradient=False):
        if gradient:
            return (self.u, [ [np.ones(np.shape(self.u[0])),
                               #self.gradient] ])
                               self.add_to_gradient] ])
        else:
            return self.u


    def stop_optimization(self):
        #raise Exception("Not implemented for " + str(self.__class__))
        pass


    
