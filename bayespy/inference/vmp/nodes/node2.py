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
This module contains a sketch of a new implementation of the framework.
"""

class Node():
    """
    Base class for all nodes.

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
        parent_plates = [self.plates_from_parent(index) 
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
            

        # Children
        self.children = list()

    @staticmethod
    def _compute_dims_from_parents(*parents):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    @staticmethod
    def _compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

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

    def _set_mask(self, mask):
        # Sub-classes may want to overwrite this
        pass
    
    def _update_mask(self):
        # Combine masks from children
        mask = False
        for (child, index) in self.children:
            mask = np.logical_or(mask, child._mask_to_parent(index))
        # Set the mask of this node
        self._set_mask(mask)
        # Tell parents to update their masks
        for parent in self.parents:
            parent._update_mask()

    def _mask_to_parent(self, index):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _message_to_child(self):
        return self.get_moments()
    
    def _message_to_parent(self, index):
        # Compute the message, check plates, apply mask and sum over some plates
        if index >= len(self.parents):
            raise ValueError("Parent index larger than the number of parents")

        # Compute the message
        m = self._get_message_to_parent(index)
        my_mask = self._get_mask()

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
                dim_parent = len(self.parents[index].dims[i])
                if dim_parent > 0:
                    plates_m = shape_m[:-dim_parent]
                else:
                    plates_m = shape_m

                # Compute the multiplier (multiply by the number of
                # plates for which both the message and parent have
                # single plates)
                plates_self = self.plates_to_parent(index)
                try:
                    ## print('node.msg', self.name, plates_self, 
                    ##                           plates_m,
                    ##                           parent.plates)
                    r = self.plate_multiplier(plates_self, 
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

    def _message_from_parents(self, ignore=None):
        return [parent._message_to_child() 
                if ind != ignore else
                None
                for (ind,parent) in enumerate(self.parents)]

    def get_moments(self):
        raise NotImplementedError()


class Stochastic(Node):
    """
    Base class for nodes that are stochastic.

    u
    mask
    observed
    """

    def __init__(self, *args, initialize=True, **kwargs):

        super().__init__(*args,
                         dims=self.compute_dims(*args),
                         **kwargs)

        axes = len(self.plates)*(1,)
        self.u = [utils.nans(axes+dim) for dim in self.dims]

        # Terms for the lower bound (G for latent and F for observed)
        #self.g = 0
        #self.f = 0

        # Not observed
        self.observed = False

        # By default, ignore all plates
        self.mask = False

        if initialize:
            self.initialize_from_prior()

    def get_moments(self):
        return self.u

    def _get_message_to_parent(self, index):
        u_parents = self._message_from_parents(ignore=index)
        return self._compute_message_to_parent(index, self.u, *u_parents)

    @staticmethod
    def _compute_mask_to_parent(index, mask):
        # Sub-classes may want to overwrite this if they do something to plates.
        return mask
    
    def _mask_to_parent(self, index):
        return self._compute_mask_to_parent(index, self.mask)

    def _set_mask(self, mask):
        self.mask = np.logical_or(mask, self.observed)
    
    def _set_moments(self, u, mask=True):
        # Store the computed moments u but do not change moments for
        # observations, i.e., utilize the mask.
        for ind in range(len(u)):
            # Add axes to the mask for the variable dimensions (mask
            # contains only axes for the plates).
            u_mask = utils.add_trailing_axes(mask, self.ndims[ind])

            # Enlarge self.u[ind] as necessary so that it can store the
            # broadcasted result.
            sh = utils.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
            self.u[ind] = utils.repeat_to_shape(self.u[ind], sh)

            # TODO/FIXME/BUG: The mask of observations is not used, observations
            # may be overwritten!!! ???
            
            # Use mask to update only unobserved plates and keep the
            # observed as before
            np.copyto(self.u[ind],
                      u[ind],
                      where=u_mask)

            # Make sure u has the correct number of dimensions:
            shape = self.get_shape(ind)
            ndim = len(shape)
            ndim_u = np.ndim(self.u[ind])
            if ndim > ndim_u:
                self.u[ind] = utils.add_leading_axes(u[ind], ndim - ndim_u)
            elif ndim < ndim_u:
                raise Exception("The size of the variable's moments array "
                                "is larger than it should be based on the "
                                "plates and dimension information. Check "
                                "that you have provided plates properly.")

    def update(self):
        if not np.all(self.observed):
            u_parents = self._message_from_parents()
            m_children = self._message_from_children()
            self._update_distribution_and_lowerbound(m_children, *u_parents)

    def observe(y, mask=True):
        """
        Fix moments, compute f and propagate mask.
        """

        # Compute fixed moments
        (u, f) = self._compute_fixed_moments_and_f(x, mask=mask)

        # Check the dimensionality of the observations
        for (i,v) in enumerate(u):
            # This is what the dimensionality "should" be
            s = self.plates + self.dims[i]
            t = np.shape(v)
            if s != t:
                msg = "Dimensionality of the observations incorrect."
                msg += "\nShape of input: " + str(t)
                msg += "\nExpected shape: " + str(s)
                msg += "\nCheck plates."
                raise Exception(msg)

        # Set the moments
        self._set_moments(u, mask=mask)
        
        # TODO/FIXME: Use the mask?
        self.f = f

        # Observed nodes should not be ignored
        self.observed = mask
        self._update_mask()

    def lowerbound():
        # Sub-class should implement this
        raise NotImplementedError()

    @staticmethod
    def _compute_fixed_moments_and_f(x, mask=True):
        # Sub-classes should implement this
        raise NotImplementedError()

    @staticmethod
    def _compute_message_to_parent(index, u_self, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _update_distribution_and_lowerbound(self, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

class Deterministic(Node):
    """
    Base class for nodes that are deterministic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _mask_to_parent(self, index):
        # Sub-classes should implement this
        mask = False
        for (child, index) in self.children:
            mask = np.logical_or(mask, child._mask_to_parent(index))
        return self._compute_mask_to_parent(index, mask)

    def get_moments(self):
        u_parents = [parent._message_to_child() for parent in self.parents]
        return self._compute_moments(*u_parents)

    def _get_message_to_parent(self, index):
        u_parents = self._message_from_parents(ignore=index)
        m_children = self._message_from_children()
        return self._compute_message_to_parent(index, m_children, *u_parents)
        
    
    def _compute_moments(*u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

    @staticmethod
    def _compute_message_to_parent(index, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()
