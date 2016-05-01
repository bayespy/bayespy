################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

from bayespy.utils import misc

import h5py

from .node import Node

class Distribution():
    """
    A base class for the VMP formulas of variables.

    Sub-classes implement distribution specific computations.

    If a sub-class maps the plates differently, it needs to overload the
    following methods:

        * compute_weights_to_parent

        * plates_to_parent

        * plates_from_parent
    """


    def compute_message_to_parent(self, parent, index, u_self, *u_parents):
        """
        Compute the message to a parent node.
        """
        raise NotImplementedError()


    def compute_weights_to_parent(self, index, weights):
        """
        Maps the mask to the plates of a parent.
        """
        # Sub-classes may need to overwrite this method
        return weights


    def plates_to_parent(self, index, plates):
        """
        Resolves the plate mapping to a parent.

        Given the plates of the node's moments, this method returns the plates
        that the message to a parent has for the parent's distribution.
        """
        return plates

    def plates_from_parent(self, index, plates):
        """
        Resolve the plate mapping from a parent.

        Given the plates of a parent's moments, this method returns the plates
        that the moments has for this distribution.
        """
        return plates


    def random(self, *params, plates=None):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()

        
class Stochastic(Node):
    """
    Base class for nodes that are stochastic.

    u
    observed

    Sub-classes must implement:
       _compute_message_to_parent(parent, index, u_self, *u_parents)
       _update_distribution_and_lowerbound(self, m, *u)
       lowerbound(self)
       _compute_dims
       initialize_from_prior()
    

    If you want to be able to observe the variable:
       _compute_fixed_moments_and_f

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_weights_to_parent(index, weights)
       _compute_plates_to_parent(self, index, plates)
       _compute_plates_from_parent(self, index, plates)
    
    """

    # Sub-classes must over-write this
    _distribution = None

    def __init__(self, *args, initialize=True, dims=None, **kwargs):

        self._id = Node._id_counter
        Node._id_counter += 1

        super().__init__(*args,
                         dims=dims,
                         **kwargs)

        # Initialize moment array
        axes = len(self.plates)*(1,)
        self.u = [misc.nans(axes+dim) for dim in dims]

        # Not observed
        self.observed = False

        self.ndims = [len(dim) for dim in self.dims]

        if initialize:
            self.initialize_from_prior()


    def _get_id_list(self):
        """
        Returns the stochastic ID list.

        This method is used to check that same stochastic nodes are not direct
        parents of a node several times. It is only valid if there are
        intermediate stochastic nodes.

        To put it another way: each ID corresponds to one factor q(..) in the
        posterior approximation. Different IDs mean different factors, thus they
        mean independence. The parents must have independent factors.

        Stochastic nodes should return their unique ID. Deterministic nodes
        should return the IDs of their parents. Constant nodes should return
        empty list of IDs.
        """
        return [self._id]

    
    def _compute_plates_to_parent(self, index, plates):
        return self._distribution.plates_to_parent(index, plates)

    def _compute_plates_from_parent(self, index, plates):
        return self._distribution.plates_from_parent(index, plates)


    def _compute_weights_to_parent(self, index, weights):
        return self._distribution.compute_weights_to_parent(index, weights)


    def get_moments(self):
        # Just for safety, do not return a reference to the moment list of this
        # node but instead create a copy of the list. 
        return [ui for ui in self.u]

    def _get_message_and_mask_to_parent(self, index, u_parent=None):
        u_parents = self._message_from_parents(exclude=index)
        u_parents[index] = u_parent
        m = self._distribution.compute_message_to_parent(self.parents[index], 
                                                         index, 
                                                         self.u, 
                                                         *u_parents)
        mask = self._distribution.compute_weights_to_parent(index, self.mask) != 0
        return (m, mask)

    def _set_mask(self, mask):
        self.mask = np.logical_or(mask, self.observed)


    def _check_shape(self, u, broadcast=True):

        if len(u) != len(self.dims):
            raise ValueError("Incorrect number of arrays")

        for (dimsi, ui) in zip(self.dims, u):
            sh_true = self.plates + dimsi
            sh = np.shape(ui)
            ndim = len(dimsi)
            errmsg = (
                "Shape of the given array not equal to the shape of the node.\n"
                "Received shape: {0}\n"
                "Expected shape: {1}\n"
                "Check plates."
                .format(sh, sh_true)
            )
            if not broadcast:
                if sh != sh_true:
                    raise ValueError(errmsg)
            else:
                if ndim == 0:
                    if not misc.is_shape_subset(sh, sh_true):
                        raise ValueError(errmsg)
                else:
                    plates_ok = misc.is_shape_subset(sh[:-ndim], self.plates)
                    dims_ok = (sh[-ndim:] == dimsi)
                    if not (plates_ok and dims_ok):
                        raise ValueError(errmsg)

        return


    def _set_moments(self, u, mask=True, broadcast=True):

        self._check_shape(u, broadcast=broadcast)

        # Store the computed moments u but do not change moments for
        # observations, i.e., utilize the mask.
        for ind in range(len(u)):
            # Add axes to the mask for the variable dimensions (mask
            # contains only axes for the plates).
            u_mask = misc.add_trailing_axes(mask, self.ndims[ind])

            # Enlarge self.u[ind] as necessary so that it can store the
            # broadcasted result.
            sh = misc.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
            self.u[ind] = misc.repeat_to_shape(self.u[ind], sh)

            # TODO/FIXME/BUG: The mask of observations is not used, observations
            # may be overwritten!!! ???
            
            # Hah, this function is used to set the observations! The caller
            # should be careful what mask he uses! If you want to set only
            # latent variables, then use such a mask.
            
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
                self.u[ind] = misc.add_leading_axes(u[ind], ndim - ndim_u)
            elif ndim < ndim_u:
                # This should not ever happen because we already checked the
                # shape at the beginning of the function.
                raise RuntimeError(
                    "This error should not happen. Fix shape checking."
                    "The size of the variable %s's %s-th moment "
                    "array is %s which is larger than it should "
                    "be, that is, %s, based on the plates %s and "
                    "dimension %s. Check that you have provided "
                    "plates properly."
                    % (self.name,
                       ind,
                       np.shape(self.u[ind]), 
                       shape,
                       self.plates,
                       self.dims[ind]))


    def update(self, annealing=1.0):
        if not np.all(self.observed):
            u_parents = self._message_from_parents()
            m_children = self._message_from_children()
            if annealing != 1.0:
                m_children = [annealing * m for m in m_children]
            self._update_distribution_and_lowerbound(m_children, *u_parents)


    def observe(self, x, mask=True):
        """
        Fix moments, compute f and propagate mask.
        """
        raise NotImplementedError()

    def unobserve(self):
        # Update mask
        self.observed = False
        self._update_mask()

    def lowerbound(self):
        # Sub-class should implement this
        raise NotImplementedError()

    def _update_distribution_and_lowerbound(self, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()


    def save(self, filename):
        # Open HDF5 file
        h5f = h5py.File(filename, 'w')
        try:
            # Write each node
            nodegroup = h5f.create_group('nodes')
            if self.name == '':
                raise ValueError("In order to save nodes, they must have "
                                 "(unique) names.")
            self._save(nodegroup.create_group(self.name))
        finally:
            # Close file
            h5f.close()


    def _save(self, group):
        """
        Save the state of the node into a HDF5 file.

        group can be the root
        """
        for i in range(len(self.u)):
            misc.write_to_hdf5(group, self.u[i], 'u%d' % i)
        misc.write_to_hdf5(group, self.observed, 'observed')
        return


    def load(self, filename):
        h5f = h5py.File(filename, 'r')
        try:
            self._load(h5f['nodes'][self.name])
        finally:
            h5f.close()
        return


    def _load(self, group):
        """
        Load the state of the node from a HDF5 file.
        """
        # TODO/FIXME: Check that the shapes are correct!
        for i in range(len(self.u)):
            ui = group['u%d' % i][...]
            self.u[i] = ui

        old_observed = self.observed
        self.observed = group['observed'][...]
        # Update masks if necessary
        if np.any(old_observed != self.observed):
            self._update_mask()


    def random(self):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()


    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        print(str(self))


    def __str__(self):
        """
        
        """
        raise NotImplementedError("String representation not yet implemented for "
                                  "node class %s" % (self.__class__.__name__))
