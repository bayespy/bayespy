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

    If you want to be able to observe the variable:
       _compute_fixed_moments_and_f

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_mask_to_parent(index, mask)
       _plates_to_parent(self, index)
       _plates_from_parent(self, index)
    
    """

    def __init__(self, *args, initialize=True, **kwargs):

        super().__init__(*args,
                         dims=self.compute_dims(*args),
                         **kwargs)

        # Initialize moment array
        axes = len(self.plates)*(1,)
        self.u = [utils.nans(axes+dim) for dim in self.dims]

        # Not observed
        self.observed = False

        if initialize:
            self.initialize_from_prior()

    # TODO: Write the initialization method.

    def get_moments(self):
        return self.u

    ## @staticmethod
    ## def _compute_message_to_parent(index, u_self, *u_parents):
    ##     # Sub-classes should implement this
    ##     raise NotImplementedError()

    def _get_message_and_mask_to_parent(self, index):
        u_parents = self._message_from_parents(exclude=index)
        m = self._compute_message_to_parent(self.parents[index], index, self.u, *u_parents)
        mask = self._compute_mask_to_parent(index, self.mask)
        return (m, mask)

    ## def get_mask(self):
    ##     return self.mask
    
    ## def _get_message_mask(self):
    ##     return self.mask
    
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
            
            # Hah, this function is used to set the observations! The caller
            # should be careful what mask he uses! If you want to set only
            # latent variables, then use such a mask.
            
            # Use mask to update only unobserved plates and keep the
            # observed as before
            np.copyto(self.u[ind],
                      u[ind],
                      where=u_mask)

            # Make sure u has the correct number of dimensions:
            # TODO/FIXME: Maybe it would be good to also check that u has a
            # shape that is a sub-shape of get_shape.
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

    def observe(self, x, mask=True):
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

    def unobserve(self):
        # Update mask
        self.observed = False
        self._update_mask()

    def lowerbound(self):
        # Sub-class should implement this
        raise NotImplementedError()

    ## @staticmethod
    ## def _compute_fixed_moments_and_f(x, mask=True):
    ##     # Sub-classes should implement this
    ##     raise NotImplementedError()

    def _update_distribution_and_lowerbound(self, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()



    def save(self, group):
        """
        Save the state of the node into a HDF5 file.

        group can be the root
        """
        ## if name is None:
        ##     name = self.name
        ## subgroup = group.create_group(name)
        
        for i in range(len(self.u)):
            utils.write_to_hdf5(group, self.u[i], 'u%d' % i)
        utils.write_to_hdf5(group, self.observed, 'observed')

    def load(self, group):
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

