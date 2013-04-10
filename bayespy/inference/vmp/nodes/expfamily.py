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

from .stochastic import Stochastic

class ExponentialFamily(Stochastic):
    """
    A base class for nodes using natural parameterization `phi`.

    phi

    Sub-classes must implement the following static methods:
       _compute_message_to_parent(index, u_self, *u_parents)
       _compute_phi_from_parents(*u_parents, mask)
       _compute_moments_and_cgf(phi, mask)

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_mask_to_parent(index, mask)
       _plates_to_parent(self, index)
       _plates_from_parent(self, index)
    
    """

    def __init__(self, *args, initialize=True, **kwargs):

        super().__init__(*args,
                         initialize=initialize,
                         **kwargs)

        if not initialize:
            axes = len(self.plates)*(1,)
            self.phi = [utils.nans(axes+dim) for dim in self.dims]

        # Terms for the lower bound (G for latent and F for observed)
        self.g = np.array(0)
        self.f = np.array(0)


    def initialize_from_prior(self):
        if not np.all(self.observed):

            # Messages from parents
            #u_parents = [parent.message_to_child() for parent in self.parents]
            u_parents = self._message_from_parents()

            # Update natural parameters using parents
            self._update_phi_from_parents(*u_parents)

            # Update moments
            mask = np.logical_not(self.observed)
            (u, g) = self._compute_moments_and_cgf(self.phi, mask=mask)
            # TODO/FIXME/BUG: You should use observation mask in order to not
            # overwrite them!
            self._set_moments_and_cgf(u, g, mask=mask)


    def initialize_from_parameters(self, *args):
        # Get the moments of the parameters if they were fixed to the
        # given values.
        #u_parents = self.compute_fixed_parameter_moments(*args)
        u_parents = list()
        for (ind, x) in enumerate(args):
            u = self.parameter_distributions[ind].compute_fixed_moments(x)
            u_parents.append(u)
        # Update natural parameters
        self._update_phi_from_parents(*u_parents)
        # Update moments
        # TODO/FIXME: Use the mask of observations!
        mask = np.logical_not(self.observed)
        (u, g) = self._compute_moments_and_cgf(self.phi, mask=mask)
        self._set_moments_and_cgf(u, g, mask=mask)

    def initialize_from_value(self, x):
        # Update moments from value
        if np.shape(x) != self.plates + self.get_shape_of_value():
            raise ValueError("Invalid shape of the value for initialization.")
        mask = np.logical_not(self.observed)
        (u, f) = self._compute_fixed_moments_and_f(x, mask=mask)
        self._set_moments_and_cgf(u, np.inf, mask=mask)

    def initialize_from_random(self):
        self.initialize_from_prior()
        X = self.random()
        self.initialize_from_value(X)

    def _update_phi_from_parents(self, *u_parents):

        # TODO/FIXME: Could this be combined to the function
        # _update_distribution_and_lowerbound ?
        # No, because some initialization methods may want to use this.

        # This makes correct broadcasting
        self.phi = self._compute_phi_from_parents(*u_parents)
        self.phi = list(self.phi)
        # Make sure phi has the correct number of axes. It makes life
        # a bit easier elsewhere.
        for i in range(len(self.phi)):
            axes = len(self.plates) + self.ndims[i] - np.ndim(self.phi[i])
            if axes > 0:
                # Add axes
                self.phi[i] = utils.add_leading_axes(self.phi[i], axes)
            elif axes < 0:
                # Remove extra leading axes
                first = -(len(self.plates)+self.ndims[i])
                sh = np.shape(self.phi[i])[first:]
                self.phi[i] = np.reshape(self.phi[i], sh)

    def _set_moments_and_cgf(self, u, g, mask=True):
        self._set_moments(u, mask=mask)
        # TODO/FIXME: Apply mask to g too!!
        self.g = g

    def _update_distribution_and_lowerbound(self, m_children, *u_parents):

        # Update phi first from parents..
        self._update_phi_from_parents(*u_parents)
        # .. then just add children's message
        for i in range(len(self.phi)):
            self.phi[i] = self.phi[i] + m_children[i]

        # Update u and g
        self._update_moments_and_cgf()

    def _update_moments_and_cgf(self):
        """
        Update moments and cgf based on current phi.
        """
        # Mask for plates to update (i.e., unobserved plates)
        update_mask = np.logical_not(self.observed)

        # Compute the moments (u) and CGF (g)...
        (u, g) = self._compute_moments_and_cgf(self.phi, mask=update_mask)
        # ... and store them
        self._set_moments_and_cgf(u, g, mask=update_mask)
            
    def lower_bound_contribution(self, gradient=False):
        # Compute E[ log p(X|parents) - log q(X) ] over q(X)q(parents)
        
        # Messages from parents
        #u_parents = [parent.message_to_child() for parent in self.parents]
        u_parents = self._message_from_parents()
        phi = self._compute_phi_from_parents(*u_parents)
        # G from parents
        L = self._compute_cgf_from_parents(*u_parents)
        # L = g
        # G for unobserved variables (ignored variables are handled
        # properly automatically)
        latent_mask = np.logical_not(self.observed)
        #latent_mask = np.logical_and(self.mask, np.logical_not(self.observed))
        # F for observed, G for latent
        L = L + np.where(self.observed, self.f, -self.g)
        for (phi_p, phi_q, u_q, dims) in zip(phi, self.phi, self.u, self.dims):
            # Form a mask which puts observed variables to zero and
            # broadcasts properly
            latent_mask_i = utils.add_axes(latent_mask,
                                   len(self.plates) - np.ndim(latent_mask),
                                   len(dims))
            axis_sum = tuple(range(-len(dims),0))

            # Compute the term
            phi_q = np.where(latent_mask_i, phi_q, 0)
            # TODO/FIXME: Use einsum here?
            Z = np.sum((phi_p-phi_q) * u_q, axis=axis_sum)

            L = L + Z

        return (np.sum(np.where(self.mask, L, 0))
                * self._plate_multiplier(self.plates,
                                         np.shape(L),
                                         np.shape(self.mask)))
        #return L

    @classmethod
    def _compute_logpdf(cls, u, phi, g, f):
        """ Compute E[log p(X)] given E[u], E[phi], E[g] and
        E[f]. Does not sum over plates."""

        # TODO/FIXME: Should I take into account what is latent or
        # observed, or what is even totally ignored (by the mask).
        L = g + f
        for (phi_i, u_i, ndims_i) in zip(phi, u, cls.ndims):
        #for (phi_i, u_i, len_dims_i) in zip(phi, u, len_dims):
            # Axes to sum (dimensions of the variable, not the plates)
            axis_sum = tuple(range(-ndims_i,0))
            #axis_sum = tuple(range(-len_dims_i,0))
            # Compute the term
            # TODO/FIXME: Use einsum!
            L = L + np.sum(phi_i * u_i, axis=axis_sum)
        return L

    def save(self, group):
        """
        Save the state of the node into a HDF5 file.

        group can be the root
        """
        ## if name is None:
        ##     name = self.name
        ## subgroup = group.create_group(name)
        
        for i in range(len(self.phi)):
            utils.write_to_hdf5(group, self.phi[i], 'phi%d' % i)
        utils.write_to_hdf5(group, self.f, 'f')
        utils.write_to_hdf5(group, self.g, 'g')
        super().save(group)
    
    def load(self, group):
        """
        Load the state of the node from a HDF5 file.
        """
        # TODO/FIXME: Check that the shapes are correct!
        for i in range(len(self.phi)):
            phii = group['phi%d' % i][...]
            self.phi[i] = phii
            
        self.f = group['f'][...]
        self.g = group['g'][...]
        super().load(group)

        

