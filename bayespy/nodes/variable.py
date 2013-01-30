######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
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


# nodes/
#   gp/
#

# nodes.random.Gaussian
# nodes.random.Wishart
# nodes.random.Mixture
# nodes.deterministic.Dot
# nodes.deterministic.Constant

# or
# nodes.variable.Gaussian
# nodes.variable.Wishart
# nodes.variable.Dot
# nodes.variable.Constant
# nodes.gp.GaussianProcess
# nodes.gp.Constant

#print(Node)
#print(Node.Node)

# MAP/ML:
# X = MAP(prior=Gaussian(mu,Cov))
# X = ML()
# or:
# X = Delta(prior=Gaussian(mu,Cov))

class Variable(Node):

    # Overwrite this
    ndims = None
    ndims_parents = None
    parameter_distributions = None
    

    @classmethod
    def compute_logpdf(cls, u, phi, g, f):
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

    @staticmethod
    def compute_fixed_parameter_moments(*args):
        """ Compute the moments of the distribution parameters for
        fixed values."""
        raise NotImplementedError()


    @staticmethod
    def compute_phi_from_parents(u_parents):
        """ Compute E[phi] over q(parents) """
        raise NotImplementedError()

    @staticmethod
    def compute_g_from_parents(u_parents):
        """ Compute E[g(phi)] over q(parents) """
        raise NotImplementedError()

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        """ Compute E[u] and g(phi) for given phi """
        raise NotImplementedError()

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        raise NotImplementedError()

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        raise NotImplementedError()

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    def __init__(self, *args, initialize=True, **kwargs):

        super().__init__(*args,
                         dims=self.compute_dims(*args),
                         **kwargs)

        # Natural parameters and moments. Note that the size of the
        # array self.u or self.phi doesn't actually need to equal
        # self.dims+self.plates, it will be broadcasted automatically
        # (although this might be extremely rare: it would mean that
        # there are identical posterior distributions over
        # plates). Anyway, for this reason (and some others), we need
        # to explicityly have the dimensionalities dims and plates? Do
        # we need dims? Yes, because you need to be able to check what
        # plates are missing.
        #self.phi = [np.array([]) for i in range(len(self.dims))]
        #self.u = [np.array([]) for i in range(len(self.dims))]
        #self.phi = [np.array(0.0) for i in range(len(self.dims))]
        #self.u = [np.array(0.0) for i in range(len(self.dims))]
        axes = len(self.plates)*(1,)
        self.phi = [utils.nans(axes+dim) for dim in self.dims]
        self.u = [utils.nans(axes+dim) for dim in self.dims]
#        self.u = [np.array(0.0) for i in range(len(self.dims))]

        # Terms for the lower bound (G for latent and F for observed)
        self.g = 0
        self.f = 0

        # Not observed
        self.observed = False

        # By default, ignore all plates
        self.mask = False

        if initialize:
            self.initialize_from_prior()

    def get_vb_term(self):
        return {self.lower_bound_contribution}

    def get_message(self, index, u_parents):
        return (self.compute_message(index, self.u, u_parents),
                self.mask)
        ## return (self.message(index, u_parents),
        ##         self.mask)

    def initialize_from_prior(self):
        if not np.all(self.observed):

            # Messages from parents
            #u_parents = [parent.message_to_child() for parent in self.parents]
            u_parents = self.moments_from_parents()

            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            # Update moments
            (u, g) = self.compute_u_and_g(self.phi, mask=True)
            self.update_u_and_g(u, g, mask=True)


    def initialize_from_parameters(self, *args):
        # Get the moments of the parameters if they were fixed to the
        # given values.
        #u_parents = self.compute_fixed_parameter_moments(*args)
        u_parents = list()
        for (ind, x) in enumerate(args):
            #print(self.parents[ind].__class__)
            u = self.parameter_distributions[ind].compute_fixed_moments(x)
            #u = self.parents[ind].compute_fixed_moments(x)
            #(u, _) = self.parents[ind].compute_fixed_u_and_f(x)
            u_parents.append(u)
        # Update natural parameters
        self.update_phi_from_parents(u_parents)
        # Update moments
        # TODO/FIXME: Use the mask of observations!
        (u, g) = self.compute_u_and_g(self.phi, mask=True)
        self.update_u_and_g(u, g, mask=True)

    # TODO: Initialization where you could give the distribution
    # values in standard parameterization!

    ## def initialize_from_value(self, x):
    ##     # Update moments from value
    ##     (u, f) = self.compute_fixed_u_and_f(x)
    ##     self.update_u_and_g(u, np.nan, mask=True)

    ## def initialize_from_random(self):
    ##     self.initialize_from_prior()
    ##     self.initialize_from_value(self.random())

    def update(self):
        if not np.all(self.observed):

            # Messages from parents
            u_parents = [parent.message_to_child() for parent in self.parents]
                
            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            #print('update, phi', self.name, self.phi)
            # Update natural parameters using children (just add the
            # messages to phi)
            for (child,index) in self.children:
                #print('ExpFam.update:', self.name)

                # TODO/FIXME: These m are nans..
                (m, mask) = child.message_to_parent(index)

                # Combine masks
                #
                # TODO: Maybe you would like to compute a new mask
                # at every update?
                self.mask = np.logical_or(self.mask,mask)
                for i in range(len(self.phi)):
                    ## try:
                    ##     # Try exploiting broadcasting rules
                    ##     #
                    ##     # TODO/FIXME: This has the problem that if phi
                    ##     # is updated from parents such that phi is a
                    ##     # view into parent's moments thus modifying
                    ##     # phi would modify parents moments.. Maybe one
                    ##     # should check that nobody else is viewing
                    ##     # phi?
                    ##     self.phi[i] += m[i]
                    ## except ValueError:
                    ##     self.phi[i] = self.phi[i] + m[i]
                    #print('update2, phi', i, self.name, self.phi, m[i])

                    # TODO/FIXME: You should take into account the
                    # mask when adding the message!!!!???
                    self.phi[i] = self.phi[i] + m[i]

            # Mask for plates to update (i.e., unobserved plates)
            update_mask = np.logical_not(self.observed)
            # Compute u and g
            (u, g) = self.compute_u_and_g(self.phi, mask=update_mask)
            # Update moments
            self.update_u_and_g(u, g, mask=update_mask)

    def phi_from_parents(self, gradient=False):
        # Messages from parents
        u_parents = [parent.message_to_child(gradient=gradient)
                     for parent in self.parents]
        # Compute and return phi
        return self.compute_phi_from_parents(u_parents,
                                             gradient=gradient)

    def update_phi_from_parents(self, u_parents):
        # This makes correct broadcasting
        self.phi = self.compute_phi_from_parents(u_parents)
        #print('update_phi, phi', self.phi)
        # Make sure phi has the correct number of axes. It makes life
        # a bit easier elsewhere.
        for i in range(len(self.phi)):
            axes = len(self.plates) + self.ndims[i] - np.ndim(self.phi[i])
            ## print('update_phi_from_parents:')
            ## print(np.shape(self.phi[i]))
            ## print(self.plates)
            ## print(self.dims[i])
            if axes > 0:
                # Add axes
                self.phi[i] = utils.add_leading_axes(self.phi[i], axes)
            elif axes < 0:
                # Remove extra leading axes
                first = -(len(self.plates)+self.ndims[i])
                sh = np.shape(self.phi[i])[first:]
                self.phi[i] = np.reshape(self.phi[i], sh)
            #print(np.shape(self.phi[i]))
                                                 
        ## for i in range(len(self.phi)):
        ##     self.phi[i].fill(0)
        ##     try:
        ##         # Try exploiting broadcasting rules
        ##         self.phi[i] += phi[i]
        ##     except ValueError:
        ##         self.phi[i] = self.phi[i] + phi[i]
    
    def get_moments(self):
        return self.u

    def get_children_vb_bound(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def start_optimization(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def stop_optimization(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def message(self, index, u_parents):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def update_u(self, u, mask=True):
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
            elif ndim < np.ndim(self.u[ind]):
                raise Exception("Weird, this shouldn't happen.. :)")


    def update_u_and_g(self, u, g, mask=True):

        self.update_u(u, mask=mask)
        # TODO/FIXME: Apply mask to g too!!
        self.g = g
        

        ## # Store the computed moments u but do not change moments for
        ## # observations, i.e., utilize the mask.
        ## for ind in range(len(u)):
        ##     # Add axes to the mask for the variable dimensions (mask
        ##     # contains only axes for the plates).
        ##     u_mask = utils.add_trailing_axes(mask, self.ndims[ind])

        ##     # Enlarge self.u[ind] as necessary so that it can store the
        ##     # broadcasted result.
        ##     sh = utils.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
        ##     self.u[ind] = utils.repeat_to_shape(self.u[ind], sh)

        ##     # Use mask to update only unobserved plates and keep the
        ##     # observed as before
        ##     np.copyto(self.u[ind],
        ##               u[ind],
        ##               where=u_mask)

        ##     # TODO/FIXME: Apply mask to g too!!
        ##     self.g = g

        ##     # Make sure u has the correct number of dimensions:
        ##     shape = self.get_shape(ind)
        ##     ndim = len(shape)
        ##     ndim_u = np.ndim(self.u[ind])
        ##     if ndim > ndim_u:
        ##         self.u[ind] = utils.add_leading_axes(u[ind], ndim - ndim_u)
        ##     elif ndim < np.ndim(self.u[ind]):
        ##         raise Exception("Weird, this shouldn't happen.. :)")

    def lower_bound_contribution(self, gradient=False):
        # Compute E[ log p(X|parents) - log q(X) ] over q(X)q(parents)
        
        # Messages from parents
        u_parents = [parent.message_to_child() for parent in self.parents]
        phi = self.compute_phi_from_parents(u_parents)
        # G from parents
        L = self.compute_g_from_parents(u_parents)
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
                * self.plate_multiplier(self.plates,
                                        np.shape(L),
                                        np.shape(self.mask)))
        #return L
            
    def fix_u_and_f(self, u, f, mask=True):
        #print(u)
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
            #self.phi[i] = 0


        self.update_u(u, mask=mask)
        
        ## for i in range(len(self.u)):
        ##     obs_mask = utils.add_axes(mask,
        ##                         len(self.plates) - np.ndim(mask),
        ##                         len(self.dims[i]))

        ##     # TODO/FIXME: Use copyto!
        ##     self.u[i] = (obs_mask * u[i]
        ##                  + np.logical_not(obs_mask) * self.u[i])

        # TODO/FIXME: Use the mask?
        self.f = f
        
    def observe(self, x, mask=True):
        (u, f) = self.compute_fixed_u_and_f(x)
        #print(u)
        self.fix_u_and_f(u, f, mask=mask)

        # Observed nodes should not be ignored
        self.observed = mask
        self.mask = np.logical_or(self.mask, self.observed)

        #print('observe', self.name, self.u)
        ## print(x)
        ## print(mask)
        ## print(u)

    def integrated_logpdf_from_parents(self, index):

        """ Approximates the posterior predictive pdf \int
        p(x|parents) q(parents) dparents in log-scale as \int
        q(parents_i) exp( \int q(parents_\i) \log p(x|parents)
        dparents_\i ) dparents_i."""

        raise NotImplementedError()
