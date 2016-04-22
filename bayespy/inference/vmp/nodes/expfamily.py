################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import warnings

import numpy as np

from bayespy.utils import misc

from .node import ensureparents
from .stochastic import Stochastic, Distribution

class ExponentialFamilyDistribution(Distribution):
    """
    Sub-classes implement distribution specific computations.
    """

    #
    # The following methods are for ExponentialFamily distributions
    #

    def compute_message_to_parent(self, parent, index, u_self, *u_parents):
        raise NotImplementedError()

    def compute_phi_from_parents(self, *u_parents, mask=True):
        raise NotImplementedError()

    def compute_moments_and_cgf(self, phi, mask=True):
        raise NotImplementedError()

    #
    # The following methods are for Mixture class
    #

    def compute_cgf_from_parents(self, *u_parents):
        raise NotImplementedError()
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        raise NotImplementedError()

    def compute_logpdf(self, u, phi, g, f, ndims):
        """ Compute E[log p(X)] given E[u], E[phi], E[g] and
        E[f]. Does not sum over plates."""

        # TODO/FIXME: Should I take into account what is latent or
        # observed, or what is even totally ignored (by the mask).
        L = g + f
        for (phi_i, u_i, ndims_i) in zip(phi, u, ndims):
            # Axes to sum (dimensions of the variable, not the plates)
            axis_sum = tuple(range(-ndims_i,0))
            # Compute the term
            # TODO/FIXME: Use einsum!
            L = L + np.sum(phi_i * u_i, axis=axis_sum)
        return L


    def compute_gradient(self, g, u, phi):
        r"""
        Compute the standard gradient with respect to the natural parameters.
        """

        raise NotImplementedError("Standard gradient not yet implemented for %s"
                                  % (self.__class__.__name__))



def useconstructor(__init__):
    def constructor_decorator(self, *args, **kwargs):
        if (self.dims is None or
            self._distribution is None or
            self._moments is None or 
            self._parent_moments is None):

            (args, kwargs, dims, plates, dist, stats, pstats) = \
              self._constructor(*args, **kwargs)
            
            self.dims = dims
            self._distribution = dist
            self._moments = stats
            self._parent_moments = pstats
            self.plates = plates

        __init__(self, *args, **kwargs)

    return constructor_decorator

class ExponentialFamily(Stochastic):
    """
    A base class for nodes using natural parameterization `phi`.

    phi

    Sub-classes must implement the following static methods:
       _compute_message_to_parent(index, u_self, *u_parents)
       _compute_phi_from_parents(*u_parents, mask)
       _compute_moments_and_cgf(phi, mask)
       _compute_fixed_moments_and_f(x, mask=True)

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_weights_to_parent(index, weights)
       _compute_plates_to_parent(self, index, plates)
       _compute_plates_from_parent(self, index, plates)
    
    """

    # Sub-classes should overwrite this (possibly using _constructor)
    dims = None
    
    # Sub-classes should overwrite this
    _distribution = None

    @useconstructor
    def __init__(self, *parents, initialize=True, **kwargs):

        self.annealing = 1.0

        # Terms for the lower bound (G for latent and F for observed)
        self.g = np.array(np.nan)
        self.f = np.array(np.nan)

        super().__init__(*parents,
                         initialize=initialize,
                         dims=self.dims,
                         **kwargs)

        if not initialize:
            axes = len(self.plates)*(1,)
            self.phi = [misc.nans(axes+dim) for dim in self.dims]


    @classmethod
    @ensureparents
    def _constructor(cls, *parents, **kwargs):
        """
        Constructs distribution and moments objects.

        If __init__ uses useconstructor decorator, this method is called to
        construct distribution and moments objects.

        The method is given the same inputs as __init__. For some nodes, some of
        these can't be "static" class attributes, then the node class must
        overwrite this method to construct the objects manually.

        The point of distribution class is to move general distribution but
        not-node specific code. The point of moments class is to define the
        messaging protocols.
        """
        parent_plates = [cls._distribution.plates_from_parent(ind, parent.plates)
                         for (ind, parent) in enumerate(parents)]
        return (parents,
                kwargs,
                cls.dims,
                cls._total_plates(kwargs.get('plates'), *parent_plates),
                cls._distribution, 
                cls._moments, 
                cls._parent_moments)

    def _initialize_from_parent_moments(self, *u_parents):
        if not np.all(self.observed):
            # Update natural parameters using parents
            self._update_phi_from_parents(*u_parents)

            # Update moments
            mask = np.logical_not(self.observed)
            (u, g) = self._distribution.compute_moments_and_cgf(self.phi,
                                                                mask=mask)
            # TODO/FIXME/BUG: You should use observation mask in order to not
            # overwrite them!
            self._set_moments_and_cgf(u, g, mask=mask)
        

    def initialize_from_prior(self):
        u_parents = self._message_from_parents()
        self._initialize_from_parent_moments(*u_parents)


    def initialize_from_parameters(self, *args):
        u_parents = [p_mom.compute_fixed_moments(x) 
                     for (p_mom, x) in zip(self._parent_moments, args)]
        self._initialize_from_parent_moments(*u_parents)
        

    def initialize_from_value(self, x, *args):
        # Update moments from value
        mask = np.logical_not(self.observed)
        u = self._moments.compute_fixed_moments(x, *args)
        # Check that the shape is correct
        for i in range(len(u)):
            ndim = len(self.dims[i])
            if ndim > 0:
                if np.shape(u[i])[-ndim:] != self.dims[i]:
                    raise ValueError("The initial value for node %s has invalid shape %s."
                                     % (np.shape(x)))
        self._set_moments_and_cgf(u, np.inf, mask=mask)

    def initialize_from_random(self):
        """
        Set the variable to a random sample from the current distribution.
        """
        #self.initialize_from_prior()
        X = self.random()
        self.initialize_from_value(X)


    def _update_phi_from_parents(self, *u_parents):

        # TODO/FIXME: Could this be combined to the function
        # _update_distribution_and_lowerbound ?
        # No, because some initialization methods may want to use this.

        # This makes correct broadcasting
        self.phi = self._distribution.compute_phi_from_parents(*u_parents)
        #self.phi = self._compute_phi_from_parents(*u_parents)
        self.phi = list(self.phi)
        # Make sure phi has the correct number of axes. It makes life
        # a bit easier elsewhere.
        for i in range(len(self.phi)):
            axes = len(self.plates) + self.ndims[i] - np.ndim(self.phi[i])
            if axes > 0:
                # Add axes
                self.phi[i] = misc.add_leading_axes(self.phi[i], axes)
            elif axes < 0:
                # Remove extra leading axes
                first = -(len(self.plates)+self.ndims[i])
                sh = np.shape(self.phi[i])[first:]
                self.phi[i] = np.reshape(self.phi[i], sh)
            # Check that the shape is correct
            if not misc.is_shape_subset(np.shape(self.phi[i]),
                                         self.get_shape(i)):
                raise ValueError("Incorrect shape of phi[%d] in node class %s. "
                                 "Shape is %s but it should be broadcastable "
                                 "to shape %s."
                                 % (i,
                                    self.__class__.__name__,
                                    np.shape(self.phi[i]),
                                    self.get_shape(i)))

    def _set_moments_and_cgf(self, u, g, mask=True):
        self._set_moments(u, mask=mask)

        self.g = np.where(mask, g, self.g)

        return


    def get_riemannian_gradient(self):
        r"""
        Computes the Riemannian/natural gradient.
        """
        u_parents = self._message_from_parents()
        m_children = self._message_from_children()
        
        # TODO/FIXME: Put observed plates to zero?
        # Compute the gradient
        phi = self._distribution.compute_phi_from_parents(*u_parents)
        for i in range(len(self.phi)):
            phi[i] = self.annealing * (phi[i] + m_children[i]) - self.phi[i]
            phi[i] = phi[i] * np.ones(self.get_shape(i))

        return phi


    def get_gradient(self, rg):
        r""" Computes gradient with respect to the natural parameters.

        The function takes the Riemannian gradient as an input.  This is for
        three reasons: 1) You probably want to use the Riemannian gradient
        anyway so this helps avoiding accidental use of this function.  2) The
        gradient is computed by using the Riemannian gradient and chain rules.
        3) Probably you need both Riemannian and normal gradients anyway so you
        can provide it to this function to avoid re-computing it."""
            
        g = self._distribution.compute_gradient(rg, self.u, self.phi)
        for i in range(len(g)):
            g[i] /= self.annealing
        return g


    ## def update_parameters(self, d, scale=1.0):
    ##     r"""
    ##     Update the parameters of the VB distribution given a change.

    ##     The parameters should be such that they can be used for
    ##     optimization, that is, use log transformation for positive
    ##     parameters.
    ##     """
    ##     phi = self.get_parameters()
    ##     for i in range(len(phi)):
    ##         phi[i] = phi[i] + scale*d[i]
    ##     self.set_parameters(phi)
    ##     return


    def get_parameters(self):
        r"""
        Return parameters of the VB distribution.

        The parameters should be such that they can be used for
        optimization, that is, use log transformation for positive
        parameters.
        """
        return [np.copy(p) for p in self.phi]
            


    def _decode_parameters(self, x):
        return [np.copy(p) for p in x]


    def set_parameters(self, x):
        r"""
        Set the parameters of the VB distribution.

        The parameters should be such that they can be used for
        optimization, that is, use log transformation for positive
        parameters.
        """
        self.phi = self._decode_parameters(x)
        self._update_moments_and_cgf()
        return


    def _update_distribution_and_lowerbound(self, m_children, *u_parents):

        # Update phi first from parents..
        self._update_phi_from_parents(*u_parents)
        # .. then just add children's message
        self.phi = [self.annealing * (phi + m)
                    for (phi, m) in zip(self.phi, m_children)]

        # Update u and g
        self._update_moments_and_cgf()


    def _update_moments_and_cgf(self):
        """
        Update moments and cgf based on current phi.
        """
        # Mask for plates to update (i.e., unobserved plates)
        update_mask = np.logical_not(self.observed)

        # Compute the moments (u) and CGF (g)...
        (u, g) = self._distribution.compute_moments_and_cgf(self.phi,
                                                            mask=update_mask)
        # ... and store them
        self._set_moments_and_cgf(u, g, mask=update_mask)


    def observe(self, x, *args, mask=True):
        """
        Fix moments, compute f and propagate mask.
        """

        # Compute fixed moments
        (u, f) = self._distribution.compute_fixed_moments_and_f(x, *args,
                                                                mask=mask)

        # # Check the dimensionality of the observations
        # self._check_shape()
        # for (i,v) in enumerate(u):
        #     # This is what the dimensionality "should" be
        #     s = self.plates + self.dims[i]
        #     t = np.shape(v)
        #     if s != t:
        #         msg = "Dimensionality of the observations incorrect."
        #         msg += "\nShape of input: " + str(t)
        #         msg += "\nExpected shape: " + str(s)
        #         msg += "\nCheck plates."
        #         raise Exception(msg)

        # Set the moments. Shape checking is done there.
        self._set_moments(u, mask=mask, broadcast=False)

        self.f = np.where(mask, f, self.f)

        # Observed nodes should not be ignored
        self.observed = mask
        self._update_mask()

    def lower_bound_contribution(self, gradient=False, ignore_masked=True):
        r"""Compute E[ log p(X|parents) - log q(X) ]

        If deterministic annealing is used, the term E[ -log q(X) ] is
        divided by the anneling coefficient.  That is, phi and cgf of q
        are multiplied by the temperature (inverse annealing
        coefficient).
        
        """

        # Annealing temperature
        T = 1 / self.annealing
        
        # Messages from parents
        u_parents = self._message_from_parents()
        phi = self._distribution.compute_phi_from_parents(*u_parents)
        # G from parents
        L = self._distribution.compute_cgf_from_parents(*u_parents)

        # G for unobserved variables (ignored variables are handled properly
        # automatically)
        latent_mask = np.logical_not(self.observed)

        # G and F
        if np.all(self.observed):
            z = np.nan
        elif T == 1:
            z = -self.g
        else:
            z = -T * self.g
            ## TRIED THIS BUT IT WAS WRONG:
            ## z = -T * self.g + (1-T) * self.f
            ## if np.any(np.isnan(self.f)):
            ##     warnings.warn("F(x) not implemented for node %s. This "
            ##                   "is required for annealed lower bound "
            ##                   "computation." % self.__class__.__name__)
            ##
            ## It was wrong because the optimal q distribution has f which is
            ## weighted by 1/T and here the f of q is weighted by T so the
            ## total weight is 1, thus it cancels out with f of p.

        L = L + np.where(self.observed, self.f, z)

        for (phi_p, phi_q, u_q, dims) in zip(phi, self.phi, self.u, self.dims):
            # Form a mask which puts observed variables to zero and
            # broadcasts properly
            latent_mask_i = misc.add_trailing_axes(
                                misc.add_leading_axes(
                                    latent_mask,
                                    len(self.plates) - np.ndim(latent_mask)),
                                len(dims))
            axis_sum = tuple(range(-len(dims),0))

            # Compute the term
            phi_q = np.where(latent_mask_i, phi_q, 0)
            # Apply annealing
            phi_diff = phi_p - T * phi_q
            # Handle 0 * -inf
            phi_diff = np.where(u_q != 0, phi_diff, 0)
            # TODO/FIXME: Use einsum here?
            Z = np.sum(phi_diff * u_q, axis=axis_sum)

            L = L + Z

        if ignore_masked:
            return (np.sum(np.where(self.mask, L, 0))
                    * self.broadcasting_multiplier(self.plates,
                                                   np.shape(L),
                                                   np.shape(self.mask))
                    * np.prod(self.plates_multiplier))
        else:
            return (np.sum(L)
                    * self.broadcasting_multiplier(self.plates,
                                                   np.shape(L))
                    * np.prod(self.plates_multiplier))


    def logpdf(self, X, mask=True):
        """
        Compute the log probability density function Q(X) of this node.
        """
        if mask is not True:
            raise NotImplementedError('Mask not yet implemented')
        (u, f) = self._distribution.compute_fixed_moments_and_f(X, mask=mask)
        Z = 0
        for (phi_d, u_d, dims) in zip(self.phi, u, self.dims):
            axis_sum = tuple(range(-len(dims),0))
            # TODO/FIXME: Use einsum here?
            Z = Z + np.sum(phi_d * u_d, axis=axis_sum)
            #Z = Z + misc.sum_multiply(phi_d, u_d, axis=axis_sum)

        return (self.g + f + Z)
        

    def pdf(self, X, mask=True):
        """
        Compute the probability density function of this node.
        """
        return np.exp(self.logpdf(X, mask=mask))
        

    def _save(self, group):
        """
        Save the state of the node into a HDF5 file.

        group can be the root
        """
        ## if name is None:
        ##     name = self.name
        ## subgroup = group.create_group(name)
        
        for i in range(len(self.phi)):
            misc.write_to_hdf5(group, self.phi[i], 'phi%d' % i)
        misc.write_to_hdf5(group, self.f, 'f')
        misc.write_to_hdf5(group, self.g, 'g')
        super()._save(group)

    
    def _load(self, group):
        """
        Load the state of the node from a HDF5 file.
        """
        # TODO/FIXME: Check that the shapes are correct!
        for i in range(len(self.phi)):
            phii = group['phi%d' % i][...]
            self.phi[i] = phii
            
        self.f = group['f'][...]
        self.g = group['g'][...]
        super()._load(group)


    def random(self):
        """
        Draw a random sample from the distribution.
        """
        return self._distribution.random(*(self.phi), plates=self.plates)
