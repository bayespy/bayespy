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
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.spatial.distance as distance

from bayespy.utils import utils

from .expfamily import ExponentialFamily
from .constant import Constant
from .categorical import Categorical

def Mixture(distribution, cluster_plate=-1):

    if cluster_plate >= 0:
        raise Exception("Give negative value for axis index cluster_plates")

    class _Mixture(ExponentialFamily):
        """

        Mixtured distributions must implement:
           _compute_cgf_from_parents
           _compute_fixed_moments_and_f(x, mask)
           _compute_dims

        Sub-classes of ExponentialFamily should implement anyway:
           _compute_message_to_parent
           _compute_phi_from_parents
           _compute_moments_and_cgf(phi, mask)

        ExponentialFamily implements already:
           _compute_logpdf
        """

        ndims = distribution.ndims

        @staticmethod
        def _compute_phi_from_parents(*u_parents):

            # Compute weighted average of the parameters

            # Cluster parameters
            Phi = distribution._compute_phi_from_parents(*(u_parents[1:]))
            # Contributions/weights/probabilities
            P = u_parents[0][0]

            phi = list()
            
            for ind in range(len(Phi)):
                # Compute element-wise product and then sum over K clusters.
                # Note that the dimensions aren't perfectly aligned because
                # the cluster dimension (K) may be arbitrary for phi, and phi
                # also has dimensions (Dd,..,D0) of the parameters.
                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                # Shape(p)      = [Nn,..,N0,K]
                # Shape(result) = [Nn,..,N0,Dd,..,D0]
                # General broadcasting rules apply for Nn,..,N0, that is,
                # preceding dimensions may be missing or dimension may be
                # equal to one. Probably, shape(phi) has lots of missing
                # dimensions and/or dimensions that are one.

                if cluster_plate < 0:
                    cluster_axis = cluster_plate - distribution.ndims[ind]
                #else:
                #    cluster_axis = cluster_plate

                # Move cluster axis to the last:
                # Shape(phi)    = [Nn,..,N0,Dd,..,D0,K]
                phi.append(utils.moveaxis(Phi[ind], cluster_axis, -1))

                # Add axes to p:
                # Shape(p)      = [Nn,..,N0,K,1,..,1]
                p = utils.add_trailing_axes(P, distribution.ndims[ind])
                # Move cluster axis to the last:
                # Shape(p)      = [Nn,..,N0,1,..,1,K]
                p = utils.moveaxis(p, -(distribution.ndims[ind]+1), -1)
                #print('Mixture.compute_phi, p:', np.sum(p, axis=-1))
                #print('mixture.compute_phi shapes:')
                #print(np.shape(p))
                #print(np.shape(phi[ind]))
                
                # Now the shapes broadcast perfectly and we can sum
                # p*phi over the last axis:
                # Shape(result) = [Nn,..,N0,Dd,..,D0]
                phi[ind] = utils.sum_product(p, phi[ind], axes_to_sum=-1)
                
            return phi

        @staticmethod
        def _compute_cgf_from_parents(*u_parents):

            # Compute weighted average of g over the clusters.

            # Shape(g)      = [Nn,..,K,..,N0]
            # Shape(p)      = [Nn,..,N0,K]
            # Shape(result) = [Nn,..,N0]

            # Compute g for clusters:
            # Shape(g)      = [Nn,..,K,..,N0]
            g = distribution._compute_cgf_from_parents(*(u_parents[1:]))
            
            # Move cluster axis to last:
            # Shape(g)      = [Nn,..,N0,K]
            g = utils.moveaxis(g, cluster_plate, -1)

            # Cluster assignments/contributions/probabilities/weights:
            # Shape(p)      = [Nn,..,N0,K]
            p = u_parents[0][0]
            
            # Weighted average of g over the clusters. As p and g are
            # properly aligned, you can just sum p*g over the last
            # axis and utilize broadcasting:
            # Shape(result) = [Nn,..,N0]
            #print('mixture.compute_g_from_parents p and g:', np.shape(p), np.shape(g))
            g = utils.sum_product(p, g, axes_to_sum=-1)

            #print('mixture.compute_g_from_parents g:', np.sum(g), np.shape(g))

            return g

        @staticmethod
        def _compute_moments_and_cgf(phi, mask=True):
            return distribution._compute_moments_and_cgf(phi, mask=mask)

        @staticmethod
        def _compute_fixed_moments_and_f(x, mask=True):
            """ Compute u(x) and f(x) for given x. """
            return distribution._compute_fixed_moments_and_f(x, mask=True)

        @staticmethod
        def _compute_message_to_parent(parent, index, u, *u_parents):
            """ . """

            #print('Mixture.compute_message:')
            
            if index == 0:

                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                # Shape(L)      = [Nn,..,K,..,N0]
                # Shape(u)      = [Nn,..,N0,Dd,..,D0]
                # Shape(result) = [Nn,..,N0,K]

                # Compute g:
                # Shape(g)      = [Nn,..,K,..,N0]
                g = distribution._compute_cgf_from_parents(*(u_parents[1:]))
                # Reshape(g):
                # Shape(g)      = [Nn,..,N0,K]
                g = utils.moveaxis(g, cluster_plate, -1)

                # Compute phi:
                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                phi = distribution._compute_phi_from_parents(*(u_parents[1:]))
                # Reshape phi:
                # Shape(phi)    = [Nn,..,N0,K,Dd,..,D0]
                for ind in range(len(phi)):
                    phi[ind] = utils.moveaxis(phi[ind],
                                              cluster_plate-distribution.ndims[ind],
                                              -1-distribution.ndims[ind])

                # Reshape u:
                # Shape(u)      = [Nn,..,N0,1,Dd,..,D0]
                u_self = list()
                for ind in range(len(u)):
                    u_self.append(np.expand_dims(u[ind],
                                                 axis=(-1-distribution.ndims[ind])))
                    
                # Compute logpdf:
                # Shape(L)      = [Nn,..,N0,K]
                L = distribution._compute_logpdf(u_self, phi, g, 0)
                
                # Sum over other than the cluster dimensions? No!
                # Hmm.. I think the message passing method will do
                # that automatically

                ## print(np.shape(phi[0]))
                ## print(np.shape(u_self[0]))
                ## print(np.shape(g))
                ## print(np.shape(L))
                
                return [L]

            elif index >= 1:

                # Parent index for the distribution used for the
                # mixture.
                index = index - 1

                # Reshape u:
                # Shape(u)      = [Nn,..1,..,N0,Dd,..,D0]
                u_self = list()
                for ind in range(len(u)):
                    if cluster_plate < 0:
                        cluster_axis = cluster_plate - distribution.ndims[ind]
                    else:
                        cluster_axis = cluster_plate
                    u_self.append(np.expand_dims(u[ind], axis=cluster_axis))
                    
                # Message from the mixed distribution
                m = distribution._compute_message_to_parent(index, 
                                                            u_self, 
                                                            *(u_parents[1:]))

                # Weigh the messages with the responsibilities
                for i in range(len(m)):

                    # Shape(m)      = [Nn,..,K,..,N0,Dd,..,D0]
                    # Shape(p)      = [Nn,..,N0,K]
                    # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
                    
                    # Number of axes for the variable dimensions for
                    # the parent message.
                    D = distribution.ndims_parents[index][i]

                    # Responsibilities for clusters are the first
                    # parent's first moment:
                    # Shape(p)      = [Nn,..,N0,K]
                    p = u_parents[0][0]
                    # Move the cluster axis to the proper place:
                    # Shape(p)      = [Nn,..,K,..,N0]
                    p = utils.moveaxis(p, -1, cluster_plate)
                    # Add axes for variable dimensions to the contributions
                    # Shape(p)      = [Nn,..,K,..,N0,1,..,1]
                    p = utils.add_trailing_axes(p, D)

                    if cluster_plate < 0:
                        # Add the variable dimensions
                        cluster_axis = cluster_plate - D

                    # Add axis for clusters:
                    # Shape(m)      = [Nn,..,1,..,N0,Dd,..,D0]
                    #m[i] = np.expand_dims(m[i], axis=cluster_axis)
                        
                    #
                    # TODO: You could do summing here already so that
                    # you wouldn't compute huge matrices as
                    # intermediate result. Use einsum.

                    ## print(np.shape(m[i]))
                    ## print(np.shape(p))

                    # Compute the message contributions for each
                    # cluster:
                    # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
                    m[i] = m[i] * p

                    #print(np.shape(m[i]))
                    
                return m

        @staticmethod
        def compute_dims(*parents):
            """ Compute the dimensions of phi and u. """
            return distribution.compute_dims(*parents[1:])

        def __init__(self, z, *args, **kwargs):
            # Check for constant mu
            if np.isscalar(z) or isinstance(z, np.ndarray):
                z = ConstantCategorical(z)
            # Construct
            super().__init__(z, *args,
                             **kwargs)

            #_compute_mask_to_parent(index, mask)
            #_plates_to_parent(self, index)
            #_plates_from_parent(self, index)
        def _plates_to_parent(self, index):
            if index == 0:
                return self.plates
            else:
                if cluster_plate < 0:
                    plates = list(self.plates)
                    if cluster_plate < 0:
                        k = len(self.plates) + cluster_plate + 1
                    else:
                        k = cluster_plate
                    plates.insert(k, self.parents[0].dims[0][0])
                    plates = tuple(plates)
                    #print('plates_to_parent', cluster_plate,  plates)
                    ## plates = (self.plates[:cluster_plate] +
                    ##           self.parents[0].dims[0] +
                    ##           self.plates[cluster_plate:])
                return plates

        def _plates_from_parent(self, index):
            if index == 0:
                return self.parents[index].plates
            else:
                plates = list(self.parents[index].plates)
                plates.pop(cluster_plate)
                return tuple(plates)
            
        def integrated_logpdf_from_parents(self, x, index):

            """ Approximates the posterior predictive pdf \int
            p(x|parents) q(parents) dparents in log-scale as \int
            q(parents_i) exp( \int q(parents_\i) \log p(x|parents)
            dparents_\i ) dparents_i."""

            if index == 0:
                # Integrate out the cluster assignments

                # First, integrate the cluster parameters in log-scale
                
                # compute_logpdf(cls, u, phi, g, f):

                # Shape(x) = [M1,..,Mm,N1,..,Nn,D1,..,Dd]
                # Add the cluster axis to x:
                # Shape(x) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
                cluster_axis = cluster_plate - distribution.ndim_observations
                x = np.expand_dims(x, axis=cluster_axis)

                u_parents = self._message_from_parents()
                #u_parents = self.moments_from_parents()
                
                # Shape(u) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
                # Shape(f) = [M1,..,Mm,N1,..,1,..,Nn]
                (u, f) = distribution._compute_fixed_moments_and_f(x)
                # Shape(phi) = [N1,..,K,..,Nn,D1,..,Dd]
                phi = distribution._compute_phi_from_parents(*(u_parents[1:]))
                # Shape(g) = [N1,..,K,..,Nn]
                g = distribution._compute_cgf_from_parents(*(u_parents[1:]))
                # Shape(lpdf) = [M1,..,Mm,N1,..,K,..,Nn]
                lpdf = distribution._compute_logpdf(u, phi, g, f)

                # From logpdf to pdf, but avoid over/underflow
                lpdf_max = np.max(lpdf, axis=cluster_plate, keepdims=True)
                pdf = np.exp(lpdf-lpdf_max)

                # Move cluster axis to be the last:
                # Shape(pdf) = [M1,..,Mm,N1,..,Nn,K]
                pdf = utils.moveaxis(pdf, cluster_plate, -1)

                #print('integrated_logpdf', pdf)
                
                # Cluster assignments/probabilities/weights
                # Shape(p) = [N1,..,Nn,K]
                p = u_parents[0][0]

                #self.parents[0].show()
                #print('integrated_logpdf, p:', p)
                
                # Weighted average. TODO/FIXME: Use einsum!
                # Shape(pdf) = [M1,..,Mm,N1,..,Nn]
                pdf = np.sum(pdf * p, axis=cluster_plate)

                #print('integrated_logpdf', pdf)
                
                # Back to log-scale (add the overflow fix!)
                lpdf_max = np.squeeze(lpdf_max, axis=cluster_plate)
                lpdf = np.log(pdf) + lpdf_max

                return lpdf

            raise NotImplementedError()
        
    return _Mixture

    ## def show(self):
    ##     p = self.u[0] #np.exp(self.phi[0])
    ##     #p /= np.sum(p, axis=-1, keepdims=True)
    ##     print("Categorical(p)")
    ##     print("  p = ")
    ##     print(p)

    ## def observe(self, x):
    ##     # TODO: You could check that x has proper dimensions
    ##     x = np.array(x, dtype=np.int)
        
    ##     # Initial array of zeros
    ##     d = self.dims[0][0]
    ##     self.u[0] = np.zeros(np.shape(x)+(d,))
        
    ##     # Compute indices
    ##     x += d*np.arange(np.size(x),dtype=int).reshape(np.shape(x))
    ##     x = x[...,np.newaxis]
    ##     # Set 1 to elements corresponding to the observations
    ##     np.put(self.u[0], x, 1)
    ##     self.show()
        
    ##     self.fix_u_and_f(self.u, 0)
