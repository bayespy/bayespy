######################################################################
# Copyright (C) 2011,2012,2014 Jaakko Luttinen
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

import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.spatial.distance as distance

from bayespy.utils import utils

from .expfamily import ExponentialFamily, \
                       ExponentialFamilyDistribution, \
                       useconstructor
                       
from .categorical import Categorical, \
                         CategoricalMoments

class MixtureDistribution(ExponentialFamilyDistribution):

    def __init__(self, distribution, cluster_plate, n_clusters):
        self.distribution = distribution
        self.cluster_plate = cluster_plate
        self.ndims = distribution.ndims
        self.ndims_parents = distribution.ndims_parents
        self.K = n_clusters

    def compute_message_to_parent(self, parent, index, u, *u_parents):

        if index == 0:

            # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
            # Shape(L)      = [Nn,..,K,..,N0]
            # Shape(u)      = [Nn,..,N0,Dd,..,D0]
            # Shape(result) = [Nn,..,N0,K]

            # Compute g:
            # Shape(g)      = [Nn,..,K,..,N0]
            g = self.distribution.compute_cgf_from_parents(*(u_parents[1:]))
            # Reshape(g):
            # Shape(g)      = [Nn,..,N0,K]
            g = utils.moveaxis(g, self.cluster_plate, -1)

            # Compute phi:
            # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
            phi = self.distribution.compute_phi_from_parents(*(u_parents[1:]))
            # Move phi axis:
            # Shape(phi)    = [Nn,..,N0,K,Dd,..,D0]
            for ind in range(len(phi)):
                if self.cluster_plate < 0:
                    axis_from = self.cluster_plate-self.distribution.ndims[ind]
                else:
                    raise RuntimeError("Cluster plate axis must be negative")
                axis_to = -1-self.distribution.ndims[ind]
                if np.ndim(phi[ind]) >= abs(axis_from):
                    # Cluster plate axis exists, move it to the correct position
                    phi[ind] = utils.moveaxis(phi[ind], axis_from, axis_to)
                else:
                    # No cluster plate axis, just add a new axis to the correct
                    # position, if phi has something on that axis
                    if np.ndim(phi[ind]) >= abs(axis_to):
                        phi[ind] = np.expand_dims(phi[ind], axis=axis_to)

            # Reshape u:
            # Shape(u)      = [Nn,..,N0,1,Dd,..,D0]
            u_self = list()
            for ind in range(len(u)):
                u_self.append(np.expand_dims(u[ind],
                                             axis=(-1-self.distribution.ndims[ind])))

            # Compute logpdf:
            # Shape(L)      = [Nn,..,N0,K]
            L = self.distribution.compute_logpdf(u_self, phi, g, 0)

            # Sum over other than the cluster dimensions? No!
            # Hmm.. I think the message passing method will do
            # that automatically

            m = [L]

            return m

        elif index >= 1:

            # Parent index for the distribution used for the
            # mixture.
            index = index - 1

            # Reshape u:
            # Shape(u)      = [Nn,..1,..,N0,Dd,..,D0]
            u_self = list()
            for ind in range(len(u)):
                if self.cluster_plate < 0:
                    cluster_axis = self.cluster_plate - self.distribution.ndims[ind]
                else:
                    cluster_axis = self.cluster_plate
                u_self.append(np.expand_dims(u[ind], axis=cluster_axis))

            # Message from the mixed distribution
            m = self.distribution.compute_message_to_parent(parent,
                                                        index, 
                                                        u_self, 
                                                        *(u_parents[1:]))

            # Weigh the messages with the responsibilities
            for i in range(len(m)):

                # Shape(m)      = [Nn,..,K,..,N0,Dd,..,D0]
                # Shape(p)      = [Nn,..,N0,K]
                # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]

                # Number of axes for the variable dimensions for
                # the parent message.
                D = self.distribution.ndims_parents[index][i]

                # Responsibilities for clusters are the first
                # parent's first moment:
                # Shape(p)      = [Nn,..,N0,K]
                p = u_parents[0][0]
                # Move the cluster axis to the proper place:
                # Shape(p)      = [Nn,..,K,..,N0]
                p = utils.atleast_nd(p, abs(self.cluster_plate))
                p = utils.moveaxis(p, -1, self.cluster_plate)
                # Add axes for variable dimensions to the contributions
                # Shape(p)      = [Nn,..,K,..,N0,1,..,1]
                p = utils.add_trailing_axes(p, D)

                if self.cluster_plate < 0:
                    # Add the variable dimensions
                    cluster_axis = self.cluster_plate - D

                # Add axis for clusters:
                # Shape(m)      = [Nn,..,1,..,N0,Dd,..,D0]
                #m[i] = np.expand_dims(m[i], axis=cluster_axis)

                #
                # TODO: You could do summing here already so that
                # you wouldn't compute huge matrices as
                # intermediate result. Use einsum.

                # Compute the message contributions for each
                # cluster:
                # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
                m[i] = m[i] * p

            return m

    def compute_mask_to_parent(self, index, mask):
        if index == 0:
            return mask
        else:
            if self.cluster_plate >= 0:
                raise ValueError("Cluster plate axis must be negative")
            if np.ndim(mask) >= abs(self.cluster_plate):
                mask = np.expand_dims(mask, axis=self.cluster_plate)
            return self.distribution.compute_mask_to_parent(index-1, mask)

    def compute_phi_from_parents(self, *u_parents, mask=True):

        # Compute weighted average of the parameters

        # Cluster parameters
        Phi = self.distribution.compute_phi_from_parents(*(u_parents[1:]))
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

            if self.cluster_plate < 0:
                cluster_axis = self.cluster_plate - self.distribution.ndims[ind]
            else:
                raise RuntimeError("Cluster plate should be negative")

            # Move cluster axis to the last:
            # Shape(phi)    = [Nn,..,N0,Dd,..,D0,K]
            if np.ndim(Phi[ind]) >= abs(cluster_axis):
                phi.append(utils.moveaxis(Phi[ind], cluster_axis, -1))
            else:
                phi.append(Phi[ind][...,None])

            # Add axes to p:
            # Shape(p)      = [Nn,..,N0,K,1,..,1]
            p = utils.add_trailing_axes(P, self.distribution.ndims[ind])
            # Move cluster axis to the last:
            # Shape(p)      = [Nn,..,N0,1,..,1,K]
            p = utils.moveaxis(p, -(self.distribution.ndims[ind]+1), -1)

            # Now the shapes broadcast perfectly and we can sum
            # p*phi over the last axis:
            # Shape(result) = [Nn,..,N0,Dd,..,D0]
            phi[ind] = utils.sum_product(p, phi[ind], axes_to_sum=-1)

        return phi

    def compute_moments_and_cgf(self, phi, mask=True):
        return self.distribution.compute_moments_and_cgf(phi, mask=mask)

    def compute_cgf_from_parents(self, *u_parents):

        # Compute weighted average of g over the clusters.

        # Shape(g)      = [Nn,..,K,..,N0]
        # Shape(p)      = [Nn,..,N0,K]
        # Shape(result) = [Nn,..,N0]

        # Compute g for clusters:
        # Shape(g)      = [Nn,..,K,..,N0]
        g = self.distribution.compute_cgf_from_parents(*(u_parents[1:]))

        # Move cluster axis to last:
        # Shape(g)      = [Nn,..,N0,K]
        g = utils.moveaxis(g, self.cluster_plate, -1)

        # Cluster assignments/contributions/probabilities/weights:
        # Shape(p)      = [Nn,..,N0,K]
        p = u_parents[0][0]

        # Weighted average of g over the clusters. As p and g are
        # properly aligned, you can just sum p*g over the last
        # axis and utilize broadcasting:
        # Shape(result) = [Nn,..,N0]
        g = utils.sum_product(p, g, axes_to_sum=-1)

        return g
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        """ Compute u(x) and f(x) for given x. """
        return self.distribution.compute_fixed_moments_and_f(x, mask=True)

    def plates_to_parent(self, index, plates):
        if index == 0:
            return plates
        else:

            # Add the cluster plate axis
            plates = list(plates)
            if self.cluster_plate < 0:
                knd = len(plates) + self.cluster_plate + 1
            else:
                raise RuntimeError("Cluster plate axis must be negative")
            plates.insert(knd, self.K)
            plates = tuple(plates)

            return self.distribution.plates_to_parent(index-1, plates)

    def plates_from_parent(self, index, plates):
        if index == 0:
            return plates
        else:
            plates = self.distribution.plates_from_parent(index-1, plates)
            
            # Remove the cluster plate, if the parent has it
            plates = list(plates)
            if len(plates) >= abs(self.cluster_plate):
                plates.pop(self.cluster_plate)
            return tuple(plates)


class Mixture(ExponentialFamily):


    @classmethod
    def _constructor(cls, z, node_class, *args, cluster_plate=-1, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        if cluster_plate >= 0:
            raise ValueError("Cluster plate axis must be negative")
        
        # Get the stuff for the mixed distribution
        (parents, _, dims, mixture_plates, distribution, moments, parent_moments) = \
          node_class._constructor(*args)

        # Check that at least one of the parents has the cluster plate axis
        if len(mixture_plates) < abs(cluster_plate):
            raise ValueError("The mixed distribution does not have a plates "
                             "axis for the cluster plate axis")
        
        # Resolve the number of clusters
        mixture_plates = list(mixture_plates)
        K = mixture_plates.pop(cluster_plate)
        
        # Convert a node to get the number of clusters
        z = cls._ensure_moments(z, CategoricalMoments(K))
        if z.dims[0][0] != K:
            raise ValueError("Inconsistent number of clusters")

        plates = cls._total_plates(kwargs.get('plates'), mixture_plates, z.plates)
        
        # Convert the distribution to a mixture
        distribution = MixtureDistribution(distribution, cluster_plate, K)

        # Add cluster assignments to parents
        parent_moments = (CategoricalMoments(K),) + parent_moments

        parents = [z] + list(parents)

        return (parents,
                kwargs,
                dims,
                plates,
                distribution, 
                moments, 
                parent_moments)
    

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
            cluster_axis = self.cluster_plate - \
                           self._moments.ndim_observations
            x = np.expand_dims(x, axis=cluster_axis)

            u_parents = self._message_from_parents()

            # Shape(u) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
            # Shape(f) = [M1,..,Mm,N1,..,1,..,Nn]
            (u, f) = self._distribution.distribution.compute_fixed_moments_and_f(x)
            # Shape(phi) = [N1,..,K,..,Nn,D1,..,Dd]
            phi = self._distribution.distribution.compute_phi_from_parents(*(u_parents[1:]))
            # Shape(g) = [N1,..,K,..,Nn]
            g = self._distribution.distribution.compute_cgf_from_parents(*(u_parents[1:]))
            # Shape(lpdf) = [M1,..,Mm,N1,..,K,..,Nn]
            lpdf = self._distribution.distribution.compute_logpdf(u, phi, g, f)

            # From logpdf to pdf, but avoid over/underflow
            lpdf_max = np.max(lpdf, axis=self.cluster_plate, keepdims=True)
            pdf = np.exp(lpdf-lpdf_max)

            # Move cluster axis to be the last:
            # Shape(pdf) = [M1,..,Mm,N1,..,Nn,K]
            pdf = utils.moveaxis(pdf, self.cluster_plate, -1)

            # Cluster assignments/probabilities/weights
            # Shape(p) = [N1,..,Nn,K]
            p = u_parents[0][0]

            # Weighted average. TODO/FIXME: Use einsum!
            # Shape(pdf) = [M1,..,Mm,N1,..,Nn]
            pdf = np.sum(pdf * p, axis=self.cluster_plate)

            # Back to log-scale (add the overflow fix!)
            lpdf_max = np.squeeze(lpdf_max, axis=self.cluster_plate)
            lpdf = np.log(pdf) + lpdf_max

            return lpdf

        raise NotImplementedError()
