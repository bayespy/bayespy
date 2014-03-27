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
                         CategoricalStatistics

class MixtureDistribution(ExponentialFamilyDistribution):

    def __init__(self, distribution, cluster_plate):
        self.distribution = distribution
        self.cluster_plate = cluster_plate
        self.ndims = distribution.ndims
        self.ndims_parents = distribution.ndims_parents

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
            # Reshape phi:
            # Shape(phi)    = [Nn,..,N0,K,Dd,..,D0]
            for ind in range(len(phi)):
                phi[ind] = utils.moveaxis(phi[ind],
                                          self.cluster_plate-self.distribution.ndims[ind],
                                          -1-self.distribution.ndims[ind])

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
            #else:
            #    cluster_axis = cluster_plate

            # Move cluster axis to the last:
            # Shape(phi)    = [Nn,..,N0,Dd,..,D0,K]
            phi.append(utils.moveaxis(Phi[ind], cluster_axis, -1))

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

class Mixture(ExponentialFamily):

    @useconstructor
    def __init__(self, z, node_class, *args, cluster_plate=-1, **kwargs):
        self.cluster_plate = cluster_plate
        super().__init__(z, *args, **kwargs)

    @classmethod
    def _constructor(cls, z, node_class, *args, cluster_plate=-1, **kwargs):
        """
        Constructs distribution and statistics objects.
        """
        if cluster_plate >= 0:
            raise Exception("Give negative value for axis index cluster_plates")
        
        # TODO/FIXME: Constant (non-node, numeric) z is a bit
        # non-trivial. Constant z is an array of cluster assignments, e.g.,
        # [2,0,2,3,1,2]. However, it is not possible to know the number of
        # clusters from this. It could be 4 but it could be 10 as well.
        # Thus, we can't create a constant node from z because the number of
        # clusters is required by constant categorical. It might be possible
        # to infer the number of clusters from the cluster_plate axis of the
        # other parameters but that is not trivial. A simple solution is to
        # require the user to provide the number of clusters when it can't
        # be inferred otherwise. Anyway, this is left for future work.

        if not hasattr(z, '_convert'):
            raise NotImplementedError("Constant cluster assignments are "
                                      "not yet implemented")
            
        # Convert a node to get the number of clusters
        z = z._convert(CategoricalStatistics)
        K = z.dims[0][0]

        # Get the stuff for the mixed distribution
        (dims, distribution, statistics, parent_statistics) = \
          node_class._constructor(*args)

        # Convert the distribution to a mixture
        distribution = MixtureDistribution(distribution, cluster_plate)

        # Add cluster assignments to parents
        parent_statistics = (CategoricalStatistics(K),) + parent_statistics
          
        return (dims, 
                distribution, 
                statistics, 
                parent_statistics)

    def _plates_to_parent(self, index):
        if index == 0:
            return self.plates
        else:
            if self.cluster_plate < 0:
                plates = list(self.plates)
                if self.cluster_plate < 0:
                    k = len(self.plates) + self.cluster_plate + 1
                else:
                    k = self.cluster_plate
                plates.insert(k, self.parents[0].dims[0][0])
                plates = tuple(plates)

            return plates

    def _plates_from_parent(self, index):
        if index == 0:
            return self.parents[index].plates
        else:
            plates = list(self.parents[index].plates)
            # Remove the cluster plate, if the parent has it
            if len(plates) >= abs(self.cluster_plate):
                plates.pop(self.cluster_plate)
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
            cluster_axis = self.cluster_plate - \
                           self._statistics.ndim_observations
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
