######################################################################
# Copyright (C) 2012-2014 Jaakko Luttinen
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
This module contains VMP nodes for Gaussian Markov chains.
"""

import numpy as np
import scipy

import matplotlib.pyplot as plt
import bayespy.plot.plotting as bpplt

from bayespy.utils import utils
from bayespy.utils import linalg

from .node import Node, message_sum_multiply
from .deterministic import Deterministic
from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution
from .expfamily import useconstructor
from .gaussian import Gaussian, GaussianStatistics
from .wishart import Wishart, WishartStatistics
from .gamma import Gamma, GammaStatistics
from .node import Statistics, ensureparents

class GaussianMarkovChainStatistics(Statistics):

    def compute_fixed_moments(self, x):
        raise NotImplementedError()
        
    def converter(self, statistics_class):
        if statistics_class is GaussianStatistics:
            return _MarkovChainToGaussian
        return super().converter(statistics_class)
    
class TemplateGaussianMarkovChainDistribution(ExponentialFamilyDistribution):
    """
    Sub-classes implement distribution specific computations.
    """

    ndims = (2, 3, 3)
    
    def __init__(self, N):
        self.N = N
        super().__init__()

    def compute_message_to_parent(self, parent, index, u_self, *u_parents):
        raise NotImplementedError()

    def compute_mask_to_parent(self, index, mask):
        raise NotImplementedError()

    def compute_phi_from_parents(self, *u_parents, mask=True):
        raise NotImplementedError()

    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and the cumulant-generating function.

        This basically performs the filtering and smoothing for the variable.

        Parameters
        ----------
        phi

        Returns
        -------
        u
        g
        """

        # Solve the Kalman filtering and smoothing problem
        y = phi[0]
        A = -2*phi[1]
        # Don't multiply phi[2] by two because it is a sum of the super- and
        # sub-diagonal blocks so we would need to divide by two anyway.
        B = -phi[2]

        (CovXnXn, CovXpXn, Xn, ldet) = linalg.block_banded_solve(A, B, y)

        # Compute moments
        u0 = Xn
        u1 = CovXnXn + Xn[...,:,np.newaxis] * Xn[...,np.newaxis,:]
        u2 = CovXpXn + Xn[...,:-1,:,np.newaxis] * Xn[...,1:,np.newaxis,:]
        u = [u0, u1, u2]

        # Compute cumulant-generating function
        g = -0.5 * np.einsum('...ij,...ij', u[0], phi[0]) + 0.5*ldet
        
        return (u, g)

    def compute_cgf_from_parents(self, *u_parents):
        raise NotImplementedError()
        
    def compute_fixed_moments_and_f(self, x, mask=True):
        """
        Compute u(x) and f(x) for given x.
        """
        u0 = x
        u1 = x[...,:,np.newaxis] * x[...,np.newaxis,:]
        u2 = x[...,:-1,:,np.newaxis] * x[...,1:,np.newaxis,:]
        u = [u0, u1, u2]

        f = -0.5 * np.shape(x)[-2] * np.shape(x)[-1] * np.log(2*np.pi)
        return (u, f)



# TODO/FIXME: The plates of masks are not handled properly! Try having
# a plate of GMCs and then the message mask to A or v..

class _TemplateGaussianMarkovChain(ExponentialFamily):
    r"""
    VMP abstract node for Gaussian Markov chain.

    This is a general base class for different Gaussian Markov chain nodes.

    Output is Gaussian variables with mean, covariance and one-step
    cross-covariance.

    self.phi and self.u are defined in a particular way but otherwise the parent
    nodes may vary.

    Child classes must implement the following methods:
        _plates_to_parent
        _plates_from_parent

    See also
    --------
    bayespy.inference.vmp.nodes.gaussian.Gaussian
    bayespy.inference.vmp.nodes.wishart.Wishart

    """

    #_statistics_class = GaussianMarkovChainStatistics
    _statistics = GaussianMarkovChainStatistics()
                                

    # phi[0] is (N,D), phi[1] is (N,D,D), phi[2] is (N-1,D,D)
    # Observations are a set of vectors (thus 2-D matrix):
    ndim_observations = 2
    
    def __init__(self, *parents, n=None, **kwargs):
        """
        """
        # Construct
        super().__init__(*parents, **kwargs)

    def get_shape_of_value(self):
        # Dimensionality of a realization
        return self.dims[0]
    
    def _plates_to_parent(self, index):
        """
        Computes the plates of this node with respect to a parent.

        Child classes must implement this.

        Parameters:
        -----------
        index : int
            The index of the parent node to use.
        """
        raise NotImplementedError()

    def _plates_from_parent(self, index):
        """
        Compute the plates using information of a parent node.

        Child classes must implement this.

        Parameters
        ----------
        index : int
            Index of the parent to use.
        """
        raise NotImplementedError()

    def random(self):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()

    def rotate(self, R, inv=None, logdet=None):

        if inv is not None:
            invR = inv
        else:
            invR = np.linalg.inv(R)

        if logdet is not None:
            logdetR = logdet
        else:
            logdetR = np.linalg.slogdet(R)[1]

        # It would be more efficient and simpler, if you just rotated the
        # moments and didn't touch phi. However, then you would need to call
        # update() before lower_bound_contribution. This is more error-safe.

        # Transform parameters
        self.phi[0] = linalg.mvdot(invR.T, self.phi[0])
        self.phi[1] = linalg.dot(invR.T, self.phi[1], invR)
        self.phi[2] = linalg.dot(invR.T, self.phi[2], invR)

        N = self.dims[0][0]

        if False:
            self._update_moments_and_cgf()
        else:
            # Transform moments and g
            u0 = linalg.mvdot(R, self.u[0])
            u1 = linalg.dot(R, self.u[1], R.T)
            u2 = linalg.dot(R, self.u[2], R.T)
            self.u = [u0, u1, u2]
            self.g -= N*logdetR

            
def _compute_cgf_for_gaussian_markov_chain(mumu, Lambda, logdet_Lambda, 
                                           logdet_v, N):
    """
    Compute CGF using the moments of the parents.
    """
        
    g0 = -0.5 * np.einsum('...ij,...ij->...', mumu, Lambda)
        
    g1 = 0.5 * logdet_Lambda
    if np.ndim(logdet_v) == 1:
        g1 = g1 + 0.5 * (N-1) * np.sum(logdet_v, axis=-1)
    elif np.shape(logdet_v)[-2] == 1:
        g1 = g1 + 0.5 * (N-1) * np.sum(logdet_v, axis=(-1,-2))
    else:
        g1 = g1 + 0.5 * np.sum(logdet_v, axis=(-1,-2))

    return g0 + g1
    
class GaussianMarkovChainDistribution(TemplateGaussianMarkovChainDistribution):
    """
    Sub-classes implement distribution specific computations.
    """

    ndims_parents = [(1, 2), (2, 0), (1, 2), (0, 0)]

    def compute_message_to_parent(self, parent, index, u, u_mu, u_Lambda, u_A, u_v):
        """
        Compute a message to a parent.

        Parameters:
        -----------
        index : int
            Index of the parent requesting the message.
        u : list of ndarrays
            Moments of this node.
        u_mu : list of ndarrays
            Moments of parent `mu`.
        u_Lambda : list of ndarrays
            Moments of parent `Lambda`.
        u_A : list of ndarrays
            Moments of parent `A`.
        u_v : list of ndarrays
            Moments of parent `v`.
        u_N : list of ndarrays
            Moments of parent `N`.
        """
        
        if index == 0:   # mu
            raise NotImplementedError()
        elif index == 1: # Lambda
            raise NotImplementedError()
        elif index == 2: # A
            XnXn = u[1]
            XpXn = u[2]
            v = u_v[0]
            m0 = v[...,np.newaxis] * XpXn.swapaxes(-1,-2)
            # The following message matrix could be huge, so let's use a help
            # function which computes sum(v*XnXn) without computing the huge
            # v*XnXn explicitly.
            m1 = -0.5 * message_sum_multiply(parent.plates,
                                             parent.dims[1],
                                             v[...,np.newaxis,np.newaxis],
                                             XnXn[..., :-1, np.newaxis, :, :])
                                      
            #m1 = -0.5 * v[...,np.newaxis,np.newaxis] * XnXn[..., :-1, np.newaxis, :, :]
        elif index == 3: # v
            XnXn = u[1] # (...,N,D,D)
            XpXn = u[2] # (...,N-1,D,D)
            A = u_A[0]  # (...,N-1,D,D)
            AA = u_A[1] # (...,N-1,D,D,D)
            m0 = (- 0.5*np.einsum('...ii->...i', XnXn[...,1:,:,:])
                  + np.einsum('...ik,...ki->...i', A, XpXn)
                  - 0.5*np.einsum('...ikl,...kl->...i', AA, XnXn[...,:-1,:,:]))
            m1 = 0.5
        elif index == 4: # N
            raise NotImplementedError()

        return [m0, m1]

    def compute_mask_to_parent(self, index, mask):

        if index == 0:   # mu
            return mask
        elif index == 1: # Lambda
            return mask
        elif index == 2: # A
            return mask[...,np.newaxis,np.newaxis]
        elif index == 3: # v
            return mask[...,np.newaxis,np.newaxis]
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_mu, u_Lambda, u_A, u_v, mask=True):
        """
        Compute the natural parameters using parents' moments.

        Parameters
        ----------
        u_parents : list of list of arrays
           List of parents' lists of moments.

        Returns
        -------
        phi : list of arrays
           Natural parameters.
        dims : tuple
           Shape of the variable part of phi.

        """

        # Dimensionality of the Gaussian states
        D = np.shape(u_mu[0])[-1]

        # Number of time instances in the process
        N = self.N
        
        # TODO/FIXME: Take into account plates!
        phi0 = np.zeros((N,D))
        phi1 = np.zeros((N,D,D))
        phi2 = np.zeros((N-1,D,D))

        # Parameters for x0
        mu = u_mu[0]         # (..., D)
        Lambda = u_Lambda[0] # (..., D, D)
        phi0[...,0,:] = np.einsum('...ik,...k->...i', Lambda, mu)
        phi1[...,0,:,:] = Lambda

        # Helpful variables (show shapes in comments)
        A = u_A[0]  # (..., N-1, D, D)
        AA = u_A[1] # (..., N-1, D, D, D)
        v = u_v[0]  # (..., N-1, D)

        # Diagonal blocks: -0.5 * (V_i + A_{i+1}' * V_{i+1} * A_{i+1})
        phi1[..., 1:, :, :] = v[...,np.newaxis]*np.identity(D)
        phi1[..., :-1, :, :] += np.einsum('...kij,...k->...ij', AA, v)
        phi1 *= -0.5

        # Super-diagonal blocks: 0.5 * A.T * V
        # However, don't multiply by 0.5 because there are both super- and
        # sub-diagonal blocks (sum them together)
        phi2[..., :, :, :] = np.einsum('...ji,...j->...ij', A, v)

        return (phi0, phi1, phi2)

    def compute_cgf_from_parents(self, u_mu, u_Lambda, u_A, u_v):
        """
        Compute CGF using the moments of the parents.
        """
        return _compute_cgf_for_gaussian_markov_chain(u_mu[1],
                                                      u_Lambda[0],
                                                      u_Lambda[1],
                                                      u_v[1],
                                                      self.N)



class GaussianMarkovChain(_TemplateGaussianMarkovChain):
    r"""
    VMP node for Gaussian Markov chain.

    Parents are:
    `mu` is the mean of x0 (Gaussian)
    `Lambda` is the precision of x0 (Wishart)
    `A` is the dynamic matrix (Gaussian)
    `v` is the diagonal precision of the innovation (Gamma)
    An additional dummy parent is created:
    'N' is the number of time instances

    Output is Gaussian variables.

    Time dimension is over the last plate.

    Hmm.. The number of time instances is one more than the plates in
    A and V. Input N -> Output N+1.

    .. bayesnet::

       \node[latent] (x1) {$\mathbf{x}_1$};
       \node[latent, right=of x1] (x2) {$\mathbf{x}_2$};
       \node[right=of x2] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$\mathbf{x}_n$};
       \edge {x1}{x2};
       \edge {x2}{dots};
       \edge {dots}{xn};


    See also
    --------
    bayespy.inference.vmp.nodes.gaussian.Gaussian
    bayespy.inference.vmp.nodes.wishart.Wishart

    """

    _parent_statistics = (GaussianStatistics(1),
                          WishartStatistics(),
                          GaussianStatistics(1),
                          GammaStatistics())

    @ensureparents
    @useconstructor
    def __init__(self, mu, Lambda, A, v, n=None, **kwargs):
        """
        `mu` is the mean of x_0
        `Lambda` is the precision of x_0
        `A` is the dynamic matrix
        `v` is the diagonal precision of the innovation
        """
        
        # Construct
        super().__init__(mu, Lambda, A, v, **kwargs)

    @classmethod
    def _constructor(cls, mu, Lambda, A, v, n=None, **kwargs):
        """
        Constructs distribution and statistics objects.
        
        Compute the dimensions of phi and u.

        The plates and dimensions of the parents should be:
        mu:     (...)                    and D-dimensional
        Lambda: (...)                    and D-dimensional
        A:      (...,1,D) or (...,N-1,D) and D-dimensional
        v:      (...,1,D) or (...,N-1,D) and 0-dimensional
        N:      ()                       and 0-dimensional (dummy parent)

        Check that the dimensionalities of the parents are proper.
        For instance, A should be a collection of DxD matrices, thus
        the dimensionality and the last plate should both equal D.
        Similarly, `v` should be a collection of diagonal innovation
        matrix elements, thus the last plate should equal D.
        """
        
        n_A = 1
        if len(A.plates) >= 2:
            n_A = A.plates[-2]
        n_v = 1
        if len(v.plates) >= 2:
            n_v = v.plates[-2]
        if n_v != n_A and n_v != 1 and n_A != 1:
            raise Exception("Plates of A and v are giving different number of time instances")
        n_A = max(n_v, n_A)
        if n is None:
            if n_A == 1:
                raise Exception("""The number of time instances could not be determined
                                 automatically. Give the number of
                                 time instances.""")
            n = n_A + 1
        elif n_A != 1 and n_A+1 != n:
            raise Exception("The number of time instances must match "
                            "the number of last plates of parents: "
                            "%d != %d+1" % (n, n_A))
                                
        D = mu.dims[0][0]
        #M = N.get_moments()[0]
        M = n

        # Check mu
        if mu.dims != ( (D,), (D,D) ):
            raise Exception("First parent has wrong dimensionality")
        # Check Lambda
        if Lambda.dims != ( (D,D), () ):
            raise Exception("Second parent has wrong dimensionality")
        # Check A
        if A.dims != ( (D,), (D,D) ):
            raise Exception("Third parent has wrong dimensionality")
        if len(A.plates) == 0 or A.plates[-1] != D:
            raise Exception("Third parent should have a last plate "
                            "equal to the dimensionality of the "
                            "system.")
        if (len(A.plates) >= 2 
            and A.plates[-2] != 1
            and A.plates[-2] != M-1):
            raise ValueError("The second last plate of the third "
                             "parent should have length equal to one or "
                             "N-1, where N is the number of time "
                             "instances.")
        # Check v
        if v.dims != ( (), () ):
            raise Exception("Fourth parent has wrong dimensionality")
        if len(v.plates) == 0 or v.plates[-1] != D:
            raise Exception("Fourth parent should have a last plate "
                            "equal to the dimensionality of the "
                            "system.")
        if (len(v.plates) >= 2 
            and v.plates[-2] != 1
            and v.plates[-2] != M-1):
            raise ValueError("The second last plate of the fourth "
                             "parent should have length equal to one or "
                             "N-1 where N is the number of time "
                             "instances.")

        
        dims = ( (M,D), (M,D,D), (M-1,D,D) )
        distribution = GaussianMarkovChainDistribution(M)

        return ( dims,
                 distribution, 
                 cls._statistics, 
                 cls._parent_statistics)

    
    def _plates_to_parent(self, index):
        """
        Computes the plates of this node with respect to a parent.

        If this node has plates (...), the latent dimensionality is D
        and the number of time instances is N, the plates with respect
        to the parents are:
          mu:     (...)
          Lambda: (...)
          A:      (...,N-1,D)
          v:      (...,N-1,D)

        Parameters:
        -----------
        index : int
            The index of the parent node to use.
        """

        N = self.dims[0][0]
        D = self.dims[0][1]
        
        if index == 0:   # mu
            return self.plates
        elif index == 1: # Lambda
            return self.plates
        elif index == 2: # A
            return self.plates + (N-1,D)
            #raise NotImplementedError()
        elif index == 3: # v
            return self.plates + (N-1,D)
        elif index == 4: # N
            return ()
        raise ValueError("Invalid parent index.")
        #raise NotImplementedError()

    def _plates_from_parent(self, index):
        """
        Compute the plates using information of a parent node.

        If the plates of the parents are:
          mu:     (...)
          Lambda: (...)
          A:      (...,N-1,D)
          v:      (...,N-1,D)
          N:      ()
        the resulting plates of this node are (...)

        Parameters
        ----------
        index : int
            Index of the parent to use.
        """
        if index == 0:   # mu
            return self.parents[0].plates
        elif index == 1: # Lambda
            return self.parents[1].plates
        elif index == 2: # A
            return self.parents[2].plates[:-2]
        elif index == 3: # v
            return self.parents[3].plates[:-2]
        elif index == 4: # N
            return ()
        raise ValueError("Invalid parent index.")


class DriftingGaussianMarkovChainDistribution(TemplateGaussianMarkovChainDistribution):
    """
    Sub-classes implement distribution specific computations.
    """

    ndims_parents = ( (1, 2), 
                      (2, 0),
                      (2, 4),
                      (1, 2),
                      (0, 0) )

    def compute_message_to_parent(self, parent, index, u, u_mu, u_Lambda, u_B,
                                   u_S, u_v):
        """
        Compute a message to a parent.

        Parameters:
        -----------
        index : int
            Index of the parent requesting the message.
        u : list of ndarrays
            Moments of this node.
        u_mu : list of ndarrays
            Moments of parent `mu`.
        u_Lambda : list of ndarrays
            Moments of parent `Lambda`.
        u_B : list of ndarrays
            Moments of parent `B`.
        u_S : list of ndarrays
            Moments of parent `S`.
        u_v : list of ndarrays
            Moments of parent `v`.
        """
        
        if index == 0:   # mu
            raise NotImplementedError()
        elif index == 1: # Lambda
            raise NotImplementedError()
        elif index == 2: # B, (...,D)x(D,K)
            XnXn = u[1] # (...,N,D,D)
            XpXn = u[2] # (...,N,D,D)
            S = u_S[0]  # (...,N,K)
            SS = u_S[1] # (...,N,K,K)
            v = u_v[0]  # (...,N,D)

            # m0: (...,D,D,K)
            m0 = np.einsum('...nji,...nk,...ni->...ijk',
                           XpXn,
                           np.atleast_2d(S),
                           np.atleast_2d(v))
            
            # m1: (...,D,D,K,D,K)

            if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")

            m1 = np.einsum('...nij,...nkl->...ikjl',
                           XnXn[...,:-1,:,:],
                           np.atleast_3d(SS))
            m1 = -0.5 * np.einsum('...ikjl,...d->...dikjl',
                                  m1,
                                  np.atleast_2d(v)[...,0,:])

        elif index == 3: # S, (...,N-1)x(K)
            XnXn = u[1] # (...,N,D,D)
            XpXn = u[2] # (...,N,D,D)
            B = u_B[0]  # (...,D,D,K)
            BB = u_B[1] # (...,D,D,K,D,K)
            v = u_v[0]  # (...,N,D)

            # m0: (...,N,K)
            m0 = np.einsum('...nji,...ijk,...ni->...nk',
                           XpXn,
                           B,
                           np.atleast_2d(v))
            
            # m1: (...,N,K,K)

            if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")
            m1 = np.einsum('...dikjl,...d->...ikjl',
                           BB,
                           np.atleast_2d(v)[...,0,:])
            m1 = -0.5 * np.einsum('...nij,...ikjl->...nkl',
                                  XnXn[...,:-1,:,:],
                                  m1)

        elif index == 4: # v
            raise NotImplementedError()
        elif index == 5: # N
            raise NotImplementedError()

        return [m0, m1]


    def compute_mask_to_parent(self, index, mask):

        if index == 0: # mu
            return mask
        elif index == 1: # Lambda
            return mask
        elif index == 2: # B
            return mask[...,np.newaxis] # new plate axis for D
        elif index == 3: # S
            return mask[...,np.newaxis] # new plate axis for N
        elif index == 4: # v
            return mask[...,np.newaxis,np.newaxis] # new plate axis for N and D
        elif index == 5: # N
            return mask
        else:
            raise ValueError("Invalid index")


    def compute_phi_from_parents(self, u_mu, u_Lambda, u_B, u_S, u_v,
                                 mask=True):
        """
        Compute the natural parameters using parents' moments.

        Parameters
        ----------
        u_parents : list of list of arrays
           List of parents' lists of moments.

        Returns
        -------
        phi : list of arrays
           Natural parameters.
        dims : tuple
           Shape of the variable part of phi.

        """

        # Dimensionality of the Gaussian states
        D = np.shape(u_mu[0])[-1]

        # Number of time instances in the process
        N = self.N

        # Helpful variables (show shapes in comments)
        mu = u_mu[0]         # (..., D)
        Lambda = u_Lambda[0] # (..., D, D)
        B = u_B[0]           # (..., D, D, K)
        BB = u_B[1]          # (..., D, D, K, D, K)
        S = u_S[0]           # (..., N-1, K) or (..., 1, K)
        SS = u_S[1]          # (..., N-1, K, K)
        v = u_v[0]           # (..., N-1, D) or (..., 1, D)

        # TODO/FIXME: Take into account plates!
        plates_phi0 = utils.broadcasted_shape(np.shape(mu)[:-1],
                                              np.shape(Lambda)[:-2])
        plates_phi1 = utils.broadcasted_shape(np.shape(Lambda)[:-2],
                                              np.shape(v)[:-2],
                                              np.shape(BB)[:-5],
                                              np.shape(SS)[:-3])
        plates_phi2 = utils.broadcasted_shape(np.shape(B)[:-3],
                                              np.shape(S)[:-2],
                                              np.shape(v)[:-2])
        phi0 = np.zeros(plates_phi0 + (N,D))
        phi1 = np.zeros(plates_phi1 + (N,D,D))
        phi2 = np.zeros(plates_phi2 + (N-1,D,D))

        # Parameters for x0
        phi0[...,0,:] = np.einsum('...ik,...k->...i', Lambda, mu)
        phi1[...,0,:,:] = Lambda


        # Diagonal blocks: -0.5 * (V_i + A_{i+1}' * V_{i+1} * A_{i+1})
        phi1[..., 1:, :, :] = v[...,np.newaxis]*np.identity(D)
        if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
            raise Exception("This implementation is not efficient if "
                            "innovation noise is time-dependent.")
            phi1[..., :-1, :, :] += np.einsum('...dikjl,...kl,...d->...ij', 
                                              BB[...,None,:,:,:,:,:],
                                              SS,
                                              v)
        else:
            # We know that S does not have the D plate so we can sum that plate
            # axis out
            v_BB = np.einsum('...dikjl,...d->...ikjl',
                             BB[...,None,:,:,:,:,:],
                             v)
            phi1[..., :-1, :, :] += np.einsum('...ikjl,...kl->...ij', 
                                              v_BB,
                                              SS)
            
        #phi1[..., :-1, :, :] += np.einsum('...kij,...k->...ij', AA, v)
        phi1 *= -0.5

        # Super-diagonal blocks: 0.5 * A.T * V
        # However, don't multiply by 0.5 because there are both super- and
        # sub-diagonal blocks (sum them together)
        phi2[..., :, :, :] = np.einsum('...jik,...k,...j->...ij', 
                                       B[...,None,:,:,:],
                                       S,
                                       v)
        #phi2[..., :, :, :] = np.einsum('...ji,...j->...ij', A, v)

        return (phi0, phi1, phi2)

    def compute_cgf_from_parents(self, u_mu, u_Lambda, u_B, u_S, u_v):
        """
        Compute CGF using the moments of the parents.
        """

        return _compute_cgf_for_gaussian_markov_chain(u_mu[1],
                                                      u_Lambda[0],
                                                      u_Lambda[1],
                                                      u_v[1],
                                                      self.N)



class DriftingGaussianMarkovChain(_TemplateGaussianMarkovChain):
    r"""
    VMP node for drifting Gaussian Markov chain.

    Parents are:
    `mu` is the mean of x0 (Gaussian)
    `Lambda` is the precision of x0 (Wishart)
    `B` is the template dynamic matrices
    `S` is the temporal weights for the template dynamic matrices
    `v` is the diagonal precision of the innovation (Gamma)
    An additional dummy parent is created:
    'N' is the number of time instances

    Not efficient if `v` is time dependent.

    Output is Gaussian Markov chain variables.

    .. bayesnet::

       TODO: FIX THIS

       \node[latent] (x1) {$\mathbf{x}_1$};
       \node[latent, right=of x1] (x2) {$\mathbf{x}_2$};
       \node[right=of x2] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$\mathbf{x}_n$};
       \edge {x1}{x2};
       \edge {x2}{dots};
       \edge {dots}{xn};


    See also
    --------
    bayespy.inference.vmp.nodes.gaussian_markov_chain.GaussianMarkovChain
    bayespy.inference.vmp.nodes.gaussian.Gaussian
    bayespy.inference.vmp.nodes.wishart.Wishart

    """

    _parent_statistics = (GaussianStatistics(1),
                          WishartStatistics(),
                          GaussianStatistics(2),
                          GaussianStatistics(1),
                          GammaStatistics())

    @ensureparents
    @useconstructor
    def __init__(self, mu, Lambda, B, S, v, n=None, **kwargs):
        """
        `mu` is the mean of x_0, (...)x(D)
        `Lambda` is the precision of x_0, (...)x(D,D)
        `B` is the dynamic matrices, (...,D)x(D,K)
        `S` is the temporal weights for the dynamic matrices, (...,N)x(K)
        `v` is the diagonal precision of the innovation, (...,D)x()
        """

        # Construct
        super().__init__(mu, Lambda, B, S, v, **kwargs)


    @classmethod
    def _constructor(cls, mu, Lambda, B, S, v, n=None, **kwargs):
        """
        Constructs distribution and statistics objects.
        
        Compute the dimensions of phi and u.

        The plates and dimensions of the parents should be:
        mu:     (...)                    and D-dimensional
        Lambda: (...)                    and D-dimensional
        B:      (...,D)                  and D-dimensional
        S:      (...,N-1)                and D-dimensional
        v:      (...,1,D) or (...,N-1,D) and 0-dimensional
        N:      ()                       and 0-dimensional (dummy parent)

        Check that the dimensionalities of the parents are proper.
        """

        # A dummy wrapper for the number of time instances.
        n_S = 1
        if len(S.plates) >= 1:
            n_S = S.plates[-1]
        n_v = 1
        if len(v.plates) >= 2:
            n_v = v.plates[-2]
        if n_v != n_S and n_v != 1 and n_S != 1:
            raise Exception(
                "Plates of A and v are giving different number of time "
                "instances")
        n_S = max(n_v, n_S)
        if n is None:
            if n_S == 1:
                raise Exception(
                    "The number of time instances could not be determined "
                    "automatically. Give the number of time instances.")
                                 
            n = n_S + 1
        elif n_S != 1 and n_S+1 != n:
            raise Exception(
                "The number of time instances must match the number of last "
                "plates of parents:" "%d != %d+1" 
                % (n, n_S))
                                
        D = mu.dims[0][0]
        K = B.dims[0][-1]
        M = n #N.get_moments()[0]

        # Check mu
        if mu.dims != ( (D,), (D,D) ):
            raise ValueError("First parent has wrong dimensionality")
        # Check Lambda
        if Lambda.dims != ( (D,D), () ):
            raise ValueError("Second parent has wrong dimensionality")
        # Check B
        if B.dims != ( (D,K), (D,K,D,K) ):
            raise ValueError("Third parent has wrong dimensionality")
        if len(B.plates) == 0 or B.plates[-1] != D:
            raise ValueError("Third parent should have a last plate "
                             "equal to the dimensionality of the "
                             "system.")
        if S.dims != ( (K,), (K,K) ):
            raise ValueError("Fourth parent has wrong dimensionality %s, "
                             "should be %s"
                             % (S.dims, ( (K,), (K,K) )))
        if (len(S.plates) >= 1
            and S.plates[-1] != 1
            and S.plates[-1] != M-1):
            raise ValueError("The last plate of the fourth "
                             "parent should have length equal to one or "
                             "N-1, where N is the number of time "
                             "instances.")
        # Check v
        if v.dims != ( (), () ):
            raise Exception("Fifth parent has wrong dimensionality")
        if len(v.plates) == 0 or v.plates[-1] != D:
            raise Exception("Fifth parent should have a last plate "
                            "equal to the dimensionality of the "
                            "system.")
        if (len(v.plates) >= 2 
            and v.plates[-2] != 1
            and v.plates[-2] != M-1):
            raise ValueError("The second last plate of the fifth "
                             "parent should have length equal to one or "
                             "N-1 where N is the number of time "
                             "instances.")

        
        dims = ( (M,D), (M,D,D), (M-1,D,D) )
        distribution = DriftingGaussianMarkovChainDistribution(M)

        return (dims,
                distribution,
                cls._statistics,
                cls._parent_statistics)
    

    def _plates_to_parent(self, index):
        """
        Computes the plates of this node with respect to a parent.

        If this node has plates (...), the latent dimensionality is D
        and the number of time instances is N, the plates with respect
        to the parents are:
          mu:     (...)
          Lambda: (...)
          A:      (...,N-1,D)
          v:      (...,N-1,D)

        Parameters:
        -----------
        index : int
            The index of the parent node to use.
        """

        N = self.dims[0][0]
        D = self.dims[0][1]
        
        if index == 0:   # mu
            return self.plates
        elif index == 1: # Lambda
            return self.plates
        elif index == 2: # B
            return self.plates + (D,)
        elif index == 3: # S
            return self.plates + (N-1,)
        elif index == 4: # v
            return self.plates + (N-1,D)
        elif index == 5: # N
            return ()
        else:
            raise ValueError("Invalid parent index.")

    def _plates_from_parent(self, index):
        """
        Compute the plates using information of a parent node.

        If the plates of the parents are:
          mu:     (...)
          Lambda: (...)
          B:      (...,D)
          S:      (...,N-1)
          v:      (...,N-1,D)
          N:      ()
        the resulting plates of this node are (...)

        Parameters
        ----------
        index : int
            Index of the parent to use.
        """
        if index == 0:   # mu
            return self.parents[0].plates
        elif index == 1: # Lambda
            return self.parents[1].plates
        elif index == 2: # B, remove last plate D
            return self.parents[2].plates[:-1]
        elif index == 3: # S, remove last plate N-1
            return self.parents[3].plates[:-1]
        elif index == 4: # v, remove last plates N-1,D
            return self.parents[4].plates[:-2]
        else:
            raise ValueError("Invalid parent index.")


            


class _MarkovChainToGaussian(Deterministic):
    """
    Transform a Gaussian Markov chain node into a Gaussian node.

    This node is deterministic.
    """

    _statistics = GaussianStatistics(1)
    _parent_statistics = (GaussianMarkovChainStatistics(),)

    def __init__(self, X, **kwargs):

        # Check for constant n
        if utils.is_numeric(X):
            X = Constant(GaussianMarkovChain)(X)

        # Make the time dimension a plate dimension...
        #plates = X.plates + (X.dims[0][0],)
        # ... and remove it from the variable dimensions
        dims = ( X.dims[0][-1:], X.dims[1][-2:] )
        super().__init__(X,
        #plates=plates,
                         dims=dims,
                         **kwargs)

    def _plates_to_parent(self, index):
        """
        Return the number of plates to the parent node.

        Normally, the parent sees the same number of plates as the
        node itself.  However, now that one of the variable dimensions
        of the parents corresponds to a plate in this node, it is
        necessary to fix it here: the last plate is ignored when
        calculating plates with respect to the parent.

        Parent:
        Plates = (...)
        Dims = (N, ...)
        This node:
        Plates = (..., N)
        Dims = (...)
        """
        return self.plates[:-1]
        
    def _plates_from_parent(self, index):
        # Sub-classes may want to overwrite this if they manipulate plates
        if index != 0:
            raise ValueError("Invalid parent index.")

        parent = self.parents[0]
        plates = parent.plates + (parent.dims[0][0],)
        return plates

    def _compute_moments(self, u):
        """
        Transform the moments of a GMC to moments of a Gaussian.

        There is no need to worry about the plates and variable
        dimensions because the child node is free to interpret the
        axes as it pleases.  However, the Gaussian moments contain
        only <X(n)> and <X(n)*X(n)> but not <X(n-1)X(n)>, thus the
        last moment is discarded.
        """

        # Get the moments from the parent Gaussian Markov Chain
        #u = self.parents[0].get_moments() #message_to_child()

        # Send only moments <X(n)> and <X(n)X(n)> but not <X(n-1)X(n)>
        return u[:2]

    def _compute_mask_to_parent(self, index, mask):
        # Remove the last axis of the mask
        if np.ndim(mask) >= 1:
            mask = np.any(mask, axis=-1)
        return mask
        

    @staticmethod
    def _compute_message_to_parent(index, m_children, *u_parents):
        """
        Transform a message to a Gaussian into a message to a GMC.

        The messages to a Gaussian are almost correct, there are only
        two minor things to be done:
        
        1) The last plate is changed into a variable/time dimension.
        Because a message mask is applied for plates only, the last
        axis of the mask must be applied to the message because the
        last plate is changed to a variable/time dimension.
        
        2) Because the message does not contain <X(n-1)X(n)> part,
        we'll put the last/third message to None meaning that it is
        empty.

        Parameters:
        -----------
        index : int
            Index of the parent requesting the message.
        u_parents : list of list of ndarrays
            List of parents' moments.

        Returns:
        --------
        m : list of ndarrays
            Message as a list of arrays.
        mask : boolean ndarray
            Mask telling which plates should be taken into account.
        """

        # Add the third empty message
        return [m_children[0], m_children[1], None]
