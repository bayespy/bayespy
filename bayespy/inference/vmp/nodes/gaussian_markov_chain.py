######################################################################
# Copyright (C) 2012-2013 Jaakko Luttinen
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

"""
This module contains VMP nodes for Gaussian Markov chains.
"""

import numpy as np

from bayespy.utils import utils

from .node import Node
from .variable import Variable
from .constant import Constant, ConstantNumeric
from .gaussian import Gaussian
from .wishart import Wishart
from .gamma import Gamma

class GaussianMarkovChain(Variable):
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

    # phi[0] is (N,D), phi[1] is (N,D,D), phi[2] is (N-1,D,D)
    ndims = (2, 3, 3)
    ndims_parents = [(1, 2), (2, 0), (1, 2), (0, 0)]
    # Observations are a set of vectors (thus 2-D matrix):
    ndim_observations = 2
    
    @staticmethod
    def compute_fixed_moments(x):
        raise NotImplementedError()

    @staticmethod
    def compute_phi_from_parents(u_parents):
        #def compute_phi_from_parents(u_mu, u_Lambda, u_A, u_v, N):
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

        u_mu = u_parents[0]
        u_Lambda = u_parents[1]
        u_A = u_parents[2]
        u_v = u_parents[3]
        u_N = u_parents[4]


        # Dimensionality of the Gaussian states
        D = np.shape(u_mu[0])[-1]

        # Number of time instances in the process
        N = u_N[0]
        
        # TODO/FIXME: Take into account plates!
        phi0 = np.zeros((N,D))
        phi1 = np.zeros((N,D,D))
        phi2 = np.zeros((N-1,D,D))

        # Parameters for x0
        mu = u_mu[0]
        Lambda = u_Lambda[0]
        phi0[...,0,:] = np.dot(Lambda, mu)
        phi1[...,0,:,:] = Lambda

        # TODO/FIXME: Take into account the covariance of A!
        A = u_A[0]
        v = u_v[0]

        # Diagonal blocks: -0.5 * (V_i + A_{i+1}' * V_{i+1} * A_{i+1})
        phi1[..., 1:, :, :] = v[...,np.newaxis]*np.identity(D)
        phi1[..., :-1, :, :] += np.einsum('...ki,...k,...kj->...ij', A, v, A)
        #phi1[..., :-1, :, :] += np.dot(A.T, v[...,np.newaxis]*A)
        phi1 *= -0.5

        # Super-diagonal blocks: 0.5 * A.T * V
        phi2[..., :, :, :] = np.einsum('...,...ji,...j->...ij', 0.5, A, v)

        return (phi0, phi1, phi2)

    @staticmethod
    def compute_g_from_parents(u_parents):
        raise NotImplementedError()

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        """
        Compute the moments and the cumulant-generating function.

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
        B = -2*phi[2]
        (CovXnXn, CovXpXn, Xn, ldet) = utils.block_banded_solve(A, B, y)
        
        # Compute moments
        u0 = Xn
        u1 = CovXnXn + Xn[...,:,np.newaxis] * Xn[...,np.newaxis,:]
        u2 = CovXpXn + Xn[...,:-1,:,np.newaxis] * Xn[...,1:,np.newaxis,:]
        u = [u0, u1, u2]

        # Compute cumulant-generating function
        g = -0.5 * np.einsum('...ij,...ij', u[0], phi[0]) + 0.5*ldet
        
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        raise NotImplementedError()

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        raise NotImplementedError()

    @staticmethod
    def compute_dims(mu, Lambda, A, v, N):
        """ Compute the dimensions of phi and u. """
        D = mu.dims[0][0]
        M = N.get_moments()[0]
        return ( (M,D), (M,D,D), (M-1,D,D) )

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    def __init__(self, mu, Lambda, A, v, N=None, plates=(), **kwargs):
        """
        `mu` is the mean of x_0
        `Lambda` is the precision of x_0
        `A` is the dynamic matrix
        `v` is the diagonal precision of the innovation
        """
        self.parameter_distributions = (Gaussian, Wishart, Gaussian, Gamma)
        
        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = Constant(Gaussian)(mu)

        # Check for constant Lambda
        if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
            Lambda = Constant(Wishart)(Lambda)

        # Check for constant A
        if np.isscalar(A) or isinstance(A, np.ndarray):
            A = Constant(Gaussian)(A)

        # Check for constant V
        if np.isscalar(v) or isinstance(v, np.ndarray):
            v = Constant(Gamma)(v)

        # You could check whether the dimensions of mu and Lambda
        # match (and Lambda is square)
        if Lambda.dims[0][-1] != mu.dims[0][-1]:
            raise Exception("Dimensionalities of mu and Lambda do not match.")

        # A dummy wrapper for the number of time instances.
        N_A = 1
        if len(A.plates) >= 2:
            N_A = A.plates[-2]
        N_v = 1
        if len(v.plates) >= 2:
            N_v = b.plates[-2]
        if N_v != N_A and N_v != 1 and N_A != 1:
            raise Exception("Plates of A and v are giving different number of time instances")
        N_A = max(N_v, N_A)
        if N is None:
            if N_A == 1:
                raise Exception("""The number of time instances could not be determined
                                 automatically. Give the number of
                                 time instances.""")
            N = ConstantNumeric(N_A+1, 0)
        else:
            if N_A != 1 and N_A+1 != N:
                raise Exception("""The number of time instances must match the number of last plates of
                                parents: %d != %d+1""" % (N, N_A))
            N = ConstantNumeric(N, 0)

        # Construct
        super().__init__(mu, Lambda, A, v, N,
                         plates=plates,
                         **kwargs)

    def random(self):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()

    def as_gaussian(self):
        return _MarkovChainToGaussian(self)


class _MarkovChainToGaussian(Node):
    """
    Transform a Gaussian Markov chain node into a Gaussian node.

    This node is deterministic.
    """

    def __init__(self, X, **kwargs):

        # Check for constant n
        if utils.is_numeric(X):
            X = Constant(GaussianMarkovChain)(X)

        # Make the time dimension a plate dimension...
        plates = X.plates + (X.dims[0][0],)
        # ... and remove it from the variable dimensions
        dims = ( X.dims[0][-1:], X.dims[1][-2:] )
        super().__init__(X,
                         plates=plates,
                         dims=dims,
                         **kwargs)

    def plates_to_parent(self, index):
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
        
    def get_moments(self):
        """
        Transform the moments of a GMC to moments of a Gaussian.

        There is no need to worry about the plates and variable
        dimensions because the child node is free to interpret the
        axes as it pleases.  However, the Gaussian moments contain
        only <X(n)> and <X(n)*X(n)> but not <X(n-1)X(n)>, thus the
        last moment is discarded.
        """

        # Get the moments from the parent Gaussian Markov Chain
        u = self.parents[0].message_to_child()

        # Send only moments <X(n)> and <X(n)X(n)> but not <X(n-1)X(n)>
        return u[:2]

    def get_message(self, index, u_parents):
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
        """
        
        (m, mask) = self.message_from_children()

        # TODO/FIXME: Apply and remove the last axis of the mask
        if np.ndim(mask) >= 1:
            mask = mask[:-1]

        # Add the third empty message
        m = [m[0], m[1], None]

        return (m, mask)
                





































class GaussianMarkovChainFull(Variable):
    r"""
    VMP node for Gaussian Markov chain.

    Parents are:
    `mu` is the mean of x0
    `Lambda` is the precision of x0
    `A` is the dynamic matrix
    `V` is the precision of the innovation
    An additional dummy parent is created:
    'N' is the number of time instances

    Output is Gaussian variables where each time instance is
    independent.

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

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    
    ## @staticmethod
    ## def compute_fixed_parameter_moments(*args):
    ##     """ Compute the moments of the distribution parameters for
    ##     fixed values."""
    ##     mu = args[0]
    ##     Lambda = args[1]
    ##     u_mu = Gaussian.compute_fixed_moments(mu)
    ##     u_Lambda = Wishart.compute_fixed_moments(Lambda)
    ##     return (u_mu, u_Lambda)

    @staticmethod
    def compute_fixed_moments(x):
        raise NotImplementedError()
        """ Compute moments for fixed x. """
        return [x, utils.m_outer(x,x)]

    @staticmethod
    def compute_phi_from_parents(u_mu, u_Lambda, u_A, u_V, N):
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
        
        # TODO/FIXME: Take into account plates!
        phi0 = np.zeros((N,D))
        phi1 = np.zeros((N,D,D))
        phi2 = np.zeros((N-1,D,D))

        # Parameters for x0
        mu = u_mu[0]
        Lambda = u_Lambda[0]
        phi0[...,0,:] = np.dot(Lambda, mu)
        phi1[...,0,:,:] = -0.5*Lambda

        # TODO/FIXME: Take into account the covariance of A!
        A = u_A[0]
        V = u_V[0]

        # Diagonal blocks
        phi1[..., 1:, :, :] = V
        phi1[..., :-1, :, :] += np.dot(A.T, np.dot(V, A))
        phi1 *= -0.5

        # Super-diagonal blocks
        phi2[..., :, :, :] = 0.5 * np.dot(A.T, V)

        return (phi0, phi1, phi2)

    @staticmethod
    def compute_g_from_parents(u_parents):
        raise NotImplementedError()
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * logdet_Lambda)
        ## g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
        ##      + 0.5 * np.sum(logdet_Lambda))
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        """
        Compute the moments and the cumulant-generating function.

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
        B = -2*phi[2]
        (CovXnXn, CovXpXn, Xn, ldet) = utils.block_banded_solve(A, B, y)
        
        # Compute moments
        u0 = Xn
        u1 = CovXnXn + np.dot(Xn, Xn.T)
        u2 = CovXpXn + np.dot(Xn[...,:-1,:], Xn[...,1:,:].T)
        u = [u0, u1, u2]
        
        # Compute cumulant-generating function
        k = np.shape(Xn)[-2] * np.shape(Xn)[-1]
        g = (-0.5 * np.einsum('...ij,...ij', u[0], phi[0]) + 0.5 *
             ldet)
        
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        raise NotImplementedError()
        k = np.shape(x)[-1]
        u = [x, utils.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        raise NotImplementedError()
        if index == 0:
            return [utils.m_dot(u_parents[1][0], u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = utils.m_outer(u[0], u_parents[0][0])
            return [-0.5 * (u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]

    @staticmethod
    def compute_dims(mu, Lambda, A, V, N):
        """ Compute the dimensions of phi and u. """
        D = mu.dims[0][0]
        M = N.get_moments()[0]
        return ( (M,D), (M,D,D), (M-1,D,D) )

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()
        d = np.shape(x)[-1]
        return [(d,), (d,d)]

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda, A, V, N=None, plates=(), **kwargs):
        """
        `mu` is the mean of x_0
        `Lambda` is the precision of x_0
        `A` is the dynamic matrix
        `V` is the precision of the innovation
        """
        self.parameter_distributions = (Gaussian, Wishart, Gaussian, Wishart)
        
        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = Constant(Gaussian)(mu)

        # Check for constant Lambda
        if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
            Lambda = Constant(Wishart)(Lambda)

        # Check for constant A
        if np.isscalar(A) or isinstance(A, np.ndarray):
            A = Constant(Gaussian)(A)

        # Check for constant V
        if np.isscalar(V) or isinstance(V, np.ndarray):
            V = Constant(Wishart)(V)

        # You could check whether the dimensions of mu and Lambda
        # match (and Lambda is square)
        if Lambda.dims[0][-1] != mu.dims[0][-1]:
            raise Exception("Dimensionalities of mu and Lambda do not match.")

        # A dummy wrapper for the number of time instances.
        M = utils.broadcasted_shape(mu.plates, Lambda.plates, A.plates, V.plates)
        if N is None:
            if len(M) == 0 or M[-1] == 1:
                raise Exception("Give the number of time instances")
            N = ConstantNumeric(M[-1], 0)
        else:
            if len(M) > 0 and M[-1] != 1 and M[-1] != N:
                raise Exception("Number of time instances must match the number of last plates of parents")
            N = ConstantNumeric(N, 0)

        # Construct
        super().__init__(mu, Lambda, A, V, N,
                         plates=plates,
                         **kwargs)

    def random(self):
        raise NotImplementedError()
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = utils.m_chol(-2*self.phi[1])
        mu = self.u[0]
        z = np.random.normal(0, 1, self.get_shape(0))
        # Compute mu + U'*z
        #return mu + np.einsum('...ij,...i->...j', U, z)
        #scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=False)
        z = utils.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
        #return self.u[0] + utils.m_chol_solve(U, z)

    def show(self):
        raise NotImplementedError()
