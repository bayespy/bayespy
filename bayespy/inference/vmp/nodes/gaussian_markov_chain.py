################################################################################
# Copyright (C) 2012-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
This module contains VMP nodes for Gaussian Markov chains.
"""

import numpy as np
import scipy

from bayespy.utils import misc
from bayespy.utils import linalg

from .node import Node, message_sum_multiply
from .deterministic import Deterministic
from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution
from .expfamily import useconstructor
from .gaussian import (Gaussian,
                       GaussianMoments,
                       GaussianWishartMoments,
                       GaussianGammaMoments,
                       WrapToGaussianGamma,
                       WrapToGaussianWishart)
from .wishart import Wishart, WishartMoments
from .gamma import Gamma, GammaMoments
from .categorical import CategoricalMoments
from .node import Moments, ensureparents


class GaussianMarkovChainMoments(Moments):


    def __init__(self, N, D):
        self.N = N
        self.D = D
        return super().__init__()


    def compute_fixed_moments(self, x):
        u0 = x
        u1 = x[...,:,np.newaxis] * x[...,np.newaxis,:]
        u2 = x[...,:-1,:,np.newaxis] * x[...,1:,np.newaxis,:]
        return [u0, u1, u2]


    def rotate(self, u, R, logdet=None):

        if logdet is None:
            logdet = np.linalg.slogdet(R)[1]

        N = np.shape(u[0])[-2]

        # Transform moments and g
        u0 = linalg.mvdot(R, u[0])
        u1 = linalg.dot(R, u[1], R.T)
        u2 = linalg.dot(R, u[2], R.T)
        u = [u0, u1, u2]
        dg = -N * logdet

        return (u, dg)


class TemplateGaussianMarkovChainDistribution(ExponentialFamilyDistribution):
    """
    Sub-classes implement distribution specific computations.
    """

    
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.moments = GaussianMarkovChainMoments(N, D)
        super().__init__()

    def compute_message_to_parent(self, parent, index, u_self, *u_parents):
        raise NotImplementedError()

    def compute_weights_to_parent(self, index, weights):
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

    def plates_to_parent(self, index, plates):
        """
        Computes the plates of this node with respect to a parent.

        Child classes must implement this.

        Parameters
        -----------
        index : int
            The index of the parent node to use.
        """
        raise NotImplementedError()

    def plates_from_parent(self, index, plates):
        """
        Compute the plates using information of a parent node.

        Child classes must implement this.

        Parameters
        ----------
        index : int
            Index of the parent to use.
        """
        raise NotImplementedError()


    def rotate(self, u, phi, R, inv=None, logdet=None):

        (u, dg) = self.moments.rotate(u, R, logdet=logdet)

        # It would be more efficient and simpler, if you just rotated the
        # moments and didn't touch phi. However, then you would need to call
        # update() before lower_bound_contribution. This is more error-safe.

        if inv is None:
            inv = np.linalg.inv(R)

        # Transform parameters
        phi0 = linalg.mvdot(inv.T, phi[0])
        phi1 = linalg.dot(inv.T, phi[1], inv)
        phi2 = linalg.dot(inv.T, phi[2], inv)
        phi = [phi0, phi1, phi2]

        return (u, phi, dg)


    def compute_rotation_bound(self, u, u_mu_Lambda, u_A_V, R, inv=None, logdet=None):

        (Lambda_mu, Lambda_mumu, Lambda, logdetLambda) = u_mu_Lambda
        (V_A, V_AA, V, logdetV) = u_A_V

        V = misc.make_diag(V, ndim=1)

        R_XnXn = linalg.dot(R, self.XnXn)
        R_XpXp = linalg.dot(R, self.XpXp)
        R_X0X0 = linalg.dot(R, self.X0X0)

        tracedot(dot(Lambda, R_X0X0), R.T)
        tracedot(dot(V, R_XnXn), R.T)
        tracedot(dot(V_AA, R_XpXp), R.T)
        tracedot(dot(V_A, R_XpXn), R.T)
        (N - 1) * logdetV
        2 * N * logdetR

        logp = random.gaussian_logpdf(
            Lambda_R_X0X0_R + V_R_XnXn_R,
            V_A_R_XpXn_R,
            V_AA_R_XpXp_R,
            (N - 1) * logdetV + 2 * N * logdetR
        )

        logH = random.gaussian_entropy(
            -2 * M * logdetR,
            0
        )

        dlogp

        dlogH

        return (L, dL)


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

    def random(self, *phi, plates=None):
        raise NotImplementedError()


def _compute_cgf_for_gaussian_markov_chain(mumu_Lambda, logdet_Lambda,
                                           logdet_nu, N):
    """
    Compute CGF using the moments of the parents.
    """

    g0 = -0.5 * mumu_Lambda #np.einsum('...ij,...ij->...', mumu, Lambda)

    g1 = 0.5 * logdet_Lambda
    if np.ndim(logdet_nu) == 1:
        g1 = g1 + 0.5 * (N-1) * np.sum(logdet_nu, axis=-1)
    elif np.shape(logdet_nu)[-2] == 1:
        g1 = g1 + 0.5 * (N-1) * np.sum(logdet_nu, axis=(-1,-2))
    else:
        g1 = g1 + 0.5 * np.sum(logdet_nu, axis=(-1,-2))

    return g0 + g1


class GaussianMarkovChainDistribution(TemplateGaussianMarkovChainDistribution):
    r"""
    Implementation of VMP formulas for Gaussian Markov chain

    The log probability density function of the prior:

    .. todo::  Fix inputs and their weight matrix in the equations.

    .. math::

       \log p(\mathbf{X} | \boldsymbol{\mu}, \mathbf{\Lambda},
       \mathbf{A}, \mathbf{B}, \boldsymbol{\nu})
       =& \log \mathcal{N}(\mathbf{x}_0|\boldsymbol{\mu}, \mathbf{\Lambda})
       + \sum^N_{n=1} \log \mathcal{N}(
         \mathbf{x}_n | \mathbf{Ax}_{n-1} + \mathbf{Bu}_n,
                        \mathrm{diag}(\boldsymbol{\nu}))
       \\
       =&
       - \frac{1}{2} \mathbf{x}_0^T \mathbf{\Lambda} \mathbf{x}_0
       + \frac{1}{2} \mathbf{x}_0^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \boldsymbol{\mu}^T \mathbf{\Lambda} \mathbf{x}_0
       - \frac{1}{2} \boldsymbol{\mu}^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \log|\mathbf{\Lambda}|
       \\
       &
       - \frac{1}{2} \sum^N_{n=1} \mathbf{x}_n^T \mathrm{diag}(\boldsymbol{\nu}) \mathbf{x}_n
       + \frac{1}{2} \sum^N_{n=1} \mathbf{x}_n^T \mathrm{diag}(\boldsymbol{\nu}) \mathbf{A} \mathbf{x}_{n-1}
       + \frac{1}{2} \sum^N_{n=1} \mathbf{x}_{n-1}^T\mathbf{A}^T \mathrm{diag}(\boldsymbol{\nu}) \mathbf{x}_n
       - \frac{1}{2} \sum^N_{n=1} \mathbf{x}_{n-1}^T\mathbf{A}^T \mathrm{diag}(\boldsymbol{\nu}) \mathbf{A} \mathbf{x}_{n-1}
       \\ &
       + \sum^N_{n=1} \sum^D_{d=1} \log\nu_d - \frac{1}{2} (N+1) D \log(2\pi)
       \\
       =&
       \begin{bmatrix}
         \mathbf{x}_0 \\ \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_{N-1} \\ \mathbf{x}_N
       \end{bmatrix}^T
       \begin{bmatrix}
         -\frac{1}{2}\mathbf{\Lambda} - \frac{1}{2}\mathbf{A}\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T
         &
         \frac{1}{2} \mathbf{A}^T\mathrm{diag}(\boldsymbol{\nu})
         &
         &
         &
         \\
         \frac{1}{2} \mathrm{diag}(\boldsymbol{\nu}) \mathbf{A}
         &
         -\frac{1}{2} \mathrm{diag}(\boldsymbol{\nu})
         - \frac{1}{2}\mathbf{A}^T\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T
         &
         \frac{1}{2} \mathbf{A}^T\mathrm{diag}(\boldsymbol{\nu})
         &
         &
         \\
         &
         \ddots
         &
         \ddots
         &
         \ddots
         &
         \\
         &
         &
         \frac{1}{2} \mathrm{diag}(\boldsymbol{\nu}) \mathbf{A}
         &
         -\frac{1}{2} \mathrm{diag}(\boldsymbol{\nu})
         - \frac{1}{2}\mathbf{A}^T\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T
         &
         \frac{1}{2} \mathbf{A}^T\mathrm{diag}(\boldsymbol{\nu})
         \\
         &
         &
         &
         \frac{1}{2} \mathrm{diag}(\boldsymbol{\nu}) \mathbf{A}
         &
         -\frac{1}{2} \mathrm{diag}(\boldsymbol{\nu})
       \end{bmatrix}
       \begin{bmatrix}
         \mathbf{x}_0 \\ \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_{N-1} \\ \mathbf{x}_N
       \end{bmatrix}
       \\
       &
       + \frac{1}{2} \mathbf{x}_0^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \boldsymbol{\mu}^T \mathbf{\Lambda} \mathbf{x}_0
       - \frac{1}{2} \boldsymbol{\mu}^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \log|\mathbf{\Lambda}|
       + \sum^N_{n=1} \sum^D_{d=1} \log\nu_d - \frac{1}{2} (N+1) D \log(2\pi)


    For simplicity, :math:`\boldsymbol{\nu}` and :math:`\mathbf{A}` are assumed
    not to depend on :math:`n` in the above equation, but this distribution
    class supports that dependency. One only needs to do the following
    replacements in the equations: :math:`\boldsymbol{\nu} \leftarrow \boldsymbol{\nu}_n` and
    :math:`\mathbf{A} \leftarrow \mathbf{A}_n`, where :math:`n=1,\ldots,N`.

    .. math::

       u(\mathbf{X}) &=
       \begin{bmatrix}
         \begin{bmatrix} \mathbf{x}_0 & \ldots & \mathbf{x}_N \end{bmatrix}
         \\
         \begin{bmatrix} \mathbf{x}_0\mathbf{x}_0^T & \ldots & \mathbf{x}_N\mathbf{x}_N^T \end{bmatrix}
         \\
         \begin{bmatrix} \mathbf{x}_0\mathbf{x}_1^T & \ldots & \mathbf{x}_{N-1}\mathbf{x}_N^T \end{bmatrix}
       \end{bmatrix}
       \\
       \phi(\boldsymbol{\mu}, \mathbf{\Lambda}, \mathbf{A}, \boldsymbol{\nu}) &=
       \begin{bmatrix}
         \begin{bmatrix}
           \mathbf{\Lambda} \boldsymbol{\mu} & \mathbf{0} & \ldots & \mathbf{0}
         \end{bmatrix}
         \\
         \begin{bmatrix}
           -\frac{1}{2}\mathbf{\Lambda} - \frac{1}{2} \mathbf{A}\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T &
           -\frac{1}{2}\mathrm{diag}(\boldsymbol{\nu}) - \frac{1}{2} \mathbf{A}\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T &
           \ldots &
           -\frac{1}{2}\mathrm{diag}(\boldsymbol{\nu}) - \frac{1}{2} \mathbf{A}\mathrm{diag}(\boldsymbol{\nu})\mathbf{A}^T &
           -\frac{1}{2}\mathrm{diag}(\boldsymbol{\nu})
         \end{bmatrix}
         \\
         \begin{bmatrix}
           \mathbf{A}^T \mathrm{diag}(\boldsymbol{\nu}) & \ldots & \mathbf{A}^T \mathrm{diag}(\boldsymbol{\nu}) 
         \end{bmatrix}
       \end{bmatrix}
       \\
       g(\boldsymbol{\mu}, \mathbf{\Lambda}, \mathbf{A}, \boldsymbol{\nu}) &=
       \frac{1}{2}\log|\mathbf{\Lambda}| + \frac{1}{2} \sum^N_{n=1}\sum^D_{d=1}\log\nu_d
       \\
       f(\mathbf{X}) &= -\frac{1}{2} (N+1) D \log(2\pi)

    The log probability denisty function of the posterior approximation:

    .. math::

       \log q(\mathbf{X}) &=
       \begin{bmatrix}
         \mathbf{x}_0
         \\
         \mathbf{x}_1
         \\
         \vdots
         \\
         \mathbf{x}_{N-1}
         \\
         \mathbf{x}_N
       \end{bmatrix}^T
       \begin{bmatrix}
         \mathbf{\Phi}_0^{(2)} & \frac{1}{2}\mathbf{\Phi}_1^{(3)} & & &
         \\
         \frac{1}{2}{\mathbf{\Phi}_1^{(3)}}^T & \mathbf{\Phi}_1^{(2)} & \frac{1}{2}\mathbf{\Phi}_2^{(3)} & &
         \\
         & \ddots & \ddots & \ddots &
         \\
         & & \frac{1}{2}{\mathbf{\Phi}_{N-1}^{(3)}}^T & \mathbf{\Phi}_{N-1}^{(2)} & \frac{1}{2}\mathbf{\Phi}_N^{(3)}
         \\
         & & & \frac{1}{2}{\mathbf{\Phi}_N^{(3)}}^T & \mathbf{\Phi}_N^{(2)}
       \end{bmatrix}
       \begin{bmatrix}
         \mathbf{x}_0
         \\
         \mathbf{x}_1
         \\
         \vdots
         \\
         \mathbf{x}_{N-1}
         \\
         \mathbf{x}_N
       \end{bmatrix}
       + \ldots

    """


    def compute_message_to_parent(self, parent, index, u, u_mu_Lambda, u_A_nu, *u_inputs):
        r"""
        Compute a message to a parent.

        Parameters
        ----------
        index : int
            Index of the parent requesting the message.
        u : list of ndarrays
            Moments of this node.
        u_mu_Lambda : list of ndarrays
            Moments of parents :math:`(\boldsymbol{\mu}, \mathbf{\Lambda})`.
        u_A_nu : list of ndarrays
            Moments of parents :math:`(\mathbf{A}, \boldsymbol{\nu})`.
        u_inputs : list of ndarrays
            Moments of input signals.
        """

        D = np.shape(u[0])[-1]

        if index == 0:   # (mu, Lambda) -- GaussianWishartMoments
            x0 = u[0][...,0,:]
            x0x0 = u[1][...,0,:,:]
            m0 = x0
            m1 = -0.5
            m2 = -0.5 * x0x0
            m3 = 0.5
            return [m0, m1, m2, m3]

        elif index == 1: # (A, nu) -- GaussianGammaMoments
            XnXn = u[1]
            XpXn = u[2]

            # (..., N-1, D, D)
            m0 = XpXn.swapaxes(-1,-2)
            # (..., N-1, D, D, D)
            m1 = -0.5 * XnXn[..., :-1, None, :, :]
            # (..., N-1, D)
            m2 = -0.5 * np.einsum('...ii->...i', XnXn[...,1:,:,:])
            # (..., N-1, D)
            m3 = 0.5

            if len(u_inputs):
                Xn = u[0]
                z = u_inputs[0][0]
                zz = u_inputs[0][1]
                D_inputs = np.shape(z)[-1]
                m0_B = Xn[...,1:,:,None] * z[...,None,:]
                m1_BB = -0.5 * zz[..., None, :, :]
                m1_AB = -0.5 * Xn[..., :-1, None, :, None] * z[..., None, None, :]

                # Construct full message arrays from blocks
                m0 = np.concatenate([m0, m0_B], axis=-1)
                row1 = np.concatenate([m1, m1_AB], axis=-1)
                row2 = np.concatenate([m1_AB.swapaxes(-1,-2), m1_BB], axis=-1)
                m1 = np.concatenate([row1, row2], axis=-2)

            return [m0, m1, m2, m3]


        #     m1 = 0.5
        elif index == 2: # input signals
            raise NotImplementedError()

        raise IndexError("Parent index out of bounds")


    def compute_weights_to_parent(self, index, weights):

        if index == 0:   # mu_Lambda
            return weights
        elif index == 1: # A_nu
            return weights[...,np.newaxis,np.newaxis]
        elif index == 2: # input signals
            return weights[...,np.newaxis]
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_mu_Lambda, u_A_nu, *u_inputs, mask=True):
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
        D = np.shape(u_mu_Lambda[0])[-1]

        # Number of time instances in the process
        N = self.N

        # Helpful variables (show shapes in comments)
        Lambda_mu = u_mu_Lambda[0]   # (..., D)
        Lambda = u_mu_Lambda[2]      # (..., D, D)
        nu_A = u_A_nu[0][...,:D]     # (..., N-1, D, D)
        nu_AA = u_A_nu[1][...,:D,:D] # (..., N-1, D, D, D)
        nu_B = u_A_nu[0][...,D:]     # (..., N-1, D, inputs)
        nu_BB = u_A_nu[1][...,D:,D:] # (..., N-1, D, inputs, inputs)
        nu_AB = u_A_nu[1][...,:D,D:] # (..., N-1, D, D, inputs)
        nu = u_A_nu[2]               # (..., N-1, D)
        # mu = u_mu[0]           # (..., D)
        # Lambda = u_Lambda[0]   # (..., D, D)
        # A = u_A[0][...,:D]     # (..., N-1, D, D)
        # AA = u_A[1][...,:D,:D] # (..., N-1, D, D, D)
        # B = u_A[0][...,D:]     # (..., N-1, D, inputs)
        # BB = u_A[1][...,D:,D:] # (..., N-1, D, inputs, inputs)
        # AB = u_A[1][...,:D,D:] # (..., N-1, D, D, inputs)
        # v = u_v[0]             # (..., N-1, D)
        if len(u_inputs):
            inputs = u_inputs[0][0]
        else:
            inputs = None

        # Allocate memory (take into account effective plates)
        if inputs is not None:
            plates_phi0 = misc.broadcasted_shape(np.shape(Lambda_mu)[:-1],
                                                 np.shape(nu_B)[:-3],
                                                 np.shape(nu_AB)[:-4])
        else:
            plates_phi0 = misc.broadcasted_shape(np.shape(Lambda_mu)[:-1])

        plates_phi1 = misc.broadcasted_shape(np.shape(Lambda)[:-2],
                                             np.shape(nu_AA)[:-4])
        plates_phi2 = misc.broadcasted_shape(np.shape(nu_A)[:-3])

        phi0 = np.zeros(plates_phi0+(N,D))
        phi1 = np.zeros(plates_phi1+(N,D,D))
        phi2 = np.zeros(plates_phi2+(N-1,D,D))

        # Parameters for x0
        phi0[...,0,:] = Lambda_mu #np.einsum('...ik,...k->...i', Lambda, mu)
        phi1[...,0,:,:] = -0.5 * Lambda

        # Effect of the input signals
        if inputs is not None:
            phi0[...,1:,:] += np.einsum('...ij,...j->...i', nu_B, inputs)
            phi0[...,:-1,:] -= np.einsum(
                '...ij,...j->...i',
                np.sum(nu_AB, axis=-3),
                inputs
            )

        # Diagonal blocks: -0.5 * (V_i + A_{i+1}' * V_{i+1} * A_{i+1})
        phi1[..., 1:, :, :] = -0.5 * misc.diag(nu, ndim=1)
        phi1[..., :-1, :, :] += -0.5 * np.sum(nu_AA, axis=-3) #np.einsum('...kij,...k->...ij', AA, v)
        #phi1 *= -0.5

        # Super-diagonal blocks: 0.5 * A.T * V
        # However, don't multiply by 0.5 because there are both super- and
        # sub-diagonal blocks (sum them together)
        phi2[..., :, :, :] = linalg.transpose(nu_A, ndim=1) # np.einsum('...ji,...j->...ij', A, v)

        return (phi0, phi1, phi2)

    def compute_cgf_from_parents(self, u_mu_Lambda, u_A_nu, *u_inputs):
        """
        Compute CGF using the moments of the parents.
        """
        g = _compute_cgf_for_gaussian_markov_chain(u_mu_Lambda[1],
                                                   u_mu_Lambda[3],
                                                   u_A_nu[3],
                                                   self.N)

        if len(u_inputs):
            D = np.shape(u_mu_Lambda[0])[-1]
            uu = u_inputs[0][1]
            nu_BB = u_A_nu[1][...,D:,D:]
            nu = u_A_nu[2]
            #BB_v = np.einsum('...d,...dij->...ij', v, BB)
            g_inputs = -0.5 * np.einsum(
                '...ij,...ij->...',
                uu,
                np.sum(nu_BB, axis=-3)
                #BB_v
            )
            # Sum over time axis
            if np.ndim(g_inputs) == 0 or np.shape(g_inputs)[-1] == 1:
                g_inputs *= self.N
            else:
                g_inputs = np.sum(g_inputs, axis=-1)
            g = g + g_inputs

        return g


    def plates_to_parent(self, index, plates):
        """
        Computes the plates of this node with respect to a parent.

        If this node has plates (...), the latent dimensionality is D
        and the number of time instances is N, the plates with respect
        to the parents are:
          (mu, Lambda): (...)
          (A, nu):      (...,N-1,D)

        Parameters
        ----------
        index : int
            The index of the parent node to use.
        """

        if index == 0:   # (mu, Lambda)
            return plates
        elif index == 1: # (A, nu)
            return plates + (self.N-1, self.D)
        elif index == 2: # input signals
            return plates + (self.N-1,)
        else:
            raise ValueError("Invalid parent index.")

    def plates_from_parent(self, index, plates):
        """
        Compute the plates using information of a parent node.

        If the plates of the parents are:
          (mu, Lambda): (...)
          (A, nu):      (...,N-1,D)
        the resulting plates of this node are (...)

        Parameters
        ----------
        index : int
            Index of the parent to use.
        """
        if index == 0:   # (mu, Lambda)
            return plates
        elif index == 1: # (A, nu)
            return plates[:-2]
        elif index == 2: # input signals
            return plates[:-1]
        else:
            raise ValueError("Invalid parent index.")


class GaussianMarkovChain(_TemplateGaussianMarkovChain):
    r"""
    Node for Gaussian Markov chain random variables.

    In a simple case, the graphical model can be presented as:

    .. bayesnet::

       \tikzstyle{latent} += [minimum size=30pt];

       \node[latent] (x0) {$\mathbf{x}_0$};
       \node[latent, right=of x0] (x1) {$\mathbf{x}_1$};
       \node[right=of x1] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$\mathbf{x}_{N-1}$};
       \edge {x0}{x1};
       \edge {x1}{dots};
       \edge {dots}{xn};

       \node[latent, above left=1 and 0.1 of x0] (mu) {$\boldsymbol{\mu}$};
       \node[latent, above right=1 and 0.1 of x0] (Lambda) {$\mathbf{\Lambda}$};
       \node[latent, above left=1 and 0.1 of dots] (A) {$\mathbf{A}$};
       \node[latent, above right=1 and 0.1 of dots] (nu) {$\boldsymbol{\nu}$};
       \edge {mu,Lambda} {x0};
       \edge {A,nu} {x1,dots,xn};

    where :math:`\boldsymbol{\mu}` and :math:`\mathbf{\Lambda}` are the mean and
    the precision matrix of the initial state, :math:`\mathbf{A}` is the state
    dynamics matrix and :math:`\boldsymbol{\nu}` is the precision of the
    innovation noise.  It is possible that :math:`\mathbf{A}` and/or
    :math:`\boldsymbol{\nu}` are different for each transition instead of being
    constant.

    The probability distribution is

    .. math::

       p(\mathbf{x}_0, \ldots, \mathbf{x}_{N-1}) = p(\mathbf{x}_0)
       \prod^{N-1}_{n=1} p(\mathbf{x}_n | \mathbf{x}_{n-1})

    where

    .. math::

       p(\mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_0 | \boldsymbol{\mu}, \mathbf{\Lambda})
       \\
       p(\mathbf{x}_n|\mathbf{x}_{n-1}) &= \mathcal{N}(\mathbf{x}_n |
       \mathbf{A}_{n-1}\mathbf{x}_{n-1}, \mathrm{diag}(\boldsymbol{\nu}_{n-1})).

    Parameters
    ----------

    mu : Gaussian-like node or (...,D)-array
        :math:`\boldsymbol{\mu}`, mean of :math:`x_0`, :math:`D`-dimensional
        with plates (...)

    Lambda : Wishart-like node or (...,D,D)-array
        :math:`\mathbf{\Lambda}`, precision matrix of :math:`x_0`,
        :math:`D\times D` -dimensional with plates (...)

    A : Gaussian-like node or (D,D)-array or (...,1,D,D)-array or (...,N-1,D,D)-array
        :math:`\mathbf{A}`, state dynamics matrix, :math:`D`-dimensional with
        plates (D,) or (...,1,D) or (...,N-1,D)

    nu : gamma-like node or (D,)-array or (...,1,D)-array or (...,N-1,D)-array
        :math:`\boldsymbol{\nu}`, diagonal elements of the precision of the
        innovation process, plates (D,) or (...,1,D) or (...,N-1,D)

    n : int, optional
        :math:`N`, the length of the chain. Must be given if :math:`\mathbf{A}`
        and :math:`\boldsymbol{\nu}` are constant over time.

    See also
    --------

    Gaussian, GaussianARD, Wishart, Gamma, SwitchingGaussianMarkovChain,
    VaryingGaussianMarkovChain, CategoricalMarkovChain
    """


    def __init__(self, mu, Lambda, A, nu, n=None, inputs=None, **kwargs):
        """
        Create GaussianMarkovChain node.
        """
        super().__init__(mu, Lambda, A, nu, n=n, inputs=inputs, **kwargs)


    @classmethod
    def _constructor(cls, mu, Lambda, A, nu, n=None, inputs=None, **kwargs):
        """
        Constructs distribution and moments objects.

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

        mu_Lambda = WrapToGaussianWishart(mu, Lambda)
        A_nu = WrapToGaussianGamma(A, nu, ndim=1)

        D = mu_Lambda.dims[0][0]

        if inputs is not None:
            inputs = cls._ensure_moments(inputs, GaussianMoments, ndim=1)

        # Check whether to use input signals or not
        if inputs is None:
            _parent_moments = (GaussianWishartMoments((D,)),
                               GaussianGammaMoments((D,)))
        else:
            K = inputs.dims[0][0]
            _parent_moments = (GaussianWishartMoments((D,)),
                               GaussianGammaMoments((D,)),
                               GaussianMoments((K,)))

        # Time instances from input signals
        if inputs is not None and len(inputs.plates) >= 1:
            n_inputs = inputs.plates[-1]
        else:
            n_inputs = 1
        # Time instances from state dynamics matrix
        if len(A_nu.plates) >= 2:
            n_A_nu = A_nu.plates[-2]
        else:
            n_A_nu = 1
        # Check consistency of the number of time instances
        if n_inputs != n_A_nu and n_inputs != 1 and n_A_nu != 1:
            raise Exception("Plates of parents are giving different number of time instances")
        n_parents = max(n_A_nu, n_inputs)
        if n is None:
            if n_parents == 1:
                raise Exception("The number of time instances could not be "
                                "determined automatically. Give the number of "
                                "time instances.")
            n = n_parents + 1
        elif n_parents != 1 and n_parents+1 != n:
            raise Exception("The number of time instances must match "
                            "the number of last plates of parents: "
                            "%d != %d+1" % (n, n_parents))

        # Dimensionality of the states
        D = mu_Lambda.dims[0][0]
        # Number of states
        M = n
        # Dimensionality of the inputs
        if inputs is None:
            D_inputs = 0
        else:
            D_inputs = inputs.dims[0][0]

        # Check (mu, Lambda)
        if mu_Lambda.dims != ( (D,), (), (D, D), () ):
            raise Exception("Initial state parameters have wrong dimensionality")
        # Check (A, nu)
        if A_nu.dims != ( (D+D_inputs,), (D+D_inputs,D+D_inputs), (), () ):
            raise Exception("Dynamics matrix has wrong dimensionality")
        if len(A_nu.plates) == 0 or A_nu.plates[-1] != D:
            raise Exception("Dynamics matrix should have a last plate "
                            "equal to the dimensionality of the "
                            "system.")
        if (len(A_nu.plates) >= 2
            and A_nu.plates[-2] != 1
            and A_nu.plates[-2] != M-1):
            raise ValueError("The second last plate of the dynamics matrix "
                             "should have length equal to one or "
                             "N-1, where N is the number of time "
                             "instances.")
        # Check input signals
        if inputs is not None:
            if inputs.dims != ( (D_inputs,), (D_inputs, D_inputs) ):
                raise ValueError("Input signals have wrong dimensionality")

        moments = GaussianMarkovChainMoments(M, D)
        dims = ( (M,D), (M,D,D), (M-1,D,D) )
        distribution = GaussianMarkovChainDistribution(M, D)

        if inputs is None:
            parents = [mu_Lambda, A_nu]
        else:
            parents = [mu_Lambda, A_nu, inputs]

        return ( parents,
                 kwargs,
                 dims,
                 cls._total_plates(kwargs.get('plates'),
                                   distribution.plates_from_parent(0, mu_Lambda.plates),
                                   distribution.plates_from_parent(1, A_nu.plates)),
                 distribution,
                 moments,
                 _parent_moments)


    def rotate(self, R, inv=None, logdet=None):

        # It would be more efficient and simpler, if you just rotated the
        # moments and didn't touch phi. However, then you would need to call
        # update() before lower_bound_contribution. This is more error-safe.
        (u, phi, dg) = self._distribution.rotate(
            self.u,
            self.phi,
            R,
            inv=inv,
            logdet=logdet
        )
        self.u = u
        self.phi = phi
        self.g = self.g + dg

        return


class VaryingGaussianMarkovChainDistribution(TemplateGaussianMarkovChainDistribution):
    """
    Sub-classes implement distribution specific computations.
    """


    def compute_message_to_parent(self, parent, index, u, u_mu, u_Lambda, u_B,
                                   u_S, u_v):
        """
        Compute a message to a parent.

        Parameters
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
            S = misc.atleast_nd(u_S[0], 2)  # (...,N,K)
            SS = misc.atleast_nd(u_S[1], 3) # (...,N,K,K)
            v = misc.atleast_nd(u_v[0], 2)  # (...,N,D)

            # m0: (...,D,D,K)
            m0 = np.einsum('...nji,...nk,...ni->...ijk',
                           XpXn,
                           S,
                           v)
            
            # m1: (...,D,D,K,D,K)

            if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")

            m1 = np.einsum('...nij,...nkl->...ikjl',
                           XnXn[...,:-1,:,:],
                           SS)
            m1 = -0.5 * np.einsum('...ikjl,...d->...dikjl',
                                  m1,
                                  v[...,0,:])

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


    def compute_weights_to_parent(self, index, weights):

        if index == 0: # mu
            return weights
        elif index == 1: # Lambda
            return weights
        elif index == 2: # B
            return weights[...,np.newaxis] # new plate axis for D
        elif index == 3: # S
            return weights[...,np.newaxis] # new plate axis for N
        elif index == 4: # v
            return weights[...,np.newaxis,np.newaxis] # new plate axis for N and D
        elif index == 5: # N
            return weights
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
        plates_phi0 = misc.broadcasted_shape(np.shape(mu)[:-1],
                                             np.shape(Lambda)[:-2])
        plates_phi1 = misc.broadcasted_shape(np.shape(Lambda)[:-2],
                                             np.shape(v)[:-2],
                                             np.shape(BB)[:-5],
                                             np.shape(SS)[:-3])
        plates_phi2 = misc.broadcasted_shape(np.shape(B)[:-3],
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

    def plates_to_parent(self, index, plates):
        """
        Computes the plates of this node with respect to a parent.

        If this node has plates (...), the latent dimensionality is D
        and the number of time instances is N, the plates with respect
        to the parents are:
          mu:     (...)
          Lambda: (...)
          A:      (...,N-1,D)
          v:      (...,N-1,D)

        Parameters
        -----------
        index : int
            The index of the parent node to use.
        """

        if index == 0:   # mu
            return plates
        elif index == 1: # Lambda
            return plates
        elif index == 2: # B
            return plates + (self.D,)
        elif index == 3: # S
            return plates + (self.N-1,)
        elif index == 4: # v
            return plates + (self.N-1,self.D)
        else:
            raise ValueError("Invalid parent index.")

    def plates_from_parent(self, index, plates):
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
            return plates
        elif index == 1: # Lambda
            return plates
        elif index == 2: # B, remove last plate D
            return plates[:-1]
        elif index == 3: # S, remove last plate N-1
            return plates[:-1]
        elif index == 4: # v, remove last plates N-1,D
            return plates[:-2]
        else:
            raise ValueError("Invalid parent index.")




class VaryingGaussianMarkovChain(_TemplateGaussianMarkovChain):
    r"""
    Node for Gaussian Markov chain random variables with time-varying dynamics.

    The node models a sequence of Gaussian variables
    :math:`\mathbf{x}_0,\ldots,\mathbf{x}_{N-1}` with linear Markovian dynamics.
    The time variability of the dynamics is obtained by modelling the state
    dynamics matrix as a linear combination of a set of matrices with
    time-varying linear combination weights.  The
    graphical model can be presented as:

    .. bayesnet::

       \tikzstyle{latent} += [minimum size=40pt];
       
       \node[latent] (x0) {$\mathbf{x}_0$};
       \node[latent, right=of x0] (x1) {$\mathbf{x}_1$};
       \node[right=of x1] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$\mathbf{x}_{N-1}$};
       \edge {x0}{x1};
       \edge {x1}{dots};
       \edge {dots}{xn};

       \node[latent, above left=1 and 0.1 of x0] (mu) {$\boldsymbol{\mu}$};
       \node[latent, above right=1 and 0.1 of x0] (Lambda) {$\mathbf{\Lambda}$};
       \node[det, below=of x1] (A0) {$\mathbf{A}_0$};
       \node[right=of A0] (Adots) {$\cdots$};
       \node[det, right=of Adots] (An) {$\mathbf{A}_{N-2}$};
       \node[latent, above=of dots] (nu) {$\boldsymbol{\nu}$};
       \edge {mu,Lambda} {x0};
       \edge {nu} {x1,dots,xn};
       \edge {A0} {x1};
       \edge {Adots} {dots};
       \edge {An} {xn};

       \node[latent, below=of A0] (s0) {$s_{0,k}$};
       \node[right=of s0] (sdots) {$\cdots$};
       \node[latent, right=of sdots] (sn) {$\mathbf{s}_{N-2,k}$};
       \node[latent, left=of s0] (B) {$\mathbf{B}_k$};
       \edge {B} {A0, Adots, An};
       \edge {s0} {A0};
       \edge {sdots} {Adots};
       \edge {sn} {An};

       \plate {K} {(B)(s0)(sdots)(sn)} {$k=0,\ldots,K-1$};

    where :math:`\boldsymbol{\mu}` and :math:`\mathbf{\Lambda}` are the mean and
    the precision matrix of the initial state, :math:`\boldsymbol{\nu}` is the
    precision of the innovation noise, and :math:`\mathbf{A}_n` are the state
    dynamics matrix obtained by mixing matrices :math:`\mathbf{B}_k` with
    weights :math:`s_{n,k}`.

    The probability distribution is

    .. math::

       p(\mathbf{x}_0, \ldots, \mathbf{x}_{N-1}) = p(\mathbf{x}_0)
       \prod^{N-1}_{n=1} p(\mathbf{x}_n | \mathbf{x}_{n-1})

    where
    
    .. math::

       p(\mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_0 | \boldsymbol{\mu}, \mathbf{\Lambda})
       \\
       p(\mathbf{x}_n|\mathbf{x}_{n-1}) &= \mathcal{N}(\mathbf{x}_n |
       \mathbf{A}_{n-1}\mathbf{x}_{n-1}, \mathrm{diag}(\boldsymbol{\nu})),
       \quad \text{for } n=1,\ldots,N-1,
       \\
       \mathbf{A}_n & = \sum^{K-1}_{k=0} s_{n,k} \mathbf{B}_k, \quad \text{for }
       n=0,\ldots,N-2.
       

    Parameters
    ----------
    
    mu : Gaussian-like node or (...,D)-array
        :math:`\boldsymbol{\mu}`, mean of :math:`x_0`, :math:`D`-dimensional
        with plates (...)
        
    Lambda : Wishart-like node or (...,D,D)-array
        :math:`\mathbf{\Lambda}`, precision matrix of :math:`x_0`,
        :math:`D\times D` -dimensional with plates (...)
        
    B : Gaussian-like node or (...,D,D,K)-array
        :math:`\{\mathbf{B}_k\}_{k=0}^{K-1}`, a set of state dynamics matrix,
        :math:`D \times K`-dimensional with plates (...,D)

    S : Gaussian-like node or (...,N-1,K)-array

        :math:`\{\mathbf{s}_0,\ldots,\mathbf{s}_{N-2}\}`, time-varying weights
        of the linear combination, :math:`K`-dimensional with plates (...,N-1)
        
    nu : gamma-like node or (...,D)-array
        :math:`\boldsymbol{\nu}`, diagonal elements of the precision of the
        innovation process, plates (...,D)

    n : int, optional
        :math:`N`, the length of the chain. Must be given if :math:`\mathbf{S}`
        does not have plates over the time domain (which would not make sense).

    See also
    --------
    
    Gaussian, GaussianARD, Wishart, Gamma, GaussianMarkovChain,
    SwitchingGaussianMarkovChain

    Notes
    -----

    Equivalent model block can be constructed with :class:`GaussianMarkovChain`
    by explicitly using :class:`SumMultiply` to compute the linear combination.
    However, that approach is not very efficient for large datasets because it
    does not utilize the structure of :math:`\mathbf{A}_n`, thus it explicitly
    computes huge moment arrays.

    References
    ----------

    :cite:`Luttinen:2014`
    
    """

    def __init__(self, mu, Lambda, B, S, nu, n=None, **kwargs):
        """
        Create VaryingGaussianMarkovChain node.
        """
        super().__init__(mu, Lambda, B, S, nu, n=n, **kwargs)


    @classmethod
    def _constructor(cls, mu, Lambda, B, S, v, n=None, **kwargs):
        """
        Constructs distribution and moments objects.
        
        Compute the dimensions of phi and u.

        The plates and dimensions of the parents should be:
        mu:     (...)                    and D-dimensional
        Lambda: (...)                    and D-dimensional
        B:      (...,D)                  and (D,K)-dimensional
        S:      (...,N-1)                and K-dimensional
        v:      (...,1,D) or (...,N-1,D) and 0-dimensional
        N:      ()                       and 0-dimensional (dummy parent)

        Check that the dimensionalities of the parents are proper.
        """

        mu = cls._ensure_moments(mu, GaussianMoments, ndim=1)
        Lambda = cls._ensure_moments(Lambda, WishartMoments, ndim=1)
        B = cls._ensure_moments(B, GaussianMoments, ndim=2)
        S = cls._ensure_moments(S, GaussianMoments, ndim=1)
        v = cls._ensure_moments(v, GammaMoments)

        (D, K) = B.dims[0]

        parent_moments = (
            GaussianMoments((D,)),
            WishartMoments((D,)),
            GaussianMoments((D, K)),
            GaussianMoments((K,)),
            GammaMoments()
        )

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
            raise ValueError("Third parent has wrong dimensionality {0}. Should be {1}.".format(B.dims[0], (D,K)))
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

        distribution = VaryingGaussianMarkovChainDistribution(M, D)
        moments = GaussianMarkovChainMoments(M, D)

        parents = [mu, Lambda, B, S, v]

        dims = ( (M,D), (M,D,D), (M-1,D,D) )

        return (parents,
                kwargs,
                dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, mu.plates),
                                  distribution.plates_from_parent(1, Lambda.plates),
                                  distribution.plates_from_parent(2, B.plates),
                                  distribution.plates_from_parent(3, S.plates),
                                  distribution.plates_from_parent(4, v.plates)),
                distribution,
                moments,
                parent_moments)



class SwitchingGaussianMarkovChainDistribution(TemplateGaussianMarkovChainDistribution):
    """
    Sub-classes implement distribution specific computations.
    """


    def __init__(self, N, D, K):
        self.K = K
        super().__init__(N, D)
        
    def compute_message_to_parent(self, parent, index, u, u_mu, u_Lambda, u_B,
                                   u_Z, u_v):
        """
        Compute a message to a parent.

        Parameters
        ----------
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
        u_Z : list of ndarrays
            Moments of parent `Z`.
        u_v : list of ndarrays
            Moments of parent `v`.
        """
        
        if index == 0:   # mu
            raise NotImplementedError()
        elif index == 1: # Lambda
            raise NotImplementedError()
        
        elif index == 2: # B, (...,K,D)x(D)
            
            XnXn = u[1]                   # (...,N,D,D)
            XpXn = u[2]                   # (...,N-1,D,D)
            Z = u_Z[0]                    # (...,N-1,K)
            v = misc.atleast_nd(u_v[0], 2)  # (...,N-1,D)

            # Check that there is no time-dependency in v and remove the axis
            if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")
            v = np.squeeze(v, axis=-2)

            # m0: (...,K,D,D)
            m0 = np.einsum('...nji,...nk,...i->...kij',
                           XpXn,
                           Z,
                           v)
            
            # m1: (...,K,D,D,D)

            m1 = np.einsum('...nij,...nk->...kij',
                           XnXn[...,:-1,:,:],
                           Z)
            m1 = -0.5 * np.einsum('...kij,...d->...kdij',
                                  m1,
                                  v)

            return [m0, m1]
    
        elif index == 3: # Z, (...,N-1)x(K)

            XnXn = u[1]                     # (...,N,D,D)
            XpXn = u[2]                     # (...,N-1,D,D)
            B = u_B[0]                      # (...,K,D,D)
            BB = u_B[1]                     # (...,K,D,D,D)
            v = misc.atleast_nd(u_v[0], 2)    # (...,N-1,D)
            logv = misc.atleast_nd(u_v[1], 2) # (...,N-1,D)

            # Check that there is no time-dependency in v and remove the axis
            if np.ndim(v) >= 2 and np.shape(v)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")
            v = np.squeeze(v, axis=-2)
            if np.ndim(logv) >= 2 and np.shape(logv)[-2] > 1:
                raise ValueError("Innovation noise is time dependent")
            logv = np.squeeze(logv, axis=-2)

            XnXn_v = np.einsum('...nii,...i->...n', 
                               XnXn[...,1:,:,:],
                               v)
            XpXn_v_B = np.einsum('...nil,...l,...kli->...nk',
                                 XpXn,
                                 v,
                                 B)
            BvB = np.einsum('...kdij,...d->...kij',
                            BB,
                            v)
            XpXp_BvB = np.einsum('...nij,...kij->...nk',
                                 XnXn[...,:-1,:,:],
                                 BvB)

            m0 = ( -0.5 * XnXn_v[...,None]
                   + XpXn_v_B
                   -0.5 * XpXp_BvB
                   +0.5 * np.sum(logv, axis=-1)[...,None,None]
                   -0.5 * self.D * np.log(2*np.pi) )

            return [m0]

        elif index == 4: # v
            raise NotImplementedError()
        elif index == 5: # N
            raise NotImplementedError()



    def compute_weights_to_parent(self, index, weights):

        if index == 0: # mu: (...)x(N,D) -> (...)x(D)
            return weights
        elif index == 1: # Lambda: (...)x(N,D) -> (...)x(D,D)
            return weights
        elif index == 2: # B: (...)x(N,D) -> (...,K,D)x(D)
            return weights[...,None,None]
        elif index == 3: # Z: (...)x(N,D) -> (...,N-1)x(K)
            return weights[...,None]
        elif index == 4: # v: (...)x(N,D) -> (...,N-1,D)x()
            return weights[...,None,None]
        else:
            raise ValueError("Invalid index")


    def compute_phi_from_parents(self, u_mu, u_Lambda, u_B, u_Z, u_v,
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
        mu = u_mu[0]                    # (..., D)
        Lambda = u_Lambda[0]            # (..., D, D)
        B = u_B[0]                      # (..., K, D, D)
        BB = u_B[1]                     # (..., K, D, D, D)
        Z = u_Z[0]                      # (..., N-1, K)
        v = misc.atleast_nd(u_v[0], 2) # (..., N-1, D) or (..., 1, D)

        # TODO/FIXME: Take into account plates!
        plates_phi0 = misc.broadcasted_shape(np.shape(mu)[:-1],
                                             np.shape(Lambda)[:-2])
        plates_phi1 = misc.broadcasted_shape(np.shape(Lambda)[:-2],
                                             np.shape(v)[:-2],
                                             np.shape(BB)[:-4],
                                             np.shape(Z)[:-2])
        plates_phi2 = misc.broadcasted_shape(np.shape(B)[:-3],
                                             np.shape(Z)[:-2],
                                             np.shape(v)[:-2])
        phi0 = np.zeros(plates_phi0 + (N,D))
        phi1 = np.zeros(plates_phi1 + (N,D,D))
        phi2 = np.zeros(plates_phi2 + (N-1,D,D))

        # Parameters for x0
        phi0[...,0,:] = np.einsum('...ik,...k->...i', Lambda, mu)
        phi1[...,0,:,:] = Lambda


        # Diagonal blocks: -0.5 * (V_i + A_{i+1}' * V_{i+1} * A_{i+1})
        phi1[..., 1:, :, :] = v[...,None]*np.identity(D)
        if np.shape(v)[-2] > 1:
            raise Exception("This implementation is not efficient if "
                            "innovation noise is time-dependent.")
            phi1[..., :-1, :, :] += np.einsum('...kdij,...nk,...nd->...nij', 
                                              BB[...,:,:,:,:],
                                              Z,
                                              v)
        else:
            # We know that S does not have the D plate so we can sum that plate
            # axis out
            v_BB = np.einsum('...kdij,...nd->...nkij',
                             BB[...,:,:,:,:],
                             v)
            phi1[..., :-1, :, :] += np.einsum('...nkij,...nk->...nij', 
                                              v_BB,
                                              Z)
            
        phi1 *= -0.5

        # Super-diagonal blocks: 0.5 * A.T * V
        # However, don't multiply by 0.5 because there are both super- and
        # sub-diagonal blocks (sum them together)
        phi2[..., :, :, :] = np.einsum('...kji,...nk,...nj->...nij', 
                                       B[...,:,:,:],
                                       Z,
                                       v)

        return (phi0, phi1, phi2)

    def compute_cgf_from_parents(self, u_mu, u_Lambda, u_B, u_Z, u_v):
        """
        Compute CGF using the moments of the parents.
        """

        return _compute_cgf_for_gaussian_markov_chain(u_mu[1],
                                                      u_Lambda[0],
                                                      u_Lambda[1],
                                                      u_v[1],
                                                      self.N)

    def plates_to_parent(self, index, plates):
        """
        Computes the plates of this node with respect to a parent.

        If this node has plates (...), the latent dimensionality is D
        and the number of time instances is N, the plates with respect
        to the parents are:
          mu:     (...)
          Lambda: (...)
          A:      (...,N-1,D)
          v:      (...,N-1,D)

        Parameters
        ----------
        index : int
            The index of the parent node to use.
        """

        if index == 0:   # mu: (...)x(N,D) -> (...)x(D)
            return plates
        elif index == 1: # Lambda: (...)x(N,D) -> (...)x(D,D)
            return plates
        elif index == 2: # B: (...)x(N,D) -> (...,K,D)x(D)
            return plates + (self.K,self.D)
        elif index == 3: # Z: (...)x(N,D) -> (...,N-1)x(K)
            return plates + (self.N-1,)
        elif index == 4: # v: (...)x(N,D) -> (...,N-1,D)x()
            return plates + (self.N-1,self.D)
        else:
            raise ValueError("Invalid parent index.")
        
        
    def plates_from_parent(self, index, plates):
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
        if index == 0:   # mu: (...)x(D) -> (...)x(N,D)
            return plates
        elif index == 1: # Lambda: (...)x(D,D) -> (...)x(N,D)
            return plates
        elif index == 2: # B: (...,K,D)x(D) -> (...)x(N,D)
            return plates[:-2]
        elif index == 3: # Z: (...,N-1)x(K) -> (...)x(N,D)
            return plates[:-1]
        elif index == 4: # v: (...,N-1,D)x() -> (...)x(N,D)
            return plates[:-2]
        else:
            raise ValueError("Invalid parent index.")




class SwitchingGaussianMarkovChain(_TemplateGaussianMarkovChain):
    r"""
    Node for Gaussian Markov chain random variables with switching dynamics.

    The node models a sequence of Gaussian variables
    :math:`\mathbf{x}_0,\ldots,\mathbf{x}_{N-1}$ with linear Markovian dynamics.
    The dynamics may change in time, which is obtained by having a set of
    matrices and at each time selecting one of them as the state dynamics
    matrix.  The graphical model can be presented as:

    .. bayesnet::

       \tikzstyle{latent} += [minimum size=40pt];
       
       \node[latent] (x0) {$\mathbf{x}_0$};
       \node[latent, right=of x0] (x1) {$\mathbf{x}_1$};
       \node[right=of x1] (dots) {$\cdots$};
       \node[latent, right=of dots] (xn) {$\mathbf{x}_{N-1}$};
       \edge {x0}{x1};
       \edge {x1}{dots};
       \edge {dots}{xn};

       \node[latent, above left=1 and 0.1 of x0] (mu) {$\boldsymbol{\mu}$};
       \node[latent, above right=1 and 0.1 of x0] (Lambda) {$\mathbf{\Lambda}$};
       \node[det, below=of x1] (A0) {$\mathbf{A}_0$};
       \node[right=of A0] (Adots) {$\cdots$};
       \node[det, right=of Adots] (An) {$\mathbf{A}_{N-2}$};
       \node[latent, above=of dots] (nu) {$\boldsymbol{\nu}$};
       \edge {mu,Lambda} {x0};
       \edge {nu} {x1,dots,xn};
       \edge {A0} {x1};
       \edge {Adots} {dots};
       \edge {An} {xn};

       \node[latent, below=of A0] (z0) {$z_0$};
       \node[right=of z0] (zdots) {$\cdots$};
       \node[latent, right=of zdots] (zn) {$z_{N-2}$};
       \node[latent, left=of z0] (B) {$\mathbf{B}_k$};
       \edge {B} {A0, Adots, An};
       \edge {z0} {A0};
       \edge {zdots} {Adots};
       \edge {zn} {An};

       \plate {K} {(B)} {$k=0,\ldots,K-1$};

    where :math:`\boldsymbol{\mu}` and :math:`\mathbf{\Lambda}` are the mean and
    the precision matrix of the initial state, :math:`\boldsymbol{\nu}` is the
    precision of the innovation noise, and :math:`\mathbf{A}_n` are the state
    dynamics matrix obtained by selecting one of the matrices
    :math:`\{\mathbf{B}_k\}^{K-1}_{k=0}` at each time.  The selections are
    provided by :math:`z_n\in\{0,\ldots,K-1\}`.  The probability distribution is

    .. math::

       p(\mathbf{x}_0, \ldots, \mathbf{x}_{N-1}) = p(\mathbf{x}_0)
       \prod^{N-1}_{n=1} p(\mathbf{x}_n | \mathbf{x}_{n-1})

    where
    
    .. math::

       p(\mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_0 | \boldsymbol{\mu}, \mathbf{\Lambda})
       \\
       p(\mathbf{x}_n|\mathbf{x}_{n-1}) &= \mathcal{N}(\mathbf{x}_n |
       \mathbf{A}_{n-1}\mathbf{x}_{n-1}, \mathrm{diag}(\boldsymbol{\nu})),
       \quad \text{for } n=1,\ldots,N-1,
       \\
       \mathbf{A}_n &= \mathbf{B}_{z_n}, \quad \text{for }
       n=0,\ldots,N-2.
       

    Parameters
    ----------
    
    mu : Gaussian-like node or (...,D)-array
        :math:`\boldsymbol{\mu}`, mean of :math:`x_0`, :math:`D`-dimensional
        with plates (...)
        
    Lambda : Wishart-like node or (...,D,D)-array
        :math:`\mathbf{\Lambda}`, precision matrix of :math:`x_0`,
        :math:`D\times D` -dimensional with plates (...)
        
    B : Gaussian-like node or (...,D,D,K)-array
        :math:`\{\mathbf{B}_k\}_{k=0}^{K-1}`, a set of state dynamics matrix,
        :math:`D \times K`-dimensional with plates (...,D)

    Z : categorical-like node or (...,N-1)-array
        :math:`\{z_0,\ldots,z_{N-2}\}`, time-dependent selection,
        :math:`K`-categorical with plates (...,N-1)
        
    nu : gamma-like node or (...,D)-array
        :math:`\boldsymbol{\nu}`, diagonal elements of the precision of the
        innovation process, plates (...,D)

    n : int, optional
        :math:`N`, the length of the chain. Must be given if :math:`\mathbf{Z}`
        does not have plates over the time domain (which would not make sense).

    See also
    --------
    
    Gaussian, GaussianARD, Wishart, Gamma, GaussianMarkovChain,
    VaryingGaussianMarkovChain, Categorical, CategoricalMarkovChain

    Notes
    -----

    Equivalent model block can be constructed with :class:`GaussianMarkovChain`
    by explicitly using :class:`Gate` to select the state dynamics matrix.
    However, that approach is not very efficient for large datasets because it
    does not utilize the structure of :math:`\mathbf{A}_n`, thus it explicitly
    computes huge moment arrays.
    """


    def __init__(self, mu, Lambda, B, Z, nu, n=None, **kwargs):
        """
        Create SwitchingGaussianMarkovChain node.
        """
        super().__init__(mu, Lambda, B, Z, nu, n=n, **kwargs)


    @classmethod
    def _constructor(cls, mu, Lambda, B, Z, v, n=None, **kwargs):
        """
        Constructs distribution and moments objects.
        
        Compute the dimensions of phi and u.

        The plates and dimensions of the parents should be:
        mu:     (...)                    and D-dimensional
        Lambda: (...)                    and D-dimensional
        B:      (...,K,D)                and D-dimensional
        Z:      (...,N-1)                and K-dimensional
        v:      (...,1,D) or (...,N-1,D) and 0-dimensional

        Check that the dimensionalities of the parents are proper.
        """

        # Infer the number of dynamic matrices
        B = cls._ensure_moments(B, GaussianMoments, ndim=2)
        K = B.plates[-2]

        mu = cls._ensure_moments(mu, GaussianMoments, ndim=1)
        Lambda = cls._ensure_moments(Lambda, WishartMoments)
        Z = cls._ensure_moments(Z, CategoricalMoments, categories=K)
        v = cls._ensure_moments(v, GammaMoments)

        parent_moments = (
            mu._moments,
            Lambda._moments,
            Z._moments,
            v._moments
        )

        # Infer the length of the chain
        n_Z = 1
        if len(Z.plates) == 0:
            raise ValueError("Z must have temporal axis on plates")
        n_Z = Z.plates[-1]
        n_v = 1
        if len(v.plates) >= 2:
            n_v = v.plates[-2]
        if n_v != n_Z and n_v != 1 and n_Z != 1:
            raise Exception(
                "Plates of Z and v are giving different number of time "
                "instances")
        n_Z = max(n_v, n_Z)
        if n is None:
            if n_Z == 1:
                raise Exception(
                    "The number of time instances could not be determined "
                    "automatically. Give the number of time instances.")
                                 
            n = n_Z + 1
        elif n_Z != 1 and n_Z+1 != n:
            raise Exception(
                "The number of time instances must match the number of last "
                "plates of parents:" "%d != %d+1" 
                % (n, n_Z))

                                
        D = mu.dims[0][0]
        K = Z.dims[0][0]
        M = n #N.get_moments()[0]

        # Check mu
        if mu.dims != ( (D,), (D,D) ):
            raise ValueError("First parent has wrong dimensionality")
        # Check Lambda
        if Lambda.dims != ( (D,D), () ):
            raise ValueError("Second parent has wrong dimensionality")
        # Check B
        if B.dims != ( (D,), (D,D) ):
            raise ValueError("Third parent has wrong dimensionality")
        if len(B.plates) < 2 or B.plates[-2:] != (K,D):
            raise ValueError("Third parent should have a last plate "
                             "equal to the dimensionality of the "
                             "system.")
        if Z.dims != ( (K,), ):
            raise ValueError("Fourth parent has wrong dimensionality %s, "
                             "should be %s"
                             % (Z.dims, ( (K,), )))
        if Z.plates[-1] != M-1:
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
        distribution = SwitchingGaussianMarkovChainDistribution(M, D, K)
        moments = GaussianMarkovChainMoments(M, D)

        parents = [mu, Lambda, B, Z, v]

        return (parents,
                kwargs,
                dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, mu.plates),
                                  distribution.plates_from_parent(1, Lambda.plates),
                                  distribution.plates_from_parent(2, B.plates),
                                  distribution.plates_from_parent(3, Z.plates),
                                  distribution.plates_from_parent(4, v.plates)),
                distribution,
                moments,
                parent_moments)


class _MarkovChainToGaussian(Deterministic):
    """
    Transform a Gaussian Markov chain node into a Gaussian node.

    This node is deterministic.
    """

    def __init__(self, X, **kwargs):

        X = self._ensure_moments(X, GaussianMarkovChainMoments)

        D = X.dims[0][-1]

        self._moments = GaussianMoments((D,))
        self._parent_moments = (X._moments,)

        super().__init__(X, dims=self._moments.dims, **kwargs)


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


    def _compute_weights_to_parent(self, index, weights):
        # Remove the last axis of the mask
        if np.ndim(weights) >= 1:
            weights = np.sum(weights, axis=-1)
        return weights


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

        Parameters
        ----------
        index : int
            Index of the parent requesting the message.
        u_parents : list of list of ndarrays
            List of parents' moments.

        Returns
        -------
        m : list of ndarrays
            Message as a list of arrays.
        mask : boolean ndarray
            Mask telling which plates should be taken into account.
        """

        # Add the third empty message
        return [m_children[0], m_children[1], None]


# Make use of the converter
GaussianMarkovChainMoments.add_converter(GaussianMoments,
                                         _MarkovChainToGaussian)
