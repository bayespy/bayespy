######################################################################
# Copyright (C) 2011-2014 Jaakko Luttinen
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
Module for the Gaussian distribution and similar distributions.
"""

import numpy as np

from scipy import special

from bayespy.utils import (random,
                           misc,
                           linalg)
from bayespy.utils.linalg import dot, mvdot

from .expfamily import (ExponentialFamily,
                        ExponentialFamilyDistribution,
                        useconstructor)
from .wishart import (WishartMoments,
                      WishartPriorMoments)
from .gamma import (GammaMoments,
                    GammaPriorMoments)
from .deterministic import Deterministic

from .node import (Moments,
                   ensureparents)



#
# MOMENTS
#

## class GaussianISOMoments(Moments):
##     """
##     Class for the moments of Gaussian ISO variables.
##     """

    
##     def __init__(self, ndim):
##         self.ndim = ndim
##         super().__init__()


##     def compute_fixed_moments(self, x):
##         """
##         Compute the moments for a fixed value
##         """
##         if np.ndim(x) < self.ndim:
##             raise ValueError("Not enough dimensions in x")
##         u0 = x
##         u1 = linalg.sum_multiply(x, x, axis=tuple(range(-self.ndim,0)))
##         u = [u0, u1]
##         return u

##     def compute_dims_from_values(self, x):
##         """
##         Return the shape of the moments for a fixed value.
##         """
##         x = misc.atleast_nd(x, self.ndim)
##         if self.ndim == 0:
##             shape = ()
##         else:
##             shape = np.shape(x)[-self.ndim:]
##         return (shape, ())
        

## class GaussianARDMoments(Moments):
##     """
##     Class for the moments of Gaussian ARD variables.
##     """

    
##     def __init__(self):
##         super().__init__()


##     def compute_fixed_moments(self, x):
##         """
##         Compute the moments for a fixed value
##         """
##         u0 = x
##         u1 = x**2
##         u = [u0, u1]
##         return u

    
##     def compute_dims_from_values(self, x):
##         """
##         Return the shape of the moments for a fixed value.
##         """
##         return ((), ())
        

class GaussianMoments(Moments):
    r"""
    Class for the moments of Gaussian variables.
    """

    
    def __init__(self, ndim):
        self.ndim = ndim
        super().__init__()


    def compute_fixed_moments(self, x):
        r"""
        Compute the moments for a fixed value
        """
        x = misc.atleast_nd(x, self.ndim)
        return [x, linalg.outer(x, x, ndim=self.ndim)]

    def compute_dims_from_values(self, x):
        r"""
        Return the shape of the moments for a fixed value.
        """
        x = misc.atleast_nd(x, self.ndim)
        if self.ndim == 0:
            shape = ()
        else:
            shape = np.shape(x)[-self.ndim:]
        return (shape, shape+shape)


class GaussianGammaISOMoments(Moments):
    r"""
    Class for the moments of Gaussian-gamma-ISO variables.
    """


    def __init__(self, ndim):
        r"""
        Create moments object for Gaussian-gamma isotropic variables

        ndim=0: scalar
        ndim=1: vector
        ndim=2: matrix
        ...
        """
        self.ndim = ndim
        super().__init__()

    
    def compute_fixed_moments(self, x, alpha):
        r"""
        Compute the moments for a fixed value

        `x` is a mean vector.
        `alpha` is a precision scale
        """

        x = np.asanyarray(x)
        alpha = np.asanyarray(alpha)

        u0 = x * misc.add_trailing_axes(alpha, self.ndim)
        u1 = (linalg.outer(x, x, ndim=self.ndim) 
              * misc.addtrailing_axes(alpha, 2*self.ndim))
        u2 = np.copy(alpha)
        u3 = np.log(alpha)
        u = [u0, u1, u2, u3]

        return u
    

    def compute_dims_from_values(self, x, alpha):
        r"""
        Return the shape of the moments for a fixed value.
        """

        if np.ndim(x) < 1:
            raise ValueError("Mean must be a vector")

        if ndim == 0:
            return ( (), (), (), () )
        
        dims = np.shape(x)[-ndim:]

        return ( dims, 2*dims, (), () )


class GaussianGammaARDMoments(Moments):
    r"""
    Class for the moments of Gaussian-gamma-ARD variables.
    """

    
    def __init__(self, ndim):
        r"""
        Create moments object for Gaussian-gamma isotropic variables

        ndim=0: scalar
        ndim=1: vector
        ndim=2: matrix
        ...
        """
        self.ndim = ndim
        super().__init__()

    
    def compute_fixed_moments(self, x, alpha):
        r"""
        Compute the moments for a fixed value

        `x` is a mean vector.
        `alpha` is a precision scale
        """

        x = np.asanyarray(x)
        alpha = np.asanyarray(alpha)

        if np.ndim(x) < self.ndim:
            raise ValueError("Not enough dimensions in x")
        if np.ndim(alpha) < self.ndim:
            raise ValueError("Not enough dimensions in alpha")
        if np.shape(x) != np.shape(alpha):
            raise ValueError("Mean and ARD scales have inconsistent shapes")

        u0 = alpha * x
        u1 = u0 * x
        u2 = np.copy(alpha)
        u3 = np.log(alpha)
        
        u = [u0, u1, u2, u3]

        return u
    

    def compute_dims_from_values(self, x, alpha):
        r"""
        Return the shape of the moments for a fixed value.
        """

        if np.ndim(x) < self.ndim:
            raise ValueError("Not enough dimensions in x")
        if np.ndim(alpha) < self.ndim:
            raise ValueError("Not enough dimensions in alpha")
        if np.shape(x) != np.shape(alpha):
            raise ValueError("Mean and ARD scales have inconsistent shapes")

        if ndim > 0:
           shape = np.shape(x)[-self.ndim:]
        else:
            shape = ()

        return ( shape, shape, shape, shape )


class GaussianWishartMoments(Moments):
    r"""
    Class for the moments of Gaussian-Wishart variables.
    """
    
    
    def compute_fixed_moments(self, x, Lambda):
        r"""
        Compute the moments for a fixed value

        `x` is a vector.
        `Lambda` is a precision matrix
        """

        x = np.asanyarray(x)
        Lambda = np.asanyarray(Lambda)

        u0 = np.einsum('...ik,...k->...i', Lambda, x)
        u1 = np.einsum('...i,...ij,...j->...', x, Lambda, x)
        u2 = np.copy(Lambda)
        u3 = linalg.logdet_cov(Lambda)

        return [u0, u1, u2, u3]
    

    def compute_dims_from_values(self, x, Lambda):
        r"""
        Return the shape of the moments for a fixed value.
        """

        if np.ndim(x) < 1:
            raise ValueError("Mean must be a vector")
        if np.ndim(Lambda) < 2:
            raise ValueError("Precision must be a matrix")

        D = np.shape(x)[-1]
        if np.shape(Lambda)[-2:] != (D,D):
            raise ValueError("Mean vector and precision matrix have "
                             "inconsistent shapes")

        return ( (D,), (), (D,D), () )



#
# DISTRIBUTIONS
#


class GaussianDistribution(ExponentialFamilyDistribution):
    r"""
    Class for the VMP formulas of Gaussian variables.

    Currently, supports only vector variables.

    Notes
    -----

    Message passing equations:

    .. math::

       \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),

    .. math::

       \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, 
       \quad \mathbf{\Lambda} \in \mathbb{R}^{D \times D},
       \quad \mathbf{\Lambda} \text{ symmetric positive definite}

    .. math::

       \log\mathcal{N}( \mathbf{x} | \boldsymbol{\mu}, \mathbf{\Lambda} )
       &= 
       - \frac{1}{2} \mathbf{x}^{\mathrm{T}} \mathbf{\Lambda} \mathbf{x}
       + \mathbf{x}^{\mathrm{T}} \mathbf{\Lambda} \boldsymbol{\mu}
       - \frac{1}{2} \boldsymbol{\mu}^{\mathrm{T}} \mathbf{\Lambda}
         \boldsymbol{\mu}
       + \frac{1}{2} \log |\mathbf{\Lambda}|
       - \frac{D}{2} \log (2\pi)
    """    

    
    def compute_message_to_parent(self, parent, index, u, u_mu_Lambda):
        r"""
        Compute the message to a parent node.

        .. math::

           \boldsymbol{\phi}_{\boldsymbol{\mu}} (\mathbf{x}, \mathbf{\Lambda})
           &=
           \left[ \begin{matrix}
             \mathbf{\Lambda} \mathbf{x} 
             \\
             - \frac{1}{2} \mathbf{\Lambda}
           \end{matrix} \right]
           \\
           \boldsymbol{\phi}_{\mathbf{\Lambda}} (\mathbf{x}, \boldsymbol{\mu})
           &=
           \left[ \begin{matrix}
             - \frac{1}{2} \mathbf{xx}^{\mathrm{T}}
             + \frac{1}{2} \mathbf{x}\boldsymbol{\mu}^{\mathrm{T}}
             + \frac{1}{2} \boldsymbol{\mu}\mathbf{x}^{\mathrm{T}}
             - \frac{1}{2} \boldsymbol{\mu\mu}^{\mathrm{T}}
             \\
             \frac{1}{2}
           \end{matrix} \right]
        """
        if index == 0:
            x = u[0]
            xx = u[1]
            m0 = x
            m1 = -0.5
            m2 = -0.5*xx
            m3 = 0.5
            return [m0, m1, m2, m3]
        else:
            raise ValueError("Index out of bounds")

    def compute_phi_from_parents(self, u_mu_Lambda, mask=True):
        r"""
        Compute the natural parameter vector given parent moments.

        .. math::

           \boldsymbol{\phi} (\boldsymbol{\mu}, \mathbf{\Lambda})
           &=
           \left[ \begin{matrix}
             \mathbf{\Lambda} \boldsymbol{\mu} 
             \\
             - \frac{1}{2} \mathbf{\Lambda}
           \end{matrix} \right]
        """
        Lambda_mu = u_mu_Lambda[0]
        Lambda = u_mu_Lambda[2]
        return [Lambda_mu,
                -0.5 * Lambda]

    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.

        .. math::
        
           \overline{\mathbf{u}}  (\boldsymbol{\phi})
           &=
           \left[ \begin{matrix}
             - \frac{1}{2} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
             \\
             \frac{1}{4} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
             \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
             - \frac{1}{2} \boldsymbol{\phi}^{-1}_2
           \end{matrix} \right]
           \\
           g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
           &=
           \frac{1}{4} \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
           \boldsymbol{\phi}_1
           + \frac{1}{2} \log | -2 \boldsymbol{\phi}_2 |
        """
        # TODO: Compute -2*phi[1] and simplify the formulas
        L = misc.m_chol(-2*phi[1])
        k = np.shape(phi[0])[-1]
        # Moments
        u0 = misc.m_chol_solve(L, phi[0])
        u1 = misc.m_outer(u0, u0) + misc.m_chol_inv(L)
        u = [u0, u1]
        # G
        g = (-0.5 * np.einsum('...i,...i', u[0], phi[0])
             + 0.5 * misc.m_chol_logdet(L))
             #+ 0.5 * np.log(2) * self.dims[0][0])
        return (u, g)

    def compute_cgf_from_parents(self, u_mu_Lambda):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`

        .. math::
           g (\boldsymbol{\mu}, \mathbf{\Lambda})
           &=
           - \frac{1}{2} \operatorname{tr}(\boldsymbol{\mu\mu}^{\mathrm{T}}
                                           \mathbf{\Lambda} )
           + \frac{1}{2} \log |\mathbf{\Lambda}|
        """
        mu_Lambda_mu = u_mu_Lambda[1]
        logdet_Lambda = u_mu_Lambda[3]
        g = -0.5*mu_Lambda_mu + 0.5*logdet_Lambda
        return g

    def compute_fixed_moments_and_f(self, x, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.

        .. math::

           \mathbf{u} (\mathbf{x})
           &=
           \left[ \begin{matrix}
             \mathbf{x}
             \\
             \mathbf{xx}^{\mathrm{T}}
           \end{matrix} \right]
           \\
           f(\mathbf{x})
           &= - \frac{D}{2} \log(2\pi)
        """
        k = np.shape(x)[-1]
        u = [x, misc.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)


    def compute_gradient(self, g, u, phi):
        r"""
        Compute the standard gradient with respect to the natural parameters.
        
        Gradient of the moments:

        .. math::

           \mathrm{d}\overline{\mathbf{u}} &=
           \begin{bmatrix}
             \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1} \phi_1
             - \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_1
             \\
             - \frac{1}{4} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1} \phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1}
             - \frac{1}{4} \phi_2^{-1} \phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1}
             + \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1}
             + \frac{1}{4} \phi_2^{-1} \mathrm{d}\phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1}
             + \frac{1}{4} \phi_2^{-1} \phi_1 \mathrm{d}\phi_1^{\mathrm{T}} \phi_2^{-1}
           \end{bmatrix}
           \\
           &=
           \begin{bmatrix}
             2 (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \mathrm{d}\phi_2 \overline{u}_1
             + (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \mathrm{d}\phi_1
             \\
             u_2 d\phi_2 u_2 - 2 u_1 u_1^T d\phi_2 u_1 u_1^T
             + 2 (u_2 - u_1 u_1^T) d\phi_1 u_1^T
           \end{bmatrix}

        Standard gradient given the gradient with respect to the moments, that
        is, given the Riemannian gradient :math:`\tilde{\nabla}`:

        .. math::

           \nabla =
           \begin{bmatrix}
             (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \tilde{\nabla}_1
             + 2 (u_2 - u_1 u_1^T) \tilde{\nabla}_2 u_1
             \\
             (u_2 - u_1 u_1^T) \tilde{\nabla}_1 u_1^T
             +  u_1 \tilde{\nabla}_1^T (u_2 - u_1 u_1^T)
             + 2 u_2 \tilde{\nabla}_2 u_2
             - 2 u_1 u_1^T \tilde{\nabla}_2 u_1 u_1^T
           \end{bmatrix}
        """
        ndim = 1
        x = u[0]
        xx = u[1]
        # Some helpful variables
        x_x = linalg.outer(x, x, ndim=ndim)
        Cov = xx - x_x
        cov_g0 = linalg.mvdot(Cov, g[0], ndim=ndim)
        cov_g0_x = linalg.outer(cov_g0, x, ndim=ndim)
        g1_x = linalg.mvdot(g[1], x, ndim=ndim)
        # Compute gradient terms
        d0 = cov_g0 + 2 * linalg.mvdot(Cov, g1_x, ndim=ndim)
        d1 = (cov_g0_x + linalg.transpose(cov_g0_x, ndim=ndim)
              + 2 * linalg.mmdot(xx,
                                 linalg.mmdot(g[1], xx, ndim=ndim),
                                 ndim=ndim)
              - 2 * x_x * misc.add_trailing_axes(linalg.inner(g1_x,
                                                              x,
                                                              ndim=ndim),
                                                 2*ndim))

        return [d0, d1]


    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = misc.m_chol(-2*phi[1])
        mu = misc.m_chol_solve(U, phi[0])
        z = np.random.normal(0, 1, plates + np.shape(mu)[-1:])
        # Compute mu + U'*z
        z = misc.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
            

class GaussianARDDistribution(ExponentialFamilyDistribution):
    r"""
    ...

    Log probability density function:

    .. math::

        \log p(x|\mu, \alpha) = -\frac{1}{2} x^T \mathrm{diag}(\alpha) x + x^T
        \mathrm{diag}(\alpha) \mu - \frac{1}{2} \mu^T \mathrm{diag}(\alpha) \mu
        + \frac{1}{2} \sum_i \log \alpha_i - \frac{D}{2} \log(2\pi)
        
    Parent has moments:

    .. math::

        \begin{bmatrix}
            \alpha \circ \mu
            \\
            \alpha \circ \mu \circ \mu
            \\
            \alpha
            \\
            \log(\alpha)
        \end{bmatrix}
    """

    def __init__(self, shape, ndim_mu):
        self.shape = shape
        self.ndim_mu = ndim_mu
        self.ndim = len(shape)
        super().__init__()
    
    def compute_message_to_parent(self, parent, index, u, u_mu_alpha):
        r"""
        ...


        .. math::

            m = 
            \begin{bmatrix}
                x
                \\
                [-\frac{1}{2}, \ldots, -\frac{1}{2}]
                \\
                -\frac{1}{2} \mathrm{diag}(xx^T)
                \\
                [\frac{1}{2}, \ldots, \frac{1}{2}]
            \end{bmatrix}
        """
        if index == 0:
            x = u[0]
            x2 = misc.get_diag(u[1], ndim=self.ndim)
            
            m0 = x
            m1 = -0.5 * np.ones(self.shape)
            m2 = -0.5 * x2
            m3 = 0.5 * np.ones(self.shape)
            m = [m0, m1, m2, m3]
            return m
        
        ## if index == 0:
        ##     x = u[0]
        ##     alpha = u_alpha[0]

        ##     axes0 = list(range(-self.ndim, -self.ndim_mu))
        ##     m0 = misc.sum_multiply(alpha, x, axis=axes0)

        ##     Alpha = misc.diag(alpha, ndim=self.ndim)
        ##     axes1 = [axis+self.ndim for axis in axes0] + axes0
        ##     m1 = -0.5 * misc.sum_multiply(Alpha, 
        ##                                   misc.identity(*self.shape),
        ##                                   axis=axes1)
        ##     return [m0, m1]

        ## elif index == 1:
        ##     x = u[0]
        ##     x2 = misc.get_diag(u[1], ndim=self.ndim)
        ##     mu = u_mu[0]
        ##     mu2 = misc.get_diag(u_mu[1], ndim=self.ndim_mu)
        ##     if self.ndim_mu == 0:
        ##         mu_shape = np.shape(mu) + (1,)*self.ndim
        ##     else:
        ##         mu_shape = (np.shape(mu)[:-self.ndim_mu] 
        ##                     + (1,)*(self.ndim-self.ndim_mu)
        ##                     + np.shape(mu)[-self.ndim_mu:])
        ##     mu = np.reshape(mu, mu_shape)
        ##     mu2 = np.reshape(mu2, mu_shape)
        ##     m0 = -0.5*x2 + x*mu - 0.5*mu2
        ##     m1 = 0.5
        ##     return [m0, m1]

        else:
            raise ValueError("Invalid parent index")


    def compute_mask_to_parent(self, index, mask):
        r"""
        Maps the mask to the plates of a parent.
        """
        if index == 0:
            if self.ndim_mu == self.ndim:
                return mask
            elif self.ndim_mu < self.ndim:
                diff = self.ndim - self.ndim_mu
                return misc.add_trailing_axes(mask, diff)
            else:
                raise RuntimeError("Parent's ndim is larger")
        else:
            raise ValueError("Invalid parent index")


    def compute_phi_from_parents(self, u_mu_alpha, mask=True):
        alpha_mu = u_mu_alpha[0]
        alpha = u_mu_alpha[2]
        #mu = u_mu[0]
        #alpha = u_alpha[0]
        ## if np.ndim(mu) < self.ndim_mu:
        ##     raise ValueError("Moment of mu does not have enough dimensions")
        ## mu = misc.add_axes(mu, 
        ##                    axis=np.ndim(mu)-self.ndim_mu, 
        ##                    num=self.ndim-self.ndim_mu)
        phi0 = alpha_mu
        phi1 = -0.5 * alpha
        if self.ndim > 0:
            # Ensure that phi is not using broadcasting for variable
            # dimension axes
            ones = np.ones(self.shape)
            phi0 = ones * phi0
            phi1 = ones * phi1

        # Make a diagonal matrix
        phi1 = misc.diag(phi1, ndim=self.ndim)
        return [phi0, phi1]

    def compute_moments_and_cgf(self, phi, mask=True):
        if self.ndim == 0:
            # Use scalar equations
            u0 = -phi[0] / (2*phi[1])
            u1 = u0**2 - 1 / (2*phi[1])
            u = [u0, u1]
            g = (-0.5 * u[0] * phi[0] + 0.5 * np.log(-2*phi[1]))

            # TODO/FIXME: You could use these equations if phi is a scalar
            # in practice although ndim>0 (because the shape can be, e.g.,
            # (1,1,1,1) for ndim=4).

        else:

            # Reshape to standard vector and matrix
            D = np.prod(self.shape)
            phi0 = np.reshape(phi[0], phi[0].shape[:-self.ndim] + (D,))
            phi1 = np.reshape(phi[1], phi[1].shape[:-2*self.ndim] + (D,D))

            # Compute the moments
            L = linalg.chol(-2*phi1)
            Cov = linalg.chol_inv(L)
            u0 = linalg.chol_solve(L, phi0)
            u1 = linalg.outer(u0, u0) + Cov

            # Compute CGF
            g = (- 0.5 * np.einsum('...i,...i', u0, phi0)
                 + 0.5 * linalg.chol_logdet(L))

            # Reshape to arrays
            u0 = np.reshape(u0, u0.shape[:-1] + self.shape)
            u1 = np.reshape(u1, u1.shape[:-2] + self.shape + self.shape)
            u = [u0, u1]

        return (u, g)


    def compute_cgf_from_parents(self, u_mu_alpha):
        r"""
        Compute the value of the cumulant generating function.
        """

        # Compute sum(mu^2 * alpha) correctly for broadcasted shapes

        alpha_mu2 = u_mu_alpha[1]
        logdet_alpha = u_mu_alpha[3]
        axes = tuple(range(-self.ndim, 0))

        # TODO/FIXME: You could use plate multiplier type of correction instead
        # of explicitly broadcasting with ones.
        if self.ndim > 0:
            alpha_mu2 = misc.sum_multiply(alpha_mu2, np.ones(self.shape),
                                          axis=axes)
        if self.ndim > 0:
            logdet_alpha = misc.sum_multiply(logdet_alpha, np.ones(self.shape),
                                             axis=axes)

        # Compute g
        g = -0.5*alpha_mu2 + 0.5*logdet_alpha

        return g

    def compute_fixed_moments_and_f(self, x, mask=True):
        r""" Compute u(x) and f(x) for given x. """
        if self.ndim > 0 and np.shape(x)[-self.ndim:] != self.shape:
            raise ValueError("Invalid shape")
        k = np.prod(self.shape)
        u = [x, linalg.outer(x, x, ndim=self.ndim)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)


    def plates_to_parent(self, index, plates):
        r"""
        Resolves the plate mapping to a parent.

        Given the plates of the node's moments, this method returns the plates
        that the message to a parent has for the parent's distribution.
        """
        if index == 0:
            if self.ndim_mu == self.ndim:
                return plates
            elif self.ndim_mu < self.ndim:
                diff = self.ndim - self.ndim_mu
                return plates + self.shape[:diff]
            else:
                raise RuntimeError("Parent's ndim is larger")
        else:
            raise ValueError("Invalid parent index")
            

    def plates_from_parent(self, index, plates):
        r"""
        Resolve the plate mapping from a parent.

        Given the plates of a parent's moments, this method returns the plates
        that the moments has for this distribution.
        """
        if index == 0:
            if self.ndim_mu == self.ndim:
                return plates
            elif self.ndim_mu < self.ndim:
                diff = self.ndim - self.ndim_mu
                return plates[:-diff]
            else:
                raise RuntimeError("Parent's ndim is larger")
        else:
            raise ValueError("Invalid parent index")


    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the Gaussian distribution.
        """
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!
        D = self.ndim
        if D == 0:
            dims = ()
        else:
            dims = np.shape(phi[0])[-D:]
            
        if np.prod(dims) == 1.0:
            # Scalar Gaussian
            phi1 = phi[1]
            if D > 0:
                # Because the covariance matrix has shape (1,1,...,1,1),
                # that is 2*D number of ones, remove the extra half of the
                # shape
                phi1 = np.reshape(phi1, np.shape(phi1)[:-2*D] + D*(1,))

            var = -0.5 / phi1
            std = np.sqrt(var)
            mu = var * phi[0]
            z = np.random.normal(0, 1, plates + dims)
            x = mu + std * z
        else:
            N = np.prod(dims)
            dims_cov = dims + dims
            # Reshape precision matrix
            plates_cov = np.shape(phi[1])[:-2*D]
            V = -2 * np.reshape(phi[1], plates_cov + (N,N))
            # Compute Cholesky
            U = linalg.chol(V)
            # Reshape mean vector
            plates_phi0 = np.shape(phi[0])[:-D]
            phi0 = np.reshape(phi[0], plates_phi0 + (N,))
            mu = linalg.chol_solve(U, phi0)
            # Compute mu + U'*z
            z = np.random.normal(0, 1, plates + (N,))
            x = mu + linalg.solve_triangular(U, z,
                                             trans='T', 
                                             lower=False)
            x = np.reshape(x, plates + dims)
        return x


    def compute_gradient(self, g, u, phi):
        r"""
        Compute the standard gradient with respect to the natural parameters.
        
        Gradient of the moments:

        .. math::

           \mathrm{d}\overline{\mathbf{u}} &=
           \begin{bmatrix}
             \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1} \phi_1
             - \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_1
             \\
             - \frac{1}{4} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1} \phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1}
             - \frac{1}{4} \phi_2^{-1} \phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1}
             + \frac{1}{2} \phi_2^{-1} \mathrm{d}\phi_2 \phi_2^{-1}
             + \frac{1}{4} \phi_2^{-1} \mathrm{d}\phi_1 \phi_1^{\mathrm{T}} \phi_2^{-1}
             + \frac{1}{4} \phi_2^{-1} \phi_1 \mathrm{d}\phi_1^{\mathrm{T}} \phi_2^{-1}
           \end{bmatrix}
           \\
           &=
           \begin{bmatrix}
             2 (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \mathrm{d}\phi_2 \overline{u}_1
             + (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \mathrm{d}\phi_1
             \\
             u_2 d\phi_2 u_2 - 2 u_1 u_1^T d\phi_2 u_1 u_1^T
             + 2 (u_2 - u_1 u_1^T) d\phi_1 u_1^T
           \end{bmatrix}

        Standard gradient given the gradient with respect to the moments, that
        is, given the Riemannian gradient :math:`\tilde{\nabla}`:

        .. math::

           \nabla =
           \begin{bmatrix}
             (\overline{u}_2 - \overline{u}_1 \overline{u}_1^{\mathrm{T}}) \tilde{\nabla}_1
             + 2 (u_2 - u_1 u_1^T) \tilde{\nabla}_2 u_1
             \\
             (u_2 - u_1 u_1^T) \tilde{\nabla}_1 u_1^T
             +  u_1 \tilde{\nabla}_1^T (u_2 - u_1 u_1^T)
             + 2 u_2 \tilde{\nabla}_2 u_2
             - 2 u_1 u_1^T \tilde{\nabla}_2 u_1 u_1^T
           \end{bmatrix}
        """
        ndim = self.ndim
        x = u[0]
        xx = u[1]
        # Some helpful variables
        x_x = linalg.outer(x, x, ndim=ndim)
        Cov = xx - x_x
        cov_g0 = linalg.mvdot(Cov, g[0], ndim=ndim)
        cov_g0_x = linalg.outer(cov_g0, x, ndim=ndim)
        g1_x = linalg.mvdot(g[1], x, ndim=ndim)
        # Compute gradient terms
        d0 = cov_g0 + 2 * linalg.mvdot(Cov, g1_x, ndim=ndim)
        d1 = (cov_g0_x + linalg.transpose(cov_g0_x, ndim=ndim)
              + 2 * linalg.mmdot(xx,
                                 linalg.mmdot(g[1], xx, ndim=ndim),
                                 ndim=ndim)
              - 2 * x_x * misc.add_trailing_axes(linalg.inner(g1_x,
                                                              x,
                                                              ndim=ndim),
                                                 2*ndim))

        return [d0, d1]


class GaussianGammaISODistribution(ExponentialFamilyDistribution):
    r"""
    Class for the VMP formulas of Gaussian-Gamma-ISO variables.

    Currently, supports only vector variables.
    """    


    def compute_message_to_parent(self, parent, index, u, u_mu_Lambda, u_a, u_b):
        r"""
        Compute the message to a parent node.
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        elif index == 2:
            raise NotImplementedError()
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_mu_Lambda, u_a, u_b, mask=True):
        r"""
        Compute the natural parameter vector given parent moments.
        """
        Lambda_mu = u_mu_Lambda[0]
        mu_Lambda_mu = u_mu_Lambda[1]
        Lambda = u_mu_Lambda[2]
        a = u_a[0]
        b = u_b[0]
        phi = [Lambda_mu,
               -0.5*Lambda,
               -0.5*mu_Lambda_mu - b,
               a]
        return phi


    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.
        """
        # Compute helpful variables
        V = -2*phi[1]
        L_V = linalg.chol(V)
        logdet_V = linalg.chol_logdet(L_V)
        mu = linalg.chol_solve(L_V, phi[0])
        Cov = linalg.chol_inv(L_V)
        a = phi[3]
        b = -phi[2] - 0.5 * linalg.inner(mu, phi[0])
        log_b = np.log(b)

        # Compute moments
        u2 = a / b
        u3 = -log_b + special.psi(a)
        u0 = u2[...,None] * mu 
        u1 = Cov + u2[...,None,None] * linalg.outer(mu, mu)
        u = [u0, u1, u2, u3]

        # Compute g
        g = 0.5*logdet_V + a*log_b - special.gammaln(a)

        return (u, g)

    
    def compute_cgf_from_parents(self, u_mu_Lambda, u_a, u_b):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        logdet_Lambda = u_mu_Lambda[3]
        a = u_a[0]
        gammaln_a = u_a[1]
        log_b = u_b[1]
        g = 0.5*logdet_Lambda + a*log_b - gammaln_a
        return g

    
    def compute_fixed_moments_and_f(self, x, alpha, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        logalpha = np.log(alpha)
        u0 = x * misc.add_trailing_axes(alpha, 1)
        u1 = linalg.outer(x, x, ndim=1) * misc.add_trailing_axes(alpha, 2)
        u2 = alpha
        u3 = logalpha
        u = [u0, u1, u2, u3]
        D = np.shape(x)[-1]
        f = (D/2 - 1) * logalpha - D/2 * np.log(2*np.pi)
        return (u, f)

    
    def random(self, *params, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()


class GaussianGammaARDDistribution(ExponentialFamilyDistribution):
    r"""
    """


    def __init__(self):
        raise NotImplementedError()

    
class GaussianWishartDistribution(ExponentialFamilyDistribution):
    r"""
    Class for the VMP formulas of Gaussian-Wishart variables.

    Currently, supports only vector variables.
    """    


    def compute_message_to_parent(self, parent, index, u, u_mu_alpha, u_V, u_n):
        r"""
        Compute the message to a parent node.
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        elif index == 2:
            raise NotImplementedError()
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_mu_alpha, u_V, u_n, mask=True):
        r"""
        Compute the natural parameter vector given parent moments.
        """
        raise NotImplementedError()


    def compute_moments_and_cgf(self, phi, mask=True):
        r"""
        Compute the moments and :math:`g(\phi)`.
        """
        raise NotImplementedError()
        return (u, g)

    
    def compute_cgf_from_parents(self, u_mu_alpha, u_V, u_n):
        r"""
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        raise NotImplementedError()
        return g

    
    def compute_fixed_moments_and_f(self, x, Lambda, mask=True):
        r"""
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        raise NotImplementedError()
        return (u, f)

    
    def random(self, *params, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()


#
# NODES
#


class Gaussian(ExponentialFamily):
    r"""
    Node for Gaussian variables.

    The node represents a :math:`D`-dimensional vector from the Gaussian
    distribution:
    
    .. math::

       \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),

    where :math:`\boldsymbol{\mu}` is the mean vector and
    :math:`\mathbf{\Lambda}` is the precision matrix (i.e., inverse of the
    covariance matrix).
    
    .. math::

       \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, 
       \quad \mathbf{\Lambda} \in \mathbb{R}^{D \times D},
       \quad \mathbf{\Lambda} \text{ symmetric positive definite}

    Parameters
    ----------

    mu : Gaussian-like node or GaussianGammaISO-like node or GaussianWishart-like node or array
        Mean vector

    Lambda : Wishart-like node or array
        Precision matrix

    See also
    --------
    
    Wishart, GaussianARD, GaussianWishart, GaussianGammaARD, GaussianGammaISO
    
    """

    _distribution = GaussianDistribution()
    _moments = GaussianMoments(1)
    _parent_moments = [GaussianWishartMoments()]


    def __init__(self, mu, Lambda, **kwargs):
        r"""
        Create Gaussian node
        """
        super().__init__(mu, Lambda, **kwargs)
    

    @classmethod
    def _constructor(cls, mu, Lambda, **kwargs):
        r"""
        Constructs distribution and moments objects.
        """

        mu_Lambda = WrapToGaussianWishart(mu, Lambda)
        
        D = mu_Lambda.dims[0][0]
        
        if mu_Lambda.dims != ( (D,), (), (D,D), () ):
            raise Exception("Parents have wrong dimensionality")

        parents = [mu_Lambda]
        dims = ( (D,), (D,D) )
        return (parents,
                kwargs,
                dims, 
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu_Lambda.plates)),
                cls._distribution, 
                cls._moments, 
                cls._parent_moments)


    def initialize_from_parameters(self, mu, Lambda):
        u = self._parent_moments[0].compute_fixed_moments(mu, Lambda)
        self._initialize_from_parent_moments(u)

        
    def __str__(self):
        mu = self.u[0]
        Cov = self.u[1] - misc.m_outer(mu, mu)
        return ("%s ~ Gaussian(mu, Cov)\n"
                "  mu = \n"
                "%s\n"
                "  Cov = \n"
                "%s\n"
                % (self.name, mu, Cov))


    def rotate(self, R, inv=None, logdet=None, Q=None):

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

        # Rotate plates, if plate rotation matrix is given. Assume that there's
        # only one plate-axis

        if Q is not None:
            # Rotate moments using Q
            self.u[0] = np.einsum('ik,kj->ij', Q, self.u[0])
            sumQ = np.sum(Q, axis=0)
            # Rotate natural parameters using Q
            self.phi[1] = np.einsum('d,dij->dij', sumQ**(-2), self.phi[1]) 
            self.phi[0] = np.einsum('dij,dj->di', -2*self.phi[1], self.u[0])

        # Transform parameters using R
        self.phi[0] = mvdot(invR.T, self.phi[0])
        self.phi[1] = dot(invR.T, self.phi[1], invR)

        if Q is not None:
            self._update_moments_and_cgf()
        else:
            # Transform moments and g using R
            self.u[0] = mvdot(R, self.u[0])
            self.u[1] = dot(R, self.u[1], R.T)
            self.g -= logdetR

    def rotate_matrix(self, R1, R2, inv1=None, logdet1=None, inv2=None, logdet2=None, Q=None):
        r"""
        The vector is reshaped into a matrix by stacking the row vectors.

        Computes R1*X*R2', which is identical to kron(R1,R2)*x (??)

        Note that this is slightly different from the standard Kronecker product
        definition because Numpy stacks row vectors instead of column vectors.

        Parameters
        ----------
        R1 : ndarray
            A matrix from the left
        R2 : ndarray
            A matrix from the right        
        """

        if Q is not None:
            # Rotate moments using Q
            self.u[0] = np.einsum('ik,kj->ij', Q, self.u[0])
            sumQ = np.sum(Q, axis=0)
            # Rotate natural parameters using Q
            self.phi[1] = np.einsum('d,dij->dij', sumQ**(-2), self.phi[1]) 
            self.phi[0] = np.einsum('dij,dj->di', -2*self.phi[1], self.u[0])

        if inv1 is None:
            inv1 = np.linalg.inv(R1)
        if logdet1 is None:
            logdet1 = np.linalg.slogdet(R1)[1]
        if inv2 is None:
            inv2 = np.linalg.inv(R2)
        if logdet2 is None:
            logdet2 = np.linalg.slogdet(R2)[1]

        D1 = np.shape(R1)[0]
        D2 = np.shape(R2)[0]

        # Reshape into matrices
        sh0 = np.shape(self.phi[0])[:-1] + (D1,D2)
        sh1 = np.shape(self.phi[1])[:-2] + (D1,D2,D1,D2)
        phi0 = np.reshape(self.phi[0], sh0)
        phi1 = np.reshape(self.phi[1], sh1)

        # Apply rotations to phi
        #phi0 = dot(inv1, phi0, inv2.T)
        phi0 = dot(inv1.T, phi0, inv2)
        phi1 = np.einsum('...ia,...abcd->...ibcd', inv1.T, phi1)
        phi1 = np.einsum('...ic,...abcd->...abid', inv1.T, phi1)
        phi1 = np.einsum('...ib,...abcd->...aicd', inv2.T, phi1)
        phi1 = np.einsum('...id,...abcd->...abci', inv2.T, phi1)

        # Reshape back into vectors
        self.phi[0] = np.reshape(phi0, self.phi[0].shape)
        self.phi[1] = np.reshape(phi1, self.phi[1].shape)

        # It'd be better to rotate the moments too..

        self._update_moments_and_cgf()


class GaussianARD(ExponentialFamily):
    r"""
    Node for Gaussian variables with ARD prior.

    The node represents a :math:`D`-dimensional vector from the Gaussian
    distribution:
    
    .. math::

       \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\alpha})),

    where :math:`\boldsymbol{\mu}` is the mean vector and
    :math:`\mathrm{diag}(\boldsymbol{\alpha})` is the diagonal precision matrix
    (i.e., inverse of the covariance matrix).
    
    .. math::

       \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, \quad \alpha_d > 0 \text{
       for } d=0,\ldots,D-1

    *Note:*  The form of the posterior approximation is a Gaussian distribution with full
    covariance matrix instead of a diagonal matrix.

    Parameters
    ----------

    mu : Gaussian-like node or GaussianGammaISO-like node or GaussianGammaARD-like node or array
        Mean vector

    alpha : gamma-like node or array
        Diagonal elements of the precision matrix

    See also
    --------
    
    Gamma, Gaussian, GaussianGammaARD, GaussianGammaISO, GaussianWishart
    """


    def __init__(self, mu, alpha, ndim=None, shape=None, **kwargs):
        r"""
        Create GaussianARD node.
        """
        super().__init__(mu, alpha, ndim=ndim, shape=shape, **kwargs)


    @classmethod
    def _constructor(cls, mu, alpha, ndim=None, shape=None, **kwargs):
        r"""
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
        # Check consistency
        if ndim is not None and shape is not None and ndim != len(shape):
            raise ValueError("Given shape and ndim inconsistent")
        if ndim is None and shape is not None:
            ndim = len(shape)

        # Infer shape of mu
        try:
            # Case: mu is a node
            mu = mu._convert(GaussianGammaARDMoments)
        except AttributeError:
            # Case: mu is constant, we can use it as a scalar
            shape_mu = ()
            mu = cls._ensure_moments(mu, GaussianMoments(0))
            mu = mu._convert(GaussianGammaARDMoments)
        else:
            shape_mu = mu.dims[0]

        ndim_mu = len(shape_mu)

        # Infer dimensionality
        if ndim is None:
            ndim = ndim_mu #max(ndim_mu, ndim_alpha)
        elif ndim < ndim_mu: # or ndim < ndim_alpha:
            raise ValueError("Parent mu has more axes")

        # Infer shape of alpha
        alpha = cls._ensure_moments(alpha, GammaMoments())
        if ndim == 0:
            shape_alpha = ()
        else:
            shape_alpha = alpha.plates[-ndim:]

        # Infer shape of the node
        #shape_bc = misc.broadcasted_shape(shape_mu, shape_alpha)
        try:
            shape_bc = misc.broadcasted_shape(mu.plates+shape_mu, alpha.plates)
        except ValueError:
            raise ValueError("Parent nodes have incompatible shapes")
            
        if ndim == 0:
            shape_bc = ()
        elif ndim > len(shape_bc):
            shape_bc = (ndim-len(shape_bc))*(1,) + shape_bc
        else:
            shape_bc = shape_bc[-ndim:]

        # By default, use the broadcasted shape
        if shape is None:
            shape = shape_bc
        
        if not misc.is_shape_subset(shape_bc, shape):
            raise ValueError("Broadcasted shape of the parents %s does not "
                             "broadcast to the given shape %s" 
                             % (shape_bc, shape))
        
        mu_alpha = WrapToGaussianGammaARD(mu, alpha)
    
        # Check shape consistency
        shape_cov = shape[-ndim_mu:] + shape[-ndim_mu:]

        moments = GaussianMoments(ndim)
        parent_moments = [mu_alpha._moments]
        distribution = GaussianARDDistribution(shape, ndim_mu)

        dims = (shape, shape+shape)
        plates = cls._total_plates(kwargs.get('plates'),
                                   distribution.plates_from_parent(0, mu_alpha.plates))

        parents = [mu_alpha]

        return (parents, 
                kwargs,
                dims,
                plates,
                distribution,
                moments,
                parent_moments)
        

    def initialize_from_parameters(self, mu, alpha):
        # Explicit broadcasting so the shapes match
        mu = mu * np.ones(np.shape(alpha))
        alpha = alpha * np.ones(np.shape(mu))
        # Compute parent moments
        u = self._parent_moments[0].compute_fixed_moments(mu, alpha)
        # Initialize distribution
        self._initialize_from_parent_moments(u)


    def initialize_from_mean_and_covariance(self, mu, Cov):
        ndim = len(self._distribution.shape)
        u = [mu, Cov + linalg.outer(mu, mu, ndim=ndim)]
        mask = np.logical_not(self.observed)
        # TODO: You could compute the CGF but it requires Cholesky of
        # Cov. Do it later.
        self._set_moments_and_cgf(u, np.nan, mask=mask)
        return

    def __str__(self):
        mu = self.u[0]
        Cov = self.u[1] - misc.m_outer(mu, mu)
        return ("%s ~ Gaussian(mu, Cov)\n"
                "  mu = \n"
                "%s\n"
                "  Cov = \n"
                "%s\n"
                % (self.name, mu, Cov))


    def rotate(self, R, inv=None, logdet=None, axis=-1, Q=None, subset=None):

        ndim = len(self._distribution.shape)
        
        if inv is not None:
            invR = inv
        else:
            invR = np.linalg.inv(R)

        if logdet is not None:
            logdetR = logdet
        else:
            logdetR = np.linalg.slogdet(R)[1]

        self.phi[0] = rotate_mean(self.phi[0], invR.T,
                                  axis=axis,
                                  ndim=ndim)
        self.phi[1] = rotate_covariance(self.phi[1], invR.T,
                                        axis=axis,
                                        ndim=ndim)
        self.u[0] = rotate_mean(self.u[0], R,
                                axis=axis,
                                ndim=ndim)
        self.u[1] = rotate_covariance(self.u[1], R, 
                                      axis=axis,
                                      ndim=ndim)
        s = list(self.dims[0])
        s.pop(axis)
        self.g -= logdetR * np.prod(s)

        return

    def rotate_plates(self, Q, plate_axis=-1):
        r"""
        Approximate rotation of a plate axis.

        Mean is rotated exactly but covariance/precision matrix is rotated
        approximately.
        """

        ndim = len(self._distribution.shape)
        
        # Rotate moments using Q
        if not isinstance(plate_axis, int):
            raise ValueError("Plate axis must be integer")
        if plate_axis >= 0:
            plate_axis -= len(self.plates)
        if plate_axis < -len(self.plates) or plate_axis >= 0:
            raise ValueError("Axis out of bounds")

        u0 = rotate_mean(self.u[0], Q, 
                         ndim=ndim+(-plate_axis),
                         axis=0)
        sumQ = misc.add_trailing_axes(np.sum(Q, axis=0),
                                      2*ndim-plate_axis-1)
        phi1 = sumQ**(-2) * self.phi[1]
        phi0 = -2 * matrix_dot_vector(phi1, u0, ndim=ndim)

        self.phi[0] = phi0
        self.phi[1] = phi1

        self._update_moments_and_cgf()

        return


class GaussianGammaISO(ExponentialFamily):
    r"""
    Node for Gaussian-gamma (isotropic) random variables.

    The prior:
    
    .. math::

        p(x, \alpha| \mu, \Lambda, a, b)

        p(x|\alpha, \mu, \Lambda) = \mathcal{N}(x | \mu, \alpha Lambda)

        p(\alpha|a, b) = \mathcal{G}(\alpha | a, b)

    The posterior approximation :math:`q(x, \alpha)` has the same Gaussian-gamma
    form.

    Currently, supports only vector variables.
    """
    
    _moments = GaussianGammaISOMoments(1)
    _parent_moments = (GaussianWishartMoments(),
                       GammaPriorMoments(),
                       GammaMoments())
    _distribution = GaussianGammaISODistribution()
    

    @classmethod
    def _constructor(cls, mu, Lambda, a, b, **kwargs):
        r"""
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        `mu` is the mean/location vector
        `alpha` is the scale
        `V` is the scale matrix
        `n` is the degrees of freedom
        """

        # Convert parent nodes
        mu_Lambda = WrapToGaussianWishart(mu, Lambda)
        a = cls._ensure_moments(a, cls._parent_moments[1])
        b = cls._ensure_moments(b, cls._parent_moments[2])

        D = mu_Lambda.dims[0][0]

        # Check shapes
        if mu_Lambda.dims != ( (D,), (), (D,D), () ):
            raise ValueError("mu and Lambda have wrong shape")
        if a.dims != ( (), () ):
            raise ValueError("a has wrong shape")
        if b.dims != ( (), () ):
            raise ValueError("b has wrong shape")

        # Shapes of the moments / natural parameters
        dims = ( (D,), (D,D), (), () )

        # List of parent nodes
        parents = [mu_Lambda, a, b]

        return (parents,
                kwargs,
                dims,
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu_Lambda.plates),
                                  cls._distribution.plates_from_parent(1, a.plates),
                                  cls._distribution.plates_from_parent(2, b.plates)),
                cls._distribution, 
                cls._moments, 
                cls._parent_moments)

    
    def plotmatrix(self):
        r"""
        Creates a matrix of marginal plots.

        On diagonal, are marginal plots of each variable. Off-diagonal plot
        (i,j) shows the joint marginal density of x_i and x_j.
        """
        import bayespy.plot as bpplt
        
        if np.prod(self.plates) != 1:
            raise ValueError("Currently, does not support plates in the node.")

        if len(self.dims[0]) != 1:
            raise ValueError("Currently, supports only vector variables")

        # Dimensionality of the Gaussian
        D = self.dims[0][0]

        # Compute standard parameters
        tau = self.u[2]
        mu = self.u[0]
        mu = mu / misc.add_trailing_axes(tau, 1)
        Cov = self.u[1] - linalg.outer(self.u[0], mu, ndim=1)
        Cov = Cov / misc.add_trailing_axes(tau, 2)
        a = self.phi[3]
        b = -self.phi[2] - 0.5*linalg.inner(self.phi[0], mu, ndim=1)

        # Create subplots
        (fig, axes) = bpplt.pyplot.subplots(D+1, D+1)

        # Plot marginal Student t distributions
        for i in range(D):
            for j in range(i+1):
                if i == j:
                    bpplt._pdf_t(*(random.gaussian_gamma_to_t(mu[i],
                                                              Cov[i,i],
                                                              a,
                                                              b,
                                                              ndim=0)),
                                 axes=axes[i,i])
                else:
                    S = Cov[np.ix_([i,j],[i,j])]
                    (m, S, nu) = random.gaussian_gamma_to_t(mu[[i,j]],
                                                            S,
                                                            a,
                                                            b)
                    bpplt._contour_t(m, S, nu, axes=axes[i,j])
                    bpplt._contour_t(m, S, nu, axes=axes[j,i], transpose=True)

        # Plot Gaussian-gamma marginal distributions
        for k in range(D):
            bpplt._contour_gaussian_gamma(mu[k], Cov[k,k], a, b, 
                                          axes=axes[D,k])
            bpplt._contour_gaussian_gamma(mu[k], Cov[k,k], a, b, 
                                          axes=axes[k,D],
                                          transpose=True)

        # Plot gamma marginal distribution
        bpplt._pdf_gamma(a, b, axes=axes[D,D])
        return axes


    def get_gaussian_mean_and_variance(self):
        r"""
        Return the mean and variance of the distribution
        """
        a = self.phi[3]
        nu = 2*a
        
        if nu <= 1:
            raise ValueError("Mean not defined for degrees of freedom <= 1")
        if nu <= 2:
            raise ValueError("Variance not defined if degrees of freedom <= 2")

        tau = self.u[2]
        tau_mu = self.u[0]
        mu = tau_mu / misc.add_trailing_axes(tau, 1)
        var = misc.get_diag(self.u[1], ndim=1) - tau_mu*mu
        var = var / misc.add_trailing_axes(tau, 1)

        var = nu / (nu-2) * var

        return (mu, var)


    def get_marginal_logpdf(self, gaussian=None, gamma=None):
        r"""
        Get the (marginal) log pdf of a subset of the variables

        Parameters
        ----------
        gaussian : list or None
            Indices of the Gaussian variables to keep or None
        gamma : bool or None
            True if keep the gamma variable, otherwise False or None

        Returns
        -------
        function
            A function which computes log-pdf
        """
        if gaussian is None and not gamma:
            raise ValueError("Must give some variables")

        # Compute standard parameters
        tau = self.u[2]
        mu = self.u[0]
        mu = mu / misc.add_trailing_axes(tau, 1)
        Cov = np.linalg.inv(-2*self.phi[1]) 
        if not np.allclose(Cov,
                           self.u[1] - linalg.outer(self.u[0], mu, ndim=1)):
            raise Exception("WAAAT")
        #Cov = Cov / misc.add_trailing_axes(tau, 2)
        a = self.phi[3]
        b = -self.phi[2] - 0.5*linalg.inner(self.phi[0], mu, ndim=1)

        if not gamma:
            # Student t distributions
            inds = list(gaussian)
            mu = mu[inds]
            Cov = Cov[np.ix_(inds, inds)]
            (mu, Cov, nu) = random.gaussian_gamma_to_t(mu,
                                                       Cov,
                                                       a,
                                                       b,
                                                       ndim=1)
            L = linalg.chol(Cov)
            logdet_Cov = linalg.chol_logdet(L)
            D = len(inds)
            def logpdf(x):
                y = x - mu
                v = linalg.chol_solve(L, y)
                z2 = linalg.inner(y, v, ndim=1)
                return random.t_logpdf(z2, logdet_Cov, nu, D)
            return logpdf
                
        elif gaussian is None:
            # Gamma distribution
            def logpdf(x):
                logx = np.log(x)
                return random.gamma_logpdf(b*x, 
                                           logx,
                                           a*logx,
                                           a*np.log(b),
                                           special.gammaln(a))
            return logpdf
        else:
            # Gaussian-gamma distribution
            inds = list(gaussian)
            mu = mu[inds]
            Cov = Cov[np.ix_(inds, inds)]
            D = len(inds)

            L = linalg.chol(Cov)
            logdet_Cov = linalg.chol_logdet(L)
            
            def logpdf(x):
                tau = x[...,-1]
                logtau = np.log(tau)
                x = x[...,:-1]

                y = x - mu
                v = linalg.chol_solve(L, y) * tau[...,None]
                z2 = linalg.inner(y, v, ndim=1)

                return (random.gaussian_logpdf(z2, 
                                               0,
                                               0,
                                               logdet_Cov + D*logtau, 
                                               D) +
                        random.gamma_logpdf(b*tau,
                                            logtau,
                                            a*logtau, 
                                            a*np.log(b), 
                                            special.gammaln(a)))

            return logpdf


class GaussianGammaARD(ExponentialFamily):
    r"""
    Node for Gaussian and gamma random variables with ARD form.

    The prior:
    
    .. math::

        p(x, \tau| \mu, \alpha, a, b) = p(x|\tau, \mu, \alpha) p(\tau|a, b)

        p(x|\alpha, \mu, \alpha) = \mathcal{N}(x | \mu, \mathrm{diag}(
        \boldsymbol{\alpha} \boldsymbol{\tau} ))

        p(\tau|a, b) = \mathcal{G}(\tau | a, b)

    The posterior approximation :math:`q(x, \tau)` has the same Gaussian-gamma
    form.

    .. warning:: Not yet implemented.

    See also
    --------
    
    Gaussian, GaussianARD, Gamma, GaussianGammaISO, GaussianWishart
    """


    def __init__(self, mu, alpha, a, b, **kwargs):
        r"""
        """
        raise NotImplementedError()

    
class GaussianWishart(ExponentialFamily):
    r"""
    Node for Gaussian-Wishart random variables.

    The prior:
    
    .. math::

        p(x, \Lambda| \mu, \alpha, V, n)

        p(x|\Lambda, \mu, \alpha) = \mathcal(N)(x | \mu, \alpha^{-1} Lambda^{-1})

        p(\Lambda|V, n) = \mathcal(W)(\Lambda | n, V)

    The posterior approximation :math:`q(x, \Lambda)` has the same Gaussian-Wishart form.

    Currently, supports only vector variables.
    """
    
    _moments = GaussianWishartMoments()
    _distribution = GaussianWishartDistribution()
    

    @classmethod
    def _constructor(cls, mu, alpha, n, V, **kwargs):
        r"""
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        `mu` is the mean/location vector
        `alpha` is the scale
        `n` is the degrees of freedom
        `V` is the scale matrix
        """

        # Convert parent nodes
        mu_alpha = WrapToGaussianGammaISO(mu, alpha)
        D = mu_alpha.dims[0][0]
        
        parent_moments = (GaussianGammaISOMoments(1),
                          WishartMoments(),
                          WishartPriorMoments(D))
        n = cls._ensure_moments(n, parent_moments[1])
        V = cls._ensure_moments(V, parent_moments[2])


        # Check shapes
        if mu_alpha.dims != ( (D,), (D,D), (), () ):
            raise ValueError("mu and alpha have wrong shape")

        if V.dims != ( (D,D), () ):
            raise ValueError("Precision matrix has wrong shape")

        if n.dims != ( (), () ):
            raise ValueError("Degrees of freedom has wrong shape")

        dims = ( (D,), (), (D,D), () )

        parents = [mu_alpha, n, V]

        return (parents,
                kwargs,
                dims,
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu_alpha.plates),
                                  cls._distribution.plates_from_parent(1, n.plates),
                                  cls._distribution.plates_from_parent(2, V.plates)),
                cls._distribution, 
                cls._moments, 
                parent_moments)

    

#
# CONVERTERS
#


## class GaussianToGaussianARD(Deterministic):
##     """
##     Converter for Gaussian moments to Gaussian ARD moments
##     """


##     def __init__(self, X, **kwargs):
##         """
##         """
##         self.ndim = X._moments.ndim
        
##         self._moments = GaussianARDMoments()
##         self._parent_moments = [GaussianMoments(self.ndim)]
    
##         dims = ( (), () )
##         super().__init__(X, dims=dims, **kwargs)
            

##     def _compute_moments(self, u_X):
##         """
##         """
##         x = u_X[0]
##         xx = u_X[1]
##         u0 = x
##         u1 = misc.get_diag(xx, ndim=self.ndim)
##         u = [u0, u1]
##         return u
    

##     def _compute_message_to_parent(self, index, m_child, u_X):
##         """
##         """
##         if index == 0:
##             m0 = m_child[0]
##             m1 = misc.diag(m_child[1], ndim=self.ndim)
##             m = [m0, m1]
##             return m
##         else:
##             raise ValueError("Invalid parent index")


##     def _compute_mask_to_parent(self, index, mask):
##         """
##         """
##         if index == 0:
##             if ndim == 0:
##                 return mask
##             else:
##                 axes = tuple(range(-self.ndim, 0))
##                 return np.any(mask, axis=axes)
##         else:
##             raise ValueError("Invalid parent index")


##     def _plates_to_parent(self, index):
##         """
##         """
##         if index == 0:
##             if ndim == 0:
##                 return self.plates
##             else:
##                 return self.plates[:-self.ndim]
##         else:
##             raise ValueError("Invalid parent index")


##     def _plates_from_parent(self, index):
##         """
##         """
##         if index == 0:
##             shape = self.parents[0].dims[0]
##             return self.parents[0].plates + shape
##         else:
##             raise ValueError("Invalid parent index")


## GaussianMoments.add_converter(GaussianARDMoments,
##                               GaussianToGaussianARD)


## class GaussianARDToGaussianISO(Deterministic):
##     """
##     Converter for Gaussian moments to Gaussian-gamma isotropic moments

##     Combines the Gaussian moments with gamma moments for a fixed value 1.
##     """



##     def __init__(self, X, **kwargs):
##         """
##         """
##         self.ndim = X._moments.ndim
        
##         self._moments = GaussianISOMoments(self.ndim)
##         self._parent_moments = [GaussianARDMoments()]
    
##         shape = X.dims[0]
##         dims = ( shape, 2*shape, (), () )
##         super().__init__(X, dims=dims, **kwargs)
            

##     def _compute_moments(self, u_X):
##         """
##         """
##         x = u_X[0]
##         xx = u_X[1]
##         u = [x, xx, 1, 0]
##         return u
    

##     def _compute_message_to_parent(self, index, m_child, u_X):
##         """
##         """
##         if index == 0:
##             m = m_child[:2]
##             return m
##         else:
##             raise ValueError("Invalid parent index")


## GaussianARDMoments.add_converter(GaussianISOMoments,
##                                  GaussianARDToGaussianISO)


class GaussianToGaussianGammaISO(Deterministic):
    r"""
    Converter for Gaussian moments to Gaussian-gamma isotropic moments

    Combines the Gaussian moments with gamma moments for a fixed value 1.
    """



    def __init__(self, X, **kwargs):
        r"""
        """
        self.ndim = X._moments.ndim
        
        self._moments = GaussianGammaISOMoments(self.ndim)
        self._parent_moments = [GaussianMoments(self.ndim)]
    
        shape = X.dims[0]
        dims = ( shape, 2*shape, (), () )
        super().__init__(X, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X):
        r"""
        """
        x = u_X[0]
        xx = u_X[1]
        u = [x, xx, 1, 0]
        return u
    

    def _compute_message_to_parent(self, index, m_child, u_X):
        r"""
        """
        if index == 0:
            m = m_child[:2]
            return m
        else:
            raise ValueError("Invalid parent index")


GaussianMoments.add_converter(GaussianGammaISOMoments,
                              GaussianToGaussianGammaISO)


class GaussianGammaISOToGaussianGammaARD(Deterministic):
    r"""
    Converter for Gaussian-gamma ISO moments to Gaussian-gamma ARD moments
    """


    def __init__(self, X, **kwargs):
        r"""
        """
        self.ndim = X._moments.ndim
        
        self._moments = GaussianGammaARDMoments(self.ndim)
        self._parent_moments = [GaussianGammaISOMoments(self.ndim)]
    
        shape = X.dims[0]
        dims = ( shape, shape, shape, shape )
        super().__init__(X, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X_alpha):
        r"""
        ...

        .. math::

            u_0 &= \alpha x
            \\
            u_1 &= \alpha \mathrm{diag}(xx^T)
            \\
            u_2 &= [\alpha, \ldots, \alpha]
            \\
            u_3 &= [\log(\alpha), \ldots, \log(\alpha)]
        """
        shape = self.dims[0]
        alpha_x = u_X_alpha[0]
        alpha_xx = misc.get_diag(u_X_alpha[1], ndim=self.ndim)
        alpha = misc.add_trailing_axes(u_X_alpha[2], self.ndim) * np.ones(shape)
        logalpha = misc.add_trailing_axes(u_X_alpha[3], self.ndim) * np.ones(shape)
        u = [alpha_x, alpha_xx, alpha, logalpha]
        return u
    

    def _compute_message_to_parent(self, index, m_child, u_X_alpha):
        r"""
        ...

        Message from the child is :math:`[m_0, m_1, m_2, m_3]`:
        
        .. math::

            \alpha m_0^T x + m_1 \alpha m_1^T \mathrm{diag}(xx^T) +
            \alpha\mathrm{sum}(m_2) + \mathrm{sum}(m_3) \log|\alpha|

        Thus, message to the first parent is (in case of Gaussian-gamma and
        Wishart parents):

        .. math::

            \tilde{m_0} &= m_0
            \\
            \tilde{m_1} &= \mathrm{diag}(m_1)
            \\
            \tilde{m_2} &= \mathrm{sum}(m_2)
            \\
            \tilde{m_3} &= \mathrm{sum}(m_3)
        """
        if index == 0:
            m0 = m_child[0]
            m1 = misc.diag(m_child[1], ndim=self.ndim)
            m2 = np.sum(m_child[2], axis=tuple(range(-self.ndim,0)))
            m3 = np.sum(m_child[3], axis=tuple(range(-self.ndim,0)))
            m = [m0, m1, m2, m3]
            return m
        else:
            raise ValueError("Invalid parent index")


GaussianGammaISOMoments.add_converter(GaussianGammaARDMoments,
                                      GaussianGammaISOToGaussianGammaARD)


class GaussianGammaARDToGaussianWishart(Deterministic):
    r"""
    """


    def __init__(self, X_alpha, **kwargs):
        raise NotImplementedError()


GaussianGammaARDMoments.add_converter(GaussianWishartMoments,
                                      GaussianGammaARDToGaussianWishart)


## class GaussianGammaISOToGamma(Deterministic):
##     """
##     """


##     def __init__(self):
##         raise NotImplementedError()


## class GaussianGammaARDToGamma(Deterministic):
##     """
##     """


##     def __init__(self):
##         raise NotImplementedError()


## class GaussianWishartToWishart(Deterministic):
##     """
##     """


##     def __init__(self):
##         raise NotImplementedError()


#
# WRAPPERS
#
# These wrappers form a single node from two nodes for messaging purposes.
#


class WrapToGaussianGammaISO(Deterministic):
    r"""
    """


    _moments = GaussianGammaISOMoments(1)
    _parent_moments = [GaussianGammaISOMoments(1),
                       GammaMoments()]
    

    @ensureparents
    def __init__(self, X, alpha, **kwargs):
        r"""
        """
        D = X.dims[0][0]
        dims = ( (D,), (D,D), (), () )
        super().__init__(X, alpha, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X, u_alpha):
        r"""
        """
        raise NotImplementedError()
    

    def _compute_message_to_parent(self, index, m_child, u_X, u_alpha):
        r"""
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _compute_mask_to_parent(self, index, mask):
        r"""
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_to_parent(self, index):
        r"""
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_from_parent(self, index):
        r"""
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


class WrapToGaussianGammaARD(Deterministic):
    r"""
    """


    def __init__(self, mu_alpha, tau, **kwargs):
        r"""
        """

        # First, just in case mu_alpha is a numeric array, convert mu_alpha to
        # (constant) Gaussian.
        ## if not isinstance(mu_alpha, Node):
        ##     raise ValueError("Mu must be a node")
        ##     if ndim is None:
        ##         raise ValueError("For non-node mu, provide ndim")
        ##     try:
        ##         mu_alpha = self._ensure_moments(mu_alpha, GaussianMoments(ndim))
        ##     except Moments.NoConverterError:
        ##         pass

        # Ensure proper moments from parents
        try:
            mu_alpha = mu_alpha._convert(GaussianGammaARDMoments)
        except AttributeError:
            raise ValueError("Mu must be a node")

        # Parent moments
        self._parent_moments = [mu_alpha._moments,
                                GammaMoments()]
        tau = self._ensure_moments(tau, self._parent_moments[1])

        ndim = len(mu_alpha.dims[0])

        if ndim == 0:
            shape = ()
        else:
            shape_mu = mu_alpha.dims[0]
            shape_tau = tau.plates[-ndim:]
            shape = misc.broadcasted_shape(shape_mu, shape_tau)
            
        self.ndim = len(shape)
        dims = ( shape, shape, shape, shape )

        self._moments = GaussianGammaARDMoments(self.ndim)
        
        super().__init__(mu_alpha, tau, dims=dims, **kwargs)


    def _compute_moments(self, u_mu_alpha, u_tau):
        r"""
        """
        mu_alpha = u_mu_alpha[0]
        mu2_alpha = u_mu_alpha[1]
        alpha = u_mu_alpha[2]
        logalpha = u_mu_alpha[3]
            
        tau = u_tau[0]
        logtau = u_tau[1]

        u0 = mu_alpha * tau
        u1 = mu2_alpha * tau
        u2 = alpha * tau
        u3 = logalpha + logtau
        u = [u0, u1, u2, u3]

        return u
    

    def _compute_message_to_parent(self, index, m_child, u_mu_alpha, u_tau):
        r"""
        ...
        
        Message from the child is :math:`[m_0, m_1, m_2, m_3]`:
        
        .. math::

            m_0^T \mathrm{diag}(\alpha \circ \tau) \mu +  
            m_1^T \mathrm{diag}(\alpha \circ \tau) (\mu \circ \mu) + 
            m_2^T (\alpha \circ \tau) + 
            m_3^T \circ (\log \alpha + \log \tau)

        Thus, message to the first parent is:

        .. math::

            \tilde{m_0} &= m_0 \circ \tau
            \\
            \tilde{m_1} &= m_1 \circ \tau
            \\
            \tilde{m_2} &= m_2 \circ \tau
            \\
            \tilde{m_3} &= m_3

        Sum those to proper shape of mu-alpha.

        The message to the second parent is:

        .. math::

            \tilde{m_0} &= \alpha \circ (m_0 \circ \mu + m_1 \circ \mu \circ \mu + m_2)
            \\
            \tilde{m_1} &= m_3
        """
        if index == 0:
            tau = u_tau[0]
            m0 = m_child[0] * tau
            m1 = m_child[1] * tau
            m2 = m_child[2] * tau
            m3 = m_child[3]
            # Sum the broadcasted variable axes. Plate axes are handled by
            # default.
            shape_mu = self.parents[0].dims[0]
            ndim_mu = len(shape_mu)
            if ndim_mu > 0 and shape_mu != self.dims[0]:
                plates_m0 = np.shape(m0)[:-ndim_mu]
                plates_m1 = np.shape(m1)[:-ndim_mu]
                plates_m2 = np.shape(m2)[:-ndim_mu]
                plates_m3 = np.shape(m3)[:-ndim_mu]
                m0 = misc.sum_to_shape(m0, plates_m0 + shape_mu)
                m1 = misc.sum_to_shape(m1, plates_m1 + shape_mu)
                m2 = misc.sum_to_shape(m2, plates_m2 + shape_mu)
                m3 = misc.sum_to_shape(m3, plates_m3 + shape_mu)
            m = [m0, m1, m2, m3]
            return m
        elif index == 1:
            ndim_mu = len(self.parents[0].dims[0])
            alpha_mu = u_mu_alpha[0]
            alpha_mu2 = u_mu_alpha[1]
            alpha = u_mu_alpha[2]
            m0 = m_child[0]*alpha_mu + m_child[1]*alpha_mu2 + m_child[2]*alpha
            m1 = m_child[3]
            m = [m0, m1]
            return m
        else:
            raise ValueError("Invalid parent index")


    def _compute_mask_to_parent(self, index, mask):
        r"""
        """
        if index == 0:
            return mask
        elif index == 1:
            return misc.add_trailing_axes(mask, self.ndim)
        else:
            raise ValueError("Invalid parent index")


    def _plates_to_parent(self, index):
        r"""
        """
        if index == 0:
            return self.plates
        elif index == 1:
            shape = self.dims[0]
            return self.plates + shape
        else:
            raise ValueError("Invalid parent index")


    def _plates_from_parent(self, index):
        r"""
        """
        if index == 0:
            return self.parents[0].plates
        elif index == 1:
            if self.ndim == 0:
                return self.parents[1].plates
            else:
                return self.parents[1].plates[:-self.ndim]
        else:
            raise ValueError("Invalid parent index")


class WrapToGaussianWishart(Deterministic):
    r"""
    Wraps Gaussian and Wishart nodes into a Gaussian-Wishart node.

    The following node combinations can be wrapped:
        * Gaussian and Wishart
        * Gaussian-gamma and Wishart
        * Gaussian-Wishart and gamma
    """


    _moments = GaussianWishartMoments()
    

    def __init__(self, X, Lambda, **kwargs):
        r"""
        """

        # Just in case X is an array, convert it to a Gaussian node first.
        try:
            X = self._ensure_moments(X, GaussianMoments(1))
        except Moments.NoConverterError:
            pass

        try:
            # Try combo Gaussian-Gamma and Wishart
            X = self._ensure_moments(X, GaussianGammaISOMoments(1))
        except ValueError:
            # Have to use Gaussian-Wishart and Gamma
            self._parent_moments = [GaussianWishartMoments(),
                                    GammaMoments()]
            X = self._ensure_moments(X, GaussianWishartMoments())
            self.wishart = False
        else:
            self._parent_moments = [GaussianGammaISOMoments(1),
                                    WishartMoments()]
            self.wishart = True

        D = X.dims[0][0]
        dims = ( (D,), (), (D,D), () )
        super().__init__(X, Lambda, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X_alpha, u_Lambda):
        r"""
        """
        if self.wishart:
            alpha_x = u_X_alpha[0]
            alpha_xx = u_X_alpha[1]
            alpha = u_X_alpha[2]
            log_alpha = u_X_alpha[3]
            Lambda = u_Lambda[0]
            logdet_Lambda = u_Lambda[1]

            D = np.prod(self.dims[0])
            
            u0 = linalg.mvdot(Lambda, alpha_x)
            u1 = linalg.inner(Lambda, alpha_xx, ndim=2)
            u2 = Lambda * misc.add_trailing_axes(alpha, 2)
            u3 = logdet_Lambda + D * log_alpha
            u = [u0, u1, u2, u3]

            return u
        else:
            raise NotImplementedError()
    

    def _compute_message_to_parent(self, index, m_child, u_X_alpha, u_Lambda):
        r"""
        ...

        Message from the child is :math:`[m_0, m_1, m_2, m_3]`:
        
        .. math::

            \alpha m_0^T \Lambda x + m_1 \alpha x^T \Lambda x
            + \mathrm{tr}(\alpha m_2 \Lambda) + m_3 (\log | \alpha \Lambda |)

        In case of Gaussian-gamma and Wishart parents:
        
        Message to the first parent (x, alpha):

        .. math::

            \tilde{m_0} &= \Lambda m_0
            \\
            \tilde{m_1} &= m_1 \Lambda
            \\
            \tilde{m_2} &= \mathrm{tr}(m_2 \Lambda)
            \\
            \tilde{m_3} &= m_3 \cdot D

        Message to the second parent (Lambda):

        .. math::

            \tilde{m_0} &= \alpha (\frac{1}{2} m_0 x^T + \frac{1}{2} x m_0^T +
            m_1 xx^T + m_2)
            \\
            \tilde{m_1} &= m_3
        """
        if index == 0:
            if self.wishart:
                # Message to Gaussian-gamma (isotropic)
                Lambda = u_Lambda[0]
                D = np.prod(self.dims[0])
                m0 = linalg.mvdot(Lambda, m_child[0])
                m1 = Lambda * misc.add_trailing_axes(m_child[1], 2)
                m2 = linalg.inner(Lambda, m_child[2], ndim=2)
                m3 = D * m_child[3]
                m = [m0, m1, m2, m3]
                return m
            else:
                # Message to Gaussian-Wishart
                raise NotImplementedError()
        elif index == 1:
            if self.wishart:
                # Message to Wishart
                alpha_x = u_X_alpha[0]
                alpha_xx = u_X_alpha[1]
                alpha = u_X_alpha[2]
                m0 = (0.5*linalg.outer(alpha_x, m_child[0], ndim=1) +
                      0.5*linalg.outer(m_child[0], alpha_x, ndim=1) +
                      alpha_xx * misc.add_trailing_axes(m_child[1], 2) +
                      misc.add_trailing_axes(alpha, 2) * m_child[2])
                m1 = m_child[3]
                m = [m0, m1]
                return m
            else:
                # Message to gamma (isotropic)
                raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")



def reshape_gaussian_array(dims_from, dims_to, x0, x1):
    r"""
    Reshape the moments Gaussian array variable.

    The plates remain unaffected.
    """
    num_dims_from = len(dims_from)
    num_dims_to = len(dims_to)

    # Reshape the first moment / mean
    num_plates_from = np.ndim(x0) - num_dims_from
    plates_from = np.shape(x0)[:num_plates_from]
    shape = (
        plates_from 
        + (1,)*(num_dims_to-num_dims_from) + dims_from
        )
    x0 = np.ones(dims_to) * np.reshape(x0, shape)

    # Reshape the second moment / covariance / precision
    num_plates_from = np.ndim(x1) - 2*num_dims_from
    plates_from = np.shape(x1)[:num_plates_from]
    shape = (
        plates_from 
        + (1,)*(num_dims_to-num_dims_from) + dims_from
        + (1,)*(num_dims_to-num_dims_from) + dims_from
        )
    x1 = np.ones(dims_to+dims_to) * np.reshape(x1, shape)

    return (x0, x1)

def transpose_covariance(Cov, ndim=1):
    r"""
    Transpose the covariance array of Gaussian array variable.

    That is, swap the last ndim axes with the ndim axes before them. This makes
    transposing easy for array variables when the covariance is not a matrix but
    a multidimensional array.
    """
    axes_in = [Ellipsis] + list(range(2*ndim,0,-1))
    axes_out = [Ellipsis] + list(range(ndim,0,-1)) + list(range(2*ndim,ndim,-1))
    return np.einsum(Cov, axes_in, axes_out)

def left_rotate_covariance(Cov, R, axis=-1, ndim=1):
    r"""
    Rotate the covariance array of Gaussian array variable.

    ndim is the number of axes for the Gaussian variable.

    For vector variable, ndim=1 and covariance is a matrix.
    """
    if not isinstance(axis, int):
        raise ValueError("Axis must be an integer")
    if axis < -ndim or axis >= ndim:
        raise ValueError("Axis out of range")

    # Force negative axis
    if axis >= 0:
        axis -= ndim

    # Rotation from left
    axes_R = [Ellipsis, ndim+abs(axis)+1, ndim+abs(axis)]
    axes_Cov = [Ellipsis] + list(range(ndim+abs(axis),
                                       0,
                                       -1))
    axes_out = [Ellipsis, ndim+abs(axis)+1] + list(range(ndim+abs(axis)-1,
                                                         0,
                                                         -1))
    Cov = np.einsum(R, axes_R, Cov, axes_Cov, axes_out)

    return Cov
    
def right_rotate_covariance(Cov, R, axis=-1, ndim=1):
    r"""
    Rotate the covariance array of Gaussian array variable.

    ndim is the number of axes for the Gaussian variable.

    For vector variable, ndim=1 and covariance is a matrix.
    """
    if not isinstance(axis, int):
        raise ValueError("Axis must be an integer")
    if axis < -ndim or axis >= ndim:
        raise ValueError("Axis out of range")

    # Force negative axis
    if axis >= 0:
        axis -= ndim

    # Rotation from right
    axes_R = [Ellipsis, abs(axis)+1, abs(axis)]
    axes_Cov = [Ellipsis] + list(range(abs(axis),
                                       0,
                                       -1))
    axes_out = [Ellipsis, abs(axis)+1] + list(range(abs(axis)-1,
                                                    0,
                                                    -1))
    Cov = np.einsum(R, axes_R, Cov, axes_Cov, axes_out)

    return Cov
    
def rotate_covariance(Cov, R, axis=-1, ndim=1):
    r"""
    Rotate the covariance array of Gaussian array variable.

    ndim is the number of axes for the Gaussian variable.

    For vector variable, ndim=1 and covariance is a matrix.
    """

    # Rotate from left and right
    Cov = left_rotate_covariance(Cov, R, ndim=ndim, axis=axis)
    Cov = right_rotate_covariance(Cov, R, ndim=ndim, axis=axis)

    return Cov

def rotate_mean(mu, R, axis=-1, ndim=1):
    r"""
    Rotate the mean array of Gaussian array variable.

    ndim is the number of axes for the Gaussian variable.

    For vector variable, ndim=1 and mu is a vector.
    """
    if not isinstance(axis, int):
        raise ValueError("Axis must be an integer")
    if axis < -ndim or axis >= ndim:
        raise ValueError("Axis out of range")

    # Force negative axis
    if axis >= 0:
        axis -= ndim

    # Rotation from right
    axes_R = [Ellipsis, abs(axis)+1, abs(axis)]
    axes_mu = [Ellipsis] + list(range(abs(axis),
                                      0,
                                      -1))
    axes_out = [Ellipsis, abs(axis)+1] + list(range(abs(axis)-1,
                                                    0,
                                                    -1))
    mu = np.einsum(R, axes_R, mu, axes_mu, axes_out)

    return mu

def array_to_vector(x, ndim=1):
    if ndim == 0:
        return x
    
    shape_x = np.shape(x)
    D = np.prod(shape_x[-ndim:])
    
    return np.reshape(x, shape_x[:-ndim] + (D,))

def array_to_matrix(A, ndim=1):
    if ndim == 0:
        return A

    shape_A = np.shape(A)
    D = np.prod(shape_A[-ndim:])
    return np.reshape(A, shape_A[:-2*ndim] + (D,D))

def vector_to_array(x, shape):
    shape_x = np.shape(x)
    return np.reshape(x, np.shape(x)[:-1] + tuple(shape))
    
def matrix_dot_vector(A, x, ndim=1):
    if ndim < 0:
        raise ValueError("ndim must be non-negative integer")
    if ndim == 0:
        return A*x

    dims_x = np.shape(x)[-ndim:]
    
    A = array_to_matrix(A, ndim=ndim)
    x = array_to_vector(x, ndim=ndim)
    
    y = np.einsum('...ik,...k->...i', A, x)

    return vector_to_array(y, dims_x)
        
