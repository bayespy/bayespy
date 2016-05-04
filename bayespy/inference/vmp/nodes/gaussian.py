################################################################################
# Copyright (C) 2011-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


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


class GaussianMoments(Moments):
    r"""
    Class for the moments of Gaussian variables.
    """


    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.dims = (shape, 2*shape)
        super().__init__()


    def compute_fixed_moments(self, x):
        r"""
        Compute the moments for a fixed value
        """
        x = misc.atleast_nd(x, self.ndim)
        return [x, linalg.outer(x, x, ndim=self.ndim)]


    @classmethod
    def from_values(cls, x, ndim):
        r"""
        Return the shape of the moments for a fixed value.
        """
        if ndim == 0:
            return cls(())
        else:
            return cls(np.shape(x)[-ndim:])


    def get_instance_conversion_kwargs(self):
        return dict(ndim=self.ndim)


    def get_instance_converter(self, ndim):
        if ndim == self.ndim or ndim is None:
            return None

        return GaussianToGaussian(self, ndim)


class GaussianToGaussian():


    def __init__(self, moments_from, ndim_to):
        if not isinstance(moments_from, GaussianMoments):
            raise ValueError()

        if ndim_to < 0:
            return ValueError("ndim_to must be non-negative")

        self.shape_from = moments_from.shape
        self.ndim_from = moments_from.ndim
        self.ndim_to = ndim_to

        if self.ndim_to > self.ndim_from:
            raise ValueError()

        if self.ndim_to == 0:
            self.moments = GaussianMoments(())
        else:
            self.moments = GaussianMoments(self.shape_from[-self.ndim_to:])

        return


    def compute_moments(self, u):
        if self.ndim_to == self.ndim_from:
            return u

        u0 = u[0]
        u1 = misc.get_diag(u[1], ndim=self.ndim_from, ndim_to=self.ndim_to)

        return [u0, u1]


    def compute_message_to_parent(self, m, u_parent):
        # Handle broadcasting in m_child
        m0 = m[0] * np.ones(self.shape_from)
        m1 = (
            misc.make_diag(m[1], ndim=self.ndim_from, ndim_from=self.ndim_to)
            * misc.identity(*self.shape_from)
        )
        return [m0, m1]


    def compute_weights_to_parent(self, weights):
        diff = self.ndim_from - self.ndim_to
        if diff == 0:
            return weights
        return np.sum(
            weights * np.ones(self.shape_from[:diff]),
            #misc.atleast_nd(weights, diff),
            axis=tuple(range(-diff, 0))
        )


    def plates_multiplier_from_parent(self, plates_multiplier):
        diff = self.ndim_from - self.ndim_to
        return plates_multiplier + diff * (1,)


    def plates_from_parent(self, plates):
        diff = self.ndim_from - self.ndim_to
        if diff == 0:
            return plates
        return plates + self.shape_from[:diff]


    def plates_to_parent(self, plates):
        diff = self.ndim_from - self.ndim_to
        if diff == 0:
            return plates
        return plates[:-diff]


class GaussianGammaMoments(Moments):
    r"""
    Class for the moments of Gaussian-gamma-ISO variables.
    """


    def __init__(self, shape):
        r"""
        Create moments object for Gaussian-gamma isotropic variables

        ndim=0: scalar
        ndim=1: vector
        ndim=2: matrix
        ...
        """
        self.shape = shape
        self.ndim = len(shape)
        self.dims = (shape, 2*shape, (), ())
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
              * misc.add_trailing_axes(alpha, 2*self.ndim))
        u2 = np.copy(alpha)
        u3 = np.log(alpha)
        u = [u0, u1, u2, u3]

        return u


    @classmethod
    def from_values(cls, x, alpha, ndim):
        r"""
        Return the shape of the moments for a fixed value.
        """
        if ndim == 0:
            shape = ( (), (), (), () )
        else:
            shape = np.shape(x)[-ndim:]
        return cls(shape)


    def get_instance_conversion_kwargs(self):
        return dict(ndim=self.ndim)


    def get_instance_converter(self, ndim):
        # FIXME/TODO: IMPLEMENT THIS CORRECTLY!
        if ndim != self.ndim:
            raise NotImplementedError(
                "Conversion to different ndim in GaussianMoments not yet "
                "implemented."
            )
        return None


class GaussianWishartMoments(Moments):
    r"""
    Class for the moments of Gaussian-Wishart variables.
    """


    def __init__(self, shape):

        self.shape = shape
        self.ndim = len(shape)
        self.dims = ( shape, (), 2*shape, () )

        super().__init__()


    def compute_fixed_moments(self, x, Lambda):
        r"""
        Compute the moments for a fixed value

        `x` is a vector.
        `Lambda` is a precision matrix
        """

        x = np.asanyarray(x)
        Lambda = np.asanyarray(Lambda)

        u0 = linalg.mvdot(Lambda, x, ndim=self.ndim)
        u1 = np.einsum(
            '...i,...ij,...j->...',
            misc.flatten_axes(x, self.ndim),
            misc.flatten_axes(Lambda, self.ndim, self.ndim),
            misc.flatten_axes(x, self.ndim)
        )
        u2 = np.copy(Lambda)
        u3 = linalg.logdet_cov(Lambda, ndim=self.ndim)

        return [u0, u1, u2, u3]


    @classmethod
    def from_values(self, x, Lambda, ndim):
        r"""
        Return the shape of the moments for a fixed value.
        """
        if ndim == 0:
            return cls(())
        else:
            if np.ndim(x) < ndim:
                raise ValueError("Mean must be a vector")
            shape = np.shape(x)[-ndim:]
            if np.shape(Lambda)[-2*ndim:] != shape + shape:
                raise ValueError("Shapes inconsistent")
            return cls(shape)



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


    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        super().__init__()


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
        L = linalg.chol(-2*phi[1], ndim=self.ndim)
        k = np.shape(phi[0])[-1]
        # Moments
        u0 = linalg.chol_solve(L, phi[0], ndim=self.ndim)
        u1 = (linalg.outer(u0, u0, ndim=self.ndim)
              + linalg.chol_inv(L, ndim=self.ndim))
        u = [u0, u1]
        # G
        g = (-0.5 * linalg.inner(u[0], phi[0], ndim=self.ndim)
             + 0.5 * linalg.chol_logdet(L, ndim=self.ndim))
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
        u = [x, linalg.outer(x, x, ndim=self.ndim)]
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
        x_x = linalg.outer(x, x, ndim=self.ndim)
        Cov = xx - x_x
        cov_g0 = linalg.mvdot(Cov, g[0], ndim=self.ndim)
        cov_g0_x = linalg.outer(cov_g0, x, ndim=self.ndim)
        g1_x = linalg.mvdot(g[1], x, ndim=self.ndim)
        # Compute gradient terms
        d0 = cov_g0 + 2 * linalg.mvdot(Cov, g1_x, ndim=self.ndim)
        d1 = (cov_g0_x + linalg.transpose(cov_g0_x, ndim=self.ndim)
              + 2 * linalg.mmdot(xx,
                                 linalg.mmdot(g[1], xx, ndim=self.ndim),
                                 ndim=self.ndim)
              - 2 * x_x * misc.add_trailing_axes(linalg.inner(g1_x,
                                                              x,
                                                              ndim=self.ndim),
                                                 2*self.ndim))

        return [d0, d1]


    def random(self, *phi, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = linalg.chol(-2*phi[1], ndim=self.ndim)
        mu = linalg.chol_solve(U, phi[0], ndim=self.ndim)
        shape = plates + self.shape
        z = np.random.randn(*shape)
        # Denote Lambda = -2*phi[1]
        # Then, Cov = inv(Lambda) = inv(U'*U) = inv(U) * inv(U')
        # Thus, compute mu + U\z
        z = linalg.solve_triangular(U, z, trans='N', lower=False, ndim=self.ndim)
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

    def __init__(self, shape):
        self.shape = shape
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
            return [m0, m1, m2, m3]
        else:
            raise ValueError("Invalid parent index")


    def compute_weights_to_parent(self, index, weights):
        r"""
        Maps the mask to the plates of a parent.
        """
        if index != 0:
            raise IndexError()
        return misc.add_trailing_axes(weights, self.ndim)


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
        if index != 0:
            raise IndexError()
        return plates + self.shape


    def plates_from_parent(self, index, plates):
        r"""
        Resolve the plate mapping from a parent.

        Given the plates of a parent's moments, this method returns the plates
        that the moments has for this distribution.
        """
        if index != 0:
            raise IndexError()

        if self.ndim == 0:
            return plates
        else:
            return plates[:-self.ndim]


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
            shape = plates + dims
            z = np.random.randn(*shape)
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
            # Compute mu + U\z
            shape = plates + (N,)
            z = np.random.randn(*shape)
            # Denote Lambda = -2*phi[1]
            # Then, Cov = inv(Lambda) = inv(U'*U) = inv(U) * inv(U')
            # Thus, compute mu + U\z
            x = mu + linalg.solve_triangular(U, z,
                                             trans='N',
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


class GaussianGammaDistribution(ExponentialFamilyDistribution):
    r"""
    Class for the VMP formulas of Gaussian-Gamma-ISO variables.

    Currently, supports only vector variables.

    Log pdf of the prior:

    .. math::

       \log p(\mathbf{x}, \tau | \boldsymbol{\mu}, \mathbf{\Lambda}, a, b) =&
       - \frac{1}{2} \tau \mathbf{x}^T \mathbf{\Lambda} \mathbf{x}
       + \frac{1}{2} \tau \mathbf{x}^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \tau \boldsymbol{\mu}^T \mathbf{\Lambda} \mathbf{x}
       - \frac{1}{2} \tau \boldsymbol{\mu}^T \mathbf{\Lambda} \boldsymbol{\mu}
       + \frac{1}{2} \log|\mathbf{\Lambda}|
       + \frac{D}{2} \log\tau
       - \frac{D}{2} \log(2\pi)
       \\ &
       - b \tau
       + a \log\tau
       - \log\tau
       + a \log b
       - \log \Gamma(a)

    Log pdf of the posterior approximation:

    .. math::

       \log q(\mathbf{x}, \tau) =&
       \tau \mathbf{x}^T \boldsymbol{\phi}_1
       + \tau \mathbf{x}^T \mathbf{\Phi}_2 \mathbf{x}
       + \tau \phi_3
       + \log\tau \phi_4
       + g(\boldsymbol{\phi}_1, \mathbf{\Phi}_2, \phi_3, \phi_4)
       + f(x, \tau)

    """


    def __init__(self, ndim):
        self.ndim = ndim
        super().__init__()


    def compute_message_to_parent(self, parent, index, u, u_mu_Lambda, u_a, u_b):
        r"""
        Compute the message to a parent node.

        - Parent :math:`(\boldsymbol{\mu}, \mathbf{\Lambda})`

          Moments:

          .. math::

             \begin{bmatrix}
               \mathbf{\Lambda}\boldsymbol{\mu}
               \\
               \boldsymbol{\mu}^T\mathbf{\Lambda}\boldsymbol{\mu}
               \\
               \mathbf{\Lambda}
               \\
               \log|\mathbf{\Lambda}|
             \end{bmatrix}

          Message:

          .. math::

             \begin{bmatrix}
               \langle \tau \mathbf{x} \rangle
               \\
               - \frac{1}{2} \langle \tau \rangle
               \\
               - \frac{1}{2} \langle \tau \mathbf{xx}^T \rangle
               \\
               \frac{1}{2}
             \end{bmatrix}

        - Parent :math:`a`:

          Moments:

          .. math::

             \begin{bmatrix}
               a
               \\
               \log \Gamma(a)
             \end{bmatrix}

          Message:

          .. math::

             \begin{bmatrix}
               \langle \log\tau \rangle + \langle \log b \rangle
               \\
               -1
             \end{bmatrix}

        - Parent :math:`b`:

          Moments:

          .. math::

             \begin{bmatrix}
               b
               \\
               \log b
             \end{bmatrix}

          Message:

          .. math::

             \begin{bmatrix}
               - \langle \tau \rangle
               \\
               \langle a \rangle
             \end{bmatrix}

        """
        x_tau = u[0]
        xx_tau = u[1]
        tau = u[2]
        logtau = u[3]

        if index == 0:
            m0 = x_tau
            m1 = -0.5 * tau
            m2 = -0.5 * xx_tau
            m3 = 0.5
            return [m0, m1, m2, m3]
        elif index == 1:
            logb = u_b[1]
            m0 = logtau + logb
            m1 = -1
            return [m0, m1]
        elif index == 2:
            a = u_a[0]
            m0 = -tau
            m1 = a
            return [m0, m1]
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
        L_V = linalg.chol(V, ndim=self.ndim)
        logdet_V = linalg.chol_logdet(L_V, ndim=self.ndim)
        mu = linalg.chol_solve(L_V, phi[0], ndim=self.ndim)
        Cov = linalg.chol_inv(L_V, ndim=self.ndim)
        a = phi[3]
        b = -phi[2] - 0.5 * linalg.inner(mu, phi[0], ndim=self.ndim)
        log_b = np.log(b)

        # Compute moments
        u2 = a / b
        u3 = -log_b + special.psi(a)
        u0 = mu * misc.add_trailing_axes(u2, self.ndim)
        u1 = Cov + (
            linalg.outer(mu, mu, ndim=self.ndim)
            * misc.add_trailing_axes(u2, 2 * self.ndim)
        )
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
        u0 = x * misc.add_trailing_axes(alpha, self.ndim)
        u1 = linalg.outer(x, x, ndim=self.ndim) * misc.add_trailing_axes(alpha, self.ndim)
        u2 = alpha
        u3 = logalpha
        u = [u0, u1, u2, u3]
        if self.ndim > 0:
            D = np.prod(np.shape(x)[-ndim:])
        else:
            D = 1
        f = (D/2 - 1) * logalpha - D/2 * np.log(2*np.pi)
        return (u, f)


    def random(self, *params, plates=None):
        r"""
        Draw a random sample from the distribution.
        """
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


class _GaussianTemplate(ExponentialFamily):


    def translate(self, b, debug=False):
        """
        Transforms the current posterior by adding a bias to the mean

        Parameters
        ----------

        b : array
        Constant to add
        """

        ndim = len(self.dims[0])

        if ndim > 0 and np.shape(b)[-ndim:] != self.dims[0]:
            raise ValueError("Bias has incorrect shape")

        x = self.u[0]
        xb = linalg.outer(x, b, ndim=ndim)
        bx = linalg.transpose(xb, ndim=ndim)
        bb = linalg.outer(b, b, ndim=ndim)
        uh = [
            self.u[0] + b,
            self.u[1] + xb + bx + bb
        ]

        Lambda = -2 * self.phi[1]
        Lambda_b = linalg.mvdot(Lambda, b, ndim=ndim)

        dg = -0.5 * (
            linalg.inner(b, Lambda_b, ndim=ndim)
            + 2 * linalg.inner(x, Lambda_b, ndim=ndim)
        )

        phih = [
            self.phi[0] + Lambda_b,
            self.phi[1]
        ]

        self._check_shape(uh)
        self._check_shape(phih)

        self.u = uh
        self.phi = phih
        self.g = self.g + dg

        # TODO: This is all just debugging stuff and can be removed
        if debug:
            uh = [ui.copy() for ui in uh]
            gh = self.g.copy()
            self._update_moments_and_cgf()
            if any(not np.allclose(uih, ui, atol=1e-6) for (uih, ui) in zip(uh, self.u)):
                raise RuntimeError("BUG")
            if not np.allclose(self.g, gh, atol=1e-6):
                raise RuntimeError("BUG")

        return


class Gaussian(_GaussianTemplate):
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

    mu : Gaussian-like node or GaussianGamma-like node or GaussianWishart-like node or array
        Mean vector

    Lambda : Wishart-like node or array
        Precision matrix

    See also
    --------

    Wishart, GaussianARD, GaussianWishart, GaussianGamma

    """


    def __init__(self, mu, Lambda, **kwargs):
        r"""
        Create Gaussian node
        """
        super().__init__(mu, Lambda, **kwargs)


    @classmethod
    def _constructor(cls, mu, Lambda, ndim=1, **kwargs):
        r"""
        Constructs distribution and moments objects.
        """

        mu_Lambda = WrapToGaussianWishart(mu, Lambda, ndim=ndim)

        shape = mu_Lambda._moments.shape

        moments = GaussianMoments(shape)
        parent_moments = (mu_Lambda._moments,)

        if mu_Lambda.dims != ( shape, (), shape+shape, () ):
            raise Exception("Parents have wrong dimensionality")

        distribution = GaussianDistribution(shape)

        parents = [mu_Lambda]
        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, mu_Lambda.plates)),
                distribution,
                moments,
                parent_moments)


    def initialize_from_parameters(self, mu, Lambda):
        u = self._parent_moments[0].compute_fixed_moments(mu, Lambda)
        self._initialize_from_parent_moments(u)


    def __str__(self):
        ndim = len(self.dims[0])
        mu = self.u[0]
        Cov = self.u[1] - linalg.outer(mu, mu, ndim=ndim)
        return ("%s ~ Gaussian(mu, Cov)\n"
                "  mu = \n"
                "%s\n"
                "  Cov = \n"
                "%s\n"
                % (self.name, mu, Cov))


    def rotate(self, R, inv=None, logdet=None, Q=None):

        # TODO/FIXME: Combine and refactor all these rotation transformations
        # into _GaussianTemplate

        if self._moments.ndim != 1:
            raise NotImplementedError("Not implemented for ndim!=1 yet")

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

        if self._moments.ndim != 1:
            raise NotImplementedError("Not implemented for ndim!=1 yet")

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


class GaussianARD(_GaussianTemplate):
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

    mu : Gaussian-like node or GaussianGamma-like node or array Mean vector

    alpha : gamma-like node or array
        Diagonal elements of the precision matrix

    See also
    --------
    
    Gamma, Gaussian, GaussianGamma, GaussianWishart
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

        mu_alpha = WrapToGaussianGamma(mu, alpha, ndim=0)

        if ndim is None:
            if shape is not None:
                ndim = len(shape)
            else:
                shape = ()
                ndim = 0
        else:
            if shape is not None:
                if ndim != len(shape):
                    raise ValueError("Given shape and ndim inconsistent")
            else:
                if ndim == 0:
                    shape = ()
                else:
                    if ndim > len(mu_alpha.plates):
                        raise ValueError(
                            "Cannot determine shape for ndim={0} because parent "
                            "full shape has ndim={1}."
                            .format(ndim, len(mu_alpha.plates))
                        )
                    shape = mu_alpha.plates[-ndim:]

        moments = GaussianMoments(shape)
        parent_moments = [GaussianGammaMoments(())]
        distribution = GaussianARDDistribution(shape)

        plates = cls._total_plates(kwargs.get('plates'),
                                   distribution.plates_from_parent(0, mu_alpha.plates))

        parents = [mu_alpha]

        return (parents,
                kwargs,
                moments.dims,
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
        Cov = self.u[1] - linalg.outer(mu, mu)
        return ("%s ~ Gaussian(mu, Cov)\n"
                "  mu = \n"
                "%s\n"
                "  Cov = \n"
                "%s\n"
                % (self.name, mu, Cov))


    def rotate(self, R, inv=None, logdet=None, axis=-1, Q=None, subset=None, debug=False):

        if Q is not None:
            raise NotImplementedError()
        if subset is not None:
            raise NotImplementedError()

        # TODO/FIXME: Combine and refactor all these rotation transformations
        # into _GaussianTemplate

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

        # TODO: This is all just debugging stuff and can be removed
        if debug:
            uh = [ui.copy() for ui in self.u]
            gh = self.g.copy()
            self._update_moments_and_cgf()
            if any(not np.allclose(uih, ui, atol=1e-6) for (uih, ui) in zip(uh, self.u)):
                raise RuntimeError("BUG")
            if not np.allclose(self.g, gh, atol=1e-6):
                raise RuntimeError("BUG")

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


class GaussianGamma(ExponentialFamily):
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


    @classmethod
    def _constructor(cls, mu, Lambda, a, b, ndim=1, **kwargs):
        r"""
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        `mu` is the mean/location vector
        `alpha` is the scale
        `V` is the scale matrix
        `n` is the degrees of freedom
        """

        # Convert parent nodes
        mu_Lambda = WrapToGaussianWishart(mu, Lambda, ndim=ndim)
        a = cls._ensure_moments(a, GammaPriorMoments)
        b = cls._ensure_moments(b, GammaMoments)

        shape = mu_Lambda.dims[0]

        distribution = GaussianGammaDistribution(ndim)

        moments = GaussianGammaMoments(shape)
        parent_moments = (
            mu_Lambda._moments,
            a._moments,
            b._moments,
        )

        # Check shapes
        if mu_Lambda.dims != ( shape, (), 2*shape, () ):
            raise ValueError("mu and Lambda have wrong shape")
        if a.dims != ( (), () ):
            raise ValueError("a has wrong shape")
        if b.dims != ( (), () ):
            raise ValueError("b has wrong shape")

        # List of parent nodes
        parents = [mu_Lambda, a, b]

        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  distribution.plates_from_parent(0, mu_Lambda.plates),
                                  distribution.plates_from_parent(1, a.plates),
                                  distribution.plates_from_parent(2, b.plates)),
                distribution,
                moments,
                parent_moments)


    def translate(self, b, debug=False):

        if self._moments.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

        tau = self.u[2]

        x = self.u[0] / tau[...,None]
        xb = linalg.outer(x, b, ndim=1)
        bx = linalg.transpose(xb, ndim=1)
        bb = linalg.outer(b, b, ndim=1)

        uh = [
            self.u[0] + tau[...,None] * b,
            self.u[1] + tau[...,None,None] * (xb + bx + bb),
            self.u[2],
            self.u[3]
        ]

        Lambda = -2 * self.phi[1]
        dtau = -0.5 * (
            np.einsum('...ij,...i,...j->...', Lambda, b, b)
            + 2 * np.einsum('...ij,...i,...j->...', Lambda, b, x)
        )
        phih = [
            self.phi[0] + np.einsum('...ij,...j->...i', Lambda, b),
            self.phi[1],
            self.phi[2] + dtau,
            self.phi[3]
        ]

        self._check_shape(uh)
        self._check_shape(phih)

        self.phi = phih
        self.u = uh

        # TODO: This is all just debugging stuff and can be removed
        if debug:
            uh = [ui.copy() for ui in uh]
            gh = self.g.copy()
            self._update_moments_and_cgf()
            if any(not np.allclose(uih, ui, atol=1e-6) for (uih, ui) in zip(uh, self.u)):
                raise RuntimeError("BUG")
            if not np.allclose(self.g, gh, atol=1e-6):
                raise RuntimeError("BUG")

        return


    def rotate(self, R, inv=None, logdet=None, debug=False):

        if self._moments.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

        if inv is None:
            inv = np.linalg.inv(R)

        if logdet is None:
            logdet = np.linalg.slogdet(R)[1]

        uh = [
            rotate_mean(self.u[0], R),
            rotate_covariance(self.u[1], R),
            self.u[2],
            self.u[3]
        ]

        phih = [
            rotate_mean(self.phi[0], inv.T),
            rotate_covariance(self.phi[1], inv.T),
            self.phi[2],
            self.phi[3]
        ]

        self._check_shape(uh)
        self._check_shape(phih)

        self.phi = phih
        self.u = uh
        self.g = self.g - logdet

        # TODO: This is all just debugging stuff and can be removed
        if debug:
            uh = [ui.copy() for ui in uh]
            gh = self.g.copy()
            self._update_moments_and_cgf()
            if any(not np.allclose(uih, ui, atol=1e-6) for (uih, ui) in zip(uh, self.u)):
                raise RuntimeError("BUG")
            if not np.allclose(self.g, gh, atol=1e-6):
                raise RuntimeError("BUG")

        return


    def plotmatrix(self):
        r"""
        Creates a matrix of marginal plots.

        On diagonal, are marginal plots of each variable. Off-diagonal plot
        (i,j) shows the joint marginal density of x_i and x_j.
        """
        import bayespy.plot as bpplt
        
        if self.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

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


    def get_gaussian_location(self):
        r"""
        Return the mean and variance of the distribution
        """
        if self._moments.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

        tau = self.u[2]
        tau_mu = self.u[0]
        return tau_mu / tau[...,None]


    def get_gaussian_mean_and_variance(self):
        r"""
        Return the mean and variance of the distribution
        """
        if self.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

        a = self.phi[3]
        nu = 2*a

        if np.any(nu <= 1):
            raise ValueError("Mean not defined for degrees of freedom <= 1")
        if np.any(nu <= 2):
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
        if self.ndim != 1:
            raise NotImplementedError("Only ndim=1 supported at the moment")

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

        moments = GaussianWishartMoments(shape)

        # Convert parent nodes
        mu_alpha = WrapToGaussianGamma(mu, alpha)
        D = mu_alpha.dims[0][0]

        n = cls._ensure_moments(n, WishartPriorMoments)
        V = cls._ensure_moments(V, WishartMoments)

        parent_moments = (
            mu_alpha._moments,
            n._moments,
            V._moments
        )

        # Check shapes
        if mu_alpha.dims != ( (D,), (D,D), (), () ):
            raise ValueError("mu and alpha have wrong shape")

        if V.dims != ( (D,D), () ):
            raise ValueError("Precision matrix has wrong shape")

        if n.dims != ( (), () ):
            raise ValueError("Degrees of freedom has wrong shape")

        parents = [mu_alpha, n, V]

        return (parents,
                kwargs,
                moments.dims,
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu_alpha.plates),
                                  cls._distribution.plates_from_parent(1, n.plates),
                                  cls._distribution.plates_from_parent(2, V.plates)),
                cls._distribution,
                moments,
                parent_moments)


#
# CONVERTERS
#


class GaussianToGaussianGamma(Deterministic):
    r"""
    Converter for Gaussian moments to Gaussian-gamma isotropic moments

    Combines the Gaussian moments with gamma moments for a fixed value 1.
    """



    def __init__(self, X, **kwargs):
        r"""
        """
        if not isinstance(X._moments, GaussianMoments):
            raise ValueError("Wrong moments, should be Gaussian")

        shape = X._moments.shape
        self.ndim = X._moments.ndim

        self._moments = GaussianGammaMoments(shape)
        self._parent_moments = [GaussianMoments(shape)]

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


GaussianMoments.add_converter(GaussianGammaMoments,
                              GaussianToGaussianGamma)


class GaussianGammaToGaussianWishart(Deterministic):
    r"""
    """


    def __init__(self, X_alpha, **kwargs):
        raise NotImplementedError()


GaussianGammaMoments.add_converter(GaussianWishartMoments,
                                   GaussianGammaToGaussianWishart)


#
# WRAPPERS
#
# These wrappers form a single node from two nodes for messaging purposes.
#


class WrapToGaussianGamma(Deterministic):
    r"""
    """


    def __init__(self, X, alpha, ndim=None, **kwargs):
        r"""
        """

        # In case X is a numerical array, convert it to Gaussian first
        try:
            X = self._ensure_moments(X, GaussianMoments, ndim=ndim)
        except Moments.NoConverterError:
            pass

        try:
            ndim = X._moments.ndim
        except AttributeError as err:
            raise TypeError("ndim needs to be given explicitly") from err

        X = self._ensure_moments(X, GaussianGammaMoments, ndim=ndim)

        if len(X.dims[0]) != ndim:
            raise RuntimeError("Conversion failed ndim.")

        shape = X.dims[0]
        dims = ( shape, 2 * shape, (), () )

        self.shape = shape
        self.ndim = len(shape)

        self._moments = GaussianGammaMoments(shape)
        self._parent_moments = [
            GaussianGammaMoments(shape),
            GammaMoments()
        ]

        super().__init__(X, alpha, dims=dims, **kwargs)


    def _compute_moments(self, u_X, u_alpha):
        r"""
        """
        (tau_x, tau_xx, tau, logtau) = u_X
        (alpha, logalpha) = u_alpha
        u0 = tau_x * misc.add_trailing_axes(alpha, self.ndim)
        u1 = tau_xx * misc.add_trailing_axes(alpha, 2 * self.ndim)
        u2 = tau * alpha
        u3 = logtau + logalpha
        return [u0, u1, u2, u3]


    def _compute_message_to_parent(self, index, m_child, u_X, u_alpha):
        r"""
        """
        if index == 0:
            alpha = u_alpha[0]
            m0 = m_child[0] * misc.add_trailing_axes(alpha, self.ndim)
            m1 = m_child[1] * misc.add_trailing_axes(alpha, 2 * self.ndim)
            m2 = m_child[2] * alpha
            m3 = m_child[3]
            return [m0, m1, m2, m3]
        elif index == 1:
            (tau_x, tau_xx, tau, logtau) = u_X
            m0 = (
                linalg.inner(m_child[0], tau_x, ndim=self.ndim)
                + linalg.inner(m_child[1], tau_xx, ndim=2*self.ndim)
                + m_child[2] * tau
            )
            m1 = m_child[3]
            return [m0, m1]
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


    def __init__(self, X, Lambda, ndim=1, **kwargs):
        r"""
        """

        # Just in case X is an array, convert it to a Gaussian node first.
        try:
            X = self._ensure_moments(X, GaussianMoments, ndim=ndim)
        except Moments.NoConverterError:
            pass

        try:
            # Try combo Gaussian-Gamma and Wishart
            X = self._ensure_moments(X, GaussianGammaMoments, ndim=ndim)
        except ValueError:
            # Have to use Gaussian-Wishart and Gamma
            X = self._ensure_moments(X, GaussianWishartMoments, ndim=ndim)
            Lambda = self._ensure_moments(Lambda, GammaMoments, ndim=ndim)
            shape = X.dims[0]
            if Lambda.dims != ((), ()):
                raise ValueError(
                    "Mean and precision have inconsistent shapes: {0} and {1}"
                    .format(
                        X.dims,
                        Lambda.dims
                    )
                )
            self.wishart = False
        else:
            # Gaussian-Gamma and Wishart
            shape = X.dims[0]
            Lambda = self._ensure_moments(Lambda, WishartMoments, ndim=ndim)
            if Lambda.dims != (2 * shape, ()):
                raise ValueError(
                    "Mean and precision have inconsistent shapes: {0} and {1}"
                    .format(
                        X.dims,
                        Lambda.dims
                    )
                )
            self.wishart = True

        self.ndim = len(shape)

        self._parent_moments = (
            X._moments,
            Lambda._moments,
        )

        self._moments = GaussianWishartMoments(shape)

        super().__init__(X, Lambda, dims=self._moments.dims, **kwargs)


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

            u0 = linalg.mvdot(Lambda, alpha_x, ndim=self.ndim)
            u1 = linalg.inner(Lambda, alpha_xx, ndim=2*self.ndim)
            u2 = Lambda * misc.add_trailing_axes(alpha, 2*self.ndim)
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
                m0 = linalg.mvdot(Lambda, m_child[0], ndim=self.ndim)
                m1 = Lambda * misc.add_trailing_axes(m_child[1], 2*self.ndim)
                m2 = linalg.inner(Lambda, m_child[2], ndim=2*self.ndim)
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
                m0 = (0.5*linalg.outer(alpha_x, m_child[0], ndim=self.ndim) +
                      0.5*linalg.outer(m_child[0], alpha_x, ndim=self.ndim) +
                      alpha_xx * misc.add_trailing_axes(m_child[1], 2*self.ndim) +
                      misc.add_trailing_axes(alpha, 2*self.ndim) * m_child[2])
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


class ConcatGaussian(Deterministic):
    """Concatenate Gaussian vectors along the variable axis (not plate axis)

    NOTE: This concatenates on the variable axis! That is, the dimensionality
    of the resulting Gaussian vector is the sum of the dimensionalities of the
    input Gaussian vectors.

    TODO: Add support for Gaussian arrays and arbitrary concatenation axis.
    """


    def __init__(self, *nodes, **kwargs):

        # Number of nodes to concatenate
        N = len(nodes)

        # This is stuff that will be useful when implementing arbitrary
        # concatenation. That is, first determine ndim.
        #
        # # Convert nodes to Gaussians (if they are not nodes, don't worry)
        # nodes_gaussian = []
        # for node in nodes:
        #     try:
        #         node_gaussian = node._convert(GaussianMoments)
        #     except AttributeError: # Moments.NoConverterError:
        #         nodes_gaussian.append(node)
        #     else:
        #         nodes_gaussian.append(node_gaussian)
        # nodes = nodes_gaussian
        #
        # # Determine shape from the first Gaussian node
        # shape = None
        # for node in nodes:
        #     try:
        #         shape = node.dims[0]
        #     except AttibuteError:
        #         pass
        #     else:
        #         break
        # if shape is None:
        #     raise ValueError("Couldn't determine shape from the input nodes")
        #
        # ndim = len(shape)

        nodes = [self._ensure_moments(node, GaussianMoments, ndim=1)
                 for node in nodes]

        D = sum(node.dims[0][0] for node in nodes)

        shape = (D,)

        self._moments = GaussianMoments(shape)

        self._parent_moments = [node._moments for node in nodes]

        # Make sure all parents are Gaussian vectors
        if any(len(node.dims[0]) != 1 for node in nodes):
            raise ValueError("Input nodes must be (Gaussian) vectors")

        self.slices = tuple(np.cumsum([0] + [node.dims[0][0] for node in nodes]))
        D = self.slices[-1]

        return super().__init__(*nodes, dims=((D,), (D, D)), **kwargs)


    def _compute_moments(self, *u_nodes):
        x = misc.concatenate(*[u[0] for u in u_nodes], axis=-1)
        xx = misc.block_diag(*[u[1] for u in u_nodes])

        # Explicitly broadcast xx to plates of x
        x_plates = np.shape(x)[:-1]
        xx = np.ones(x_plates)[...,None,None] * xx

        # Compute the cross-covariance terms using the means of each variable
        # (because covariances are zero for factorized nodes in the VB
        # approximation)
        i_start = 0
        for m in range(len(u_nodes)):
            i_end = i_start + np.shape(u_nodes[m][0])[-1]
            j_start = 0
            for n in range(m):
                j_end = j_start + np.shape(u_nodes[n][0])[-1]
                xm_xn = linalg.outer(u_nodes[m][0], u_nodes[n][0], ndim=1)
                xx[...,i_start:i_end,j_start:j_end] = xm_xn
                xx[...,j_start:j_end,i_start:i_end] = misc.T(xm_xn)
                j_start = j_end
            i_start = i_end

        return [x, xx]


    def _compute_message_to_parent(self, i, m, *u_nodes):
        r = self.slices

        # Pick the proper parts from the message array
        m0 = m[0][...,r[i]:r[i+1]]
        m1 = m[1][...,r[i]:r[i+1],r[i]:r[i+1]]

        # Handle cross-covariance terms by using the mean of the covariate node
        for (j, u) in enumerate(u_nodes):
            if j != i:
                m0 = m0 + 2 * np.einsum(
                    '...ij,...j->...i',
                    m[1][...,r[i]:r[i+1],r[j]:r[j+1]],
                    u[0]
                )

        return [m0, m1]
