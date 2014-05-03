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


import numpy as np

from bayespy import utils
from bayespy.utils.linalg import dot, mvdot

from .expfamily import ExponentialFamily
from .expfamily import ExponentialFamilyDistribution
from .expfamily import useconstructor
from .wishart import Wishart, WishartMoments
from .gamma import Gamma, GammaMoments
from .deterministic import Deterministic

from .node import Moments, ensureparents

class GaussianMoments(Moments):
    def __init__(self, ndim):
        self.ndim = ndim

    def compute_fixed_moments(self, x):
        """ Compute Gaussian moments for fixed x. """
        x = utils.utils.atleast_nd(x, self.ndim)
        return [x, utils.linalg.outer(x, x, ndim=self.ndim)]

    def compute_dims_from_values(self, x):
        x = utils.utils.atleast_nd(x, self.ndim)
        if self.ndim == 0:
            dims = ()
        else:
            dims = np.shape(x)[-self.ndim:]
        return (dims, dims+dims)
        

class GaussianDistribution(ExponentialFamilyDistribution):

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]

    def compute_message_to_parent(self, parent, index, u, *u_parents):
        if index == 0:
            return [utils.utils.m_dot(u_parents[1][0], u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = utils.utils.m_outer(u[0], u_parents[0][0])
            return [-0.5 * (u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]
        else:
            raise ValueError("Index out of bounds")

    def compute_phi_from_parents(self, u_mu, u_Lambda, mask=True):
        return [utils.utils.m_dot(u_Lambda[0], u_mu[0]),
                -0.5 * u_Lambda[0]]

    def compute_moments_and_cgf(self, phi, mask=True):
        # TODO: Compute -2*phi[1] and simplify the formulas
        L = utils.utils.m_chol(-2*phi[1])
        k = np.shape(phi[0])[-1]
        # Moments
        u0 = utils.utils.m_chol_solve(L, phi[0])
        u1 = utils.utils.m_outer(u0, u0) + utils.utils.m_chol_inv(L)
        u = [u0, u1]
        # G
        g = (-0.5 * np.einsum('...i,...i', u[0], phi[0])
             + 0.5 * utils.utils.m_chol_logdet(L))
             #+ 0.5 * np.log(2) * self.dims[0][0])
        return (u, g)

    def compute_cgf_from_parents(self, u_mu, u_Lambda):
        mu = u_mu[0]
        mumu = u_mu[1]
        Lambda = u_Lambda[0]
        logdet_Lambda = u_Lambda[1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * logdet_Lambda)
        return g

    def compute_fixed_moments_and_f(self, x, mask=True):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(x)[-1]
        u = [x, utils.utils.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)


class Gaussian(ExponentialFamily):
    r"""
    VMP node for Gaussian variable.

    The node represents a :math:`D`-dimensional vector from the
    Gaussian distribution:
    
    .. math::

       \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu},
       \mathbf{\Lambda}),

    where :math:`\boldsymbol{\mu}` is the mean vector and
    :math:`\mathbf{\Lambda}` is the precision matrix (i.e., inverse of
    the covariance matrix).
    
    .. math::

       \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, 
       \quad \mathbf{\Lambda} \in \mathbb{R}^{D \times D},
       \quad \mathbf{\Lambda} \text{ symmetric positive definite}

    Plates!

    Parent nodes? Child nodes?

    See also
    --------
    Wishart
    
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

    .. math::

       \mathbf{u} (\mathbf{x})
       &=
       \left[ \begin{matrix}
         \mathbf{x}
         \\
         \mathbf{xx}^{\mathrm{T}}
       \end{matrix} \right]
       \\
       \boldsymbol{\phi} (\boldsymbol{\mu}, \mathbf{\Lambda})
       &=
       \left[ \begin{matrix}
         \mathbf{\Lambda} \boldsymbol{\mu} 
         \\
         - \frac{1}{2} \mathbf{\Lambda}
       \end{matrix} \right]
       \\
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
       \\
       g (\boldsymbol{\mu}, \mathbf{\Lambda})
       &=
       - \frac{1}{2} \operatorname{tr}(\boldsymbol{\mu\mu}^{\mathrm{T}}
                                       \mathbf{\Lambda} )
       + \frac{1}{2} \log |\mathbf{\Lambda}|
       \\
       g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
       &=
       \frac{1}{4} \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
       \boldsymbol{\phi}_1
       + \frac{1}{2} \log | -2 \boldsymbol{\phi}_2 |
       \\
       f(\mathbf{x})
       &= - \frac{D}{2} \log(2\pi)
       \\
       \overline{\mathbf{u}}  (\boldsymbol{\phi})
       &=
       \left[ \begin{matrix}
         - \frac{1}{2} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
         \\
         \frac{1}{4} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
         \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
         - \frac{1}{2} \boldsymbol{\phi}^{-1}_2
       \end{matrix} \right]

    """

    _distribution = GaussianDistribution()
    _moments = GaussianMoments(1)
    _parent_moments = (GaussianMoments(1),
                       WishartMoments())
    

    @classmethod
    @ensureparents
    def _constructor(cls, mu, Lambda, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        D_mu = mu.dims[0][0]
        D_Lambda = Lambda.dims[0][0]

        if D_mu != D_Lambda:
            raise ValueError("Mean vector (%d-D) and precision matrix (%d-D) "
                             "have different dimensionalities"
                             % (D_mu, D_Lambda))
        D = D_mu
        
        if mu.dims != ( (D,), (D,D) ):
            raise Exception("First parent has wrong dimensionality")
        if Lambda.dims != ( (D,D), () ):
            raise Exception("Second parent has wrong dimensionality")

        parents = [mu, Lambda]
        dims = ( (D,), (D,D) )
        return (parents,
                kwargs,
                dims, 
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu.plates),
                                  cls._distribution.plates_from_parent(1, Lambda.plates)),
                cls._distribution, 
                cls._moments, 
                cls._parent_moments)


    def random(self):
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = utils.utils.m_chol(-2*self.phi[1])
        mu = self.u[0]
        z = np.random.normal(0, 1, self.get_shape(0))
        # Compute mu + U'*z
        z = utils.utils.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
            

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - utils.utils.m_outer(mu, mu)
        print("%s ~ Gaussian(mu, Cov)" % self.name)
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

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

        #logdet_old = np.sum(utils.linalg.logdet_cov(-2*self.phi[1]))
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
        """
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

        #g0 = np.sum(np.ones(self.plates)*self.g)
        self._update_moments_and_cgf()
        #g1 = np.sum(np.ones(self.plates)*self.g)

        #dg = g1 - g0







        




###
###
###


# GaussianFromGaussianAndGamma
# GaussianFromGaussianAndWishart
# GaussianGammaFromGaussianAndWishart
# GaussianGammaFromGaussianAndGamma

class GaussianARDDistribution(ExponentialFamilyDistribution):

    def __init__(self, shape, ndim_mu):
        self.shape = shape
        self.ndim_mu = ndim_mu
        self.ndim = len(shape)
        self.ndims = (self.ndim, 2*self.ndim)
        self.ndims_parents = ( (ndim_mu, 2*ndim_mu),
                               (0, 0) )
        super().__init__()
    
    def compute_message_to_parent(self, parent, index, u, u_mu, u_alpha):
        if index == 0:
            x = u[0]
            alpha = u_alpha[0]

            axes0 = list(range(-self.ndim, -self.ndim_mu))
            m0 = utils.utils.sum_multiply(alpha, x, axis=axes0)

            Alpha = utils.utils.diag(alpha, ndim=self.ndim)
            axes1 = [axis+self.ndim for axis in axes0] + axes0
            m1 = -0.5 * utils.utils.sum_multiply(Alpha, 
                                                 utils.utils.identity(*self.shape),
                                                 axis=axes1)
            return [m0, m1]

        elif index == 1:
            x = u[0]
            x2 = utils.utils.get_diag(u[1], ndim=self.ndim)
            mu = u_mu[0]
            mu2 = utils.utils.get_diag(u_mu[1], ndim=self.ndim_mu)
            if self.ndim_mu == 0:
                mu_shape = np.shape(mu) + (1,)*self.ndim
            else:
                mu_shape = (np.shape(mu)[:-self.ndim_mu] 
                            + (1,)*(self.ndim-self.ndim_mu)
                            + np.shape(mu)[-self.ndim_mu:])
            mu = np.reshape(mu, mu_shape)
            mu2 = np.reshape(mu2, mu_shape)
            m0 = -0.5*x2 + x*mu - 0.5*mu2
            m1 = 0.5
            return [m0, m1]

        else:
            raise ValueError("Index out of bounds")

    def compute_mask_to_parent(self, index, mask):
        """
        Compute the mask used for messages sent to parent[index].

        The mask tells which plates in the messages are active. This method
        is used for obtaining the mask which is used to set plates in the
        messages to parent to zero.
        """

        if index == 1:
            # Add trailing axes
            mask = np.reshape(mask, np.shape(mask) + (1,)*self.ndim)

        return mask

    def compute_phi_from_parents(self, u_mu, u_alpha, mask=True):
        mu = u_mu[0]
        alpha = u_alpha[0]
        if np.ndim(mu) < self.ndim_mu:
            raise ValueError("Moment of mu does not have enough dimensions")
        mu = utils.utils.add_axes(mu, 
                                  axis=np.ndim(mu)-self.ndim_mu, 
                                  num=self.ndim-self.ndim_mu)
        phi0 = alpha * mu
        phi1 = -0.5 * alpha
        if self.ndim > 0:
            # Ensure that phi is not using broadcasting for variable
            # dimension axes
            ones = np.ones(self.shape)
            phi0 = ones * phi0
            phi1 = ones * phi1

        # Make a diagonal matrix
        phi1 = utils.utils.diag(phi1, ndim=self.ndim)
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
            L = utils.linalg.chol(-2*phi1)
            Cov = utils.linalg.chol_inv(L)
            u0 = utils.linalg.chol_solve(L, phi0)
            u1 = utils.linalg.outer(u0, u0) + Cov

            # Compute CGF
            g = (- 0.5 * np.einsum('...i,...i', u0, phi0)
                 + 0.5 * utils.linalg.chol_logdet(L))

            # Reshape to arrays
            u0 = np.reshape(u0, u0.shape[:-1] + self.shape)
            u1 = np.reshape(u1, u1.shape[:-2] + self.shape + self.shape)
            u = [u0, u1]

        return (u, g)


    def compute_cgf_from_parents(self, u_mu, u_alpha):
        """
        Compute the value of the cumulant generating function.
        """

        # Compute sum(mu^2 * alpha) correctly for broadcasted shapes
        mumu = u_mu[1]
        alpha = u_alpha[0]
        if self.ndim == 0:
            z = mumu * alpha
        else:
            if np.ndim(alpha) == 0 and np.ndim(mumu) == 0:
                # Einsum doesn't like scalar only inputs, so we have to
                # handle them separately..
                z = mumu * alpha
            else:
                # Use ellipsis for the plates, sum other axes
                out_keys = [Ellipsis]
                # Take the diagonal of the second moment matrix mu*mu.T
                mu_keys = [Ellipsis] + 2 * list(range(self.ndim_mu,0,-1))
                # Keys for alpha
                if np.ndim(alpha) <= self.ndim:
                    # Add empty Ellipsis just to avoid errors from einsum
                    alpha_keys = [Ellipsis] + list(range(np.ndim(alpha),0,-1))
                else:
                    alpha_keys = [Ellipsis] + list(range(self.ndim,0,-1))
                # Perform the computation
                z = np.einsum(mumu, mu_keys, alpha, alpha_keys, out_keys)

            # Take into account broadcasting
            if self.ndim_mu == 0:
                shape_mumu = ()
            else:
                shape_mumu = np.shape(mumu)[-self.ndim_mu:]
            if self.ndim == 0:
                shape_alpha = ()
            else:
                shape_alpha = np.shape(alpha)[-self.ndim:]
            z *= Gaussian._plate_multiplier(self.shape,
                                            shape_mumu,
                                            shape_alpha)

        # Compute log(alpha) correctly for broadcasted alpha
        logdet_alpha = u_alpha[1]
        if np.ndim(logdet_alpha) <= self.ndim:
            dims_logalpha = np.shape(logdet_alpha)
            logdet_alpha = np.sum(logdet_alpha)
        elif self.ndim == 0:
            dims_logalpha = ()
        else:
            dims_logalpha = np.shape(logdet_alpha)[-self.ndim:]
            logdet_alpha = np.sum(logdet_alpha,
                                  axis=tuple(range(-self.ndim,0)))
        logdet_alpha *= Gaussian._plate_multiplier(self.shape,
                                                   dims_logalpha)

        # Compute cumulant generating function
        cgf = -0.5*z + 0.5*logdet_alpha

        return cgf

    def compute_fixed_moments_and_f(self, x, mask=True):
        """ Compute u(x) and f(x) for given x. """
        if self.ndim > 0 and np.shape(x)[-self.ndim:] != self.shape:
            raise ValueError("Invalid shape")
        k = np.prod(self.shape)
        u = [x, utils.linalg.outer(x, x, ndim=self.ndim)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)

    def plates_to_parent(self, index, plates):
        if index == 1:
            return plates + self.shape
        else:
            return super().plates_to_parent(index, plates)

    def plates_from_parent(self, index, plates):
        ndim = len(self.shape)
        if index == 1 and ndim > 0:
            return plates[:-ndim]
        else:
            return super().plates_from_parent(index, plates)


class GaussianARD(ExponentialFamily):
    r"""
    VMP node for Gaussian array variable.

    The node represents a :math:`(D0,D1,...,DN)`-dimensional vector from the
    Gaussian distribution:

    .. math::

       \mathbf{x} &\sim \mathcal{N}(\boldsymbol{\mu},
       \mathbf{\Lambda}),

    where :math:`\boldsymbol{\mu}` is the mean vector and
    :math:`\mathbf{\Lambda}` is the precision matrix (i.e., inverse of
    the covariance matrix).

    .. math::

       \mathbf{x},\boldsymbol{\mu} \in \mathbb{R}^{D}, 
       \quad \mathbf{\Lambda} \in \mathbb{R}^{D \times D},
       \quad \mathbf{\Lambda} \text{ symmetric positive definite}

    Plates!

    Parent nodes? Child nodes?

    See also
    --------
    Wishart

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

    .. math::

       \mathbf{u} (\mathbf{x})
       &=
       \left[ \begin{matrix}
         \mathbf{x}
         \\
         \mathbf{xx}^{\mathrm{T}}
       \end{matrix} \right]
       \\
       \boldsymbol{\phi} (\boldsymbol{\mu}, \mathbf{\Lambda})
       &=
       \left[ \begin{matrix}
         \mathbf{\Lambda} \boldsymbol{\mu} 
         \\
         - \frac{1}{2} \mathbf{\Lambda}
       \end{matrix} \right]
       \\
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
       \\
       g (\boldsymbol{\mu}, \mathbf{\Lambda})
       &=
       - \frac{1}{2} \operatorname{tr}(\boldsymbol{\mu\mu}^{\mathrm{T}}
                                       \mathbf{\Lambda} )
       + \frac{1}{2} \log |\mathbf{\Lambda}|
       \\
       g_{\boldsymbol{\phi}} (\boldsymbol{\phi})
       &=
       \frac{1}{4} \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
       \boldsymbol{\phi}_1
       + \frac{1}{2} \log | -2 \boldsymbol{\phi}_2 |
       \\
       f(\mathbf{x})
       &= - \frac{D}{2} \log(2\pi)
       \\
       \overline{\mathbf{u}}  (\boldsymbol{\phi})
       &=
       \left[ \begin{matrix}
         - \frac{1}{2} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
         \\
         \frac{1}{4} \boldsymbol{\phi}^{-1}_2 \boldsymbol{\phi}_1
         \boldsymbol{\phi}^{\mathrm{T}}_1 \boldsymbol{\phi}^{-1}_2 
         - \frac{1}{2} \boldsymbol{\phi}^{-1}_2
       \end{matrix} \right]

    """


    @classmethod
    def _constructor(cls, mu, alpha, ndim=None, shape=None, **kwargs):
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
        # Check consistency
        if ndim is not None and shape is not None and ndim != len(shape):
            raise ValueError("Given shape and ndim inconsistent")
        if ndim is None and shape is not None:
            ndim = len(shape)

        # Infer shape of mu
        try:
            shape_mu = mu.dims[0]
            if ndim is None and shape is None:
                ndim = len(shape_mu)
        except:
            if ndim is None and shape is None:
                shape_mu = np.shape(mu)
            elif ndim == 0:
                shape_mu = ()
            elif ndim is not None and ndim > 0:
                shape_mu = np.shape(mu)[-ndim:]
            else:
                raise ValueError("Can't infer the shape of the parent mu")
        ndim_mu = len(shape_mu)

        # Infer shape of alpha
        try:
            shape_alpha = alpha.plates
        except:
            shape_alpha = np.shape(alpha)
        if ndim == 0:
            shape_alpha = ()
        elif ndim is not None and ndim > 0:
            shape_alpha = shape_alpha[-ndim:]
        elif ndim is not None:
            raise ValueError("Can't infer the shape of the parent alpha")
        ndim_alpha = len(shape_alpha)

        # Infer dimensionality
        if ndim is None:
            ndim = max(ndim_mu, ndim_alpha)
        elif ndim < ndim_mu or ndim < ndim_alpha:
            raise ValueError("Parent mu has more axes")

        # Infer shape of the node
        shape_bc = utils.utils.broadcasted_shape(shape_mu, shape_alpha)
        if shape is None:
            shape = (ndim-len(shape_bc))*(1,) + shape_bc
        elif not utils.utils.is_shape_subset(shape_bc, shape):
            raise ValueError("Broadcasted shape of the parents %s does not "
                             "broadcast to the given shape %s" 
                             % (shape_bc, shape))

        ndim = len(shape)
        if shape_mu is None:
            shape_mu = shape
        ndim_mu = len(shape_mu)
    
        moments = GaussianMoments(ndim)
        parent_moments = (GaussianMoments(ndim_mu),
                          GammaMoments())
        distribution = GaussianARDDistribution(shape, ndim_mu)

        # Convert parents to proper nodes
        mu = cls._ensure_moments(mu, parent_moments[0])
        alpha = cls._ensure_moments(alpha, parent_moments[1])

        # Check consistency with respect to parent mu
        shape_mean = shape[-ndim_mu:]
        # Check mean
        if not utils.utils.is_shape_subset(mu.dims[0], shape_mean):
            raise ValueError("Parent node %s with mean shaped %s does not "
                             "broadcast to the shape %s of this node"
                             % (mu.name,
                                mu.dims[0],
                                shape))
        # Check covariance
        shape_cov = shape[-ndim_mu:] + shape[-ndim_mu:]
        if not utils.utils.is_shape_subset(mu.dims[1], shape_cov):
            raise ValueError("Parent node %s with covariance shaped %s "
                             "does not broadcast to the shape %s of this "
                             "node"
                             % (mu.name,
                                mu.dims[1],
                                shape+shape))

        # Check consistency with respect to parent alpha
        if ndim == 0:
            shape_alpha = ()
        else:
            shape_alpha = alpha.plates[-ndim:]
        if not utils.utils.is_shape_subset(shape_alpha, shape):
            raise ValueError("Parent node (precision) does not broadcast "
                             "to the shape of this node")
        if alpha.dims != ( (), () ):
            raise Exception("Second parent has wrong dimensionality")
        
        dims = (shape, shape+shape)
        plates = cls._total_plates(kwargs.get('plates'),
                                   distribution.plates_from_parent(0, mu.plates),
                                   distribution.plates_from_parent(1, alpha.plates))

        parents = [mu, alpha]

        return (parents, kwargs, dims, plates, distribution, moments, parent_moments)
        
    def initialize_from_mean_and_covariance(self, mu, Cov):
        ndim = len(self._distribution.shape)
        u = [mu, Cov + utils.linalg.outer(mu, mu, ndim=ndim)]
        mask = np.logical_not(self.observed)
        # TODO: You could compute the CGF but it requires Cholesky of
        # Cov. Do it later.
        self._set_moments_and_cgf(u, np.nan, mask=mask)
        return

    def random(self):
        """
        Draw a random sample from the Gaussian distribution.
        """
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!
        D = len(self.dims[0])
        if np.prod(self.dims[1]) == 1.0:
            # Scalar Gaussian
            phi1 = self.phi[1]
            if D > 0:
                # Because the covariance matrix has shape (1,1,...,1,1),
                # that is 2*D number of ones, remove the extra half of the
                # shape
                phi1 = np.reshape(phi1, np.shape(phi1)[:-2*D] + D*(1,))

            std = np.sqrt(-0.5 / phi1)
            mu = self.u[0]
            z = np.random.normal(0, 1, self.get_shape(0))
            x = mu + std * z
        else:
            N = np.prod(self.dims[0])
            dims_cov = self.dims[1]
            # Reshape precision matrix
            plates_cov = np.shape(self.phi[1])[:-2*D]
            V = -2 * np.reshape(self.phi[1], plates_cov + (N,N))
            # Reshape mean vector
            plates_mu = np.shape(self.u[0])[:-D]
            mu = np.reshape(self.u[0], plates_mu + (N,))
            # Compute Cholesky
            U = utils.linalg.chol(V)
            # Compute mu + U'*z
            z = np.random.normal(0, 1, self.plates + (N,))
            x = mu + utils.linalg.solve_triangular(U, z,
                                                   trans='T', 
                                                   lower=False)
            x = np.reshape(x, self.plates + self.dims[0])
        return x

    def show(self):
        raise NotImplementedError()
        mu = self.u[0]
        Cov = self.u[1] - utils.utils.m_outer(mu, mu)
        print("%s ~ Gaussian(mu, Cov)" % self.name)
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))


    def rotate(self, R, inv=None, logdet=None, axis=-1, Q=None):

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
        """
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
        sumQ = utils.utils.add_trailing_axes(np.sum(Q, axis=0),
                                             2*ndim-plate_axis-1)
        phi1 = sumQ**(-2) * self.phi[1]
        phi0 = -2 * matrix_dot_vector(phi1, u0, ndim=ndim)

        self.phi[0] = phi0
        self.phi[1] = phi1

        self._update_moments_and_cgf()

        return


def reshape_gaussian_array(dims_from, dims_to, x0, x1):
    """
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
    """
    Transpose the covariance array of Gaussian array variable.

    That is, swap the last ndim axes with the ndim axes before them. This makes
    transposing easy for array variables when the covariance is not a matrix but
    a multidimensional array.
    """
    axes_in = [Ellipsis] + list(range(2*ndim,0,-1))
    axes_out = [Ellipsis] + list(range(ndim,0,-1)) + list(range(2*ndim,ndim,-1))
    return np.einsum(Cov, axes_in, axes_out)

def left_rotate_covariance(Cov, R, axis=-1, ndim=1):
    """
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
    """
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
    """
    Rotate the covariance array of Gaussian array variable.

    ndim is the number of axes for the Gaussian variable.

    For vector variable, ndim=1 and covariance is a matrix.
    """

    # Rotate from left and right
    Cov = left_rotate_covariance(Cov, R, ndim=ndim, axis=axis)
    Cov = right_rotate_covariance(Cov, R, ndim=ndim, axis=axis)

    return Cov

def rotate_mean(mu, R, axis=-1, ndim=1):
    """
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
        
