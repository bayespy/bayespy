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
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Bayespy is distributed in the hope that it will be useful, but
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
from .constant import Constant
from .wishart import Wishart

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
    bayespy.inference.vmp.nodes.wishart.Wishart
    inference.vmp.nodes.wishart.Wishart
    vmp.nodes.wishart.Wishart
    nodes.wishart.Wishart
    wishart.Wishart
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

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    
    def __init__(self, mu, Lambda, **kwargs):

        self.parameter_distributions = (Gaussian, Wishart)
        
        # Check for constant mu
        if utils.utils.is_numeric(mu):
            mu = Constant(Gaussian)(np.atleast_1d(mu))

        # Check for constant Lambda
        if utils.utils.is_numeric(Lambda):
            Lambda = Constant(Wishart)(np.atleast_2d(Lambda))

        ## # You could check whether the dimensions of mu and Lambda
        ## # match (and Lambda is square)
        ## if Lambda.dims[0][-1] != mu.dims[0][-1]:
        ##     raise Exception("Dimensionalities of mu and Lambda do not match.")

        # Construct
        super().__init__(mu, Lambda,
                         **kwargs)

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        return [x, utils.utils.m_outer(x,x)]

    @staticmethod
    def _compute_phi_from_parents(*u_parents):
        return [utils.utils.m_dot(u_parents[1][0], u_parents[0][0]),
                -0.5 * u_parents[1][0]]

    @staticmethod
    def _compute_cgf_from_parents(u_mu, u_Lambda):
        mu = u_mu[0]
        mumu = u_mu[1]
        Lambda = u_Lambda[0]
        logdet_Lambda = u_Lambda[1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * logdet_Lambda)
        return g

    @staticmethod
    def _compute_moments_and_cgf(phi, mask=True):
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

    @staticmethod
    def _compute_fixed_moments_and_f(x, mask=True):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(x)[-1]
        u = [x, utils.utils.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)

    @staticmethod
    def _compute_message_to_parent(parent, index, u, *u_parents):
        """ . """
        if index == 0:
            return [utils.utils.m_dot(u_parents[1][0], u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = utils.utils.m_outer(u[0], u_parents[0][0])
            return [-0.5 * (u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]

    @staticmethod
    def compute_dims(mu, Lambda):
        """
        Compute the dimensions of phi and u using the parent nodes.

        Also, check that the dimensionalities of the parents are
        consistent with each other.

        Parameters:
        -----------
        mu : Node
            A VB node with ( (D,), (D,D) ) dimensional Gaussian
            output.
        Lambda: Node
            A VB node with ( (D,D), () ) dimensional Wishart output.
        """
        D = mu.dims[0][0]

        if mu.dims != ( (D,), (D,D) ):
            raise Exception("First parent has wrong dimensionality")
        if Lambda.dims != ( (D,D), () ):
            raise Exception("Second parent has wrong dimensionality")
        
        return ( (D,), (D,D) )

    @staticmethod
    def compute_dims_from_values(x):
        """
        Compute the dimensions of phi and u from a value.

        The last axis tells the dimensionality, the other axes are
        plates.

        Parameters:
        -----------
        x : ndarray
        """
        if np.ndim(x) == 0:
            raise ValueError("The value must be at least 1-D array.")
        D = np.shape(x)[-1]
        return ( (D,), (D,D) )

    def get_shape_of_value(self):
        # Dimensionality of a realization
        return self.dims[0]
    
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
        """

        if Q is not None:
            # Rotate moments using Q
            #print("Debug in rotate matrix", np.shape(self.u[0]), self.get_shape(0))
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

        #print("debug rotate", np.sum(self.u[1],axis=0), self.name)

        ## XX = np.sum(np.reshape(self.u[1], (-1,D1,D2,D1,D2)),
        ##             axis=(0,1,3))
        ## print("DEBUG", XX)
