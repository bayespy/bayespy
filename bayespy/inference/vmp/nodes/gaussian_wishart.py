######################################################################
# Copyright (C) 2014 Jaakko Luttinen
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
Module for the Gaussian-Wishart and similar distributions.
"""

import numpy as np
from scipy import special

from .expfamily import (ExponentialFamily,
                        ExponentialFamilyDistribution,
                        useconstructor)
from .gaussian import GaussianMoments
from .gamma import (GammaMoments,
                    GammaPriorMoments)
from .wishart import (WishartMoments,
                      WishartPriorMoments)
from .node import (Moments,
                   ensureparents)
from .deterministic import Deterministic

from bayespy.utils import (random,
                           misc,
                           linalg)


class GaussianGammaISOMoments(Moments):
    """
    Class for the moments of Gaussian-gamma-ISO variables.
    """


    def __init__(self, ndim):
        """
        Create moments object for Gaussian-gamma isotropic variables

        ndim=0: scalar
        ndim=1: vector
        ndim=2: matrix
        ...
        """
        self.ndim = ndim
        super().__init__()

    
    def compute_fixed_moments(self, x, alpha):
        """
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
        """
        Return the shape of the moments for a fixed value.
        """

        if np.ndim(x) < 1:
            raise ValueError("Mean must be a vector")

        if ndim == 0:
            return ( (), (), (), () )
        
        dims = np.shape(x)[-ndim:]

        return ( dims, 2*dims, (), () )


class GaussianGammaARDMoments(Moments):
    """
    Class for the moments of Gaussian-gamma-ARD variables.
    """

    
    def compute_fixed_moments(self, x, alpha):
        """
        Compute the moments for a fixed value

        `x` is a mean vector.
        `alpha` is a precision scale
        """

        x = np.asanyarray(x)
        alpha = np.asanyarray(alpha)

        if np.ndim(x) < 1:
            raise ValueError("Mean must be a vector")
        if np.ndim(alpha) < 1:
            raise ValueError("ARD scales must be a vector")

        if np.shape(x)[-1] != np.shape(alpha)[-1]:
            raise ValueError("Mean and ARD scales have inconsistent shapes")
        
        u0 = np.einsum('...i,...i->...i', alpha, x)
        u1 = np.einsum('...k,...k,...k->...k', alpha, x, x)
        u2 = np.copy(alpha)
        u3 = np.log(alpha)
        u = [u0, u1, u2, u3]

        return u
    

    def compute_dims_from_values(self, x, alpha):
        """
        Return the shape of the moments for a fixed value.
        """

        if np.ndim(x) < 1:
            raise ValueError("Mean must be a vector")
        if np.ndim(alpha) < 1:
            raise ValueError("ARD scales must be a vector")

        D = np.shape(x)[-1]

        if np.shape(alpha)[-1] != D:
            raise ValueError("Mean and ARD scales have inconsistent shapes")

        return ( (D,), (D,), (D,), (D,) )


class GaussianWishartMoments(Moments):
    """
    Class for the moments of Gaussian-Wishart variables.
    """
    
    
    def compute_fixed_moments(self, x, Lambda):
        """
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
        """
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


class GaussianGammaISODistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Gaussian-Gamma-ISO variables.

    Currently, supports only vector variables.
    """    


    def compute_message_to_parent(self, parent, index, u, u_mu_Lambda, u_a, u_b):
        """
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
        """
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
        """
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
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        logdet_Lambda = u_mu_Lambda[3]
        a = u_a[0]
        gammaln_a = u_a[1]
        log_b = u_b[1]
        g = 0.5*logdet_Lambda + a*log_b - gammaln_a
        return g

    
    def compute_fixed_moments_and_f(self, x, alpha, mask=True):
        """
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
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()


class GaussianGammaARDDistribution(ExponentialFamilyDistribution):
    """
    """


    def __init__(self):
        raise NotImplementedError()

    
class GaussianWishartDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Gaussian-Wishart variables.

    Currently, supports only vector variables.
    """    


    def compute_message_to_parent(self, parent, index, u, u_mu_alpha, u_V, u_n):
        """
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
        """
        Compute the natural parameter vector given parent moments.
        """
        raise NotImplementedError()


    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        raise NotImplementedError()
        return (u, g)

    
    def compute_cgf_from_parents(self, u_mu_alpha, u_V, u_n):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        raise NotImplementedError()
        return g

    
    def compute_fixed_moments_and_f(self, x, Lambda, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        raise NotImplementedError()
        return (u, f)

    
    def random(self, *params, plates=None):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()


class GaussianGammaISO(ExponentialFamily):
    """
    Node for Gaussian-gamma (isotropic) random variables.

    The prior:
    
    .. math::

        p(x, \alpha| \mu, \Lambda, a, b)

        p(x|\alpha, \mu, \Lambda) = \mathcal(N)(x | \mu, \alpha^{-1} Lambda^{-1})

        p(\alpha|a, b) = \mathcal(G)(\alpha | a, b)

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
        """
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

    
    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        raise NotImplementedError()


class GaussianGammaARD(ExponentialFamily):
    """
    """


    def __init__(self):
        """
        """
        raise NotImplementedError()

    
class GaussianWishart(ExponentialFamily):
    """
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
        """
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

    
    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        raise NotImplementedError()


#
# CONVERTERS
#


class GaussianToGaussianGammaISO(Deterministic):
    """
    Converter for Gaussian moments to Gaussian-gamma isotropic moments

    Combines the Gaussian moments with gamma moments for a fixed value 1.
    """



    def __init__(self, X, **kwargs):
        """
        """
        self.ndim = X._moments.ndim
        
        self._moments = GaussianGammaISOMoments(self.ndim)
        self._parent_moments = [GaussianMoments(self.ndim)]
    
        shape = X.dims[0]
        dims = ( shape, 2*shape, (), () )
        super().__init__(X, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X):
        """
        """
        x = u_X[0]
        xx = u_X[1]
        u = [x, xx, 1, 0]
        return u
    

    def _compute_message_to_parent(self, index, m_child, u_X):
        """
        """
        if index == 0:
            m = m_child[:2]
            return m
        else:
            raise ValueError("Invalid parent index")


GaussianMoments.add_converter(GaussianGammaISOMoments,
                              GaussianToGaussianGammaISO)


class GaussianGammaISOToGaussianGammaARD(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


class GaussianGammaARDToGaussianWishart(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


class GaussianGammaISOToGamma(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


class GaussianGammaARDToGamma(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


class GaussianWishartToWishart(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


#
# WRAPPERS
#
# These wrappers form a single node from two nodes for messaging purposes.
#


class WrapToGaussianGammaISO(Deterministic):
    """
    """


    _moments = GaussianGammaISOMoments(1)
    _parent_moments = [GaussianGammaISOMoments(1),
                       GammaMoments()]
    

    @ensureparents
    def __init__(self, X, alpha, **kwargs):
        """
        """
        D = X.dims[0][0]
        dims = ( (D,), (D,D), (), () )
        super().__init__(X, alpha, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X, u_alpha):
        """
        """
        raise NotImplementedError()
    

    def _compute_message_to_parent(self, index, m_child, u_X, u_alpha):
        """
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _compute_mask_to_parent(self, index, mask):
        """
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_to_parent(self, index):
        """
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_from_parent(self, index):
        """
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


class WrapToGaussianGammaARD(Deterministic):
    """
    """


    def __init__(self):
        raise NotImplementedError()


class WrapToGaussianWishart(Deterministic):
    """
    Wraps Gaussian and Wishart nodes into a Gaussian-Wishart node.

    The following node combinations can be wrapped:
        * Gaussian and Wishart
        * Gaussian-gamma and Wishart
        * Gaussian-Wishart and gamma
    """


    _moments = GaussianWishartMoments()
    

    def __init__(self, X, Lambda, **kwargs):
        """
        """

        # Just in case X is an array, convert it to a Gaussian node first.
        try:
            X = self._ensure_moments(X, GaussianMoments(1))
        except ValueError:
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
        """
        """
        if self.wishart:
            alpha_x = u_X_alpha[0]
            alpha_xx = u_X_alpha[1]
            alpha = u_X_alpha[2]
            log_alpha = u_X_alpha[3]
            Lambda = u_Lambda[0]
            logdet_Lambda = u_Lambda[1]

            D = self.dims[0][0]
            
            u0 = linalg.mvdot(Lambda, alpha_x)
            u1 = linalg.inner(Lambda, alpha_xx, ndim=2)
            u2 = Lambda * misc.add_trailing_axes(alpha, 2)
            u3 = logdet_Lambda + D * log_alpha
            u = [u0, u1, u2, u3]

            return u
        else:
            raise NotImplementedError()
    

    def _compute_message_to_parent(self, index, m_child, u_X, u_alpha):
        """
        """
        if index == 0:
            if self.wishart:
                # Message to Gaussian-gamma (isotropic)
                raise NotImplementedError()
            else:
                # Message to Gaussian-Wishart
                raise NotImplementedError()
        elif index == 1:
            if self.wishart:
                # Message to Wishart
                raise NotImplementedError()
            else:
                # Message to gamma (isotropic)
                raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")
