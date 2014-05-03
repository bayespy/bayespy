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
from .gamma import GammaMoments
from .wishart import (WishartMoments,
                      WishartPriorMoments)
from .node import (Moments,
                   ensureparents)

from bayespy.utils import random
from bayespy.utils import utils


class GaussianGammaISOMoments(Moments):
    """
    Class for the moments of Gaussian-gamma-ISO variables.
    """

    
    def compute_fixed_moments(self, x, alpha):
        """
        Compute the moments for a fixed value

        `x` is a mean vector.
        `alpha` is a precision scale
        """

        x = np.asanyarray(x)
        alpha = np.asanyarray(alpha)

        u0 = np.einsum('...,...i->...i', alpha, x)
        u1 = np.einsum('...,...i,...j->...ij', alpha, x, x)
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

        D = np.shape(x)[-1]

        return ( (D,), (D,D), (), () )


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
        raise NotImplementedError()


    def compute_moments_and_cgf(self, phi, mask=True):
        """
        Compute the moments and :math:`g(\phi)`.
        """
        raise NotImplementedError()
        return (u, g)

    
    def compute_cgf_from_parents(self, u_mu_Lambda, u_a, u_b):
        """
        Compute :math:`\mathrm{E}_{q(p)}[g(p)]`
        """
        raise NotImplementedError()
        return g

    
    def compute_fixed_moments_and_f(self, x, alpha, mask=True):
        """
        Compute the moments and :math:`f(x)` for a fixed value.
        """
        raise NotImplementedError()
        return (u, f)

    
class GaussianWishartDistribution(ExponentialFamilyDistribution):
    """
    Class for the VMP formulas of Gaussian-Wishart variables.
    """    


    def compute_message_to_parent(self, parent, index, u, u_mu, u_alpha, u_V, u_n):
        """
        Compute the message to a parent node.
        """
        if index == 0:
            raise NotImplementedError()
        elif index == 1:
            raise NotImplementedError()
        elif index == 2:
            raise NotImplementedError()
        elif index == 3:
            raise NotImplementedError()
        else:
            raise ValueError("Index out of bounds")


    def compute_phi_from_parents(self, u_mu, u_alpha, u_V, u_n, mask=True):
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

    
    def compute_cgf_from_parents(self, u_mu, u_alpha, u_V, u_n):
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

    
class GaussianWishart(ExponentialFamily):
    """
    Node for Gaussian-Wishart random variables.

    The prior:
    
    .. math::

        p(x, \Lambda| \mu, \alpha, V, n)

        p(x|\Lambda, \mu, \alpha) = \mathcal(N)(x | \mu, \alpha^{-1} Lambda^{-1})

        p(\Lambda|V, n) = \mathcal(W)(\Lambda | n, V)

    The posterior approximation :math:`q(x, \Lambda)` has the same Gaussian-Wishart form.
    """
    
    _moments = GaussianWishartMoments()
    _parent_moments = (GaussianGammaMoments(),
                       GammaMoments(),
                       WishartMoments(),
                       WishartPriorMoments())
    _distribution = GaussianWishartDistribution()
    

    @classmethod
    @ensureparents
    def _constructor(cls, mu, alpha, V, n, plates_lambda=None, plates_x=None, **kwargs):
        """
        Constructs distribution and moments objects.

        This method is called if useconstructor decorator is used for __init__.

        `mu` is the mean/location vector
        `alpha` is the scale
        `V` is the scale matrix
        `n` is the degrees of freedom
        """

        D = mu.dims[0][0]

        # Check shapes
        if mu.dims != ( (D,), (D,D), (), () ):
            raise ValueError("Mean vector has wrong shape")

        if alpha.dims != ( (), () ):
            raise ValueError("Scale has wrong shape")

        if V.dims != ( (D,D), () ):
            raise ValueError("Precision matrix has wrong shape")

        if n.dims != ( (), () ):
            raise ValueError("Degrees of freedom has wrong shape")

        dims = ( (D,), (), (D,D), () )

        return (dims,
                kwargs,
                cls._total_plates(kwargs.get('plates'),
                                  cls._distribution.plates_from_parent(0, mu.plates),
                                  cls._distribution.plates_from_parent(1, alpha.plates),
                                  cls._distribution.plates_from_parent(2, V.plates),
                                  cls._distribution.plates_from_parent(3, n.plates)),
                cls._distribution, 
                cls._moments, 
                cls._parent_moments)

    
    def random(self):
        """
        Draw a random sample from the distribution.
        """
        raise NotImplementedError()

    
    def show(self):
        """
        Print the distribution using standard parameterization.
        """
        raise NotImplementedError()
