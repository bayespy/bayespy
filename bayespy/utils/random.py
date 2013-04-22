######################################################################
# Copyright (C) 2013 Jaakko Luttinen
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
General functions random sampling and distributions.
"""

import numpy as np

from . import linalg

def mask(*shape, p=0.5):
    """
    Return a boolean array of the given shape.

    Parameters:
    -----------
    d0, d1, ..., dn : int
        Shape of the output.
    p : value in range [0,1]
        A probability that the elements are `True`.
    """
    return np.random.rand(*shape) < p

def wishart_rand(nu, V):
    """
    Draw a random sample from the Wishart distribution.

    Parameters:
    -----------
    nu : int
    """
    # TODO/FIXME: Are these correct..
    D = np.shape(V)[0]
    if nu < D:
        raise ValueError("Degrees of freedom must be equal or greater than the "
                         "dimensionality of the matrix.")
    X = np.random.multivariate_normal(np.zeros(D), V, size=nu)
    return np.dot(X, X.T)

def invwishart_rand(nu, V):
    # TODO/FIXME: Are these correct..
    return np.linalg.inv(wishart_rand(nu, V))

def covariance(D, size=()):
    """
    Draw a random covariance matrix.

    Draws from inverse-Wishart distribution. The distribution of each element is
    independent of the dimensionality of the matrix.

    C ~ Inv-W(I, D)

    Parameters:
    -----------
    D : int
        Dimensionality of the covariance matrix.

    Returns:
    --------
    C : (D,D) ndarray
        Positive-definite symmetric :math:`D\times D` matrix.
    """

    if isinstance(size, int):
        size = (size,)
        
    shape = tuple(size) + (D,D)
    C = np.random.randn(*shape)
    C = linalg.dot(C, np.swapaxes(C, -1, -2))
    return linalg.inv(C)
#return np.linalg.inv(np.dot(C, C.T))

def correlation(D):
    """
    Draw a random correlation matrix.
    """
    X = np.random.randn(D,D);
    s = np.sqrt(np.sum(X**2, axis=-1, keepdims=True))
    X = X / s
    return np.dot(X, X.T)


def gaussian_logpdf(yVy, yVmu, muVmu, logdet_V, D):
    """
    Log-density of a Gaussian distribution.

    :math:`\mathcal{G}(\mathbf{y}|\boldsymbol{\mu},\mathbf{V}^{-1})`

    Parameters:
    -----------
    yVy : ndarray or double
        :math:`\mathbf{y}^T\mathbf{Vy}`
    yVmu : ndarray or double
        :math:`\mathbf{y}^T\mathbf{V}\boldsymbol{\mu}`
    muVmu : ndarray or double
        :math:`\boldsymbol{\mu}^T\mathbf{V}\boldsymbol{\mu}`
    logdet_V : ndarray or double
        Log-determinant of the precision matrix, :math:`\log|\mathbf{V}|`.
    D : int
        Dimensionality of the distribution.
    """
    return -0.5*yVy + yVmu - 0.5*muVmu + 0.5*logdet_V - 0.5*D*np.log(2*np.pi)

def gaussian_entropy(logdet_V, D):
    """
    Compute the entropy of a Gaussian distribution.

    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters:
    -----------
    logdet_V : ndarray or double
        The log-determinant of the precision matrix.
    D : int
        The dimensionality of the distribution.
    """
    return -0.5*logdet_V + 0.5*D + 0.5*D*np.log(2*np.pi)

def gamma_logpdf(bx, logx, a_logx, a_logb, gammaln_a):
    """
    Log-density of :math:`\mathcal{G}(x|a,b)`.
    
    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters:
    -----------
    bx : ndarray
        :math:`bx`
    logx : ndarray
        :math:`\log(x)`
    a_logx : ndarray
        :math:`a \log(x)`
    a_logb : ndarray
        :math:`a \log(b)`
    gammaln_a : ndarray
        :math:`\log\Gamma(a)`
    """
    return a_logb - gammaln_a + a_logx - logx - bx
#def gamma_logpdf(a, log_b, gammaln_a, 

def gamma_entropy(a, log_b, gammaln_a, psi_a, a_psi_a):
    """
    Entropy of :math:`\mathcal{G}(a,b)`.

    If you want to get the gradient, just let each parameter be a gradient of
    that term.

    Parameters:
    -----------
    a : ndarray
        :math:`a`
    log_b : ndarray
        :math:`\log(b)`
    gammaln_a : ndarray
        :math:`\log\Gamma(a)`
    psi_a : ndarray
        :math:`\psi(a)`
    a_psi_a : ndarray
        :math:`a\psi(a)`
    """
    return a - log_b + gammaln_a + psi_a - a_psi_a

def orth(D):
    """
    Draw random orthogonal matrix.
    """
    Q = np.random.randn(D,D)
    (Q, _) = np.linalg.qr(Q)
    return Q

def svd(s):
    """
    Draw a random matrix given its singular values.
    """
    D = len(s)
    U = orth(D) * s
    V = orth(D)
    return np.dot(U, V.T)
    
