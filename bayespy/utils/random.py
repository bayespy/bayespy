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

def covariance(D):
    """
    Draw a random covariance matrix.

    Parameters:
    -----------
    D : int
        Dimensionality of the covariance matrix.

    Returns:
    --------
    C : (D,D) ndarray
        Positive-definite symmetric :math:`D\times D` matrix.
    """
    X = np.random.randn(D,D)
    return np.dot(X, X.T) / D

def wishart_rand(nu, V):
    """
    Draw a random sample from the Wishart distribution.

    Parameters:
    -----------
    nu : int
    """
    raise NotImplementedError()

def gaussian_logpdf(yVy, yVmu, muVmu, logdet_cov, D):
    """
    Compute the log probability density of a Gaussian distribution.

    Parameters:
    yVy : ndarray or double
    yVmu : ndarray or double
    muVmu : ndarray or double
    logdet_cov : ndarray or double
    D : int
    """
    return -0.5*yVy + yVmu - 0.5*muVmu - 0.5*logdet_cov - 0.5*D*np.log(2*np.pi)

def gaussian_entropy(logdet_cov, D):
    """
    Compute the entropy of a Gaussian distribution.

    Parameters:
    -----------
    logdet_cov : ndarray or double
        The log-determinant of the covariance matrix.
    D : int
        The dimensionality of the distribution.
    """
    return 0.5*logdet_cov + 0.5*D + 0.5*D*np.log(2*np.pi)
