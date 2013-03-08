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

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import h5py
import datetime
import tempfile

from bayespy import utils

def bound_rotate_gaussian_ard(V, X, alpha):
    """
    Rotate q(X) and q(alpha).

    Assume:
    p(X|alpha) = prod_m N(x_m|0,diag(alpha))
    p(alpha) = prod_d G(a_d,b_d)
    """

    # TODO/FIXME: X and alpha should NOT contain observed values!! Check that.

    cost = 0

    # Compute the sum <XX> over plates
    mask = X.mask[...,np.newaxis,np.newaxis]
    XX = utils.utils.sum_multiply(X.get_moments()[1],
                                  mask,
                                  axis=(-1,-2),
                                  sumaxis=False,
                                  keepdims=False)
    # Compute rotated second moment
    XX_V = np.einsum('ik,kl,jl', V, XX, V)
    
    # Compute q(alpha)
    a_0 = np.ravel(alpha.parents[0].get_moments()[0])
    b_0 = np.ravel(alpha.parents[1].get_moments()[0])
    a_alpha = np.ravel(alpha.phi[1])
    b_alpha = b0 + 0.5*np.diag(XX_V)
    alpha_V = a_alpha / b_alpha
    logalpha_V = - log(b_alpha) # + const

    N = np.sum(X.mask)
    (Q, R) = np.linalg.qr(V)
    logdet_V = utils.linalg.logdet_tri(R)

    # Compute entropy H(X)
    logH_X = utils.random.gaussian_entropy(-N*2*logdet_V, 
                                           0)

    # Compute entropy H(alpha)
    logH_alpha = utils.random.gamma_entropy(0,
                                            np.sum(log(b_alpha)),
                                            0,
                                            0,
                                            0)

    # Compute <log p(X|alpha)>
    logp_X = utils.random.gaussian_logpdf(np.einsum('ii,i', XX_V, alpha_V),
                                          0,
                                          0,
                                          N*np.sum(logalpha_V),
                                          0)

    # Compute <log p(alpha)>
    logp_alpha = utils.random.gamma_logpdf(b_0*np.sum(alpha_V),
                                           np.sum(logalpha_V),
                                           a_0*np.sum(logalpha_V),
                                           0,
                                           0)

    # Compute dH(X)
    dlogH_X = utils.random.gaussian_entropy(-2*N*inv_V.T,
                                            0)

    # Compute dH(alpha)
    d_log_b = np.einsum('i,ik,kj->ij', 1/b_alpha, V, XX)
    dlogH_alpha = utils.random.gamma_entropy(0,
                                             d_log_b,
                                             0,
                                             0,
                                             0)

    # Compute d<log p(X|alpha)>
    d_log_alpha = -d_log_b
    dXX_alpha = 2*np.einsum('i,ik,kj->ij', alpha_V, V, XX)
    XX_dalpha = -np.einsum('i,i,ii,ik,kj->ij', alpha_V, 1/b_alpha, XX_V, V, XX)
    dlogp_X = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                           0,
                                           0,
                                           N*d_log_alpha,
                                           0)

    # Compute d<log p(alpha)>
    d_alpha = -np.einsum('i,i,ik,kj->ij', alpha_V, 1/b_alpha, V, XX)
    dlogp_alpha = utils.random.gamma_logpdf(b_0*d_alpha,
                                            d_log_alpha,
                                            a_0*d_log_alpha,
                                            0,
                                            0)

    # Compute the bound
    bound = logp_X + logp_alpha + logH_X + logH_alpha
    d_bound = dlogp_X + dlogp_alpha + dlogH_X + dlogH_alpha
    return (bound, d_bound)

def bound_rotate_gaussian(V, X):
    """
    Rotate q(X) as X->RX.

    Assume:
    :math:`p(\mathbf{X}) = \prod^M_{m=1} 
           N(\mathbf{x}_m|0, \mathbf{\Lambda})`
    """

    # TODO/FIXME: X and alpha should NOT contain observed values!! Check that.

    # Compute the sum <XX> over plates
    mask = X.mask[...,np.newaxis,np.newaxis]
    XX = utils.utils.sum_multiply(X.get_moments()[1],
                                  mask,
                                  axis=(-1,-2),
                                  sumaxis=False,
                                  keepdims=False)
    # Compute rotated moments
    XX_V = np.einsum('ik,kl,jl', V, XX, V)
    

    N = np.sum(X.mask)
    (Q, R) = np.linalg.qr(V)
    logdet_V = utils.linalg.logdet_tri(R)

    # Compute entropy H(X)
    logH_X = utils.random.gaussian_entropy(-N*2*logdet_V, 
                                           0)

    # Compute <log p(X)>
    Lambda = X.parents[1].get_moments()[0]
    logp_X = utils.random.gaussian_logpdf(np.einsum('ij,ij', XX_V, Lambda),
                                          0,
                                          0,
                                          0,
                                          0)

    # Compute dH(X)
    dlogH_X = utils.random.gaussian_entropy(-2*N*inv_V.T,
                                            0)

    # Compute d<log p(X)>
    d_log_alpha = -d_log_b
    dXX = 2*np.einsum('ik,kl,lj->ij', Lambda, V, XX)
    dlogp_X = utils.random.gaussian_logpdf(dXX,
                                           0,
                                           0,
                                           0,
                                           0)

    # Compute the bound
    bound = logp_X + logH_X
    d_bound = dlogp_X + dlogH_X
    return (bound, d_bound)


def cost_rotate_gmc(X, A, alpha):
    pass


def rotate(C, X):
    """
    Optimize rotation of C and X.

    (C*R) * (inv(R)*X)
    """
    pass
