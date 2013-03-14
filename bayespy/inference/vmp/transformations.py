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
#from scipy import optimize

#import matplotlib.pyplot as plt
#import warnings
#import time
#import h5py
#import datetime
#import tempfile

from bayespy import utils
from bayespy.utils.linalg import dot

class RotationOptimizer():

    def __init__(self, block1, block2, D):
        self.block1 = block1
        self.block2 = block2
        self.D = D

    def rotate(self, maxiter=None, check_gradient=False, verbose=False):
        """
        Optimize the rotation of two separate model blocks jointly.

        If some variable is the dot product of two Gaussians, rotating the two
        Gaussians optimally can make the inference algorithm orders of magnitude
        faster.

        First block is rotated with :math:`\mathbf{R}` and the second with
        :math:`\mathbf{R}^{-T}`.

        Blocks must have methods: `bound(U,s,V)` and `rotate(R)`.
        """

        def cost(r):

            # Make vector-r into matrix-R
            R = np.reshape(r, (self.D,self.D))

            # Compute SVD
            (U,s,V) = np.linalg.svd(R)
            invR = np.linalg.inv(R)

            # Compute lower bound terms
            (b1,db1) = self.block1.bound(R, svd=(U,s,V), inv=invR)
            (b2,db2) = self.block2.bound(invR.T, svd=(U,1/s,V), inv=R.T)

            # Apply chain rule for the second gradient:
            # d b(invR.T) 
            # = tr(db.T * d(invR.T)) 
            # = tr(db * d(invR))
            # = -tr(db * invR * (dR) * invR) 
            # = -tr(invR * db * invR * dR)
            db2 = -dot(invR.T, db2.T, invR.T)
            #db2 = -np.einsum('ik,kl,lj->ij', invR.T, db2.T, invR.T)
            #db2 = -np.dot(invR.T, np.dot(db2.T, invR.T))

            # Compute the cost function
            c = -(b1+b2)
            dc = -(db1+db2)

            return (c, np.ravel(dc))

        if check_gradient:
            print("Rotator: Check gradient.")
            R0 = np.random.randn(self.D*self.D)
            utils.optimize.check_gradient(cost, R0)
            utils.optimize.check_gradient(cost, np.ravel(np.identity(self.D)))

        # Run optimization
        r0 = np.ravel(np.identity(self.D))
        r = utils.optimize.minimize(cost, r0)
        R = np.reshape(r, (self.D,self.D))

        # Apply the optimal rotation
        invR = np.linalg.inv(R)
        logdetR = np.linalg.slogdet(R)[1]
        self.block1.rotate(R, inv=invR, logdet=logdetR)
        self.block2.rotate(invR.T, inv=R.T, logdet=-logdetR)

class RotateGaussianARD():

    def __init__(self, X, alpha):
        self.X = X
        self.alpha = alpha

    def rotate(self, R, svd=None):
        pass

    def bound(self, R, svd=None):
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
        XX_R = dot(R, XX, R.T)
        #XX_R = np.einsum('ik,kl,jl', R, XX, R)

        # Compute q(alpha)
        a_0 = np.ravel(alpha.parents[0].get_moments()[0])
        b_0 = np.ravel(alpha.parents[1].get_moments()[0])
        a_alpha = np.ravel(alpha.phi[1])
        b_alpha = b0 + 0.5*np.diag(XX_R)
        alpha_R = a_alpha / b_alpha
        logalpha_R = - log(b_alpha) # + const

        N = np.sum(X.mask)
        #(Q, R) = np.linalg.qr(R)
        if not svd:
            (U,s,V) = np.linalg.svd(R)
        logdet_R = np.sum(np.log(np.abs(s)))
        #utils.linalg.logdet_tri(R)

        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-N*2*logdet_R, 
                                               0)

        # Compute entropy H(alpha)
        logH_alpha = utils.random.gamma_entropy(0,
                                                np.sum(log(b_alpha)),
                                                0,
                                                0,
                                                0)

        # Compute <log p(X|alpha)>
        logp_X = utils.random.gaussian_logpdf(np.einsum('ii,i', XX_R, alpha_R),
                                              0,
                                              0,
                                              N*np.sum(logalpha_R),
                                              0)

        # Compute <log p(alpha)>
        logp_alpha = utils.random.gamma_logpdf(b_0*np.sum(alpha_R),
                                               np.sum(logalpha_R),
                                               a_0*np.sum(logalpha_R),
                                               0,
                                               0)

        # Compute dH(X)
        dlogH_X = utils.random.gaussian_entropy(-2*N*inv_R.T,
                                                0)

        # Compute dH(alpha)
        d_log_b = np.einsum('i,ik,kj->ij', 1/b_alpha, R, XX)
        dlogH_alpha = utils.random.gamma_entropy(0,
                                                 d_log_b,
                                                 0,
                                                 0,
                                                 0)

        # Compute d<log p(X|alpha)>
        d_log_alpha = -d_log_b
        dXX_alpha = 2*np.einsum('i,ik,kj->ij', alpha_R, R, XX)
        XX_dalpha = -np.einsum('i,i,ii,ik,kj->ij', alpha_R, 1/b_alpha, XX_R, R, XX)
        dlogp_X = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                               0,
                                               0,
                                               N*d_log_alpha,
                                               0)

        # Compute d<log p(alpha)>
        d_alpha = -np.einsum('i,i,ik,kj->ij', alpha_R, 1/b_alpha, R, XX)
        dlogp_alpha = utils.random.gamma_logpdf(b_0*d_alpha,
                                                d_log_alpha,
                                                a_0*d_log_alpha,
                                                0,
                                                0)

        # Compute the bound
        bound = logp_X + logp_alpha + logH_X + logH_alpha
        d_bound = dlogp_X + dlogp_alpha + dlogH_X + dlogH_alpha
        return (bound, d_bound)

class RotateGaussian():

    def __init__(self, X):
        self.X = X

    def rotate(self, R, inv=None, logdet=None):
        self.X.rotate(R, inv=inv, logdet=logdet)

    def bound(self, R, svd=None, inv=None, verbose=False, lu=None):
        """
        Rotate q(X) as X->RX.

        Assume:
        :math:`p(\mathbf{X}) = \prod^M_{m=1} 
               N(\mathbf{x}_m|0, \mathbf{\Lambda})`
        """

        # TODO/FIXME: X and alpha should NOT contain observed values!! Check
        # that.

        # TODO/FIXME: Allow non-zero prior mean!

        # Assume constant mean and precision matrix over plates..

        N = np.sum(self.X.mask)

        # Compute the sum <XX> over plates
        mask = self.X.mask[...,np.newaxis,np.newaxis]
        XX = utils.utils.sum_multiply(self.X.get_moments()[1],
                                      mask,
                                      axis=(-1,-2),
                                      sumaxis=False,
                                      keepdims=False)
        # Compute rotated moments
        #XX_R = utils.utils.symm(np.einsum('ik,kl,jl->ij', R, XX, R))
        XX_R = dot(R, XX, R.T)


        #if svd is None:
        #    (U,s,V) = np.linalg.svd(R)
        #else:
        (U,s,V) = svd

        ## if inv is None:
        ##     inv_R = np.linalg.inv(R)
        ## else:
        inv_R = inv

        logdet_R = np.sum(np.log(np.abs(s)))

        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-2*N*logdet_R, 
                                               0)

        # Compute <log p(X)>
        Lambda = self.X.parents[1].get_moments()[0]
        logp_X = utils.random.gaussian_logpdf(np.einsum('ij,ij', XX_R, Lambda),
                                              0,
                                              0,
                                              0,
                                              0)

        # Compute dH(X)
        dlogH_X = utils.random.gaussian_entropy(-2*N*inv_R.T,
                                                0)

        # Compute d<log p(X)>
        #dXX = 2*np.einsum('ik,kl,lj->ij', Lambda, R, XX)
        dXX = 2*dot(Lambda, R, XX)
        dlogp_X = utils.random.gaussian_logpdf(dXX,
                                               0,
                                               0,
                                               0,
                                               0)

        # Compute the bound
        bound = logp_X + logH_X
        d_bound = dlogp_X + dlogH_X

        #bound = logp_X
        #d_bound = dlogp_X

        #if verbose:
        #    print("debug:", logp_X, logH_X, bound)

        if np.isnan(bound) or np.any(np.isnan(d_bound)):
            raise Exception("Gradient contains nans!")

        return (bound, d_bound)
