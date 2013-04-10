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
import warnings
import scipy

from bayespy import utils
from bayespy.utils.linalg import dot, tracedot

class RotationOptimizer():

    def __init__(self, block1, block2, D):
        self.block1 = block1
        self.block2 = block2
        self.D = D

    def rotate(self, 
               maxiter=10, 
               check_gradient=False,
               verbose=False,
               check_bound=None,
               check_bound_terms=None):
        """
        Optimize the rotation of two separate model blocks jointly.

        If some variable is the dot product of two Gaussians, rotating the two
        Gaussians optimally can make the inference algorithm orders of magnitude
        faster.

        First block is rotated with :math:`\mathbf{R}` and the second with
        :math:`\mathbf{R}^{-T}`.

        Blocks must have methods: `bound(U,s,V)` and `rotate(R)`.
        """

        I = np.identity(self.D)
        piv = np.arange(self.D)
        
        def cost(r):

            # Make vector-r into matrix-R
            R = np.reshape(r, (self.D,self.D))

            # Compute SVD
            invR = np.linalg.inv(R)
            logdetR = np.linalg.slogdet(R)[1]

            # Compute lower bound terms
            (b1,db1) = self.block1.bound(R, logdet=logdetR, inv=invR)
            (b2,db2) = self.block2.bound(invR.T, logdet=-logdetR, inv=R.T)

            # Apply chain rule for the second gradient:
            # d b(invR.T) 
            # = tr(db.T * d(invR.T)) 
            # = tr(db * d(invR))
            # = -tr(db * invR * (dR) * invR) 
            # = -tr(invR * db * invR * dR)
            db2 = -dot(invR.T, db2.T, invR.T)

            # Compute the cost function
            c = -(b1+b2)
            dc = -(db1+db2)

            return (c, np.ravel(dc))

        def get_bound_terms(r):
            """
            Returns a dictionary of bound terms for the nodes.
            """
            # Make vector-r into matrix-R
            R = np.reshape(r, (self.D,self.D))

            # Compute SVD
            invR = np.linalg.inv(R)
            logdetR = np.linalg.slogdet(R)[1]

            # Compute lower bound terms
            dict1 = self.block1.get_bound_terms(R, logdet=logdetR, inv=invR)
            dict2 = self.block2.get_bound_terms(invR.T, logdet=-logdetR, inv=R.T)

            dict1.update(dict2)
            
            return dict1

        self.block1.setup()
        self.block2.setup()
        
        if check_gradient:
            R = np.random.randn(self.D, self.D)
            utils.optimize.check_gradient(cost, np.ravel(R))

        # Initial rotation is identity matrix
        r0 = np.ravel(np.identity(self.D))

        if check_bound is not None and check_bound is not False:
            bound_begin = check_bound()
            (cost_begin, _) = cost(r0)
        if check_bound_terms:
            bound_terms_begin = get_bound_terms(r0)
            true_bound_terms_begin = check_bound_terms()

        # Run optimization
        #print("Initial cost:", cost(r0)[0])
        r = utils.optimize.minimize(cost, r0, maxiter=maxiter, verbose=verbose)

        if check_bound is not None and check_bound is not False:
            (cost_end, _) = cost(r)
        if check_bound_terms:
            bound_terms_end = get_bound_terms(r)

        # Apply the optimal rotation
        R = np.reshape(r, (self.D,self.D))
        invR = np.linalg.inv(R)
        logdetR = np.linalg.slogdet(R)[1]
        self.block1.rotate(R, inv=invR, logdet=logdetR)
        self.block2.rotate(invR.T, inv=R.T, logdet=-logdetR)

        # Check that the cost function and the true lower bound changed equally
        if check_bound is not None and check_bound is not False:
            bound_end = check_bound()
            bound_change = bound_end - bound_begin
            cost_change = cost_end - cost_begin
            if not np.allclose(bound_change, -cost_change):
                print(cost_end, cost_begin, bound_end, bound_begin)
                warnings.warn("Rotation cost function is not consistent with "
                              "the true lower bound. Bound changed %g but "
                              "optimized function changed %g."  
                              % (bound_change, -cost_change))
            # Check that we really have improved the bound.
            if bound_change < 0:
                warnings.warn("Rotation made the lower bound worse by %g. "
                              "Probably a bug in the rotation functions."
                              % (bound_change,))
            ## if cost_change > 0:
            ##     raise Exception("WTF! Bug in optimization..")
                
        if check_bound_terms:
            true_bound_terms_end = check_bound_terms()
            for rot in bound_terms_begin.keys():
                opt_bound_change = (bound_terms_end[rot] 
                                    - bound_terms_begin[rot])
                true_bound_change = 0
                for node in rot.nodes():
                    true_bound_change += (true_bound_terms_end[node] 
                                         - true_bound_terms_begin[node])
                if not np.allclose(opt_bound_change, true_bound_change):
                    print(rot.nodes()[0].name, opt_bound_change, true_bound_change)
                    warnings.warn("Rotation cost function is not consistent "
                                  "with the true lower bound for node %s. "
                                  "Bound changed %g but optimized function "
                                  "changed %g."  
                                  % (rot.nodes()[0].name,
                                     true_bound_change,
                                     opt_bound_change))
                    raise Exception()
                

class RotateGaussian():

    def __init__(self, X):
        self.X = X

    def rotate(self, R, inv=None, logdet=None):
        self.X.rotate(R, inv=inv, logdet=logdet)

    def setup(self):
        """
        This method should be called just before optimization.
        """
        
        mask = self.X.mask[...,np.newaxis,np.newaxis]

        # Number of plates
        self.N = self.X.plates[0] #np.sum(mask)

        # Compute the sum <XX> over plates
        self.XX = utils.utils.sum_multiply(self.X.get_moments()[1],
                                           mask,
                                           axis=(-1,-2),
                                           sumaxis=False,
                                           keepdims=False)
        # Parent's moments
        self.Lambda = self.X.parents[1].get_moments()[0]

    def _compute_bound(self, R, logdet=None, inv=None, gradient=False):
        
        """
        Rotate q(X) as X->RX: q(X)=N(R*mu, R*Cov*R')

        Assume:
        :math:`p(\mathbf{X}) = \prod^M_{m=1} 
               N(\mathbf{x}_m|0, \mathbf{\Lambda})`
        """

        # TODO/FIXME: X and alpha should NOT contain observed values!! Check
        # that.

        # TODO/FIXME: Allow non-zero prior mean!

        # Assume constant mean and precision matrix over plates..

        # Compute rotated moments
        XX_R = dot(R, self.XX, R.T)

        inv_R = inv
        logdet_R = logdet

        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-2*self.N*logdet_R, 
                                               0)

        # Compute <log p(X)>
        logp_X = utils.random.gaussian_logpdf(np.vdot(XX_R, self.Lambda),
                                              0,
                                              0,
                                              0,
                                              0)

        # Compute the bound
        bound = logp_X + logH_X

        if gradient:

            # Compute dH(X)
            dlogH_X = utils.random.gaussian_entropy(-2*self.N*inv_R.T,
                                                    0)

            # Compute d<log p(X)>
            dXX = 2*dot(self.Lambda, R, self.XX)
            dlogp_X = utils.random.gaussian_logpdf(dXX,
                                                   0,
                                                   0,
                                                   0,
                                                   0)

            d_bound = dlogp_X + dlogH_X

            return (bound, d_bound)

        else:
            return bound

    def bound(self, R, logdet=None, inv=None):
        return self._compute_bound(R, 
                                   logdet=logdet,
                                   inv=inv,
                                   gradient=True)

    def get_bound_terms(self, R, logdet=None, inv=None):
        bound = self._compute_bound(R, 
                                    logdet=logdet,
                                    inv=inv,
                                    gradient=False)
        
        return {self: bound}

    def nodes(self):
        return [self.X]
        
class RotateGaussianARD():

    def __init__(self, X, alpha):
        self.node_X = X
        self.node_alpha = alpha

    def nodes(self):
        return [self.node_X, self.node_alpha]

    def rotate(self, R, inv=None, logdet=None, Q=None):
        self.node_X.rotate(R, inv=inv, logdet=logdet, Q=Q)
        self.node_alpha.update()

    def setup(self, rotate_plates=False):
        """
        This method should be called just before optimization.

        If using Q, set rotate_plates to True.
        """
        
        mask = self.node_X.mask[...,np.newaxis,np.newaxis]

        # Number of plates
        self.N = self.node_X.plates[0] #np.sum(mask)

        if not rotate_plates:
            # Compute the sum <XX> over plates
            self.XX = utils.utils.sum_multiply(self.node_X.get_moments()[1],
                                               mask,
                                               axis=(-1,-2),
                                               sumaxis=False,
                                               keepdims=False)
        else:
            self.X = self.node_X.get_moments()[0] * self.node_X.mask[...,np.newaxis]
            XX = self.node_X.get_moments()[1] * mask
            self.CovX = XX - utils.linalg.outer(self.X, self.X)
            
        # Parent's moments
        self.a = np.ravel(self.node_alpha.phi[1])
        # TODO/FIXME: Handle vector valued parents a0 and b0
        self.a0 = self.node_alpha.parents[0].get_moments()[0]
        self.b0 = self.node_alpha.parents[1].get_moments()[0]

    def _compute_bound(self, R, logdet=None, inv=None, Q=None, gradient=False):
        """
        Rotate q(X) and q(alpha).

        Assume:
        p(X|alpha) = prod_m N(x_m|0,diag(alpha))
        p(alpha) = prod_d G(a_d,b_d)
        """

        # TODO/FIXME: X and alpha should NOT contain observed values!! Check that.

        #
        # Transform the distributions and moments
        #

        # Compute rotated second moment
        if Q is not None:
            # Rotate plates
            sumQ = np.sum(Q, axis=0)
            QX = np.einsum('ik,kj->ij', Q, self.X)
            XX = (np.einsum('ki,kj->ij', QX, QX)
                  + np.einsum('d,dij->ij', sumQ**2, self.CovX))
            logdet_Q = np.sum(np.log(np.abs(sumQ)))
        else:
            XX = self.XX
            logdet_Q = 0

        XX_R = dot(R, XX, R.T)


        # Compute q(alpha)
        a_alpha = self.a
        b_alpha = self.b0 + 0.5*np.diag(XX_R)
        alpha_R = a_alpha / b_alpha
        logalpha_R = - np.log(b_alpha) # + const

        logdet_R = logdet
        inv_R = inv

        N = self.N
        D = np.shape(R)[0]

        #
        # Compute the cost
        #
        
        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-2*N*logdet_R - 2*D*logdet_Q, 
                                               0)

        # Compute entropy H(alpha)
        logH_alpha = utils.random.gamma_entropy(0,
                                                np.sum(np.log(b_alpha)),
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
        logp_alpha = utils.random.gamma_logpdf(self.b0*np.sum(alpha_R),
                                               np.sum(logalpha_R),
                                               self.a0*np.sum(logalpha_R),
                                               0,
                                               0)

        # Compute the bound
        bound = (
            logp_X
            + logp_alpha
            + logH_X
            + logH_alpha
            )

        if not gradient:
            return bound

        #
        # Compute the gradient with respect R
        #

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
        dlogp_alpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                d_log_alpha,
                                                self.a0*d_log_alpha,
                                                0,
                                                0)

        dR_bound = (
            dlogp_X
            + dlogp_alpha
            + dlogH_X
            + dlogH_alpha
            )

        if Q is None:
            return (bound, dR_bound)

        #
        # Compute the gradient with respect to Q (if Q given)
        #

        def d_helper(v):
            R_v_R = np.einsum('ki,k,kj->ij', R, v, R)
            tr_R_v_R_Cov = np.einsum('ij,dji->d', R_v_R, self.CovX)
            return (dot(QX, R_v_R, self.X.T)
                    + sumQ * tr_R_v_R_Cov)
            

        # Compute dH(X)
        dQ_logHX = utils.random.gaussian_entropy(-2*D/sumQ,
                                                 0)

        # Compute dH(alpha)
        ## R_b_R = np.einsum('ki,k,kj->ij', R, 1/b_alpha, R)
        ## tr_R_b_R_Cov = np.einsum('ij,dji->d', R_b_R, self.CovX)
        ## d_log_b = (dot(QX, R_b_R, self.X.T)
        ##            + sumQ * tr_R_b_R_Cov)
        d_log_b = d_helper(1/b_alpha)
        dQ_logHalpha = utils.random.gamma_entropy(0,
                                                  d_log_b,
                                                  0,
                                                  0,
                                                  0)

        # Compute d<log p(X|alpha)>
        dXX_alpha = 2*d_helper(alpha_R)
        XX_dalpha = -d_helper(np.diag(XX_R)*alpha_R/b_alpha)
        d_log_alpha = -d_log_b
        dQ_logpX = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                                0,
                                                0,
                                                N*d_log_alpha,
                                                0)


        # Compute d<log p(alpha)>
        ## R_alpha_b_R = np.einsum('ki,k,kj->ij', 
        ##                         R, alpha_R/b_alpha, R)
        ## tr_R_alpha_b_R_Cov = np.einsum('ij,dji->d', 
        ##                                R_alpha_b_R, self.CovX)
        ## d_alpha = (dot(QX, R_alpha_b_R, self.X.T)
        ##            + sumQ * tr_R_alpha_b_R_Cov)
        d_alpha = -d_helper(alpha_R/b_alpha)
        dQ_logpalpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                 d_log_alpha,
                                                 self.a0*d_log_alpha,
                                                 0,
                                                 0)

        dQ_bound = (
            dQ_logpX
            + dQ_logpalpha
            + dQ_logHX
            + dQ_logHalpha
            )

        return (bound, dR_bound, dQ_bound)



    def bound(self, R, logdet=None, inv=None, Q=None):
        return self._compute_bound(R, 
                                   logdet=logdet, 
                                   inv=inv, 
                                   Q=Q,
                                   gradient=True)
            
    def get_bound_terms(self, R, logdet=None, inv=None, Q=None):
        bound = self._compute_bound(R, 
                                    logdet=logdet, 
                                    inv=inv, 
                                    Q=Q,
                                    gradient=False)
        
        return {self: bound}
        
class RotateGaussianMarkovChain():
    """
    Assume the following model.

    Constant, unit isotropic innovation noise.

    A may vary in time.

    No plates for X.
    """

    def __init__(self, X, A, A_rotator):
        self.X_node = X
        self.A_node = A
        self.A_rotator = A_rotator

    def nodes(self):
        # A node is in the A rotator.
        return [self.X_node]

    def rotate(self, R, inv=None, logdet=None):
        self.X_node.rotate(R, inv=inv, logdet=logdet)
        self.A_rotator.rotate(inv.T, inv=R.T, logdet=-logdet, Q=R)

    def setup(self):
        """
        This method should be called just before optimization.
        """
        
        # Compute the sum of the moments over time
        (self.X, self.XnXn, self.XpXn) = self.X_node.get_moments()
        self.X0 = self.X[0,:]
        self.X0X0 = self.XnXn[0,:,:]
        self.XpXp = np.sum(self.XnXn[:-1,:,:], axis=0)
        self.XnXn = np.sum(self.XnXn[1:,:,:], axis=0)
        self.XpXn = np.sum(self.XpXn, axis=0)

        self.N = np.shape(self.X)[-2]

        # Get moments of A
        (self.A, AA) = self.X_node.parents[2].get_moments()
        self.CovA = AA - self.A[...,:,np.newaxis]*self.A[...,np.newaxis,:]
        self.A_XpXn = dot(self.A, self.XpXn)
        self.A_XpXp_A = dot(self.A, self.XpXp, self.A.T)
        self.CovA_XpXp = np.einsum('dij,ij->d', self.CovA, self.XpXp)
        
        # Get moments of the fixed parameter nodes
        mu = self.X_node.parents[0].get_moments()[0]
        self.Lambda = self.X_node.parents[1].get_moments()[0]
        self.Lambda_mu_X0 = np.outer(np.dot(self.Lambda,mu), self.X0)

        self.A_rotator.setup(rotate_plates=True)

        # Innovation noise is assumed to be I
        #self.v = self.X_node.parents[3].get_moments()[0]

    def _compute_bound(self, R, logdet=None, inv=None, gradient=False):
        """
        Rotate q(X) as X->RX: q(X)=N(R*mu, R*Cov*R')

        Assume:
        :math:`p(\mathbf{X}) = \prod^M_{m=1} 
               N(\mathbf{x}_m|0, \mathbf{\Lambda})`

        Assume unit innovation noise covariance.
        """

        # TODO/FIXME: X and alpha should NOT contain observed values!! Check
        # that.

        # TODO/FIXME: Allow non-zero prior mean!

        # Assume constant mean and precision matrix over plates..

        invR = inv
        logdetR = logdet

        # Transform moments of X:
        # Transform moments of A:
        Lambda_R_X0X0 = dot(self.Lambda, R, self.X0X0)
        sumr = np.sum(R, axis=0)
        R_CovA_XpXp = sumr * self.CovA_XpXp
        RA_XpXp_A = dot(R, self.A_XpXp_A)
        R_XnXn = dot(R, self.XnXn)

        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-2*self.N*logdetR, 
                                               0)

        # Compute <log p(X)>
        yy = tracedot(R_XnXn, R.T) + tracedot(Lambda_R_X0X0, R.T)
        yz = tracedot(dot(R,self.A_XpXn),R.T) + tracedot(self.Lambda_mu_X0, R.T)
        zz = tracedot(RA_XpXp_A, R.T) + np.dot(R_CovA_XpXp, sumr) #RR_CovA_XpXp
        logp_X = utils.random.gaussian_logpdf(yy,
                                              yz,
                                              zz,
                                              0,
                                              0)

        # Compute dH(X)
        dlogH_X = utils.random.gaussian_entropy(-2*self.N*invR.T,
                                                0)

        # Compute the bound
        bound = logp_X + logH_X
        
        if gradient:
            # Compute d<log p(X)>
            dyy = 2 * (R_XnXn + Lambda_R_X0X0)
            dyz = dot(R, self.A_XpXn + self.A_XpXn.T) + self.Lambda_mu_X0
            dzz = 2 * (RA_XpXp_A + R_CovA_XpXp)
            dlogp_X = utils.random.gaussian_logpdf(dyy,
                                                   dyz,
                                                   dzz,
                                                   0,
                                                   0)

            d_bound = dlogp_X + dlogH_X

            return (bound, d_bound)

        else:
            return bound

    def bound(self, R, logdet=None, inv=None):
        (bound_X, d_bound_X) = self._compute_bound(R,
                                                   logdet=logdet,
                                                   inv=inv,
                                                   gradient=True)
        
        # Compute cost and gradient from A
        (bound_A, dR_bound_A, dQ_bound_A) = self.A_rotator.bound(inv.T, 
                                                                 inv=R.T,
                                                                 logdet=-logdet,
                                                                 Q=R)
        # TODO/FIXME: Also apply the gradient of invR.T to the result
        dR_bound_A = -dot(inv.T, dR_bound_A.T, inv.T)

        # Compute the bound
        bound = bound_X + bound_A
        d_bound = d_bound_X + dR_bound_A + dQ_bound_A

        return (bound, d_bound)

    def get_bound_terms(self, R, logdet=None, inv=None):
        bound_dict = self.A_rotator.get_bound_terms(inv.T, 
                                                    inv=R.T,
                                                    logdet=-logdet,
                                                    Q=R)
        
        bound_X = self._compute_bound(R,
                                      logdet=logdet,
                                      inv=inv,
                                      gradient=False)
        
        bound_dict.update({self: bound_X})
        
        return bound_dict
