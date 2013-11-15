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

import numpy as np
import warnings
import scipy

from bayespy import utils
from bayespy.utils.linalg import dot, tracedot

from .nodes import gaussian

class RotationOptimizer():

    def __init__(self, block1, block2, D):
        self.block1 = block1
        self.block2 = block2
        self.D = D

    def rotate(self, 
               maxiter=10, 
               check_gradient=False,
               verbose=False,
               check_bound=False,
               check_bound_terms=False):
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

        def get_true_bound_terms():
            D = {}
            for node in self.block1.nodes():
                L = node.lower_bound_contribution()
                D[node] = L
            for node in self.block2.nodes():
                L = node.lower_bound_contribution()
                D[node] = L
            return D

        def get_true_bound():
            nodes = set()
            for node in self.block1.nodes():
                nodes.add(node)
            for node in self.block2.nodes():
                nodes.add(node)
            L = 0
            for node in nodes:
                L += node.lower_bound_contribution()
            return L
                

        self.block1.setup()
        self.block2.setup()
        
        if check_gradient:
            R = np.random.randn(self.D, self.D)
            err = utils.optimize.check_gradient(cost, np.ravel(R), 
                                                verbose=False)
            if err > 1e-3:
                warnings.warn("Rotation gradient has relative error %g" % err)

        # Initial rotation is identity matrix
        r0 = np.ravel(np.identity(self.D))

        (cost_begin, _) = cost(r0)
        ## if check_bound is not None and check_bound is not False:
        ##     bound_begin = check_bound()
        if check_bound:
            bound_begin = get_true_bound()
        if check_bound_terms:
            bound_terms_begin = get_bound_terms(r0)
            true_bound_terms_begin = get_true_bound_terms()
            #true_bound_terms_begin = check_bound_terms()

        # Run optimization
        r = utils.optimize.minimize(cost, r0, maxiter=maxiter, verbose=verbose)

        #if check_bound is not None and check_bound is not False:
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
        cost_change = cost_end - cost_begin
        ## if check_bound is not None and check_bound is not False:
        ##     bound_end = check_bound()
        if check_bound:
            bound_end = get_true_bound()
            bound_change = bound_end - bound_begin
            if not np.allclose(bound_change, -cost_change):
                warnings.warn("Rotation cost function is not consistent with "
                              "the true lower bound. Bound changed %g but "
                              "optimized function changed %g."  
                              % (bound_change, -cost_change))
            # Check that we really have improved the bound.
            if bound_change < 0:
                warnings.warn("Rotation made the true lower bound worse by %g. "
                              "Probably a bug in the rotation functions."
                              % (bound_change,))
        # Check that we really have improved the bound.
        if cost_change > 0:
            warnings.warn("Rotation optimization made the cost function worse "
                          "by %g. Probably a bug in the gradient of the "
                          "rotation functions."
                          % (cost_change,))
                
        if check_bound_terms:
            true_bound_terms_end = get_true_bound_terms()
            #true_bound_terms_end = check_bound_terms()
            for rot in bound_terms_begin.keys():
                opt_bound_change = (bound_terms_end[rot] 
                                    - bound_terms_begin[rot])
                true_bound_change = 0
                for node in rot.nodes():
                    try:
                        true_bound_change += (true_bound_terms_end[node] 
                                              - true_bound_terms_begin[node])
                    except KeyError:
                        raise Exception("The node %s is part of the "
                                        "transformation but not part of the "
                                        "model. Check your VB construction." 
                                        % node.name)
                if not np.allclose(opt_bound_change, true_bound_change):
                    warnings.warn("Rotation cost function is not consistent "
                                  "with the true lower bound for node %s. "
                                  "Bound changed %g but optimized function "
                                  "changed %g."  
                                  % (rot.nodes()[0].name,
                                     true_bound_change,
                                     opt_bound_change))
                    pass
                

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


    
def covariance_to_variance(C, ndim=1, covariance_axis=None):
    # Force None to empty list
    if covariance_axis is None:
        covariance_axis = []

    # Force a list from integer
    if isinstance(covariance_axis, int):
        covariance_axis = [covariance_axis]

    # Force positive axis indices
    covariance_axis = [axis + ndim if axis < 0 else axis 
                       for axis in covariance_axis]
    
    # Make a set of the axes
    covariance_axis = set(covariance_axis)

    keys = [i+ndim if i in covariance_axis else i for i in range(ndim)]
    keys += [i+2*ndim if i in covariance_axis else i for i in range(ndim)]
    out_keys = sorted(list(set(keys)))

    return np.einsum(C, [Ellipsis]+keys, [Ellipsis]+out_keys)

class RotateGaussianArrayARD():
    """
    Class for computing the cost of rotating a Gaussian array with ARD prior.

    The model:

    alpha ~ N(a, b)
    X ~ N(mu, alpha)

    X can be an array (e.g., GaussianArrayARD).

    Transform q(X) and q(alpha) by rotating X.

    Requirements:
    * X and alpha do not contain any observed values
    """
    def __init__(self, X, *alpha, axis=-1):
        if len(alpha) == 0:
            alpha = X.parents[1]
            self.update_alpha = False
        elif len(alpha) == 1:
            alpha = alpha[0]
            self.update_alpha = True
        else:
            raise ValueError("Too many arguments")
        self.node_X = X
        self.node_alpha = alpha
        self.node_mu = X.parents[0]
        self.ndim = len(X.dims[0])

        # Force negative rotation axis indexing
        if not isinstance(axis, int):
            raise ValueError("Axis must be integer")
        if axis >= 0:
            axis -= self.ndim
        if axis < -self.ndim or axis >= 0:
            raise ValueError("Axis out of bounds")
        self.axis = axis

    def nodes(self):
        if self.update_alpha:
            return [self.node_X, self.node_alpha]
        else:
            return [self.node_X]

    def rotate(self, R, inv=None, logdet=None, Q=None):
        self.node_X.rotate(R, inv=inv, logdet=logdet, Q=Q)
        if self.update_alpha:
            self.node_alpha.update()

    def setup(self, rotate_plates=False):
        """
        This method should be called just before optimization.

        For efficiency, sum over axes that are not in mu, alpha nor rotation.

        If using Q, set rotate_plates to True.
        """
        
        # Get the mean parameter. It will not be rotated.
        (mu, mumu) = self.node_mu.get_moments()
        # For simplicity, force mu to have the same shape as X
        (mu, mumu) = gaussian.reshape_gaussian_array(self.node_mu.dims[0],
                                                     self.node_X.dims[0],
                                                     mu,
                                                     mumu)

        (X, XX) = self.node_X.get_moments()

        # Take diagonal of covariances to variances for axes that are not in R
        # (and move those axes to be the last)
        XX = covariance_to_variance(XX,
                                    ndim=self.ndim,
                                    covariance_axis=self.axis)
        mumu = covariance_to_variance(mumu,
                                      ndim=self.ndim, 
                                      covariance_axis=self.axis)
        
        # Move axes of X and mu and compute their outer product
        X = utils.utils.moveaxis(X, self.axis, -1)
        mu = utils.utils.moveaxis(mu, self.axis, -1)
        Xmu = utils.linalg.outer(X, mu, ndim=1)
        D = np.shape(X)[-1]
        
        # Move axes of alpha related variables
        def safe_move_axis(x):
            if np.ndim(x) >= -self.axis:
                return utils.utils.moveaxis(x, self.axis, -1)
            else:
                return x[...,np.newaxis]
        if self.update_alpha:
            a = safe_move_axis(self.node_alpha.phi[1])
            a0 = safe_move_axis(self.node_alpha.parents[0].get_moments()[0])
            b0 = safe_move_axis(self.node_alpha.parents[1].get_moments()[0])
        else:
            alpha = safe_move_axis(self.node_alpha.get_moments()[0])
            logalpha = safe_move_axis(self.node_alpha.get_moments()[1])

        # Move plates of alpha
        plates_alpha = list(self.node_alpha.plates)
        if len(plates_alpha) >= -self.axis:
            plate = plates_alpha.pop(self.axis)
            plates_alpha.append(plate)
        else:
            plates_alpha.append(1)
            
        plates_X = list(self.node_X.get_shape(0))
        plates_X.pop(self.axis)

        # Sum axes that are not in the plates of alpha
        def sum_to_alpha(V):
            plates_to = plates_alpha[:-1] + [D,D]
            Y = (utils.utils.sum_to_shape(V, plates_to) *
                 self.node_X._plate_multiplier(plates_X, np.shape(V)[:-2]))
            return Y
        XX = sum_to_alpha(XX)
        mumu = sum_to_alpha(mumu)
        Xmu = sum_to_alpha(Xmu)

        self.XX = XX
        self.mumu = mumu
        self.Xmu = Xmu
        if self.update_alpha:
            self.a = a
            self.a0 = a0
            self.b0 = b0
        else:
            self.alpha = alpha
            self.logalpha =  logalpha

        self.plates_X = plates_X
        self.plates_alpha = plates_alpha
    

    def _compute_bound(self, R, logdet=None, inv=None, Q=None, gradient=False):
        """
        Rotate q(X) and q(alpha).

        Assume:
        p(X|alpha) = prod_m N(x_m|0,diag(alpha))
        p(alpha) = prod_d G(a_d,b_d)
        """

        #
        # Transform the distributions and moments
        #

        # Compute rotated second moment
        if Q is not None:
            raise NotImplementedError()
            # Rotate plates
            sumQ = np.sum(Q, axis=0)
            QX = np.einsum('ik,kj->ij', Q, self.X)
            XX = (np.einsum('ki,kj->ij', QX, QX)
                  + np.einsum('d,dij->ij', sumQ**2, self.CovX))
            logdet_Q = np.sum(np.log(np.abs(sumQ)))
            X = QX
            X_mu = utils.utils.sum_multiply(X[...,:,np.newaxis],
                                            self.mu[...,np.newaxis,:],
                                            axis=(-1,-2),
                                            sumaxis=False,
                                            keepdims=False)
        else:
            #X = self.X
            XX = self.XX
            mumu = self.mumu
            Xmu = self.Xmu
            logdet_Q = 0

        plates_alpha = self.plates_alpha
        plates_X = self.plates_X
        
        # Compute transformed moments
        mumu = np.einsum('...ii->...i', mumu)
        RXmu = np.einsum('...ik,...ki->...i', R, Xmu)
        RXX = np.einsum('...ik,...kj->...ij', R, XX)
        RXXR = np.einsum('...ik,...ik->...i', RXX, R)

        # <(X-mu) * (X-mu)'>_R
        XmuXmu = (RXXR - 2*RXmu + mumu)

        # Compute q(alpha)
        if self.update_alpha:
            # Parameters
            a0 = self.a0
            b0 = self.b0
            a = self.a
            b = self.b0 + 0.5*XmuXmu
            # Some expectations
            alpha = a / b
            logb = np.log(b)
            logalpha = -logb # + const
            b0_alpha = b0 * alpha
            a0_logalpha = a0 * logalpha
        else:
            alpha = self.alpha
            logalpha = self.logalpha
        
        #
        # Compute the cost
        #

        def sum_plates(V, *plates):
            full_plates = utils.utils.broadcasted_shape(*plates)
            return (np.sum(V) * self.node_X._plate_multiplier(full_plates,
                                                              np.shape(V)))

        XmuXmu_alpha = XmuXmu * alpha

        logdet_R = logdet
        inv_R = inv

        D = np.shape(R)[0]

        # Compute entropy H(X)
        # logH_X = utils.random.gaussian_entropy(-2*N*logdet_R - 2*D*logdet_Q, 
        logH_X = utils.random.gaussian_entropy(-2*sum_plates(logdet_R,
                                                             plates_X),
                                               0)

        # Compute <log p(X|alpha)>
        logp_X = utils.random.gaussian_logpdf(sum_plates(XmuXmu_alpha,
                                                         plates_alpha,
                                                         (D,)),
                                              0,
                                              0,
                                              sum_plates(logalpha,
                                                         plates_X + [D]),
                                              0)

        if self.update_alpha:
            # Compute entropy H(alpha)
            logH_alpha = utils.random.gamma_entropy(0,
                                                    sum_plates(logb,
                                                               plates_alpha),
                                                    0,
                                                    0,
                                                    0)

            # Compute <log p(alpha)>
            logp_alpha = utils.random.gamma_logpdf(sum_plates(b0_alpha,
                                                              plates_alpha),
                                                   sum_plates(logalpha,
                                                              plates_alpha),
                                                   sum_plates(a0_logalpha,
                                                              plates_alpha),
                                                   0,
                                                   0)
        else:
            logH_alpha = 0
            logp_alpha = 0

        # Compute the bound
        bound = (0
        + logp_X
        + logp_alpha
        + logH_X
        + logH_alpha
            )

        if not gradient:
            return bound

        #
        # Compute the gradient with respect R
        #

        def sum_plates(V, plates):
            ones = np.ones(np.shape(R))
            return (utils.utils.sum_multiply(V, ones,
                                             axis=(-1,-2),
                                             sumaxis=False,
                                             keepdims=False) *
                    self.node_X._plate_multiplier(plates,
                                                  np.shape(V)[:-2]))

        D_XmuXmu      = 2*RXX - 2*gaussian.transpose_covariance(Xmu)

        DXmuXmu_alpha = np.einsum('...i,...ij->...ij', 
                                  alpha,
                                  D_XmuXmu)
        if self.update_alpha:
            XmuXmu_Dalpha = -np.einsum('...i,...i,...i,...ij->...ij', 
                                       alpha, 
                                       1/b, 
                                       XmuXmu, 
                                       D_XmuXmu)
            D_b0_alpha     = -np.einsum('...i,...i,...i,...ij->...ij', 
                                        b0,
                                        alpha,
                                        1/b,
                                        D_XmuXmu)
            D_logb         = np.einsum('...i,...ij->...ij', 
                                       1/b,
                                       D_XmuXmu)
            D_logalpha     = -D_logb
            D_a0_logalpha  = a0 * D_logalpha
        else:
            XmuXmu_Dalpha = 0
            D_logalpha = 0
            
        D_XmuXmu_alpha = DXmuXmu_alpha + XmuXmu_Dalpha
        D_logR         = inv_R.T
        
        
        # Compute dH(X)
        dlogH_X = utils.random.gaussian_entropy(-2*sum_plates(D_logR,
                                                              plates_X),
                                                0)

        # Compute d<log p(X|alpha)>
        dlogp_X = utils.random.gaussian_logpdf(sum_plates(D_XmuXmu_alpha,
                                                          plates_alpha[:-1]),
                                               0,
                                               0,
                                               sum_plates(D_logalpha,
                                                          plates_X),
                                               0)

        if self.update_alpha:
            # Compute dH(alpha)
            dlogH_alpha = utils.random.gamma_entropy(0,
                                                     sum_plates(D_logb,
                                                                plates_alpha[:-1]),
                                                     0,
                                                     0,
                                                     0)

            # Compute d<log p(alpha)>
            dlogp_alpha = utils.random.gamma_logpdf(sum_plates(D_b0_alpha,
                                                               plates_alpha[:-1]),
                                                    sum_plates(D_logalpha,
                                                               plates_alpha[:-1]),
                                                    sum_plates(D_a0_logalpha,
                                                               plates_alpha[:-1]),
                                                    0,
                                                    0)
        else:
            dlogH_alpha = 0
            dlogp_alpha = 0

        dR_bound = (0*dlogp_X
        + dlogp_X
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
            mu_v_R = np.einsum('ik,k,kj', self.mu, v, R)
            return (dot(QX, R_v_R, self.X.T)
                    + sumQ * tr_R_v_R_Cov
                    - dot(mu_v_R, self.X.T))
            

        # Compute dH(X)
        dQ_logHX = utils.random.gaussian_entropy(-2*D/sumQ,
                                                 0)

        # Compute dH(alpha)
        d_log_b = d_helper(1/b_alpha)
        dQ_logHalpha = utils.random.gamma_entropy(0,
                                                  d_log_b,
                                                  0,
                                                  0,
                                                  0)

        # Compute d<log p(X|alpha)>
        dXX_alpha = 2*d_helper(alpha_R)
        XX_dalpha = -d_helper(np.diag(XmuXmu_R)*alpha_R/b_alpha)
        d_log_alpha = -d_log_b
        dQ_logpX = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                                0,
                                                0,
                                                N*d_log_alpha,
                                                0)


        # Compute d<log p(alpha)>
        d_alpha = -d_helper(alpha_R/b_alpha)
        dQ_logpalpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                 d_log_alpha,
                                                 self.a0*d_log_alpha,
                                                 0,
                                                 0)

        dQ_bound = (0*dQ_logpX
        + dQ_logpX
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

    :math:`A` may vary in time.
    
    Shape of A: (N,D,D)
    Shape of AA: (N,D,D,D)

    No plates for X.
    """

    def __init__(self, X, A, A_rotator):
        self.X_node = X
        self.A_node = A
        self.A_rotator = A_rotator
        self.N = X.dims[0][0]

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
        
        # Get moments of X
        (X, XnXn, XpXn) = self.X_node.get_moments()
        XpXp = XnXn[:-1,:,:]

        # Get moments of A (and make sure they include time axis)
        (A, AA) = self.X_node.parents[2].get_moments()
        A = utils.utils.atleast_nd(A, 3)
        AA = utils.utils.atleast_nd(AA, 4)
        CovA = AA - A[...,:,np.newaxis]*A[...,np.newaxis,:]

        #
        # Expectations with respect to X
        #
        
        self.X0 = X[0,:]
        self.X0X0 = XnXn[0,:,:]
        #self.XpXp = np.sum(XpXp, axis=0)
        self.XnXn = np.sum(XnXn[1:,:,:], axis=0)
        #self.XpXn = np.sum(XpXn, axis=0)

        #
        # Expectations with respect to A and X
        #

        # Compute: \sum_n <A_n> <x_{n-1} x_n^T>
        self.A_XpXn = np.sum(dot(A, XpXn),
                             axis=0)

        # Compute: \sum_n <A_n> <x_{n-1} x_{n-1}^T> <A_n>^T
        self.A_XpXp_A = np.sum(dot(A, XpXp, utils.utils.T(A)),
                               axis=0)

        # Compute: \sum_n tr(CovA_n <x_{n-1} x_{n-1}^T>)
        self.CovA_XpXp = np.einsum('ndij,nij->d', CovA, XpXp)
        
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
        R_XnXn = dot(R, self.XnXn)
        RA_XpXp_A = dot(R, self.A_XpXp_A)
        sumr = np.sum(R, axis=0)
        R_CovA_XpXp = sumr * self.CovA_XpXp

        ## if not gradient:
        ##     print("DEBUG TOO", dot(R_XnXn,R.T))

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
        bound = (0
                 + logp_X 
                 + logH_X
                 )

        # TODO/FIXME: There might be a very small error in the gradient?
        
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

            d_bound = (0*dlogp_X
                       + dlogp_X 
                       + dlogH_X
                       )

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

    
class RotateDriftingMarkovChain():
    """
    Assume the following model.

    Constant, unit isotropic innovation noise.

    :math:`A_n = \sum_k B_k s_{kn}`
    
    Shape of B: (D,D*K)
    Shape of BB: (D,D*K,D*K)
    Shape of S: (N,K)
    Shape of SS: (N,K,K)

    No plates for X.
    """

    def __init__(self, X, B, S, B_rotator):
        self.X_node = X
        self.B_node = B
        self.S_node = S
        self.B_rotator = B_rotator
        self.N = X.dims[0][0]
        self.D = X.dims[0][-1]
        self.K = S.dims[0][-1]

    def nodes(self):
        # B node is in the B rotator.
        # S is not rotated.
        return [self.X_node]

    def rotate(self, R, inv=None, logdet=None):
        self.X_node.rotate(R, inv=inv, logdet=logdet)
        self.B_rotator.rotate(inv.T, inv=R.T, logdet=-logdet, Q=R)

    def setup(self):
        """
        This method should be called just before optimization.
        """
        
        # Get moments of X
        (X, XnXn, XpXn) = self.X_node.get_moments()
        XpXp = XnXn[:-1,:,:]

        # Get moments of B and S
        (B, BB) = self.B_node.get_moments()
        u_S = self.S_node.get_moments()
        S = u_S[0]
        SS = u_S[1]

        CovB = BB - B[...,:,np.newaxis]*B[...,np.newaxis,:]
        B = np.reshape(B, (self.D,self.D,self.K))
        CovB = np.reshape(CovB, (self.D,self.D,self.K,self.D,self.K))

        #
        # Expectations with respect to X
        #
        
        self.X0 = X[0,:]
        self.X0X0 = XnXn[0,:,:]
        #self.XpXp = np.sum(XpXp, axis=0)
        self.XnXn = np.sum(XnXn[1:,:,:], axis=0)
        #self.XpXn = np.sum(XpXn, axis=0)

        #
        # Expectations with respect to A and X
        #

        # Compute: \sum_n <A_n> <x_{n-1} x_n^T>
        S_XpXn = np.einsum('nk,nij->kij', S, XpXn)
        self.A_XpXn = np.einsum('dik,kij->dj', B, S_XpXn)

        # Compute: \sum_n <A_n> <x_{n-1} x_{n-1}^T> <A_n>^T
        SS_XpXp = np.einsum('nkl,nij->ikjl', SS, XpXp)
        B_SS_XpXp = np.einsum('dik,ikjl->djl', B, SS_XpXp)
        self.A_XpXp_A = np.einsum('djl,ejl->de', B_SS_XpXp, B)
        #np.sum(dot(A, XpXp, utils.utils.T(A)),
        #                       axis=0)

        # Compute: \sum_n tr(CovA_n <x_{n-1} x_{n-1}^T>)
        self.CovA_XpXp = np.einsum('dikjl,ikjl->d', CovB, SS_XpXp)
        #self.CovA_XpXp = np.einsum('ndij,nij->d', CovA, XpXp)
        
        # Get moments of the fixed parameter nodes
        mu = self.X_node.parents[0].get_moments()[0]
        self.Lambda = self.X_node.parents[1].get_moments()[0]
        self.Lambda_mu_X0 = np.outer(np.dot(self.Lambda,mu), self.X0)

        self.B_rotator.setup(rotate_plates=True)

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
        R_XnXn = dot(R, self.XnXn)
        RA_XpXp_A = dot(R, self.A_XpXp_A)
        sumr = np.sum(R, axis=0)
        R_CovA_XpXp = sumr * self.CovA_XpXp

        ## if not gradient:
        ##     print("DEBUG TOO", dot(R_XnXn,R.T))

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
        bound = (0
                 + logp_X 
                 + logH_X
                 )

        #if not gradient:
        #    print("Debug in transformations", bound)
        
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

            d_bound = (0*dlogp_X
                       + dlogp_X 
                       + dlogH_X
                       )

            return (bound, d_bound)

        else:
            return bound

    def bound(self, R, logdet=None, inv=None):
        (bound_X, d_bound_X) = self._compute_bound(R,
                                                   logdet=logdet,
                                                   inv=inv,
                                                   gradient=True)
        
        # Compute cost and gradient from A
        (bound_B, dR_bound_B, dQ_bound_B) = self.B_rotator.bound(inv.T, 
                                                                 inv=R.T,
                                                                 logdet=-logdet,
                                                                 Q=R)
        # TODO/FIXME: Also apply the gradient of invR.T to the result
        dR_bound_B = -dot(inv.T, dR_bound_B.T, inv.T)

        # Compute the bound
        bound = bound_X + bound_B
        d_bound = d_bound_X + dR_bound_B + dQ_bound_B

        return (bound, d_bound)

    def get_bound_terms(self, R, logdet=None, inv=None):
        bound_dict = self.B_rotator.get_bound_terms(inv.T, 
                                                    inv=R.T,
                                                    logdet=-logdet,
                                                    Q=R)
        
        bound_X = self._compute_bound(R,
                                      logdet=logdet,
                                      inv=inv,
                                      gradient=False)
        
        bound_dict.update({self: bound_X})
        
        return bound_dict

    ## def get_gradient_terms(self, R, logdet=None, inv=None):
    ##     grad_dict = self.B_rotator.get_gradient_terms(inv.T, 
    ##                                                   inv=R.T,
    ##                                                   logdet=-logdet,
    ##                                                   Q=R)
        
    ##     (_, grad_X) = self._compute_bound(R,
    ##                                       logdet=logdet,
    ##                                       inv=inv,
    ##                                       gradient=True)
        
    ##     grad_dict.update({self: grad_X})
        
    ##     return grad_dict



































###############################
#
# DEPRECATED STUFF BELOW
#
###############################

    


class RotateGaussianARD():
    # THIS CLASS SHOULD BECOME DEPRECATED BY ROTATEGAUSSIANARRAYARD!!!

    # TODO: Let the mean be non-zero!
    
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
        
        # Get the mean parameter. It is not rotated.
        node_mu = self.node_X.parents[0]
        (mu, self.mumu) = node_mu.get_moments()

        mask = self.node_X.mask[...,np.newaxis,np.newaxis]

        # Number of plates
        self.N = self.node_X.plates[0] #np.sum(mask)

        self.X = self.node_X.get_moments()[0] * self.node_X.mask[...,np.newaxis]
        if not rotate_plates:
            # Compute the sum <XX> over plates
            self.XX = utils.utils.sum_multiply(self.node_X.get_moments()[1],
                                               mask,
                                               axis=(-1,-2),
                                               sumaxis=False,
                                               keepdims=False)
            self.Xmu = utils.utils.sum_multiply(self.X[...,:,np.newaxis],
                                                mu[...,np.newaxis,:],
                                                axis=(-1,-2),
                                                sumaxis=False,
                                                keepdims=False)
        else:
            XX = self.node_X.get_moments()[1] * mask
            self.CovX = XX - utils.linalg.outer(self.X, self.X)
            self.mu = np.atleast_2d(mu)
            #self.Xmu = self.X[...,:,np.newaxis] * mu[...,np.newaxis,:]
            
        # Parent's moments
        self.a = np.ravel(self.node_alpha.phi[1])
        # TODO/FIXME: Handle vector valued parents a0 and b0
        self.a0 = self.node_alpha.parents[0].get_moments()[0]
        self.b0 = self.node_alpha.parents[1].get_moments()[0]

        # Compute <X*mu'> (summed over plates)
        #self.mu = np.atleast_2d(self.mu)
        if len(node_mu.plates) == 0 or node_mu.plates[0] == 1:
            self.mumu = self.N * np.diag(utils.utils.sum_to_dim(self.mumu, 2))
            #print("no plates", node_mu.plates)
        else:
            self.mumu = np.diag(utils.utils.sum_to_dim(self.mumu, 2))
            #print("yes plates", node_mu.plates, np.shape(self.mumu))

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
            X = QX
            X_mu = utils.utils.sum_multiply(X[...,:,np.newaxis],
                                            self.mu[...,np.newaxis,:],
                                            axis=(-1,-2),
                                            sumaxis=False,
                                            keepdims=False)
        else:
            X = self.X
            XX = self.XX
            logdet_Q = 0
            X_mu = self.Xmu

        # TODO/FIXME: X can be summed to the plates of mu!?
        RX_mu = dot(R, X_mu)

        XmuXmu_R = (dot(R, XX, R.T) - RX_mu - RX_mu.T + self.mumu)

        # Compute q(alpha)
        a_alpha = self.a
        b_alpha = self.b0 + 0.5*np.diag(XmuXmu_R)
        #b_alpha = self.b0 + 0.5*np.diag(RXXR)
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
        logp_X = utils.random.gaussian_logpdf(np.einsum('ii,i', XmuXmu_R, alpha_R),
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
        bound = (0
        + logp_X
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
        dXmuXmu_R = 2*np.dot(R, XX) - 2*X_mu.T
        d_log_b = np.einsum('i,ij->ij', 1/b_alpha, dXmuXmu_R)
        dlogH_alpha = utils.random.gamma_entropy(0,
                                                 d_log_b,
                                                 0,
                                                 0,
                                                 0)

        # Compute d<log p(X|alpha)>
        # TODO/FIXME: Fix these gradients!
        d_log_alpha = -d_log_b
        dXmuXmu_alpha = np.einsum('i,ij->ij', alpha_R, dXmuXmu_R)
        XmuXmu_dalpha = -np.einsum('i,i,ii,ij->ij', alpha_R, 1/b_alpha, XmuXmu_R, dXmuXmu_R)
        dlogp_X = utils.random.gaussian_logpdf(dXmuXmu_alpha + XmuXmu_dalpha,
                                               0,
                                               0,
                                               N*d_log_alpha,
                                               0)

        # Compute d<log p(alpha)>
        d_alpha = -np.einsum('i,i,ij->ij', alpha_R, 1/b_alpha, dXmuXmu_R)
        dlogp_alpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                d_log_alpha,
                                                self.a0*d_log_alpha,
                                                0,
                                                0)

        dR_bound = (0*dlogp_X
        + dlogp_X
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
            mu_v_R = np.einsum('ik,k,kj', self.mu, v, R)
            return (dot(QX, R_v_R, self.X.T)
                    + sumQ * tr_R_v_R_Cov
                    - dot(mu_v_R, self.X.T))
            

        # Compute dH(X)
        dQ_logHX = utils.random.gaussian_entropy(-2*D/sumQ,
                                                 0)

        # Compute dH(alpha)
        d_log_b = d_helper(1/b_alpha)
        dQ_logHalpha = utils.random.gamma_entropy(0,
                                                  d_log_b,
                                                  0,
                                                  0,
                                                  0)

        # Compute d<log p(X|alpha)>
        dXX_alpha = 2*d_helper(alpha_R)
        XX_dalpha = -d_helper(np.diag(XmuXmu_R)*alpha_R/b_alpha)
        d_log_alpha = -d_log_b
        dQ_logpX = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                                0,
                                                0,
                                                N*d_log_alpha,
                                                0)


        # Compute d<log p(alpha)>
        d_alpha = -d_helper(alpha_R/b_alpha)
        dQ_logpalpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                 d_log_alpha,
                                                 self.a0*d_log_alpha,
                                                 0,
                                                 0)

        dQ_bound = (0*dQ_logpX
        + dQ_logpX
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
        



class RotateGaussianMatrixARD():

    # THIS WILL BE DEPRECATED AND REPLACED BY RotateGaussianArrayARD !!!!
    
    """
    Assume the following model:

    :math:`p(X|\alpha) = \prod_{dk} N(x_{dk} | 0, 1 / \alpha_{dk})`

    :math:`p(\alpha) = \prod_k G(\alpha_{dk} | a, b)`

    Reshape vector x to matrix X and multiply either on the left or on the
    right: R*X or X*R'.
    """
    def __init__(self, X, alpha, D1, D2, axis='rows'):
        self.X_node = X
        self.alpha_node = alpha

        self.D1 = D1
        self.D2 = D2
        ## D = X.dims[0][0]
        ## self.D2 = alpha.plates[0]
        ## self.D1 = D / self.D2

        # Whether to multiply on the left or on the right
        if axis == 'rows':
            self.axis = 'rows'
        elif axis == 'cols':
            self.axis = 'cols'
        else:
            raise Exception("Unknown axis")

    def nodes(self):
        return [self.X_node, self.alpha_node]

    def rotate(self, R, inv=None, logdet=None, Q=None):
        if self.axis == 'rows':
            # Multiply on the left by R
            R1 = R
            inv1 = inv
            logdet1 = logdet
            R2 = np.identity(self.D2)
            inv2 = R2
            logdet2 = 0
        else:
            # Multiply on the right by R'
            R1 = np.identity(self.D1)
            inv1 = R1
            logdet1 = 0
            R2 = R
            inv2 = inv
            logdet2 = logdet
            
        self.X_node.rotate_matrix(R1, R2, 
                                  inv1=inv1,
                                  logdet1=logdet1,
                                  inv2=inv2, 
                                  logdet2=logdet2,
                                  Q=Q)

        self.alpha_node.update()
        #print("debug in rotazione", self.alpha_node.get_moments()[0])

    def setup(self, rotate_plates=False):
        """
        This method should be called just before optimization.
        """
        
        
        #mask = self.X_node.mask[...,np.newaxis,np.newaxis]

        # Number of plates
        self.N = self.X_node.plates[0] #np.sum(mask)

        if not rotate_plates:
            # Compute the sum <XX> over plates
            self.XX = utils.utils.sum_multiply(self.X_node.get_moments()[1],
                                               self.X_node.mask[...,np.newaxis,np.newaxis],
                                               axis=(-1,-2),
                                               sumaxis=False,
                                               keepdims=False)
        else:
            self.X = (self.X_node.get_moments()[0] 
                      * self.X_node.mask[...,np.newaxis])
            XX = (self.X_node.get_moments()[1] 
                  * self.X_node.mask[...,np.newaxis,np.newaxis])
            self.CovX = XX - utils.linalg.outer(self.X, self.X)
            
        # Parent's moments
        self.a = np.ravel(self.alpha_node.phi[1])
        #self.b = -np.ravel(self.alpha_node.phi[0])
        # TODO/FIXME: Handle vector valued parents a0 and b0
        self.a0 = self.alpha_node.parents[0].get_moments()[0]
        self.b0 = self.alpha_node.parents[1].get_moments()[0]

    def _compute_bound(self, R, logdet=None, inv=None, Q=None, gradient=False):
        """
        Rotate q(X) and q(alpha).
        """

        # TODO/FIXME: X and alpha should NOT contain observed values!! Check that.

        #
        # Transform the distributions and moments
        #

        D1 = self.D1
        D2 = self.D2
        N = self.N
        D = D1 * D2

        # Compute rotated second moment
        if Q is not None:
            X = np.reshape(self.X, (N,D1,D2))
            CovX = np.reshape(self.CovX, (N,D1,D2,D1,D2))
            # Rotate plates
            sumQ = np.sum(Q, axis=0)
            QX = np.einsum('ik,kab->iab', Q, X)
            logdet_Q = np.sum(np.log(np.abs(sumQ)))

            if self.axis == 'cols':
                # Sum "rows"
                #X = np.einsum('nkj->nj', X)
                # Rotate "columns"
                X_R = np.einsum('jk,nik->nij', R, X)
                r_CovX_r = np.einsum('bk,bl,nakal->nab', R, R, CovX)
                XX = (np.einsum('kai,kaj->aij', QX, QX)
                      + np.einsum('d,daiaj->aij', sumQ**2, CovX))
            else:
                # Rotate "rows"
                #print("OR THE BUG IS HERE...")
                X_R = np.einsum('ik,nkj->nji', R, X)
                r_CovX_r = np.einsum('ak,al,nkblb->nba', R, R, CovX)
                XX = (np.einsum('kib,kjb->bij', QX, QX)
                      + np.einsum('d,dibjb->bij', sumQ**2, CovX))

            Q_X_R = np.einsum('nk,kij->nij', Q, X_R)

        else:
            # Reshape into matrix form
            sh = (D1,D2,D1,D2)
            XX = np.reshape(self.XX, sh)
            if self.axis == 'cols':
                XX = np.einsum('aiaj->aij', XX)
            else:
                XX = np.einsum('ibjb->bij', XX)
                
            logdet_Q = 0

        # Reshape vector to matrix
        if self.axis == 'cols':

            # Apply rotation on the right
            R_XX = np.einsum('ik,akj->aij', R, XX)
            r_XX_r = np.einsum('ik,il,akl->ai', R, R, XX)

            if Q is not None:
                # Debug stuff:
                #print("debug for Q", np.shape(XX_R))
                r_XQQX_r = np.einsum('ab->ab', r_XX_r) #r_XX_r #np.einsum('abab->b', XX_R)
                
            # Compute q(alpha)
            a_alpha = self.a
            b_alpha = self.b0 + 0.5 * r_XX_r #np.einsum('ab->b', r_XX_r)
            alpha_R = a_alpha / b_alpha
            logalpha_R = -np.log(b_alpha) # + const

            logdet_R = logdet
            inv_R = inv

            # Bug in here:
            XX_dalpha = -np.einsum('ab,ab,abj->bj', alpha_R/b_alpha, r_XX_r, R_XX)

            #XX = np.einsum('aiaj->ij', XX)
            #XX_R = np.einsum('aiaj->ij', XX_R)

            #print("THERE")
            alpha_R_XX = np.einsum('ai,aij->ij', alpha_R, R_XX) # BUG IN HERE? In gradient
            dalpha_R_XX = np.einsum('ai,aij->ij', alpha_R/b_alpha, R_XX)
            invb_R_XX = np.einsum('ai,aij->ij', 1/b_alpha, R_XX)
            #dalpha_RXXR = np.einsum('i,ii->i', alpha_R/b_alpha, XX_R)

            ND = self.N * D1
            
        else:
            # Apply rotation on the left

            R_XX = np.einsum('ik,bkj->bij', R, XX)
            r_XX_r = np.einsum('ik,il,bkl->bi', R, R, XX)

            if Q is not None:
                # Debug stuff:
                #print("debug for Q", np.shape(XX_R))
                r_XQQX_r = np.einsum('bi->bi', r_XX_r) #np.einsum('abab->b', XX_R)
                
            # Compute q(alpha)
            a_alpha = self.a
            b_alpha = self.b0 + 0.5 * r_XX_r #np.einsum('bi->b', r_XX_r)
            #b_alpha = self.b0 + 0.5 * np.einsum('abab->b', XX_R)
            alpha_R = a_alpha / b_alpha
            logalpha_R = -np.log(b_alpha) # + const

            logdet_R = logdet
            inv_R = inv

            #print("HERE IS THE BUG SOMEWHERE")
            # Compute: <alpha>_* R <
            #print(np.shape(alpha_R), np.shape(R), np.shape(XX), np.shape(R_XX), QX.shape, D1, D2)
            alpha_R_XX = np.einsum('bi,bij->ij', alpha_R, R_XX)
            dalpha_R_XX = np.einsum('bi,bij->ij', alpha_R/b_alpha, R_XX)
            invb_R_XX = np.einsum('bi,bij->ij', 1/b_alpha, R_XX)
            #dalpha_RXXR = np.einsum('b,ibib->i', alpha_R/b_alpha, XX_R)
            XX_dalpha = -np.einsum('ba,ba,baj->aj', alpha_R/b_alpha, r_XX_r, R_XX)

            #XX = np.einsum('ibjb->ij', XX)
            #XX_R = np.nan * np.einsum('ibjb->ij', XX_R)

            ND = self.N * D2


        #
        # Compute the cost
        #
        
        # Compute entropy H(X)
        logH_X = utils.random.gaussian_entropy(-2*ND*logdet_R - 2*D*logdet_Q, 
                                               0)

        # Compute entropy H(alpha)
        logH_alpha = utils.random.gamma_entropy(0,
                                                np.sum(np.log(b_alpha)),
                                                0,
                                                0,
                                                0)

        # Compute <log p(X|alpha)>
        #logp_X = utils.random.gaussian_logpdf(np.einsum('ii,i', XX_R, alpha_R),
        logp_X = utils.random.gaussian_logpdf(tracedot(alpha_R_XX, R.T),
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
        bound = (0
                 + logp_X
                 + logp_alpha
                 + logH_X
                 + logH_alpha
                 )

        if not gradient:
            return bound

        #
        # Compute the gradient with respect to R
        #

        # Compute dH(X)
        dlogH_X = utils.random.gaussian_entropy(-2*ND*inv_R.T,
                                                0)

        # Compute dH(alpha)
        d_log_b = invb_R_XX #np.einsum('i,ik,kj->ij', 1/b_alpha, R, XX)
        dlogH_alpha = utils.random.gamma_entropy(0,
                                                 d_log_b,
                                                 0,
                                                 0,
                                                 0)

        # Compute d<log p(X|alpha)>
        d_log_alpha = -d_log_b
        dXX_alpha =  2*alpha_R_XX #np.einsum('i,ik,kj->ij', alpha_R, R, XX)
        #dalpha_xx = np.einsum('id,di', dalpha_R_XX, R.T)
        #XX_dalpha = -np.einsum('i,ik,kj', dalpha_RXXR, R, XX) # BUG IS IN THIS TERM!!!!
        #np.einsum('i,i,ii,ik,kj->ij', alpha_R, 1/b_alpha, XX_R, R, XX)

        # TODO/FIXME: This gradient term seems to have a bug.
        #
        # DEBUG: If you set these gradient terms to zero, the gradient is more
        # accurate..?!
        #dXX_alpha = 0
        #XX_dalpha = 0
        dlogp_X = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                               0,
                                               0,
                                               N*d_log_alpha,
                                               0)

        # Compute d<log p(alpha)>
        d_alpha = -dalpha_R_XX #np.einsum('i,i,ik,kj->ij', alpha_R, 1/b_alpha, R, XX)
        dlogp_alpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                d_log_alpha,
                                                self.a0*d_log_alpha,
                                                0,
                                                0)

        dR_bound = (0*dlogp_X
                    + dlogp_X
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
            return (np.einsum('iab,ab,jab->ij', Q_X_R, v, X_R)
                    + np.einsum('n,ab,nab->n', sumQ, v, r_CovX_r))
            

        # Compute dH(X)
        dQ_logHX = utils.random.gaussian_entropy(-2*D/sumQ,
                                                 0)

        # Compute dH(alpha)
        d_log_b = d_helper(1/b_alpha)
        dQ_logHalpha = utils.random.gamma_entropy(0,
                                                  d_log_b,
                                                  0,
                                                  0,
                                                  0)

        # Compute d<log p(X|alpha)>
        dXX_alpha = 2*d_helper(alpha_R)
        XX_dalpha = -d_helper(r_XQQX_r*alpha_R/b_alpha)
        d_log_alpha = -d_log_b
        dQ_logpX = utils.random.gaussian_logpdf(dXX_alpha + XX_dalpha,
                                                0,
                                                0,
                                                N*d_log_alpha,
                                                0)


        # Compute d<log p(alpha)>
        d_alpha = -d_helper(alpha_R/b_alpha)
        dQ_logpalpha = utils.random.gamma_logpdf(self.b0*d_alpha,
                                                 d_log_alpha,
                                                 self.a0*d_log_alpha,
                                                 0,
                                                 0)

        dQ_bound = (0*dQ_logpX
                    + dQ_logpX
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
        
