################################################################################
# Copyright (C) 2011-2012 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import itertools
import numpy as np
#import scipy as sp
#import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
#import scipy.special as special
#import matplotlib.pyplot as plt
#import time
#import profile
#import scipy.spatial.distance as distance
import scipy.sparse as sp

from bayespy.utils import misc as utils
from . import node as EF
from . import CovarianceFunctions as CF

class CovarianceMatrix:
    def cholesky(self):
        pass

def multiply(A, B):
    return np.multiply(A,B)



# m prior mean function
# k prior covariance function
# x data inputs
# z processed data outputs (z = inv(Cov) * (y-m(x)))
# U data covariance Cholesky factor
def gp_posterior_moment_function(m, k, x, y, k_sparse=None, pseudoinputs=None, noise=None):

    # Prior
    # FIXME: We are ignoring the covariance of mu now..
    mu = m(x)[0]
    ## if np.ndim(mu) == 1:
    ##     mu = np.asmatrix(mu).T
    ## else:
    ##     mu = np.asmatrix(mu)
    
    K_noise = None
    
    if noise != None:
        if K_noise is None:
            K_noise = noise
        else:
            K_noise += noise
            
    if k_sparse != None:
        if K_noise is None:
            K_noise = k_sparse(x,x)[0]
        else:
            K_noise += k_sparse(x,x)[0]

    if pseudoinputs != None:
        p = pseudoinputs
        #print('in pseudostuff')
        #print(K_noise)
        #print(np.shape(K_noise))
        K_pp = k(p,p)[0]
        K_xp = k(x,p)[0]
        U = utils.chol(K_noise)

        # Compute Lambda
        Lambda = K_pp + np.dot(K_xp.T, utils.chol_solve(U, K_xp))
        U_lambda = utils.chol(Lambda)

        # Compute statistics for posterior predictions
        #print(np.shape(U_lambda))
        #print(np.shape(y))
        z = utils.chol_solve(U_lambda,
                       np.dot(K_xp.T,
                              utils.chol_solve(U,
                                         y - mu)))
        U = utils.chol(K_pp)

        # Now we can forget the location of the observations and
        # consider only the pseudoinputs when predicting.
        x = p

        
    else:
        K = K_noise
        if K is None:
            K = k(x,x)[0]
        else:
            try:
                K += k(x,x)[0]
            except:
                K = K + k(x,x)[0]

        # Compute posterior GP
        N = len(y)
        U = None
        z = None
        if N > 0:
            U = utils.chol(K)
            z = utils.chol_solve(U, y-mu)

    def get_moments(h, covariance=1, mean=True):

        K_xh = k(x, h)[0]
        if k_sparse != None:
            try:
                # This may not work, for instance, if either one is a
                # sparse matrix.
                K_xh += k_sparse(x, h)[0]
            except:
                K_xh = K_xh + k_sparse(x, h)[0]
        
        # NumPy has problems when mixing matrices and arrays.
        # Matrices may appear, for instance, when you sum an array and
        # a sparse matrix.  Make sure the result is either an array or
        # a sparse matrix (not dense matrix!), because matrix objects
        # cause lots of problems:
        #
        # array.dot(array) = array
        # matrix.dot(array) = matrix
        # sparse.dot(array) = array
        if not sp.issparse(K_xh):
            K_xh = np.asarray(K_xh)

        # Function for computing posterior moments
        if mean:
            # Mean vector
            # FIXME: Ignoring the covariance of prior mu
            m_h = m(h)[0]
            
            if z != None:
                m_h += K_xh.T.dot(z)
                
        else:
            m_h = None

        # Compute (co)variance matrix/vector
        if covariance:
            if covariance == 1:
                ## Compute variance vector
                
                k_h = k(h)[0]
                if k_sparse != None:
                    k_h += k_sparse(h)[0]
                if U != None:
                    if isinstance(K_xh, np.ndarray):
                        k_h -= np.einsum('i...,i...',
                                         K_xh,
                                         utils.chol_solve(U, K_xh))
                    else:
                        # TODO: This isn't very efficient way, but
                        # einsum doesn't work for sparse matrices..
                        # This may consume A LOT of memory for sparse
                        # matrices.
                        k_h -= np.asarray(K_xh.multiply(utils.chol_solve(U, K_xh))).sum(axis=0)
                if pseudoinputs != None:
                    if isinstance(K_xh, np.ndarray):
                        k_h += np.einsum('i...,i...',
                                         K_xh,
                                         utils.chol_solve(U_lambda, K_xh))
                    else:
                        # TODO: This isn't very efficient way, but
                        # einsum doesn't work for sparse matrices..
                        # This may consume A LOT of memory for sparse
                        # matrices.
                        k_h += np.asarray(K_xh.multiply(utils.chol_solve(U_lambda, K_xh))).sum(axis=0)
                # Ensure non-negative variances        
                k_h[k_h<0] = 0
                
                return (m_h, k_h)
                    
            elif covariance == 2:
                ## Compute full covariance matrix
                
                K_hh = k(h,h)[0]
                if k_sparse != None:
                    K_hh += k_sparse(h)[0]
                if U != None:
                    K_hh -= K_xh.T.dot(utils.chol_solve(U,K_xh))
                    #K_hh -= np.dot(K_xh.T, utils.chol_solve(U,K_xh))
                if pseudoinputs != None:
                    K_hh += K_xh.T.dot(utils.chol_solve(U_lambda, K_xh))
                    #K_hh += np.dot(K_xh.T, utils.chol_solve(U_lambda, K_xh))
                return (m_h, K_hh)
        else:
            return (m_h, None)


    return get_moments


# Constant function using GP mean protocol
class Constant(EF.Node):
    def __init__(self, f, **kwargs):

        self.f = f
        EF.Node.__init__(self, dims=[(np.inf,)], **kwargs)

    def message_to_child(self, gradient=False):

        # Wrapper
        def func(x, gradient=False):
            if gradient:
                return ([self.f(x), None], [])
            else:
                return [self.f(x), None]

        return func

#class MultiDimensional(EF.NodeVariable):
#    """ A multi-dimensional Gaussian process f(x). """

## class ToGaussian(EF.NodeVariable):

##     """ Deterministic node which transform a Gaussian process into
##     finite-dimensional Gaussian variable. """

##     def __init__(self, f, x, **kwargs):
##         EF.NodeVariable.__init__(self,
##                                  f,
##                                  x,
##                                  plates=
##                                  dims=
    
# Deterministic node for creating a set of GPs which can be used as a
# mean function to a general GP node.
class Multiple(EF.Node):

    def __init__(self, GPs, **kwargs):

        # Ignore plates
        EF.NodeVariable.__init__(self,
                                 *GPs,
                                 plates=(),
                                 dims=[(np.inf,), (np.inf,np.inf)],
                                 **kwargs)
    
    def message_to_parent(self, index):
        raise Exception("not implemented yet")

    def message_to_child(self, gradient=False):
        u = [parent.message_to_child() for parent in self.parents]
        
        def get_moments(xh, **kwargs):
            mh_all = []
            khh_all = []
            for i in range(len(self.parents)):
                xi = np.array(xh[i])
                #print(xi)
                #print(np.shape(xi))
                #print(xi)
                # FIXME: We are ignoring the covariance of mu now..
                if gradient:
                    ((mh, khh), dm) = u[i](xi, **kwargs)
                else:
                    (mh, khh) = u[i](xi, **kwargs)
                #mh = u[i](xi, **kwargs)[0]
                #print(mh)
                #print(mh_all)
                ## print(mh)
                ## print(khh)
                ## print(np.shape(mh))
                mh_all = np.concatenate([mh_all, mh])
                #print(np.shape(mh_all))
                if khh != None:
                    print(khh)
                    raise Exception('Not implemented yet for covariances')
                    #khh_all = np.concatenate([khh_all, khh])

            # FIXME: Compute gradients!
            if gradient:
                return ([mh_all, khh_all], [])
            else:
                return [mh_all, khh_all]
                
            #return [mh_all, khh_all]
        
        return get_moments


# Gaussian process distribution
class GaussianProcess(EF.Node):

    def __init__(self, m, k, k_sparse=None, pseudoinputs=None, **kwargs):

        self.x = np.array([])
        self.f = np.array([])
        ## self.x_obs = np.zeros((0,1))
        ## self.f_obs = np.zeros((0,))

        if pseudoinputs != None:
            pseudoinputs = EF.NodeConstant([pseudoinputs],
                                           dims=[np.shape(pseudoinputs)])

        # By default, posterior == prior
        self.m = None #m
        self.k = None #k

        if isinstance(k, list) and isinstance(m, list):
            if len(k) != len(m):
                raise Exception('The number of mean and covariance functions must be equal.')
            k = CF.Multiple(k)
            m = Multiple(m)
        elif isinstance(k, list):
            D = len(k)
            k = CF.Multiple(k)
            m = Multiple(D*[m])
        elif isinstance(m, list):
            D = len(m)
            k = CF.Multiple(D*[k])
            m = Multiple(m)

        # Ignore plates
        EF.NodeVariable.__init__(self,
                                 m,
                                 k,
                                 k_sparse,
                                 pseudoinputs,
                                 plates=(),
                                 dims=[(np.inf,), (np.inf,np.inf)],
                                 **kwargs)
            

    def __call__(self, x, covariance=None):
        if not covariance:
            return self.u(x, covariance=False)[0]
        elif covariance.lower() == 'vector':
            return self.u(x, covariance=1)
        elif covariance.lower() == 'matrix':
            return self.u(x, covariance=2)
        else:
            raise Exception("Unknown covariance type requested")
            

    def message_to_parent(self, index):
        if index == 0:
            k = self.parents[1].message_to_child()[0]
            K = k(self.x, self.x)
            return [self.x,
                    self.mu,
                    K]
        if index == 1:
            raise Exception("not implemented yet")

    def message_to_child(self):
        if self.observed:
            raise Exception("Observable GP should not have children.")
        return self.u

    def get_parameters(self):
        return self.u

    def observe(self, x, f):

        self.observed = True
        
        self.x = x
        self.f = f
        ## if np.ndim(f) == 1:
        ##     self.f = np.asmatrix(f).T
        ## else:
        ##     self.f = np.asmatrix(f)

    # You might want:
    # - mean for x
    # - covariance (and mean) for x
    # - variance (and mean) for x
    # - i.e., mean and/or (co)variance for x
    # - covariance for x1 and x2

            
        
    def lower_bound_contribution(self, gradient=False):

        # Get moment functions from parents
        m = self.parents[0].message_to_child(gradient=gradient)
        k = self.parents[1].message_to_child(gradient=gradient)
        if self.parents[2]:
            k_sparse = self.parents[2].message_to_child(gradient=gradient)
        else:
            k_sparse = None
        if self.parents[3]:
            pseudoinputs = self.parents[3].message_to_child(gradient=gradient)
            #pseudoinputs = self.parents[3].message_to_child(gradient=gradient)[0]
        else:
            pseudoinputs = None
        ## m = self.parents[0].message_to_child(gradient=gradient)[0]
        ## k = self.parents[1].message_to_child(gradient=gradient)[0]

        # Compute the parameters (covariance matrices etc) using
        # parents' moment functions
        DKs_xx = []
        DKd_xx = []
        DKd_xp = []
        DKd_pp = []
        Dxp = []
        Dmu = []
        if gradient:
            # FIXME: We are ignoring the covariance of mu now..
            ((mu, _), Dmu) = m(self.x, gradient=True)
            ## if k_sparse:
            ##     ((Ks_xx,), DKs_xx) = k_sparse(self.x, self.x, gradient=True)
            if pseudoinputs:
                ((Ks_xx,), DKs_xx) = k_sparse(self.x, self.x, gradient=True)
                ((xp,), Dxp) = pseudoinputs
                ((Kd_pp,), DKd_pp) = k(xp,xp, gradient=True)
                ((Kd_xp,), DKd_xp) = k(self.x, xp, gradient=True)
            else:
                ((K_xx,), DKd_xx) = k(self.x, self.x, gradient=True)
                if k_sparse:
                    ((Ks_xx,), DKs_xx) = k_sparse(self.x, self.x, gradient=True)
                    try:
                        K_xx += Ks_xx
                    except:
                        K_xx = K_xx + Ks_xx
                
        else:
            # FIXME: We are ignoring the covariance of mu now..
            (mu, _) = m(self.x)
            ## if k_sparse:
            ##     (Ks_xx,) = k_sparse(self.x, self.x)
            if pseudoinputs:
                (Ks_xx,) = k_sparse(self.x, self.x)
                (xp,) = pseudoinputs
                (Kd_pp,) = k(xp, xp)
                (Kd_xp,) = k(self.x, xp)
            else:
                (K_xx,) = k(self.x, self.x)
                if k_sparse:
                    (Ks_xx,) = k_sparse(self.x, self.x)
                    try:
                        K_xx += Ks_xx
                    except:
                        K_xx = K_xx + Ks_xx


        mu = mu[0]
        #K = K[0]

        # Log pdf
        if self.observed:
            ## Log pdf for directly observed GP
            
            f0 = self.f - mu
            
            #print('hereiam')
            #print(K)
            if pseudoinputs:

                ## Pseudo-input approximation

                # Decompose the full-rank sparse/noise covariance matrix
                try:
                    Us_xx = utils.cholesky(Ks_xx)
                except linalg.LinAlgError:
                    print('Noise/sparse covariance not positive definite')
                    return -np.inf

                # Use Woodbury-Sherman-Morrison formula with the
                # following notation:
                #
                # y2 = f0' * inv(Kd_xp*inv(Kd_pp)*Kd_xp' + Ks_xx) * f0
                #
                # z = Ks_xx \ f0
                # Lambda = Kd_pp + Kd_xp'*inv(Ks_xx)*Kd_xp
                # nu = inv(Lambda) * (Kd_xp' * (Ks_xx \ f0))
                # rho = Kd_xp * inv(Lambda) * (Kd_xp' * (Ks_xx \ f0))
                #
                # y2 = f0' * z - z' * rho
                
                z = Us_xx.solve(f0)
                Lambda = Kd_pp + np.dot(Kd_xp.T,
                                        Us_xx.solve(Kd_xp))
                ## z = utils.chol_solve(Us_xx, f0)
                ## Lambda = Kd_pp + np.dot(Kd_xp.T,
                ##                         utils.chol_solve(Us_xx, Kd_xp))
                try:
                    U_Lambda = utils.cholesky(Lambda)
                    #U_Lambda = utils.chol(Lambda)
                except linalg.LinAlgError:
                    print('Lambda not positive definite')
                    return -np.inf

                nu = U_Lambda.solve(np.dot(Kd_xp.T, z))
                #nu = utils.chol_solve(U_Lambda, np.dot(Kd_xp.T, z))
                rho = np.dot(Kd_xp, nu)

                y2 = np.dot(f0, z) - np.dot(z, rho)

                # Use matrix determinant lemma
                #
                # det(Kd_xp*inv(Kd_pp)*Kd_xp' + Ks_xx)
                # = det(Kd_pp + Kd_xp'*inv(Ks_xx)*Kd_xp)
                #   * det(inv(Kd_pp)) * det(Ks_xx)
                # = det(Lambda) * det(Ks_xx) / det(Kd_pp)
                try:
                    Ud_pp = utils.cholesky(Kd_pp)
                    #Ud_pp = utils.chol(Kd_pp)
                except linalg.LinAlgError:
                    print('Covariance of pseudo inputs not positive definite')
                    return -np.inf
                logdet = (U_Lambda.logdet()
                          + Us_xx.logdet()
                          - Ud_pp.logdet())
                ## logdet = (utils.logdet_chol(U_Lambda)
                ##           + utils.logdet_chol(Us_xx)
                ##           - utils.logdet_chol(Ud_pp))

                # Compute the log pdf
                
                L = gaussian_logpdf(y2,
                                    0,
                                    0,
                                    logdet,
                                    np.size(self.f))

                # Add the variational cost of the pseudo-input
                # approximation

                # Compute gradients

                
                
                for (dmu, func) in Dmu:
                    # Derivative w.r.t. mean vector
                    d = np.nan
                    # Send the derivative message
                    func(d)
                    
                for (dKs_xx, func) in DKs_xx:
                    # Compute derivative w.r.t. covariance matrix
                    d = np.nan
                    # Send the derivative message
                    func(d)
                    
                for (dKd_xp, func) in DKd_xp:
                    # Compute derivative w.r.t. covariance matrix
                    d = np.nan
                    # Send the derivative message
                    func(d)

                V = Ud_pp.solve(Kd_xp.T)
                Z = Us_xx.solve(V.T)
                ## V = utils.chol_solve(Ud_pp, Kd_xp.T)
                ## Z = utils.chol_solve(Us_xx, V.T)
                for (dKd_pp, func) in DKd_pp:
                    # Compute derivative w.r.t. covariance matrix
                    d = (0.5 * np.trace(Ud_pp.solve(dKd_pp))
                         - 0.5 * np.trace(U_Lambda.solve(dKd_pp))
                         + np.dot(nu, np.dot(dKd_pp, nu))
                         + np.trace(np.dot(dKd_pp,
                                    np.dot(V,Z))))
                    ## d = (0.5 * np.trace(utils.chol_solve(Ud_pp, dKd_pp))
                    ##      - 0.5 * np.trace(utils.chol_solve(U_Lambda, dKd_pp))
                    ##      + np.dot(nu, np.dot(dKd_pp, nu))
                    ##      + np.trace(np.dot(dKd_pp,
                    ##                 np.dot(V,Z))))
                    # Send the derivative message
                    func(d)
                    
                for (dxp, func) in Dxp:
                    # Compute derivative w.r.t. covariance matrix
                    d = np.nan
                    # Send the derivative message
                    func(d)
                
                

            else:
                
                ## Full exact (no pseudo approximations)
                
                try:
                    U = utils.cholesky(K_xx)
                    #U = utils.chol(K_xx)
                except linalg.LinAlgError:
                    print('non positive definite, return -inf')
                    return -np.inf
                z = U.solve(f0)
                #z = utils.chol_solve(U, f0)
                #print(K)
                L = utils.gaussian_logpdf(np.dot(f0, z),
                                          0,
                                          0,
                                          U.logdet(),
                                          ## utils.logdet_chol(U),
                                          np.size(self.f))

                for (dmu, func) in Dmu:
                    # Derivative w.r.t. mean vector
                    d = -np.sum(z)
                    # Send the derivative message
                    func(d)

                for (dK, func) in DKd_xx:
                    # Compute derivative w.r.t. covariance matrix
                    #
                    # TODO: trace+chol_solve should be handled better
                    # for sparse matrices.  Use sparse-inverse!
                    d = 0.5 * (dK.dot(z).dot(z)
                               - U.trace_solve_gradient(dK))
                               ## - np.trace(U.solve(dK)))
                    ## d = 0.5 * (dK.dot(z).dot(z)
                    ##            - np.trace(utils.chol_solve(U, dK)))
                    #print('derivate', d, dK)
                    ## d = 0.5 * (np.dot(z, np.dot(dK, z))
                    ##            - np.trace(utils.chol_solve(U, dK)))
                    #
                    # Send the derivative message
                    func(d)
                    
                for (dK, func) in DKs_xx:
                    # Compute derivative w.r.t. covariance matrix
                    d = 0.5 * (dK.dot(z).dot(z)
                               - U.trace_solve_gradient(dK))
                               ## - np.trace(U.solve(dK)))
                    ## d = 0.5 * (dK.dot(z).dot(z)
                    ##            - np.trace(utils.chol_solve(U, dK)))
                    ## d = 0.5 * (np.dot(z, np.dot(dK, z))
                    ##            - np.trace(utils.chol_solve(U, dK)))
                    # Send the derivative message
                    func(d)

        else:
            ## Log pdf for latent GP
            raise Exception('Not implemented yet')

        return L

        ## Let f1 be observed and f2 latent function values.

        # Compute <log p(f1,f2|m,k)>
    
        #L = gaussian_logpdf(sum_product(np.outer(self.f,self.f) + self.Cov,
                                        

        # Compute <log q(f2)>
        
            


    def update(self):

        # Messages from parents
        m = self.parents[0].message_to_child()
        k = self.parents[1].message_to_child()
        if self.parents[2]:
            k_sparse = self.parents[2].message_to_child()
        else:
            k_sparse = None
        if self.parents[3]:
            pseudoinputs = self.parents[3].message_to_child()[0]
        else:
            pseudoinputs = None
                
        ## m = self.parents[0].message_to_child()[0]
        ## k = self.parents[1].message_to_child()[0]

        if self.observed:

            # Observations of this node
            self.u = gp_posterior_moment_function(m,
                                                  k,
                                                  self.x,
                                                  self.f,
                                                  k_sparse=k_sparse,
                                                  pseudoinputs=pseudoinputs)

        else:

            x = np.array([])
            y = np.array([])
            # Messages from children
            for (child,index) in self.children:
                (msg, mask) = child.message_to_parent(index)

                # Ignoring masks and plates..

                # m[0] is the inputs
                x = np.concatenate((x, msg[0]), axis=-2)

                # m[1] is the observations
                y = np.concatenate((y, msg[1]))

                # m[2] is the covariance matrix
                V = linalg.block_diag(V, msg[2])

            self.u = gp_posterior_moment_function(m, k, x, y, covariance=V)
            self.x = x
            self.f = y
            




# At least for now, simplify this GP node such that a GP is either
# observed or latent. If it is observed, it doesn't take messages from
# children, actually, it should not even have children!





    ## # Pseudo for GPFA:
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k3 = gp_cov_rq(magnitude=.., lengthscale=.., alpha=..)
    ## f = NodeGPSet(0, [k1,k2,k3]) # assumes block diagonality
    ## # f = NodeGPSet(0, [[k11,k12,k13],[k21,k22,k23],[k31,k32,k33]])
    ## X = GaussianFromGP(f, [ [[t0,0],[t0,1],[t0,2]], [t1,0],[t1,1],[t1,2], ..])
    ## ...
    

    ## # Construct a sum of GPs if interested only in the sum term
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k = gp_cov_sum(k1, k2)
    ## f = NodeGP(0, k)
    ## f.observe(x, y)
    ## f.update()
    ## (mp, kp) = f.get_parameters()
    
    

    ## # Construct a sum of GPs when interested also in the individual
    ## # GPs:
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k3 = gp_cov_delta(magnitude=theta3)
    ## f = NodeGPSum(0, [k1,k2,k3])
    ## x = np.array([1,2,3,4,5,6,7,8,9,10])
    ## y = np.sin(x[0]) + np.random.normal(0, 0.1, (10,))
    ## # Observe the sum (index 0)
    ## f.observe((0,x), y)
    ## # Inference
    ## f.update()
    ## (mp, kp) = f.get_parameters()
    ## # Mean of the sum
    ## mp[0](...)
    ## # Mean of the individual terms
    ## mp[1](...)
    ## mp[2](...)
    ## mp[3](...)
    ## # Covariance of the sum
    ## kp[0][0](..., ...)
    ## # Other covariances
    ## kp[1][1](..., ...)
    ## kp[2][2](..., ...)
    ## kp[3][3](..., ...)
    ## kp[1][2](..., ...)
    ## kp[1][3](..., ...)
    ## kp[2][3](..., ...)
