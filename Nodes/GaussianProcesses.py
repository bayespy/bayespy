import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import matplotlib.pyplot as plt
import time
import profile
import scipy.spatial.distance as distance

from Nodes.ExponentialFamily import *
from utils import *
import Nodes.CovarianceFunctions as CF

# m prior mean function
# k prior covariance function
# x data inputs
# z processed data outputs (z = inv(Cov) * (y-m(x)))
# U data covariance Cholesky factor
def gp_posterior_moment_function(m, k, x, y, noise=None):

    # Prior
    # FIXME: We are ignoring the covariance of mu now..
    mu = m(x)[0]
    K = k(x,x)[0]
    if noise != None:
        K += noise

    #print('hereiamagain')
    #print(K)

    # Compute posterior GP

    N = len(y)
    U = None
    z = None
    if N > 0:
        U = chol(K)
        z = chol_solve(U, y-mu)

    # DEBUG STUFF:
    #zed = np.ones(np.shape(zed))
    #print(z)

    #K = np.identity(np.shape(K)[0])
    #y = np.ones(np.shape(y))
    #U = chol(K)
    #z = chol_solve(U, y)

    def get_moments(xh, covariance=1, mean=True):
        #print(k)
        kh = k(x, xh)[0]
        #(kh,) = k(x, xh)
        #kh = np.ones(np.shape(kh))
        #print(kh)
        #print(np.shape(kh))
        #kh = kh.copy()

        #print('get_moments')
        
        # Function for computing posterior moments
        if mean:
            # Mean vector
            # FIXME: Ignoring the covariance of prior mu
            mh = m(xh)[0]
            ## print(mh)
            ## print(np.shape(mh))
            ## print(np.shape(kh))
            ## print(np.shape(z))
            ## print(np.shape(kh))
            ## print(np.shape(mh))
            #kh = kh.copy()
            #print(z)
            #zed = zed.copy()
            #print(np.dot(kh.T,z).squeeze().shape)

            p = np.dot(kh.T,z)
            #p = p.reshape((50,1,5))
            #print(np.shape(p))
            #p = p.squeeze()
            #print(np.shape(p))
            
            if z != None:
                #mh = mh + np.dot(kh.T, z)
                mh += np.dot(kh.T, z).squeeze()
            print(np.shape(mh))
                
        else:
            mh = None
        if covariance:
            if covariance == 1:
                # Variance vector
                khh = k(xh)[0]
                if U != None:
                    khh -= np.einsum('i...,i...', kh, chol_solve(U, kh))
            elif covariance == 2:
                # Full covariance matrix
                khh = k(xh,xh)[0]
                if U != None:
                    khh -= np.dot(kh.T, chol_solve(U,kh))
        else:
            khh = None

        return [mh, khh]

    return get_moments


# Constant function using GP mean protocol
class Constant(Node):
    def __init__(self, f, **kwargs):

        self.f = f
        Node.__init__(self, dims=[(np.inf,)], **kwargs)

    def message_to_child(self, gradient=False):

        # Wrapper
        def func(x, gradient=False):
            if gradient:
                return ([self.f(x), None], [])
            else:
                return [self.f(x), None]

        return func
    
# Gaussian process distribution
class GaussianProcess(NodeVariable):

    def __init__(self, m, k, **kwargs):

        self.x = np.array([])
        self.f = np.array([])
        ## self.x_obs = np.zeros((0,1))
        ## self.f_obs = np.zeros((0,))

        # By default, posterior == prior
        self.m = m
        self.k = k

        # Ignore plates
        NodeVariable.__init__(self,
                              m,
                              k,
                              plates=(),
                              dims=[(np.inf,), (np.inf,np.inf)],
                              **kwargs)
    

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

        ## if np.ndim(x) == 1:
        ##     if np.shape(f) != np.shape(x):
        ##         print(np.shape(f))
        ##         print(np.shape(x))
        ##         raise Exception("Number of inputs and function values do not match")
        ## elif np.shape(f) != np.shape(x)[:-1]:
        ##     print(np.shape(f))
        ##     print(np.shape(x))
        ##     raise Exception("Number of inputs and function values do not match")

        self.observed = True
        self.x = x
        self.f = f
        ## self.x_obs = x
        ## self.f_obs = f

    # You might want:
    # - mean for x
    # - covariance (and mean) for x
    # - variance (and mean) for x
    # - i.e., mean and/or (co)variance for x
    # - covariance for x1 and x2

            
        
    def lower_bound_contribution(self, gradient=False):
        m = self.parents[0].message_to_child(gradient=gradient)
        k = self.parents[1].message_to_child(gradient=gradient)
        ## m = self.parents[0].message_to_child(gradient=gradient)[0]
        ## k = self.parents[1].message_to_child(gradient=gradient)[0]

        # Prior
        if gradient:
            (mu, dmus) = m(self.x, gradient=True)
            (K, dKs) = k(self.x, self.x, gradient=True)
        else:
            mu = m(self.x)
            K = k(self.x, self.x)
            dmus = []
            dKs = []

        # FIXME: We are ignoring the covariance of mu now..
        mu = mu[0]
        K = K[0]

        # Log pdf
        if self.observed:
            f0 = self.f - mu
            
            #print('hereiam')
            #print(K)
            try:
                U = chol(K)
            except linalg.LinAlgError:
                print('non positive definite, return -inf')
                return -np.inf
            z = chol_solve(U, f0)
            #print(K)
            L = gaussian_logpdf(np.dot(f0, z),
                                0,
                                0,
                                logdet_chol(U),
                                np.size(self.f))

            for (dmu, func) in dmus:
                # Derivative w.r.t. mean vector
                d = -np.sum(z)
                # Send the derivative message
                func += d
                #func(d)
                
            for (dK, func) in dKs:
                # Compute derivative w.r.t. covariance matrix
                d = 0.5 * (np.dot(z, np.dot(dK, z))
                           - np.trace(chol_solve(U, dK)))
                # Send the derivative message
                #print('add gradient')
                #func += d
                func(d)

        else:
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
        ## m = self.parents[0].message_to_child()[0]
        ## k = self.parents[1].message_to_child()[0]

        if self.observed:

            # Observations of this node
            self.u = gp_posterior_moment_function(m, k, self.x, self.f)

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


# Deterministic node for creating a set of GPs which can be used as a
# mean function to a general GP node.
class Multiple(NodeVariable):

    def __init__(self, GPs, **kwargs):

        # Ignore plates
        NodeVariable.__init__(self,
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
