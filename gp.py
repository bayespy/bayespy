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

from nodes import *
from utils import *

def gp_cov_se(D2, overwrite=False):
    if overwrite:
        K = D2
        K *= -0.5
        np.exp(K, out=K)
    else:
        K = np.exp(-0.5*D2)
    return K

def gp_cov_delta(N):
    return np.identity(N)
        

def squared_distance(x1, x2):
    # Reshape arrays to 2-D arrays
    sh1 = np.shape(x1)[:-1]
    sh2 = np.shape(x2)[:-1]
    d = np.shape(x1)[-1]
    x1 = np.reshape(x1, (-1,d))
    x2 = np.reshape(x2, (-1,d))
    # Compute squared Euclidean distance
    D2 = distance.cdist(x1, x2, metric='sqeuclidean')
    # Reshape the result
    D2 = np.reshape(D2, sh1 + sh2)
    return D2

def covfunc_sum(*covfuncs):

    def cov(*inputs):
        K = 0
        for k in covfuncs:
            K += k(*inputs)
        return K

    return cov

#class CovarianceFunctionDelta:
def covfunc_delta(amplitude):
    ## def __init__(self, amplitude):
    ##     # Amplitude could be a function which gives the value for
    ##     # inputs
    ##     self.amplitude = amplitude

    def cov(*inputs):
        # Compute distance and covariance matrix
        if len(inputs) == 1:
            # Only variance vector asked
            x = inputs[0]
            if np.ndim(x) == 1:
                K = np.ones(np.shape(x)) * amplitude**2
            else:
                K = np.ones(np.shape(x)[:-1]) * amplitude**2
        elif len(inputs) == 2:
            # Full covariance matrix asked
            x1 = inputs[0]
            x2 = inputs[1]
            # Number of inputs x1
            if np.ndim(x1) == 1:
                N1 = np.shape(x1)[-1]
            else:
                N1 = np.shape(x1)[-2]

            # x1 == x2?
            if x1 is x2:
                delta = True
                # Delta covariance
                K = gp_cov_delta(N1) * amplitude**2
            else:
                delta = False
                # Number of inputs x2
                if np.ndim(x2) == 1:
                    N2 = np.shape(x2)[-1]
                else:
                    N2 = np.shape(x2)[-2]
                # Zero covariance
                K = np.zeros((N1,N2))
                    
        else:
            raise Exception('Must give one or two inputs')
        return K

    return cov

def covfunc_se(amplitude, lengthscale):
        
    def cov(*args):
        
        # Compute distance and covariance matrix
        if len(args) == 1:
            x = args[0]
            if np.ndim(x) == 1:
                K = np.ones(np.shape(x))
            else:
                K = np.ones(np.shape(x)[:-1])
        elif len(args) == 2:
            x1 = args[0] / (lengthscale)
            x2 = args[1] / (lengthscale)
            if np.ndim(x1) == 1:
                x1 = np.reshape(x1, (-1,1))
            if np.ndim(x2) == 1:
                x2 = np.reshape(x2, (-1,1))
            # Compute covariance matrix
            K = squared_distance(x1, x2)
            gp_cov_se(K, overwrite=True)
        else:
            raise Exception('Must give one or two inputs')
        # Apply amplitude
        K *= amplitude**2
        return K

    return cov

class NodeCovarianceFunction(Node):

    def __init__(self, covfunc, *args, **kwargs):
        self.covfunc = covfunc

        params = list(args)
        for i in range(len(args)):
            # Check constant parameters
            if is_numeric(args[i]):
                params[i] = NodeConstant([np.asanyarray(args[i])],
                                         dims=[np.shape(args[i])])
                # TODO: Parameters could be constant functions? :)
                
        Node.__init__(self, *params, dims=[(np.inf, np.inf)], **kwargs)

    def message_to_child(self):
        params = [parent.message_to_child()[0] for parent in self.parents]
        return [self.covfunc(*params)]

class NodeCovarianceFunctionSum(NodeCovarianceFunction):
    def __init__(self, *args, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        covfunc_sum,
                                        *args,
                                        **kwargs)

class NodeCovarianceFunctionDelta(NodeCovarianceFunction):
    def __init__(self, amplitude, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        covfunc_delta,
                                        amplitude,
                                        **kwargs)

class NodeCovarianceFunctionSE(NodeCovarianceFunction):
    def __init__(self, amplitude, lengthscale, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        covfunc_se,
                                        amplitude,
                                        lengthscale,
                                        **kwargs)

class NodeConstantGaussianProcess(Node):
    def __init__(self, f, **kwargs):

        self.f = f
        Node.__init__(self, dims=[(np.inf,)], **kwargs)

    def message_to_child(self):
        return [self.f]
    
    
class NodeGaussianProcess(NodeVariable):

    def __init__(self, m, k, **kwargs):

        self.x = np.array([])
        self.f = np.array([])
        self.x_obs = np.zeros((0,1))
        self.f_obs = np.zeros((0,))

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
        return self.u #[self.m, self.k]

    #def get_parameters(self):
        #return [

    def observe(self, x, f):
        ## if np.ndim(x) == 1:
        ##     x = np.atleast_2d(x).T

        if np.ndim(x) == 1:
            if np.shape(f) != np.shape(x):
                print(np.shape(f))
                print(np.shape(x))
                raise Exception("Number of inputs and function values do not match")
        elif np.shape(f) != np.shape(x)[:-1]:
            print(np.shape(f))
            print(np.shape(x))
            raise Exception("Number of inputs and function values do not match")
        self.x_obs = x
        self.f_obs = f

    # You might want:
    # - mean for x
    # - covariance (and mean) for x
    # - variance (and mean) for x
    # - i.e., mean and/or (co)variance for x
    # - covariance for x1 and x2
            
        
    def update(self):

        # Messages from parents
        #u_parents = [parent.message_to_child() for parent in self.parents]
        #m = self.parents[0]
        #k = self.parents[1].message_to_child()[0]
        m = self.parents[0].message_to_child()[0]
        k = self.parents[1].message_to_child()[0]
                
        # Observations of this node
        x = self.x_obs
        y = self.f_obs
        V = np.zeros((len(y),len(y)))

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

        # Prior
        mu = m(x)
        K = k(x,x)

        Cov = K + V

        # Compute posterior GP
        #
        # FIXME: You would need to make a copy of m and k here,
        # otherwise things won't work correctly if the parents are
        # updated..?

        N = len(y)
        if N == 0:
            U = None
            z = None
        else:
            U = chol(Cov)
            z = chol_solve(U, y-mu)

        def get_moment_function(m, k, z, U):
            
            def get_moments(xh, covariance=1, mean=True):
                ## if np.ndim(xh) == 1:
                ##     xh = np.atleast_2d(xh).T
                    
                kh = k(x, xh)
                # Function for computing posterior moments
                if mean:
                    # Mean vector
                    mh = m(xh)
                    if z != None:
                        mh += np.dot(kh.T, z)
                else:
                    mh = None
                if covariance:
                    if covariance == 1:
                        # Variance vector
                        #khh = k(xh)
                        #print(np.shape(khh))
                        #print(np.shape(np.einsum('i...,i...', kh, chol_solve(U, kh))))
                        khh = k(xh)
                        if U != None:
                            khh -= np.einsum('i...,i...', kh, chol_solve(U, kh))
                    elif covariance == 2:
                        # Full covariance matrix
                        khh = k(xh,xh)
                        if U != None:
                            khh -= np.dot(kh.T, chol_solve(U,kh))
                else:
                    khh = None

                return [mh, khh]

            return get_moments

        self.u = get_moment_function(m, k, z, U)

        #self.m = lambda xh: m(xh) + np.dot(k(xh,x), z)
        #self.k = lambda xh1, xh2: k(xh1,xh2) - np.dot(k(xh1,x),
                                                      #chol_solve(U,
                                                                 #k(x,xh2)))

        # These are required for sending messages to parents
        self.x = x
        self.f = mu + np.dot(K,z)


