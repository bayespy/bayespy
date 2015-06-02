################################################################################
# Copyright (C) 2011-2012 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


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

from .node import Node
from .stochastic import Stochastic
from bayespy.utils.misc import *

# Computes log probability density function of the Gaussian
# distribution
def gaussian_logpdf(y_invcov_y,
                    y_invcov_mu,
                    mu_invcov_mu,
                    logdetcov,
                    D):

    return (-0.5*D*np.log(2*np.pi)
            -0.5*logdetcov
            -0.5*y_invcov_y
            +y_invcov_mu
            -0.5*mu_invcov_mu)


# m prior mean function
# k prior covariance function
# x data inputs
# z processed data outputs (z = inv(Cov) * (y-m(x)))
# U data covariance Cholesky factor
def gp_posterior_moment_function(m, k, x, y, noise=None):

    # Prior
    mu = m(x)[0]
    K = k(x,x)[0]
    if noise != None:
        K += noise

    #print('hereiamagain')
    #print(K)

    # Compute posterior GP

    N = len(y)
    if N == 0:
        U = None
        z = None
    else:
        U = chol(K)
        z = chol_solve(U, y-mu)

    def get_moments(xh, covariance=1, mean=True):
        (kh,) = k(x, xh)
        
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

# m prior mean function
# k prior covariance function
# x data inputs
# z processed data outputs (z = inv(Cov) * (y-m(x)))
# U data covariance Cholesky factor
## def gp_multi_posterior_moment_function(m, k, x, y, noise=None):

##     # Prior
##     mu = m(x)[0]
##     K = k(x,x)[0]
##     if noise != None:
##         K += noise

##     #print('hereiamagain')
##     #print(K)

##     # Compute posterior GP

##     N = len(y)
##     if N == 0:
##         U = None
##         z = None
##     else:
##         U = chol(K)
##         z = chol_solve(U, y-mu)

##     def get_moments(xh, covariance=1, mean=True):
##         (kh,) = k(x, xh)
        
##         # Function for computing posterior moments
##         if mean:
##             # Mean vector
##             mh = m(xh)
##             if z != None:
##                 mh += np.dot(kh.T, z)
##         else:
##             mh = None
##         if covariance:
##             if covariance == 1:
##                 # Variance vector
##                 khh = k(xh)
##                 if U != None:
##                     khh -= np.einsum('i...,i...', kh, chol_solve(U, kh))
##             elif covariance == 2:
##                 # Full covariance matrix
##                 khh = k(xh,xh)
##                 if U != None:
##                     khh -= np.dot(kh.T, chol_solve(U,kh))
##         else:
##             khh = None

##         return [mh, khh]

##     return get_moments

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

# General rule for the parameters for covariance functions:
#
# (value, [ [dvalue1, ...], [dvalue2, ...], [dvalue3, ...], ...])
#
# For instance,
#
# k = covfunc_se((1.0, []), (15, [ [1,update_grad] ]))
# K = k((x1, [ [dx1,update_grad] ]), (x2, []))
#
# Plain values are converted as:
# value  ->  (value, [])

def gp_standardize_input(x):
    if np.ndim(x) == 0:
        x = add_trailing_axes(x, 2)
    elif np.ndim(x) == 1:
        x = add_trailing_axes(x, 1)
    return x

def gp_preprocess_inputs(*args):
    args = list(args)
    if len(args) < 1 or len(args) > 2:
        raise Exception("Number of inputs must be one or two")
    if len(args) == 2:
        if args[0] is args[1]:
            args[0] = gp_standardize_input(args[0])
            args[1] = args[0]
        else:
            args[1] = gp_standardize_input(args[1])
            args[0] = gp_standardize_input(args[0])
    else:
        args[0] = gp_standardize_input(args[0])
        
    return args

def covfunc_delta(theta, *inputs, gradient=False):

    amplitude = theta[0]

    if gradient:
        gradient_amplitude = gradient[0]
    else:
        gradient_amplitude = []

    inputs = gp_preprocess_inputs(*inputs)

    # Compute distance and covariance matrix
    if len(inputs) == 1:
        # Only variance vector asked
        x = inputs[0]
        K = np.ones(np.shape(x)[:-1]) * amplitude**2

    else:
        # Full covariance matrix asked
        x1 = inputs[0]
        x2 = inputs[1]
        # Number of inputs x1
        N1 = np.shape(x1)[-2]

        # x1 == x2?
        if x1 is x2:
            delta = True
            # Delta covariance
            K = gp_cov_delta(N1) * amplitude**2
        else:
            delta = False
            # Number of inputs x2
            N2 = np.shape(x2)[-2]
            # Zero covariance
            K = np.zeros((N1,N2))

    # Gradient w.r.t. amplitude
    if gradient:
        for ind in range(len(gradient_amplitude)):
            gradient_amplitude[ind] = K * (2 * gradient_amplitude[ind] / amplitude)

    if gradient:
        return (K, gradient)
    else:
        return K

def covfunc_se(theta, *inputs, gradient=False):

    amplitude = theta[0]
    lengthscale = theta[1]

    ## print('in se')
    ## print(amplitude)
    ## print(lengthscale)

    if gradient:
        gradient_amplitude = gradient[0]
        gradient_lengthscale = gradient[1]
    else:
        gradient_amplitude = []
        gradient_lengthscale = []

    inputs = gp_preprocess_inputs(*inputs)

    # Compute covariance matrix
    if len(inputs) == 1:
        x = inputs[0]
        # Compute variance vector
        K = np.ones(np.shape(x)[:-1])
        K *= amplitude**2
        # Compute gradient w.r.t. lengthscale
        for ind in range(len(gradient_lengthscale)):
            gradient_lengthscale[ind] = np.zeros(np.shape(x)[:-1])
    else:
        x1 = inputs[0] / (lengthscale)
        x2 = inputs[1] / (lengthscale)
        # Compute distance matrix
        K = squared_distance(x1, x2)
        # Compute gradient partly
        if gradient:
            for ind in range(len(gradient_lengthscale)):
                gradient_lengthscale[ind] = K * ((lengthscale**-1) * gradient_lengthscale[ind])
        # Compute covariance matrix
        gp_cov_se(K, overwrite=True)
        K *= amplitude**2
        # Compute gradient w.r.t. lengthscale
        if gradient:
            for ind in range(len(gradient_lengthscale)):
                gradient_lengthscale[ind] *= K

    # Gradient w.r.t. amplitude
    if gradient:
        for ind in range(len(gradient_amplitude)):
            gradient_amplitude[ind] = K * (2 * gradient_amplitude[ind] / amplitude)

    # Return values
    if gradient:
        return (K, gradient)
    else:
        return K


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

    def message_to_child(self, gradient=False):

        params = [parent.message_to_child(gradient=gradient) for parent in self.parents]
        return self.covariance_function(*params)

    def covariance_function(self, *params):
        params = list(params)
        gradient_params = list()
        for ind in range(len(params)):
            if isinstance(params[ind], tuple):
                gradient_params.append(params[ind][1])
                params[ind] = params[ind][0][0]
            else:
                gradient_params.append([])
                params[ind] = params[ind][0]
                
        def cov(*inputs, gradient=False):

            if gradient:
                grads = [[grad[0] for grad in gradient_params[ind]]
                         for ind in range(len(gradient_params))]
                
                (K, dK) = self.covfunc(params,
                                       *inputs,
                                       gradient=grads)

                for ind in range(len(dK)):
                    for (grad, dk) in zip(gradient_params[ind], dK[ind]):
                        grad[0] = dk

                K = [K]
                dK = []
                for grad in gradient_params:
                    dK += grad
                return (K, dK)
                    
            else:
                K = self.covfunc(params,
                                 *inputs,
                                 gradient=False)
                return [K]

        return cov


class NodeCovarianceFunctionSum(NodeCovarianceFunction):
    def __init__(self, *args, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        None,
                                        *args,
                                        **kwargs)

    def covariance_function(self, *covfuncs):
        def cov(*inputs, gradient=False):
            K_sum = 0
            if gradient:
                dK_sum = list()
            for k in covfuncs:
                if gradient:
                    (K, dK) = k(*inputs, gradient=gradient)
                    dK_sum += dK
                else:
                    K = k(*inputs, gradient=gradient)
                K_sum += K[0]

            if gradient:
                return ([K_sum], dK_sum)
            else:
                return [K_sum]

        return cov


class NodeCovarianceFunctionDelta(NodeCovarianceFunction):
    def __init__(self, amplitude, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        covfunc_delta,
                                        amplitude,
                                        **kwargs)


class NodeCovarianceFunctionSquaredExponential(NodeCovarianceFunction):
    def __init__(self, amplitude, lengthscale, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        covfunc_se,
                                        amplitude,
                                        lengthscale,
                                        **kwargs)

class NodeMultiCovarianceFunction(NodeCovarianceFunction):
    
    def __init__(self, *args, **kwargs):
        NodeCovarianceFunction.__init__(self,
                                        None,
                                        *args,
                                        **kwargs)

    def covfunc(self, *covfuncs):
        def cov(*inputs, gradient=False):
            K_sum = 0
            if gradient:
                dK_sum = list()
            for k in covfuncs:
                if gradient:
                    (K, dK) = k(*inputs, gradient=gradient)
                    dK_sum += dK
                else:
                    K = k(*inputs, gradient=gradient)
                K_sum += K[0]


            if gradient:
                return ([K_sum], dK_sum)
            else:
                return [K_sum]

        return cov



class NodeConstantGaussianProcess(Node):
    def __init__(self, f, **kwargs):

        self.f = f
        Node.__init__(self, dims=[(np.inf,)], **kwargs)

    def message_to_child(self, gradient=False):

        # Wrapper
        def func(x, gradient=False):
            if gradient:
                return ([self.f(x)], [])
            else:
                return [self.f(x)]

        return func
    

# At least for now, simplify this GP node such that a GP is either
# observed or latent. If it is observed, it doesn't take messages from
# children, actually, it should not even have children!


            
#class NodeMultiGaussianProcess(NodeVariable):
class NodeMultiGaussianProcess(Stochastic):
    

    def __init__(self, m, k, **kwargs):

        self.x = []
        self.f = []

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

        if np.ndim(x) == 1:
            if np.shape(f) != np.shape(x):
                print(np.shape(f))
                print(np.shape(x))
                raise Exception("Number of inputs and function values do not match")
        elif np.shape(f) != np.shape(x)[:-1]:
            print(np.shape(f))
            print(np.shape(x))
            raise Exception("Number of inputs and function values do not match")

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

        mu = mu[0]
        K = K[0]

        # Log pdf
        if self.observed:
            # Vector of f-mu
            f0 = np.vstack([(f-m) for (f,m) in zip(self.f,mu)])
            # Full covariance matrix
            K_full = np.bmat(K)
            
            try:
                U = chol(K_full)
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
            




class NodeGaussianProcess(Stochastic):
    #class NodeGaussianProcess(NodeVariable):

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

        if np.ndim(x) == 1:
            if np.shape(f) != np.shape(x):
                print(np.shape(f))
                print(np.shape(x))
                raise Exception("Number of inputs and function values do not match")
        elif np.shape(f) != np.shape(x)[:-1]:
            print(np.shape(f))
            print(np.shape(x))
            raise Exception("Number of inputs and function values do not match")

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
            


