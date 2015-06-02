################################################################################
# Copyright (C) 2011-2012 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import itertools
import numpy as np
#import scipy as sp
import scipy.sparse as sp # prefer CSC format
#import scipy.linalg.decomp_cholesky as decomp
#import scipy.linalg as linalg
#import scipy.special as special
#import matplotlib.pyplot as plt
#import time
#import profile
import scipy.spatial.distance as dist
#import scikits.sparse.distance as spdist

from . import node as ef
from bayespy.utils import misc as utils

# Covariance matrices can be either arrays or matrices so be careful
# with products and powers! Use explicit multiply or dot instead of
# *-operator.


def gp_cov_se(D2, overwrite=False):
    if overwrite:
        K = D2
        K *= -0.5
        np.exp(K, out=K)
    else:
        K = np.exp(-0.5*D2)
    return K

def gp_cov_pp2_new(r, d, derivative=False):
    # Dimension dependent parameter
    q = 2
    j = np.floor(d/2) + q + 1

    # Polynomial coefficients
    a2 = j**2 + 4*j + 3
    a1 = 3*j + 6
    a0 = 3

    # Two parts of the covariance function
    k1 = (1-r) ** (j+2)
    k2 = (a2*r**2 + a1*r + 3)

    # TODO: Check that derivative is 0, 1 or 2!

    if derivative == 0:
        # Return covariance
        return k1 * k2 / 3

    dk1 = - (j+2) * (1-r)**(j+1)
    dk2 = 2*a2*r + a1

    if derivative == 1:
        # Return first derivative of the covariance
        return (k1 * dk2 + dk1 * k2) / 3
    
    ddk1 = (j+2) * (j+1) * (1-r)**j
    ddk2 = 2*a2

    if derivative == 2:
        # Return second derivative of the covariance
        return (ddk1*k2 + 2*dk1*dk2 + k1*ddk2) / 3

def gp_cov_pp2(r, d, gradient=False):
    # Dimension dependent parameter
    j = np.floor(d/2) + 2 + 1

    # Polynomial coefficients
    a2 = j**2 + 4*j + 3
    a1 = 3*j + 6
    a0 = 3

    # Two parts of the covariance function
    k1 = (1-r) ** (j+2)
    k2 = (a2*r**2 + a1*r + 3)

    # The covariance function
    k = k1 * k2 / 3
        
    if gradient:
        # The gradient w.r.t. r
        dk = k * (j+2) / (r-1) + k1 * (2*a2*r + a1) / 3
        return (k, dk)
    else:
        return k

def gp_cov_delta(N):
    # TODO: Use sparse matrices here!
    if N > 0:
        #print('in gpcovdelta', N, sp.identity(N).shape)
        return sp.identity(N)
    else:
        # Sparse matrices do not allow zero-length dimensions
        return np.identity(N)
    #return np.identity(N)
    #return np.asmatrix(np.identity(N))
        

def squared_distance(x1, x2):
    ## # Reshape arrays to 2-D arrays
    ## sh1 = np.shape(x1)[:-1]
    ## sh2 = np.shape(x2)[:-1]
    ## d = np.shape(x1)[-1]
    ## x1 = np.reshape(x1, (-1,d))
    ## x2 = np.reshape(x2, (-1,d))
    (m1,n1) = x1.shape
    (m2,n2) = x2.shape
    if m1 == 0 or m2 == 0:
        D2 = np.empty((m1,m2))
    else:
        # Compute squared Euclidean distance
        D2 = dist.cdist(x1, x2, metric='sqeuclidean')
    #D2 = np.asmatrix(D2)
    # Reshape the result
    #D2 = np.reshape(D2, sh1 + sh2)
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
    if np.size(x) == 0:
        x = np.reshape(x, (0,0))
    elif np.ndim(x) == 0:
        x = np.reshape(x, (1,1))
    elif np.ndim(x) == 1:
        x = np.reshape(x, (-1,1))
    elif np.ndim(x) == 2:
        x = np.atleast_2d(x)
    else:
        raise Exception("Standard GP inputs must be 2-dimensional")

    return x

def gp_preprocess_inputs(x1,x2=None):
    #args = list(args)
    #if len(args) < 1 or len(args) > 2:
        #raise Exception("Number of inputs must be one or two")
    if x2 is None:
        x1 = gp_standardize_input(x1)
        return x1
    else:
        if x1 is x2:
            x1 = gp_standardize_input(x1)
            x2 = x1
        else:
            x1 = gp_standardize_input(x1)
            x2 = gp_standardize_input(x2)
        return (x1, x2)
        
    #return args
## def gp_preprocess_inputs(x1,x2=None):
##     #args = list(args)
##     #if len(args) < 1 or len(args) > 2:
##         #raise Exception("Number of inputs must be one or two")
##     if x2 is not None: len(args) == 2:
##         if args[0] is args[1]:
##             args[0] = gp_standardize_input(args[0])
##             args[1] = args[0]
##         else:
##             args[1] = gp_standardize_input(args[1])
##             args[0] = gp_standardize_input(args[0])
##     else:
##         args[0] = gp_standardize_input(args[0])
        
##     return args

# TODO:
# General syntax for these covariance functions:
# covfunc(hyper1,
#         hyper2,
#         ...
#         hyperN,
#         x1,
#         x2=None,
#         gradient=list_of_booleans_for_each_hyperparameter)

def covfunc_zeros(x1, x2=None, gradient=False):

    inputs = gp_preprocess_inputs(*inputs)

    # Compute distance and covariance matrix
    if x2 is None:
        x1 = gp_preprocess_inputs(x1)
        # Only variance vector asked
        N = np.shape(x1)[0]
        # TODO: Use sparse matrices!
        K = np.zeros(N)
        #K = np.asmatrix(np.zeros((N,1)))

    else:
        (x1,x2) = gp_preprocess_inputs(x1,x2)
        # Full covariance matrix asked
        #x1 = inputs[0]
        #x2 = inputs[1]
        # Number of inputs x1
        N1 = np.shape(x1)[0]
        N2 = np.shape(x2)[0]

        # TODO: Use sparse matrices!
        K = np.zeros((N1,N2))
        #K = np.asmatrix(np.zeros((N1,N2)))

    if gradient is not False:
        return (K, [])
    else:
        return K

def covfunc_delta(amplitude, x1, x2=None, gradient=False):

    # Make sure that amplitude is a scalar, not an array object
    amplitude = utils.array_to_scalar(amplitude)

    ## if gradient:
    ##     gradient_amplitude = gradient[0]
    ## else:
    ##     gradient_amplitude = []

    ## inputs = gp_preprocess_inputs(*inputs)

    # Compute distance and covariance matrix
    if x2 is None:
        x1 = gp_preprocess_inputs(x1)
        # Only variance vector asked
        #x = inputs[0]
        N = np.shape(x1)[0]
        K = np.ones(N) * amplitude**2

    else:
        (x1,x2) = gp_preprocess_inputs(x1,x2)
        # Full covariance matrix asked
        #x1 = inputs[0]
        #x2 = inputs[1]
        # Number of inputs x1
        N1 = np.shape(x1)[0]

        # x1 == x2?
        if x1 is x2:
            delta = True
            # Delta covariance
            #
            # FIXME: Broadcasting doesn't work with sparse matrices,
            # so must use scalar multiplication
            K = gp_cov_delta(N1) * amplitude**2
            #K = gp_cov_delta(N1).multiply(amplitude**2)
        else:
            delta = False
            # Number of inputs x2
            N2 = np.shape(x2)[0]
            # Zero covariance
            if N1 > 0 and N2 > 0:
                K = sp.csc_matrix((N1,N2))
            else:
                K = np.zeros((N1,N2))

    # Gradient w.r.t. amplitude
    if gradient:
        # FIXME: Broadcasting doesn't work with sparse matrices,
        # so must use scalar multiplication
        gradient_amplitude = K*(2/amplitude)
        print("noise grad", gradient_amplitude)
        return (K, (gradient_amplitude,))
    else:
        return K

def covfunc_pp2(amplitude, lengthscale, x1, x2, gradient=False):

    # Make sure that hyperparameters are scalars, not an array objects
    amplitude = utils.array_to_scalar(amplitude)
    lengthscale = utils.array_to_scalar(lengthscale)
    #amplitude = theta[0]
    #lengthscale = theta[1]

    ## if gradient:
    ##     gradient_amplitude = gradient[0]
    ##     gradient_lengthscale = gradient[1]
    ## else:
    ##     gradient_amplitude = []
    ##     gradient_lengthscale = []

    ## inputs = gp_preprocess_inputs(*inputs)

    # Compute covariance matrix
    if x2 is None:
        x1 = gp_preprocess_inputs(x1)
        # Compute variance vector
        K = np.ones(np.shape(x)[:-1])
        K *= amplitude**2
        # Compute gradient w.r.t. lengthscale
        if gradient:
            gradient_lengthscale = np.zeros(np.shape(x1)[:-1])
    
    else:
        (x1,x2) = gp_preprocess_inputs(x1,x2)
        # Compute (sparse) distance matrix
        if x1 is x2:
            x1 = inputs[0] / (lengthscale)
            x2 = x1
            D2 = spdist.pdist(x1, 1.0, form="full", format="csc")
        else:
            x1 = inputs[0] / (lengthscale)
            x2 = inputs[1] / (lengthscale)
            D2 = spdist.cdist(x1, x2, 1.0, format="csc")
        r = np.sqrt(D2.data)

        N1 = np.shape(x1)[0]
        N2 = np.shape(x2)[0]
        
        # Compute the covariances
        if gradient:
            (k, dk) = gp_cov_pp2(r, np.shape(x1)[-1], gradient=True)
        else:
            k = gp_cov_pp2(r, np.shape(x1)[-1])
        k *= amplitude**2
        # Compute gradient w.r.t. lengthscale
        if gradient:
            if N1 >= 1 and N2 >= 1:
                dk *= r * (-amplitude**2 / lengthscale)
                gradient_lengthscale = sp.csc_matrix((dk, D2.indices, D2.indptr),
                                                     shape=(N1,N2))
            else:
                gradient_lengthscale = np.empty((N1,N2))
            
        # Form sparse covariance matrix
        if N1 >= 1 and N2 >= 1:
            ## K = sp.csc_matrix((k, ij), shape=(N1,N2))
            K = sp.csc_matrix((k, D2.indices, D2.indptr), shape=(N1,N2))
        else:
            K = np.empty((N1, N2))
        #print(K.__class__)

    # Gradient w.r.t. amplitude
    if gradient:
        gradient_amplitude = K * (2 / amplitude)

    # Return values
    if gradient:
        print("pp2 grad", gradient_lengthscale)
        return (K, (gradient_amplitude, gradient_lengthscale))
    else:
        return K


def covfunc_se(amplitude, lengthscale, x1, x2=None, gradient=False):

    # Make sure that hyperparameters are scalars, not an array objects
    amplitude = utils.array_to_scalar(amplitude)
    lengthscale = utils.array_to_scalar(lengthscale)

    # Compute covariance matrix
    if x2 is None:
        x1 = gp_preprocess_inputs(x1)
        #x = inputs[0]
        # Compute variance vector
        N = np.shape(x1)[0]
        K = np.ones(N)
        np.multiply(K, amplitude**2, out=K)
        # Compute gradient w.r.t. lengthscale
        if gradient:
            # TODO: Use sparse matrices?
            gradient_lengthscale = np.zeros(N)
    else:
        (x1,x2) = gp_preprocess_inputs(x1,x2)
        x1 = x1 / (lengthscale)
        x2 = x2 / (lengthscale)
        # Compute distance matrix
        K = squared_distance(x1, x2)
        # Compute gradient partly
        if gradient:
            gradient_lengthscale = np.divide(K, lengthscale)
        # Compute covariance matrix
        gp_cov_se(K, overwrite=True)
        np.multiply(K, amplitude**2, out=K)
        # Compute gradient w.r.t. lengthscale
        if gradient:
            gradient_lengthscale *= K

    # Gradient w.r.t. amplitude
    if gradient:
        gradient_amplitude = K * (2 / amplitude)

    # Return values
    if gradient:
        print("se grad", gradient_amplitude, gradient_lengthscale)
        return (K, (gradient_amplitude, gradient_lengthscale))
    else:
        return K


class CovarianceFunctionWrapper():
    def __init__(self, covfunc, *params):
        # Parse parameter values and their gradients to separate lists
        self.covfunc = covfunc
        self.params = list(params)
        self.gradient_params = list()
        ## print(params)
        for ind in range(len(params)):
            if isinstance(params[ind], tuple):
                # Parse the value and the list of gradients from the
                # form:
                #  ([value, ...], [ [grad1, ...], [grad2, ...], ... ])
                self.gradient_params.append(params[ind][1])
                self.params[ind] = params[ind][0][0]
            else:
                # No gradients, parse from the form:
                #  [value, ...]
                self.gradient_params.append([])
                self.params[ind] = params[ind][0]

    def fixed_covariance_function(self, *inputs, gradient=False):

        # What if this is called several times??

        if gradient:

            ## grads = [[grad[0] for grad in self.gradient_params[ind]]
            ##          for ind in range(len(self.gradient_params))]

            ## (K, dK) = self.covfunc(self.params,
            ##                        *inputs,
            ##                        gradient=self.gradient_params)
            arguments = tuple(self.params) + tuple(inputs)
            (K, dK) = self.covfunc(*arguments,
                                   gradient=True)
            ## (K, dK) = self.covfunc(self.params,
            ##                        *inputs,
            ##                        gradient=grads)

            DK = []
            for ind in range(len(dK)):
                # Gradient w.r.t. covariance function's ind-th
                # hyperparameter
                dk = dK[ind]
                # Chain rule: Multiply by the gradient of the
                # hyperparameter w.r.t. parent node and append the
                # list DK:
                # DK = [ (dx1_1, callback), ..., (dx1_n, callback) ]
                for grad in self.gradient_params[ind]:
                    #print(grad[0])
                    #print(grad[1:])
                    #print(dk)
                    if sp.issparse(dk):
                        print(dk.shape)
                        print(grad[0].shape)
                        DK += [ [dk.multiply(grad[0])] + grad[1:] ]
                    else:
                        DK += [ [np.multiply(dk,grad[0])] + grad[1:] ]
                    #DK += [ [np.multiply(grad[0], dk)] + grad[1:] ]
                ## DK += [ (np.multiply(grad, dk),) + grad[1:]
                ##         for grad in self.gradient_params[ind] ]
                
                ## for grad in self.gradient_params[ind]:
                ##     DK += ( (np.multiply(grad, dk),) + grad[1:] )
            ## DK = []
            ## for ind in range(len(dK)):
            ##     for (grad, dk) in zip(self.gradient_params[ind], dK[ind]):
            ##         DK += [ [dk] + grad[1:] ]

            K = [K]

            return (K, DK)

        else:
            arguments = tuple(self.params) + tuple(inputs)
            #print(arguments)
            K = self.covfunc(*arguments,
                             gradient=False)
            return [K]

class CovarianceFunction(ef.Node):


    def __init__(self, covfunc, *args, **kwargs):
        self.covfunc = covfunc

        params = list(args)
        for i in range(len(args)):
            # Check constant parameters
            if utils.is_numeric(args[i]):
                params[i] = ef.NodeConstant([np.asanyarray(args[i])],
                                            dims=[np.shape(args[i])])
                # TODO: Parameters could be constant functions? :)

        ef.Node.__init__(self, *params, dims=[(np.inf, np.inf)], **kwargs)


    def __call__(self, x1, x2):
        """ Compute covariance matrix for inputs x1 and x2. """
        covfunc = self.message_to_child()
        return covfunc(x1, x2)[0]

    def message_to_child(self, gradient=False):

        params = [parent.message_to_child(gradient=gradient) for parent in self.parents]
        covfunc = self.get_fixed_covariance_function(*params)
        return covfunc

    def get_fixed_covariance_function(self, *params):
        get_cov_func = CovarianceFunctionWrapper(self.covfunc, *params)
        return get_cov_func.fixed_covariance_function


    ## def covariance_function(self, *params):
    ##     # Parse parameter values and their gradients to separate lists
    ##     params = list(params)
    ##     gradient_params = list()
    ##     print(params)
    ##     for ind in range(len(params)):
    ##         if isinstance(params[ind], tuple):
    ##             # Parse the value and the list of gradients from the
    ##             # form:
    ##             #  ([value, ...], [ [grad1, ...], [grad2, ...], ... ])
    ##             gradient_params.append(params[ind][1])
    ##             params[ind] = params[ind][0][0]
    ##         else:
    ##             # No gradients, parse from the form:
    ##             #  [value, ...]
    ##             gradient_params.append([])
    ##             params[ind] = params[ind][0]

    ##     # This gradient_params changes mysteriously..
    ##     print('grad_params before')
    ##     if isinstance(self, SquaredExponential):
    ##         print(gradient_params)
            
    ##     def cov(*inputs, gradient=False):

    ##         if gradient:
    ##             print('grad_params after')
    ##             print(gradient_params)
    ##             grads = [[grad[0] for grad in gradient_params[ind]]
    ##                      for ind in range(len(gradient_params))]


    ##             print('CovarianceFunction.cov')
    ##             #if isinstance(self, SquaredExponential):
    ##                 #print(self.__class__)
    ##                 #print(grads)
    ##             (K, dK) = self.covfunc(params,
    ##                                    *inputs,
    ##                                    gradient=grads)

    ##             for ind in range(len(dK)):
    ##                 for (grad, dk) in zip(gradient_params[ind], dK[ind]):
    ##                     grad[0] = dk

    ##             K = [K]
    ##             dK = []
    ##             for grad in gradient_params:
    ##                 dK += grad
    ##             return (K, dK)
                    
    ##         else:
    ##             K = self.covfunc(params,
    ##                              *inputs,
    ##                              gradient=False)
    ##             return [K]

    ##     return cov


class Sum(CovarianceFunction):
    def __init__(self, *args, **kwargs):
        CovarianceFunction.__init__(self,
                                    None,
                                    *args,
                                    **kwargs)

    def get_fixed_covariance_function(self, *covfunc_parents):
        def covfunc(*inputs, gradient=False):
            K_sum = None
            if gradient:
                dK_sum = list()
            for k in covfunc_parents:
                if gradient:
                    (K, dK) = k(*inputs, gradient=gradient)
                    print("dK in sum", dK)
                    dK_sum += dK
                    #print("dK_sum in sum", dK_sum)
                else:
                    K = k(*inputs, gradient=gradient)
                if K_sum is None:
                    K_sum = K[0]
                else:
                    try:
                        K_sum += K[0]
                    except:
                        # You have to do this way, for instance, if
                        # K_sum is sparse and K[0] is dense.
                        K_sum = K_sum + K[0]

            if gradient:
                #print("dK_sum on: ", dK_sum)
                #print('covsum', dK_sum)
                return ([K_sum], dK_sum)
            else:
                return [K_sum]

        return covfunc


class Delta(CovarianceFunction):
    def __init__(self, amplitude, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_delta,
                                    amplitude,
                                    **kwargs)

class Zeros(CovarianceFunction):
    def __init__(self, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_zeros,
                                    **kwargs)


class SquaredExponential(CovarianceFunction):
    def __init__(self, amplitude, lengthscale, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_se,
                                    amplitude,
                                    lengthscale,
                                    **kwargs)

class PiecewisePolynomial2(CovarianceFunction):
    def __init__(self, amplitude, lengthscale, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_pp2,
                                    amplitude,
                                    lengthscale,
                                    **kwargs)

# TODO: Rename to Blocks or Joint ?
class Multiple(CovarianceFunction):
    
    def __init__(self, covfuncs, **kwargs):
        self.d = len(covfuncs)
        #self.sparse = sparse
        parents = [covfunc for row in covfuncs for covfunc in row]
        CovarianceFunction.__init__(self,
                                    None,
                                    *parents,
                                    **kwargs)

    def get_fixed_covariance_function(self, *covfuncs):
        def cov(*inputs, gradient=False):

            # Computes the covariance matrix from blocks which all
            # have their corresponding covariance functions

            if len(inputs) < 2:
                # For one input, return the variance vector instead of
                # the covariance matrix
                x1 = inputs[0]
                # Collect variance vectors from the covariance
                # functions corresponding to the diagonal blocks
                K = [covfuncs[i*self.d+i](x1[i], gradient=gradient)[0]
                     for i in range(self.d)]
                # Form the variance vector from the collected vectors
                if gradient:
                    raise Exception('Gradient not yet implemented.')
                else:
                    ## print("in cov multiple")
                    ## for (k,kf) in zip(K,covfuncs):
                    ##     print(np.shape(k), k.__class__, kf)
                    #K = np.vstack(K)
                    K = np.concatenate(K)
            else:
                x1 = inputs[0]
                x2 = inputs[1]

                # Collect the covariance matrix (and possibly
                # gradients) from each block.
                #print('cov mat collection begins')
                K = [[covfuncs[i*self.d+j](x1[i], x2[j], gradient=gradient)
                      for j in range(self.d)]
                      for i in range(self.d)]
                #print('cov mat collection ends')

                # Remove matrices that have zero length dimensions?
                if gradient:
                    K = [[K[i][j]
                          for j in range(self.d)
                          if np.shape(K[i][j][0][0])[1] != 0]
                          for i in range(self.d)
                          if np.shape(K[i][0][0][0])[0] != 0]
                else:
                    K = [[K[i][j]
                          for j in range(self.d)
                          if np.shape(K[i][j][0])[1] != 0]
                          for i in range(self.d)
                          if np.shape(K[i][0][0])[0] != 0]
                n_blocks = len(K)
                #print("nblocks", n_blocks)
                #print("K", K)

                # Check whether all blocks are sparse
                is_sparse = True
                for i in range(n_blocks):
                    for j in range(n_blocks):
                        if gradient:
                            A = K[i][j][0][0]
                        else:
                            A = K[i][j][0]
                        if not sp.issparse(A):
                            is_sparse = False

                if gradient:

                    ## Compute the covariance matrix and the gradients

                    # Create block matrices of zeros. This helps in
                    # computing the gradient.
                    if is_sparse:
                        # Empty sparse matrices. Some weird stuff here
                        # because sparse matrices can't have zero
                        # length dimensions.
                        Z = [[sp.csc_matrix(np.shape(K[i][j][0][0]))
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                    else:
                        # Empty dense matrices
                        Z = [[np.zeros(np.shape(K[i][j][0][0]))
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                              ## for j in range(self.d)]
                              ## for i in range(self.d)]

                    # Compute gradients block by block
                    dK = list()
                    for i in range(n_blocks):
                        for j in range(n_blocks):
                            # Store the zero block
                            z_old = Z[i][j]
                            # Go through the gradients for the (i,j)
                            # block
                            for dk in K[i][j][1]:
                                # Keep other blocks at zero and set
                                # the gradient to (i,j) block.  Form
                                # the matrix from blocks
                                if is_sparse:
                                    Z[i][j] = dk[0]
                                    dk[0] = sp.bmat(Z).tocsc()
                                else:
                                    if sp.issparse(dk[0]):
                                        Z[i][j] = dk[0].toarray()
                                    else:
                                        Z[i][j] = dk[0]
                                    #print("Z on:", Z)
                                    dk[0] = np.asarray(np.bmat(Z))
                                # Append the computed gradient matrix
                                # to the list of gradients
                                dK.append(dk)
                            # Restore the zero block
                            Z[i][j] = z_old

                    ## Compute the covariance matrix but not the
                    ## gradients

                    if is_sparse:
                        # Form the full sparse covariance matrix from
                        # blocks.  Ignore blocks having a zero-length
                        # axis because sparse matrices consider zero
                        # length as an invalid shape (BUG IN SCIPY?).
                        K = [[K[i][j][0][0]
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                        K = sp.bmat(K).tocsc()
                    else:
                        # Form the full dense covariance matrix from
                        # blocks. Transform sparse blocks to dense
                        # blocks.
                        K = [[K[i][j][0][0]
                              if not sp.issparse(K[i][j][0][0]) else
                              K[i][j][0][0].toarray()
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                        K = np.asarray(np.bmat(K))

                else:

                    ## Compute the covariance matrix but not the
                    ## gradients

                    if is_sparse:
                        # Form the full sparse covariance matrix from
                        # blocks.  Ignore blocks having a zero-length
                        # axis because sparse matrices consider zero
                        # length as an invalid shape (BUG IN SCIPY?).
                        K = [[K[i][j][0]
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                        K = sp.bmat(K).tocsc()
                    else:
                        # Form the full dense covariance matrix from
                        # blocks. Transform sparse blocks to dense
                        # blocks.
                        K = [[K[i][j][0]
                              if not sp.issparse(K[i][j][0]) else
                              K[i][j][0].toarray()
                              for j in range(n_blocks)]
                              for i in range(n_blocks)]
                        K = np.asarray(np.bmat(K))



            if gradient:
                return ([K], dK)
            else:
                return [K]

        return cov



