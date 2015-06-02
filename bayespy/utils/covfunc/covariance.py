################################################################################
# Copyright (C) 2011-2012 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import itertools
import numpy as np
#import scipy as sp
import scipy.sparse as sp # prefer CSC format
#import scipy.spatial.distance as dist

from bayespy.utils import misc
#from bayespy.utils.covfunc import distance
from scipy.spatial import distance

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
        D2 = distance.cdist(x1, x2, metric='sqeuclidean')
        #D2 = distance.cdist(x1, x2, metric='sqeuclidean')
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
    amplitude = misc.array_to_scalar(amplitude)

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

def covfunc_pp2(amplitude, lengthscale, x1, x2=None, gradient=False):

    # Make sure that hyperparameters are scalars, not an array objects
    amplitude = misc.array_to_scalar(amplitude)
    lengthscale = misc.array_to_scalar(lengthscale)
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
            x1 = x1 / (lengthscale)
            x2 = x1
            D2 = distance.sparse_pdist(x1, 1.0, form="full", format="csc")
        else:
            x1 = x1 / (lengthscale)
            x2 = x2 / (lengthscale)
            D2 = distance.sparse_cdist(x1, x2, 1.0, format="csc")
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
    amplitude = misc.array_to_scalar(amplitude)
    lengthscale = misc.array_to_scalar(lengthscale)

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


