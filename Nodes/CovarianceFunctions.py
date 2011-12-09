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
        #print(inputs[0])
        #print(inputs[1])
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


class CovarianceFunction(Node):

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


class SumCF(CovarianceFunction):
    def __init__(self, *args, **kwargs):
        CovarianceFunction.__init__(self,
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


class Delta(CovarianceFunction):
    def __init__(self, amplitude, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_delta,
                                    amplitude,
                                    **kwargs)


class SquaredExponential(CovarianceFunction):
    def __init__(self, amplitude, lengthscale, **kwargs):
        CovarianceFunction.__init__(self,
                                    covfunc_se,
                                    amplitude,
                                    lengthscale,
                                    **kwargs)

class Multiple(CovarianceFunction):
    
    def __init__(self, covfuncs, **kwargs):
        self.d = len(covfuncs)
        parents = [covfunc for row in covfuncs for covfunc in row]
        CovarianceFunction.__init__(self,
                                    None,
                                    *parents,
                                    **kwargs)

    def covariance_function(self, *covfuncs):
        def cov(*inputs, gradient=False):
            #print(inputs)
            if len(inputs) < 2:
                x1 = inputs[0]
                K = [covfuncs[i*self.d+i](x1[i], gradient=gradient)[0] for i in range(self.d)]
                # Form the variance vector from vectors
                if gradient:
                    raise Exception('Gradient not yet implemented.')
                else:
                    K = np.concatenate(K)
            else:
                x1 = inputs[0]
                x2 = inputs[1]
                K = [[covfuncs[i*self.d+j](x1[i], x2[j], gradient=gradient)
                      for j in range(self.d)]
                      for i in range(self.d)]

                if gradient:

                    # Block matrices of zeros
                    #print(K[0][0])
                    Z = [[np.zeros(np.shape(K[i][j][0][0])) for j in range(self.d)]
                         for i in range(self.d)]

                    dK = list()
                    for i in range(self.d):
                        for j in range(self.d):
                            for dk in K[i][j][1]:
                                z_old = Z[i][j]
                                Z[i][j] = dk[0]
                                dk[0] = np.array(np.bmat(Z))
                                Z[i][j] = z_old
                                dK.append(dk)

                    # The full covariance matrix
                    K = [[K[i][j][0][0] for j in range(self.d)]
                              for i in range(self.d)]
                    K = np.array(np.bmat(K))

                else:
                    # Form the matrix from blocks
                    K = [[K[i][j][0] for j in range(self.d)]
                              for i in range(self.d)]
                    K = np.array(np.bmat(K))
                


            if gradient:
                return ([K], dK)
            else:
                return [K]

        return cov



