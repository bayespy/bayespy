######################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
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

"""
General numerical functions and methods.

"""
import functools
import itertools

import numpy as np
import scipy as sp
#import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.optimize as optimize
import scipy.sparse as sparse
#import scikits.sparse.cholmod as cholmod

import tempfile as tmp

import unittest
from numpy import testing

def is_callable(f):
    return hasattr(f, '__call__')

def atleast_nd(X, d):
    if np.ndim(X) < d:
        sh = (d-np.ndim(X))*(1,) + np.shape(X)
        X = np.reshape(X, sh)
    return X

def T(X):
    """
    Transpose the matrix.
    """
    return np.swapaxes(X, -1, -2)

class TestCase(unittest.TestCase):
    """
    Simple base class for unit testing.

    Adds NumPy's features to Python's unittest.
    """

    def assertAllClose(self, A, B, msg="Arrays not almost equal"):

        self.assertEqual(np.shape(A), np.shape(B), msg=msg)
        
        testing.assert_allclose(A, B, err_msg=msg)

def symm(X):
    """
    Make X symmetric.
    """
    return 0.5 * (X + np.swapaxes(X, -1, -2))

def unique(l):
    """
    Remove duplicate items from a list while preserving order.
    """
    seen = set()
    seen_add = seen.add
    return [ x for x in l if x not in seen and not seen_add(x)]    

def tempfile(prefix='', suffix=''):
    return tmp.NamedTemporaryFile(prefix=prefix, suffix=suffix).name

def write_to_hdf5(group, data, name):
    """
    Writes the given array into the HDF5 file.
    """
    try:
        # Try using compression. It doesn't work for scalars.
        group.create_dataset(name, 
                             data=data, 
                             compression='gzip')
    except TypeError:
        group.create_dataset(name, 
                             data=data)
    except ValueError:
        raise ValueError('Could not write %s' % data)


def nans(size=()):
    return np.tile(np.nan, size)

def trues(shape):
    return np.ones(shape, dtype=np.bool)

def array_to_scalar(x):
    # This transforms an N-dimensional array to a scalar. It's most
    # useful when you know that the array has only one element and you
    # want it out as a scalar.
    return np.ravel(x)[0]

#def diag(x):

def grid(x1, x2):
    """ Returns meshgrid as a (M*N,2)-shape array. """
    (X1, X2) = np.meshgrid(x1, x2)
    return np.hstack((X1.reshape((-1,1)),X2.reshape((-1,1))))


class CholeskyDense():
    
    def __init__(self, K):
        self.U = linalg.cho_factor(K)
    
    def solve(self, b):
        if sparse.issparse(b):
            b = b.toarray()
        return linalg.cho_solve(self.U, b)

    def logdet(self):
        return 2*np.sum(np.log(np.diag(self.U[0])))

    def trace_solve_gradient(self, dK):
        return np.trace(self.solve(dK))

class CholeskySparse():
    
    def __init__(self, K):
        self.LD = cholmod.cholesky(K)

    def solve(self, b):
        if sparse.issparse(b):
            b = b.toarray()
        return self.LD.solve_A(b)

    def logdet(self):
        return self.LD.logdet()
        #np.sum(np.log(LD.D()))

    def trace_solve_gradient(self, dK):
        # WTF?! numpy.multiply doesn't work for two sparse
        # matrices.. It returns a result but it is incorrect!
        
        # Use the identity trace(K\dK)=sum(inv(K).*dK) by computing
        # the sparse inverse (lower triangular part)
        iK = self.LD.spinv(form='lower')
        return (2*iK.multiply(dK).sum()
                - iK.diagonal().dot(dK.diagonal()))
        # Multiply by two because of symmetry (remove diagonal once
        # because it was taken into account twice)
        #return np.multiply(self.LD.inv().todense(),dK.todense()).sum()
        #return self.LD.inv().multiply(dK).sum() # THIS WORKS
        #return np.multiply(self.LD.inv(),dK).sum() # THIS NOT WORK!! WTF??
        iK = self.LD.spinv()
        return iK.multiply(dK).sum()
        #return (2*iK.multiply(dK).sum()
        #        - iK.diagonal().dot(dK.diagonal()))
        #return (2*np.multiply(iK, dK).sum()
        #        - iK.diagonal().dot(dK.diagonal())) # THIS NOT WORK!!
        #return np.trace(self.solve(dK))
    
    
def cholesky(K):
    if isinstance(K, np.ndarray):
        return CholeskyDense(K)
    elif sparse.issparse(K):
        return CholeskySparse(K)
    else:
        raise Exception("Unsupported covariance matrix type")
    
def vb_optimize(x0, set_values, lowerbound, gradient=None):
    # Function for computing the lower bound
    def func(x):
        # Set the value of the nodes
        set_values(x)
        # Compute lower bound (and gradient terms)
        return -lowerbound()
        #return f

    # Function for computing the gradient of the lower bound
    def funcprime(x):
        # Collect the gradients from the nodes
        set_values(x)
        # Compute lower bound (and gradient terms)
        #lowerbound()
        return -gradient()
        #return df

    # Optimize
    if gradient != None:
        check_gradient(x0, func, funcprime, 1e-6)

        xopt = optimize.fmin_bfgs(func, x0, fprime=funcprime, maxiter=100)
        #xopt = optimize.fmin_ncg(func, x0, fprime=funcprime, maxiter=50)
    else:
        xopt = optimize.fmin_bfgs(func, x0, maxiter=100)
        #xopt = optimize.fmin_ncg(func, x0, maxiter=50)

    # Set optimal values to the nodes
    set_values(xopt)
    

# Optimizes the parameters of the given nodes.
def vb_optimize_nodes(*nodes):

    # Get cost functions
    lbs = set()
    for node in nodes:
        # Add node's cost function
        lbs |= node.get_all_vb_terms()
        #.lower_bound_contribution)
        # Add child nodes' cost functions
        #for lb in node.get_children_vb_bound():
            #lbs.add(lb)

    # Uniqify nodes?
    nodes = set(nodes)

    # Get initial value and transformation/update function
    ind = 0
    ind_all = list()
    transform_all = list()
    gradient_all = list()
    x0_all = np.array([])
    for node in nodes:
        (x0, transform, gradient) = node.start_optimization()
        # Vector of initial values
        x0 = np.atleast_1d(x0)
        x0_all = np.concatenate((x0_all, x0))
        # Indices of the vector elements that correspond to this node
        sz = np.size(x0)
        ind_all.append((ind, ind+sz))
        ind += sz
        # Function for setting the value of this node
        transform_all.append(transform)
        # Gradients
        gradient_all.append(gradient)

    # Function for changing the values of the nodes
    def set_value(x):
        for (ind, transform) in zip(ind_all, transform_all):
            # Transform/update variable
            transform(x[ind[0]:ind[1]])

    # Compute the lower bound (and the gradient)
    def lowerbound():
        l = 0
        # TODO: Put gradients to zero!
        for lb in lbs:
            l += lb(gradient=False)
        return l

    # Compute (or get) the gradient
    def gradient():
        for lb in lbs:
            lb(gradient=True)
        dl = np.zeros(np.shape(x0_all))
        for (ind, gradient) in zip(ind_all, gradient_all):
            dl[ind[0]:ind[1]] = gradient()

        return dl
            
    #vb_optimize(x0_all, set_value, lowerbound)
    vb_optimize(x0_all, set_value, lowerbound, gradient=gradient)
        
    for node in nodes:
        node.stop_optimization()
        
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


def check_gradient(x0, f, df, eps):
    #f0 = f(x0)
    grad = np.zeros(np.shape(x0))
    
    for ind in range(len(x0)):
        xmin = x0.copy()
        xmax = x0.copy()
        xmin[ind] -= eps
        xmax[ind] += eps
        
        fmin = f(xmin)
        fmax = f(xmax)

        grad[ind] = (fmax-fmin) / (2*eps)

    print('x: ' + str(x0))
    print('Numerical gradient: ' + str(grad))
    print('Exact gradient: ' + str(df(x0)))
    

def is_numeric(a):
    return (np.isscalar(a) or
            isinstance(a, list) or
            isinstance(a, np.ndarray))

def multiply_shapes(*shapes):
    """
    Compute element-wise product of lists/tuples.

    Shorter lists are concatenated with leading 1s in order to get lists with
    the same length.
    """

    # Make the shapes equal length
    shapes = make_equal_length(*shapes)

    # Compute element-wise product
    f = lambda X,Y: (x*y for (x,y) in zip(X,Y))
    shape = functools.reduce(f, shapes)

    return tuple(shape)

def make_equal_length(*shapes):
    """
    Make tuples equal length.

    Add leading 1s to shorter tuples.
    """
    
    # Get maximum length
    max_len = max((len(shape) for shape in shapes))

    # Make the shapes equal length
    shapes = ((1,)*(max_len-len(shape)) + tuple(shape) for shape in shapes)

    return shapes

def sum_to_dim(A, dim):
    """
    Sum leading axes of A such that A has dim dimensions.
    """
    dimdiff = np.ndim(A) - dim
    if dimdiff > 0:
        axes = np.arange(dimdiff)
        A = np.sum(A, axis=axes)
    return A

def sum_multiply(*args, axis=None, sumaxis=True, keepdims=False):

    # Computes sum(arg[0]*arg[1]*arg[2]*..., axis=axes_to_sum) without
    # explicitly computing the intermediate product

    if len(args) == 0:
        raise ValueError("You must give at least one input array")

    # Dimensionality of the result
    max_dim = 0
    for k in range(len(args)):
        max_dim = max(max_dim, np.ndim(args[k]))

    if sumaxis:
        if axis is None:
            # Sum all axes
            axes = []
        else:
            if np.isscalar(axis):
                axis = [axis]
            axes = [i
                    for i in range(max_dim)
                    if i not in axis and (-max_dim+i) not in axis]
    else:
        if axis is None:
            # Keep all axes
            axes = range(max_dim)
        else:
            # Find axes that are kept
            if np.isscalar(axis):
                axes = [axis]
            axes = [i if i >= 0
                    else i+max_dim
                    for i in axis]
            axes = sorted(axes)

    if len(axes) > 0 and (min(axes) < 0 or max(axes) >= max_dim):
        raise ValueError("Axis index out of bounds")

    # Form a list of pairs: the array in the product and its axes
    pairs = list()
    for i in range(len(args)):
        a = args[i]
        a_dim = np.ndim(a)
        pairs.append(a)
        pairs.append(range(max_dim-a_dim, max_dim))

    # Output axes are those which are not summed
    pairs.append(axes)

    # Compute the sum-product
    try:
        y = np.einsum(*pairs)
    except ValueError as err:
        if str(err) == ("If 'op_axes' or 'itershape' is not NULL in "
                        "theiterator constructor, 'oa_ndim' must be greater "
                        "than zero"):
            # TODO/FIXME: Handle a bug in NumPy. If all arguments to einsum are
            # scalars, it raises an error. For scalars we can just use multiply
            # and forget about summing. Hopefully, in the future, einsum handles
            # scalars properly and this try-except becomes unnecessary.
            y = functools.reduce(np.multiply, args)
        else:
            raise err

    # Restore summed axes as singleton axes
    if keepdims:
        d = 0
        s = ()
        for k in range(max_dim):
            if k in axes:
                # Axis not summed
                s = s + (np.shape(y)[d],)
                d += 1
            else:
                # Axis was summed
                s = s + (1,)
        y = np.reshape(y, s)

    return y

def sum_product(*args, axes_to_keep=None, axes_to_sum=None, keepdims=False):
    if axes_to_keep is not None:
        return sum_multiply(*args, 
                            axis=axes_to_keep, 
                            sumaxis=False,
                            keepdims=keepdims)
    else:
        return sum_multiply(*args, 
                            axis=axes_to_sum, 
                            sumaxis=True,
                            keepdims=keepdims)

def moveaxis(A, axis_from, axis_to):

    """ Move the axis number axis_from to be the axis number axis_to. """
    axes = np.arange(np.ndim(A))
    axes[axis_from:axis_to] += 1
    axes[axis_from:axis_to:-1] -= 1
    axes[axis_to] = axis_from
    return np.transpose(A, axes=axes)
    


def broadcasted_shape_from_arrays(*args):

    """ Computes the resulting shape if shapes a and b are broadcasted
    together. """

    # The dimensionality (i.e., number of axes) of the result
    dim = 0
    for a in args:
        dim = max(dim, np.ndim(a))
    S = ()
    for i in range(-dim,0):
        s = 1
        for a in args:
            if -i <= np.ndim(a):
                if s == 1:
                    s = np.shape(a)[i]
                elif np.shape(a)[i] != 1 and np.shape(a)[i] != s:
                    raise Exception("Shapes do not broadcast")
        S = S + (s,)
    return S

def is_shape_subset(sub_shape, full_shape):
    """
    """
    if len(sub_shape) > len(full_shape):
        return False
    for i in range(len(sub_shape)):
        ind = -1 - i
        if sub_shape[ind] != 1 and sub_shape[ind] != full_shape[ind]:
            return False
    return True

def broadcasted_shape(*shapes):
    """
    Get the resulting shape if the given shapes were broadcasted.

    Broadcasting rules of NumPy.
    """
    dim = 0
    for a in shapes:
        dim = max(dim, len(a))
    S = ()
    for i in range(-dim,0):
        s = 1
        for a in shapes:
            if -i <= len(a):
                if s == 1:
                    s = a[i]
                elif a[i] != 1 and a[i] != s:
                    raise ValueError("Shapes %s do not broadcast" % (shapes,))
        S = S + (s,)
    return S


## def broadcasted_shape(a,b):
##     # Computes the resulting shape if shapes a and b are broadcasted
##     # together
##     l_max = max(len(a), len(b))
##     s = ()
##     for i in range(-l_max,0):
##         if -i > len(b):
##             s += (a[i],)
##         elif -i > len(a) or a[i] == 1 or a[i] == b[i]:
##             s += (b[i],)
##         elif b[i] == 1:
##             s += (a[i],)
##         else:
##             raise Exception("Shapes %s and %s do not broadcast" % (a,b))
##     return s

    
def add_leading_axes(x, n):
    shape = (1,)*n + np.shape(x)
    return np.reshape(x, shape)
    
def add_trailing_axes(x, n):
    shape = np.shape(x) + (1,)*n
    return np.reshape(x, shape)

def add_axes(x, lead, trail):
    shape = (1,)*lead + np.shape(x) + (1,)*trail
    return np.reshape(x, shape)
    
    

def nested_iterator(max_inds):
    s = (range(i) for i in max_inds)
    return itertools.product(*s)

def squeeze_to_dim(X, dim):
    s = tuple(range(np.ndim(X)-dim))
    return np.squeeze(X, axis=s)


def axes_to_collapse(shape_x, shape_to):
    # Solves which axes of shape shape_x need to be collapsed in order
    # to get the shape shape_to
    s = ()
    for j in range(-len(shape_x), 0):
        if shape_x[j] != 1:
            if -j > len(shape_to) or shape_to[j] == 1:
                s += (j,)
            elif shape_to[j] != shape_x[j]:
                print('Shape from: ' + str(shape_x))
                print('Shape to: ' + str(shape_to))
                raise Exception('Incompatible shape to squeeze')
    return tuple(s)


def repeat_to_shape(A, s):
    # Current shape
    t = np.shape(A)
    if len(t) > len(s):
        raise Exception("Can't repeat to a smaller shape")
    # Add extra axis
    t = tuple([1]*(len(s)-len(t))) + t
    A = np.reshape(A,t)
    # Repeat
    for i in reversed(range(len(s))):
        if s[i] != t[i]:
            if t[i] != 1:
                raise Exception("Can't repeat non-singular dimensions")
            else:
                A = np.repeat(A, s[i], axis=i)
    return A

#def spinv_chol(L):
    

def chol(C):
    if sparse.issparse(C):
        # Sparse Cholesky decomposition (returns a Factor object)
        return cholmod.cholesky(C)
    else:
        # Dense Cholesky decomposition
        return linalg.cho_factor(C)[0]

def chol_solve(U, b):
    if isinstance(U, np.ndarray):
        if sparse.issparse(b):
            b = b.toarray()
        return linalg.cho_solve((U, False), b)
    elif isinstance(U, cholmod.Factor):
        if sparse.issparse(b):
            b = b.toarray()
        return U.solve_A(b)
    else:
        raise ValueError("Unknown type of Cholesky factor")

def chol_inv(U):
    if isinstance(U, np.ndarray):
        I = np.identity(np.shape(U)[-1])
        return linalg.cho_solve((U, False), I)
    elif isinstance(U, cholmod.Factor):
        raise NotImplementedError
        ## if sparse.issparse(b):
        ##     b = b.toarray()
        ## return U.solve_A(b)
    else:
        raise ValueError("Unknown type of Cholesky factor")

def chol_logdet(U):
    if isinstance(U, np.ndarray):
        return 2*np.sum(np.log(np.diag(U)))
    elif isinstance(U, cholmod.Factor):
        return np.sum(np.log(U.D()))
    else:
        raise ValueError("Unknown type of Cholesky factor")
    
def logdet_chol(U):
    if isinstance(U, np.ndarray):
        return 2*np.sum(np.log(np.diag(U)))
    elif isinstance(U, cholmod.Factor):
        return np.sum(np.log(U.D()))

def m_solve_triangular(U, B, **kwargs):
    # Allocate memory
    U = np.atleast_2d(U)
    B = np.atleast_1d(B)
    sh_u = U.shape[:-2]
    sh_b = B.shape[:-1]
    l_u = len(sh_u)
    l_b = len(sh_b)

    # Check which axis are iterated over with B along with U
    ind_b = [Ellipsis] * l_b
    l_min = min(l_u, l_b)
    jnd_b = tuple(i for i in range(-l_min,0) if sh_b[i]==sh_u[i])

    # Shape of the result (broadcasting rules)
    sh = broadcasted_shape(sh_u, sh_b)
    #out = np.zeros(np.shape(B))
    out = np.zeros(sh + B.shape[-1:])
    ## if out == None:
    ##     # Shape of the result (broadcasting rules)
    ##     sh = broadcasted_shape(sh_u, sh_b)
    ##     #out = np.zeros(np.shape(B))
    ##     out = np.zeros(sh + B.shape[-1:])
    for i in nested_iterator(np.shape(U)[:-2]):

        # The goal is to run triangular solver once for all vectors of
        # B for which the matrices of U are the same (according to the
        # broadcasting rules). Thus, we collect all the axes of B for
        # which U is singleton and form them as a 2-D matrix and then
        # run the solver once.
        
        # Select those axes of B for which U and B are not singleton
        for j in jnd_b:
            ind_b[j] = i[j]
            
        # Collect all the axes for which U is singleton
        b = B[tuple(ind_b) + (Ellipsis,)]

        # Reshape it to a 2-D (or 1-D) array
        orig_shape = b.shape
        if b.ndim > 1:
            b = b.reshape((-1, b.shape[-1]))

        # Ellipsis to all preceeding axes and ellipsis for the last
        # axis:
        if len(ind_b) < len(sh):
            ind_out = (Ellipsis,) + tuple(ind_b) + (Ellipsis,)
        else:
            ind_out = tuple(ind_b) + (Ellipsis,)

        #print('utils.m_solve_triangular', np.shape(U[i]), np.shape(b))
        out[ind_out] = linalg.solve_triangular(U[i],
                                               b.T,
                                               **kwargs).T.reshape(orig_shape)
        #out[ind_out] = out[ind_out].T.reshape(orig_shape)

        
    return out
    
    
def m_chol(C):
    # Computes Cholesky decomposition for a collection of matrices.
    # The last two axes of C are considered as the matrix.
    C = np.atleast_2d(C)
    U = np.empty(np.shape(C))
    #print('m_chol', C)
    for i in nested_iterator(np.shape(U)[:-2]):
        try:
            U[i] = linalg.cho_factor(C[i])[0]
        except np.linalg.linalg.LinAlgError:
            print(C[i])
            raise Exception("Matrix not positive definite")
    return U


def m_chol_solve(U, B, out=None):

    
    # Allocate memory
    U = np.atleast_2d(U)
    B = np.atleast_1d(B)
    sh_u = U.shape[:-2]
    sh_b = B.shape[:-1]
    l_u = len(sh_u)
    l_b = len(sh_b)

    # Check which axis are iterated over with B along with U
    ind_b = [Ellipsis] * l_b
    l_min = min(l_u, l_b)
    jnd_b = tuple(i for i in range(-l_min,0) if sh_b[i]==sh_u[i])

    if out == None:
        # Shape of the result (broadcasting rules)
        sh = broadcasted_shape(sh_u, sh_b)
        #out = np.zeros(np.shape(B))
        out = np.zeros(sh + B.shape[-1:])
    for i in nested_iterator(np.shape(U)[:-2]):

        # The goal is to run Cholesky solver once for all vectors of B
        # for which the matrices of U are the same (according to the
        # broadcasting rules). Thus, we collect all the axes of B for
        # which U is singleton and form them as a 2-D matrix and then
        # run the solver once.
        
        # Select those axes of B for which U and B are not singleton
        for j in jnd_b:
            ind_b[j] = i[j]
            
        # Collect all the axes for which U is singleton
        b = B[tuple(ind_b) + (Ellipsis,)]

        # Reshape it to a 2-D (or 1-D) array
        orig_shape = b.shape
        if b.ndim > 1:
            b = b.reshape((-1, b.shape[-1]))

        # Ellipsis to all preceeding axes and ellipsis for the last
        # axis:
        if len(ind_b) < len(sh):
            ind_out = (Ellipsis,) + tuple(ind_b) + (Ellipsis,)
        else:
            ind_out = tuple(ind_b) + (Ellipsis,)

        out[ind_out] = linalg.cho_solve((U[i], False),
                                        b.T).T.reshape(orig_shape)

        
    return out
    

def m_chol_inv(U):
    # Allocate memory
    V = np.tile(np.identity(np.shape(U)[-1]), np.shape(U)[:-2]+(1,1))
    for i in nested_iterator(np.shape(U)[:-2]):
        V[i] = linalg.cho_solve((U[i], False),
                                V[i],
                                overwrite_b=True) # This would need Fortran order
        
    return V
    

def m_chol_logdet(U):
    # Computes Cholesky decomposition for a collection of matrices.
    return 2*np.sum(np.log(np.einsum('...ii->...i', U)), axis=(-1,))


def m_digamma(a, d):
    y = 0
    for i in range(d):
        y += special.digamma(a + 0.5*(1-i))
    return y

def m_outer(A,B):
    # Computes outer product over the last axes of A and B. The other
    # axes are broadcasted. Thus, if A has shape (..., N) and B has
    # shape (..., M), then the result has shape (..., N, M)
    return A[...,np.newaxis]*B[...,np.newaxis,:]

def diagonal(A):
    return np.diagonal(A, axis1=-2, axis2=-1)

def m_dot(A,b):
    # Compute matrix-vector product over the last two axes of A and
    # the last axes of b.  Other axes are broadcasted. If A has shape
    # (..., M, N) and b has shape (..., N), then the result has shape
    # (..., M)
    
    #b = reshape(b, shape(b)[:-1] + (1,) + shape(b)[-1:])
    #return np.dot(A, b)
    return np.einsum('...ik,...k->...i', A, b)
    # TODO: Use einsum!!
    #return np.sum(A*b[...,np.newaxis,:], axis=(-1,))


def block_banded(D, B):
    """
    Construct a symmetric block-banded matrix.

    `D` contains square diagonal blocks.
    `B` contains super-diagonal blocks.

    The resulting matrix is:

    D[0],   B[0],   0,    0,    ..., 0,        0,        0
    B[0].T, D[1],   B[1], 0,    ..., 0,        0,        0
    0,      B[1].T, D[2], B[2], ..., ...,      ...,      ...
    ...     ...     ...   ...   ..., B[N-2].T, D[N-1],   B[N-1]
    0,      0,      0,    0,    ..., 0,        B[N-1].T, D[N]

    """

    D = [np.atleast_2d(d) for d in D]
    B = [np.atleast_2d(b) for b in B]

    # Number of diagonal blocks
    N = len(D)

    if len(B) != N-1:
        raise ValueError("The number of super-diagonal blocks must contain "
                         "exactly one block less than the number of diagonal "
                         "blocks")

    # Compute the size of the full matrix
    M = 0
    for i in range(N):
        if np.ndim(D[i]) != 2:
            raise ValueError("Blocks must be 2 dimensional arrays")
        d = np.shape(D[i])
        if d[0] != d[1]:
            raise ValueError("Diagonal blocks must be square")
        M += d[0]

    for i in range(N-1):
        if np.ndim(B[i]) != 2:
            raise ValueError("Blocks must be 2 dimensional arrays")
        b = np.shape(B[i])
        if b[0] != np.shape(D[i])[1] or b[1] != np.shape(D[i+1])[0]:
            raise ValueError("Shapes of the super-diagonal blocks do not match "
                             "the shapes of the diagonal blocks")

    A = np.zeros((M,M))
    k = 0

    for i in range(N-1):
        (d0, d1) = np.shape(B[i])
        # Diagonal block
        A[k:k+d0, k:k+d0] = D[i]
        # Super-diagonal block
        A[k:k+d0, k+d0:k+d0+d1] = B[i]
        # Sub-diagonal block
        A[k+d0:k+d0+d1, k:k+d0] = B[i].T

        k += d0
    A[k:,k:] = D[-1]

    return A
    

def block_banded_solve(A, B, y):
    """
    Invert symmetric, banded, positive-definite matrix.

    A contains the diagonal blocks.

    B contains the superdiagonal blocks (their transposes are the
    subdiagonal blocks).

    A and B are lists. The length of B is one smaller.

    The algorithm is basically LU decomposition.

    Computes only the diagonal and super-diagonal blocks of the
    inverse. The true inverse is dense, in general.

    Assume each block has the same size.

    Return:
    * inverse blocks
    * solution to the system
    * log-determinant
    """

    # Number of diagonal blocks
    N = len(A)

    if len(B) != N-1:
        raise ValueError("The number of super-diagonal blocks must be exactly "
                         "one less than the number of diagonal blocks")

    # Compute the size of the full matrix
    D = np.shape(A[0])[0]
    
    V = np.empty((N,D,D))
    C = np.empty((N-1,D,D))
    x = np.empty(np.shape(y))

    #
    # Forward recursion
    #
    
    # In the forward recursion, store the Cholesky factor in V. So you
    # don't need to recompute them in the backward recursion.

    # TODO/FIXME: You could store chol_solve(V[n], B[n]) in forward recursion to
    # C, because it is used in backward recursion too!

    x[0] = y[0]
    V[0] = chol(A[0])
    ldet = chol_logdet(V[0])
    for n in range(N-1):
        # Compute the solution of the system
        x[n+1] = y[n+1] - np.dot(B[n].T, chol_solve(V[n], x[n]))
        # Compute the diagonal block and store the Cholesky factor
        V[n+1] = A[n+1] - np.dot(B[n].T, chol_solve(V[n], B[n]))
        V[n+1] = 0.5*V[n+1] + 0.5*V[n+1].T
        #print('blk bnd solve', n, V[n+1])
        V[n+1] = chol(V[n+1])
        #V[n+1] = chol(A[n+1] - np.dot(B[n].T, chol_solve(V[n], B[n])))
        # Compute the log-det term here, too
        ldet += chol_logdet(V[n+1])

    #
    # Backward recursion
    #
    x[-1] = chol_solve(V[-1], x[-1])
    V[-1] = chol_inv(V[-1])
    for n in reversed(range(N-1)):
        # Compute the solution of the system
        x[n] = chol_solve(V[n], x[n] - np.dot(B[n], x[n+1]))
        # Compute the superdiagonal block of the inverse
        Z = chol_solve(V[n], B[n])
        C[n] = -np.dot(Z, V[n+1])
        # Compute the diagonal block of the inverse
        V[n] = chol_inv(V[n]) + np.dot(np.dot(Z, V[n+1]), Z.T)
        V[n] = 0.5*V[n] + 0.5*V[n].T

    return (V, C, x, ldet)
    

def kalman_filter(y, U, A, V, mu0, Cov0, out=None):
    """
    Perform Kalman filtering to obtain filtered mean and covariance.
    
    The parameters of the process may vary in time, thus they are
    given as iterators instead of fixed values.

    Parameters
    ----------
    y : (N,D) array
        "Normalized" noisy observations of the states, that is, the
        observations multiplied by the precision matrix U (and possibly
        other transformation matrices).
    U : (N,D,D) array or N-list of (D,D) arrays
        Precision matrix (i.e., inverse covariance matrix) of the observation 
        noise for each time instance.
    A : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Dynamic matrix for each time instance.
    V : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Covariance matrix of the innovation noise for each time instance.

    Returns
    -------
    mu : array
        Filtered mean of the states.
    Cov : array
        Filtered covariance of the states.

    See also
    --------
    rts_smoother
    """
    mu = mu0
    Cov = Cov0

    # Allocate memory for the results
    (N,D) = np.shape(y)
    X = np.empty((N,D))
    CovX = np.empty((N,D,D))
    
    # Update step for t=0
    M = np.dot(np.dot(Cov, U[0]), Cov) + Cov
    L = chol(M)
    mu = np.dot(Cov, chol_solve(L, np.dot(Cov,y[0]) + mu))
    Cov = np.dot(Cov, chol_solve(L, Cov))
    X[0,:] = mu
    CovX[0,:,:] = Cov
    
    #for (yn, Un, An, Vn) in zip(y, U, A, V):
    for n in range(len(y)-1): #(yn, Un, An, Vn) in zip(y, U, A, V):
        # Prediction step
        mu = np.dot(A[n], mu)
        Cov = np.dot(np.dot(A[n], Cov), A[n].T) + V[n]
        # Update step
        M = np.dot(np.dot(Cov, U[n+1]), Cov) + Cov
        L = chol(M)
        mu = np.dot(Cov, chol_solve(L, np.dot(Cov,y[n+1]) + mu))
        Cov = np.dot(Cov, chol_solve(L, Cov))

        # Force symmetric covariance (for numeric inaccuracy)
        Cov = 0.5*Cov + 0.5*Cov.T

        # Store results
        X[n+1,:] = mu
        CovX[n+1,:,:] = Cov

    return (X, CovX)


def rts_smoother(mu, Cov, A, V, removethis=None):
    """
    Perform Rauch-Tung-Striebel smoothing to obtain the posterior.

    The function returns the posterior mean and covariance of each
    state. The parameters of the process may vary in time, thus they
    are given as iterators instead of fixed values.

    Parameters
    ----------
    mu : (N,D) array
        Mean of the states from Kalman filter.
    Cov : (N,D,D) array
        Covariance of the states from Kalman filter. 
    A : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Dynamic matrix for each time instance.
    V : (N-1,D,D) array or (N-1)-list of (D,D) arrays
        Covariance matrix of the innovation noise for each time instance.

    Returns
    -------
    mu : array
        Posterior mean of the states.
    Cov : array
        Posterior covariance of the states.

    See also
    --------
    kalman_filter
    """

    N = len(mu)
    #n = N-1

    # Start from the last time instance and smoothen backwards
    x = mu[-1,:]
    Covx = Cov[-1,:,:]
    
    for n in reversed(range(N-1)):#(An, Vn) in zip(reversed(A), reversed(V)):

        #n = n - 1
        #if n <= 0:
        #    break

        # The predicted value of n
        x_p = np.dot(A[n], mu[n,:])
        Cov_p = np.dot(np.dot(A[n], Cov[n,:,:]), A[n].T) + V[n]

        # Temporary variable
        S = np.linalg.solve(Cov_p, np.dot(A[n], Cov[n,:,:]))

        # Smoothed value of n
        x = mu[n,:] + np.dot(S.T, x-x_p)
        Covx = Cov[n,:,:] + np.dot(np.dot(S.T, Covx-Cov_p), S)

        # Force symmetric covariance (for numeric inaccuracy)
        Covx = 0.5*Covx + 0.5*Covx.T

        # Store results
        mu[n,:] = x
        Cov[n,:] = Covx


    return (mu, Cov)
        
    
    
    ## x_p = A*x;
    ## Covx_p = A*Covx*A' + Q;

    ## S = (Covx*A') / Covx_p;
    ## x = x + S*(x_s-x_p);
    ## if nargout >= 2
    ##   Covx = Covx + S*(Covx_s-Covx_p)*S';
    ## end
    pass
