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
import scipy.linalg as linalg
import scipy.special as special
import scipy.optimize as optimize
import scipy.sparse as sparse

import tempfile as tmp

import unittest
from numpy import testing


def composite_function(function_list):
    """
    Construct a function composition from a list of functions.

    Given a list of functions [f,g,h], constructs a function :math:`h \circ g
    \circ f`.  That is, returns a function :math:`z`, for which :math:`z(x) =
    h(g(f(x)))`.
    """
    def composite(X):
        for function in function_list:
            X = function(X)
        return X
    return composite

    
def ceildiv(a, b):
    """
    Compute a divided by b and rounded up.
    """
    return -(-a // b)

def rmse(y1, y2, axis=None):
    return np.sqrt(np.mean((y1-y2)**2, axis=axis))

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

    def assertAllClose(self, A, B, 
                       msg="Arrays not almost equal", 
                       rtol=1e-4,
                       atol=0):

        self.assertEqual(np.shape(A), np.shape(B), msg=msg)
        testing.assert_allclose(A, B, err_msg=msg, rtol=rtol, atol=atol)
        pass

    def assertArrayEqual(self, A, B, msg="Arrays not equal"):
        self.assertEqual(np.shape(A), np.shape(B), msg=msg)
        testing.assert_array_equal(A, B, err_msg=msg)
        pass

    def assertMessage(self, M1, M2):
        
        if len(M1) != len(M2):
            self.fail("Message lists have different lengths")

        for (m1, m2) in zip(M1, M2):
            self.assertAllClose(m1, m2)

        pass

    def assertMessageToChild(self, X, u):
        self.assertMessage(X._message_to_child(), u)
        pass

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

def identity(*shape):
    return np.reshape(np.identity(np.prod(shape)), shape+shape)

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

def zipper_merge(*lists):
    """
    Combines lists by alternating elements from them.

    Combining lists [1,2,3], ['a','b','c'] and [42,666,99] results in
    [1,'a',42,2,'b',666,3,'c',99]

    The lists should have equal length or they are assumed to have the length of
    the shortest list.

    This is known as alternating merge or zipper merge.
    """
    
    return list(sum(zip(*lists), ()))

def remove_whitespace(s):
    return ''.join(s.split())
    
def is_numeric(a):
    return (np.isscalar(a) or
            isinstance(a, list) or
            isinstance(a, np.ndarray))

def isinteger(x):
    t = np.asanyarray(x).dtype.type
    return ( issubclass(t, np.integer) or issubclass(t, np.bool_) )


def is_string(s):
    return isinstance(s, str)

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


def make_equal_ndim(*arrays):
    """
    Add trailing unit axes so that arrays have equal ndim
    """
    shapes = [np.shape(array) for array in arrays]
    shapes = make_equal_length(*shapes)
    arrays = [np.reshape(array, shape)
              for (array, shape) in zip(arrays, shapes)]
    return arrays


def sum_to_dim(A, dim):
    """
    Sum leading axes of A such that A has dim dimensions.
    """
    dimdiff = np.ndim(A) - dim
    if dimdiff > 0:
        axes = np.arange(dimdiff)
        A = np.sum(A, axis=axes)
    return A


def broadcasting_multiplier(plates, *args):
    """
    Compute the plate multiplier for given shapes.

    The first shape is compared to all other shapes (using NumPy
    broadcasting rules). All the elements which are non-unit in the first
    shape but 1 in all other shapes are multiplied together.

    This method is used, for instance, for computing a correction factor for
    messages to parents: If this node has non-unit plates that are unit
    plates in the parent, those plates are summed. However, if the message
    has unit axis for that plate, it should be first broadcasted to the
    plates of this node and then summed to the plates of the parent. In
    order to avoid this broadcasting and summing, it is more efficient to
    just multiply by the correct factor. This method computes that
    factor. The first argument is the full plate shape of this node (with
    respect to the parent). The other arguments are the shape of the message
    array and the plates of the parent (with respect to this node).
    """

    # Check broadcasting of the shapes
    for arg in args:
        broadcasted_shape(plates, arg)

    # Check that each arg-plates are a subset of plates?
    for arg in args:
        if not is_shape_subset(arg, plates):
            print("Plates:", plates)
            print("Args:", args)
            raise ValueError("The shapes in args are not a sub-shape of "
                             "plates")

    r = 1
    for j in range(-len(plates),0):
        mult = True
        for arg in args:
            # if -j <= len(arg) and arg[j] != 1:
            if not (-j > len(arg) or arg[j] == 1):
                mult = False
        if mult:
            r *= plates[j]
    return r


def sum_multiply_to_plates(*arrays, to_plates=(), from_plates=None, ndim=0):
    """
    Compute the product of the arguments and sum to the target shape.
    """
    arrays = list(arrays)
    def get_plates(x):
        if ndim == 0:
            return x
        else:
            return x[:-ndim]

    plates_arrays = [get_plates(np.shape(array)) for array in arrays]
    product_plates = broadcasted_shape(*plates_arrays)

    if from_plates is None:
        from_plates = product_plates
        r = 1
    else:
        r = broadcasting_multiplier(from_plates, product_plates, to_plates)

    for ind in range(len(arrays)):
        plates_others = plates_arrays[:ind] + plates_arrays[(ind+1):]
        plates_without = broadcasted_shape(to_plates, *plates_others)
        ax = axes_to_collapse(plates_arrays[ind], #get_plates(np.shape(arrays[ind])),
                              plates_without)
        if ax:
            ax = tuple([a-ndim for a in ax])
            arrays[ind] = np.sum(arrays[ind], axis=ax, keepdims=True)

    plates_arrays = [get_plates(np.shape(array)) for array in arrays]
    product_plates = broadcasted_shape(*plates_arrays)

    ax = axes_to_collapse(product_plates, to_plates)
    if ax:
        ax = tuple([a-ndim for a in ax])
        y = sum_multiply(*arrays, axis=ax, keepdims=True)
    else:
        y = functools.reduce(np.multiply, arrays)
    y = squeeze_to_dim(y, len(to_plates) + ndim)
    return r * y


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
    """
    Move the axis `axis_from` to position `axis_to`. 
    """
    if ((axis_from < 0 and abs(axis_from) > np.ndim(A)) or
        (axis_from >= 0 and axis_from >= np.ndim(A)) or
        (axis_to < 0 and abs(axis_to) > np.ndim(A)) or
        (axis_to >= 0  and axis_to >= np.ndim(A))):

        raise ValueError("Can't move axis %d to position %d. Axis index out of "
                         "bounds for array with shape %s"
                         % (axis_from,
                            axis_to,
                            np.shape(A)))
                            
    axes = np.arange(np.ndim(A))
    axes[axis_from:axis_to] += 1
    axes[axis_from:axis_to:-1] -= 1
    axes[axis_to] = axis_from
    return np.transpose(A, axes=axes)


def safe_indices(inds, shape):
    """
    Makes sure that indices are valid for given shape.

    The shorter shape determines the length.

    For instance,

    .. testsetup::

       from bayespy.utils.misc import safe_indices

    >>> safe_indices( (3, 4, 5), (1, 6) )
    (0, 5)
    """
    m = min(len(inds), len(shape))

    if m == 0:
        return ()

    inds = inds[-m:]
    maxinds = np.array(shape[-m:]) - 1

    return tuple(np.fmin(inds, maxinds))


def broadcasted_shape(*shapes):
    """
    Computes the resulting broadcasted shape for a given set of shapes.

    Uses the broadcasting rules of NumPy.  Raises an exception if the shapes do
    not broadcast.
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

def broadcasted_shape_from_arrays(*arrays):
    """
    Computes the resulting broadcasted shape for a given set of arrays.

    Raises an exception if the shapes do not broadcast.
    """

    shapes = [np.shape(array) for array in arrays]
    return broadcasted_shape(*shapes)


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


def add_axes(X, num=1, axis=0):
    for i in range(num):
        X = np.expand_dims(X, axis=axis)
    return X
    shape = np.shape(X)[:axis] + num*(1,) + np.shape(X)[axis:]
    return np.reshape(X, shape)

    
def add_leading_axes(x, n):
    return add_axes(x, axis=0, num=n)
    

def add_trailing_axes(x, n):
    return add_axes(x, axis=-1, num=n)


## def add_axes(x, lead, trail):
##     shape = (1,)*lead + np.shape(x) + (1,)*trail
##     return np.reshape(x, shape)
    
    

def nested_iterator(max_inds):
    s = [range(i) for i in max_inds]
    return itertools.product(*s)

def first(L):
    """
    """
    for (n,l) in enumerate(L):
        if l:
            return n
    return None

def squeeze(X):
    """
    Remove leading axes that have unit length.

    For instance, a shape (1,1,4,1,3) will be reshaped to (4,1,3).
    """
    shape = np.array(np.shape(X))
    inds = np.nonzero(shape != 1)[0]
    if len(inds) == 0:
        shape = ()
    else:
        shape = shape[inds[0]:]
    return np.reshape(X, shape)
    
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

def sum_to_shape(X, s):
    """
    Sum axes of the array such that the resulting shape is as given.

    Thus, the shape of the result will be s or an error is raised.
    """
    # First, sum and remove axes that are not in s
    if np.ndim(X) > len(s):
        axes = tuple(range(-np.ndim(X), -len(s)))
    else:
        axes = ()
    Y = np.sum(X, axis=axes)

    # Second, sum axes that are 1 in s but keep the axes
    axes = ()
    for i in range(-np.ndim(Y), 0):
        if s[i] == 1:
            if np.shape(Y)[i] > 1:
                axes = axes + (i,)
        else:
            if np.shape(Y)[i] != s[i]:
                raise ValueError("Shape %s can't be summed to shape %s" %
                                 (np.shape(X), s))
    Y = np.sum(Y, axis=axes, keepdims=True)
    
    return Y

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


def multidigamma(a, d):
    """
    Returns the derivative of the log of multivariate gamma.
    """
    return np.sum(special.digamma(a[...,None] - 0.5*np.arange(d)),
                  axis=-1)

m_digamma = multidigamma

def m_outer(A,B):
    # Computes outer product over the last axes of A and B. The other
    # axes are broadcasted. Thus, if A has shape (..., N) and B has
    # shape (..., M), then the result has shape (..., N, M)
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    return A[...,np.newaxis]*B[...,np.newaxis,:]

def diagonal(A):
    return np.diagonal(A, axis1=-2, axis2=-1)

def get_diag(X, ndim=1):
    """
    Get the diagonal of an array.

    If ndim>1, take the diagonal of the last 2*ndim axes.
    """
    if ndim == 0:
        return X

    if ndim < 0:
        raise ValueError("Parameter ndim must be non-negative integer")

    if np.ndim(X) < 2*ndim:
        raise ValueError("The array does not have enough axes")

    if np.shape(X)[-ndim:] != np.shape(X)[-2*ndim:-ndim]:
        raise ValueError("The array X is not square")

    axes_out = tuple(range(np.ndim(X)-ndim, 0, -1))
    axes_dim = tuple(range(ndim, 0, -1))
    return np.einsum(X, axes_out+axes_dim, axes_out)
    

def diag(X, ndim=1):
    """
    Create a diagonal array given the diagonal elements.

    The diagonal array can be multi-dimensional. By default, the last axis is
    transformed to two axes (diagonal matrix) but this can be changed using ndim
    keyword. For instance, an array with shape (K,L,M,N) can be transformed to a
    set of diagonal 4-D tensors with shape (K,L,M,N,M,N) by giving ndim=2. If
    ndim=3, the result has shape (K,L,M,N,L,M,N), and so on.

    Diagonality means that for the resulting array Y holds:
    Y[...,i_1,i_2,..,i_ndim,j_1,j_2,..,j_ndim] is zero if i_n!=j_n for any n.
    """
    X = atleast_nd(X, ndim)
    if ndim > 0:
        I = identity(*(np.shape(X)[-ndim:]))
        X = add_axes(X, axis=np.ndim(X), num=ndim)
        X = I * X
    return X

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
        
    
def dist_haversine(c1, c2, radius=6372795):

    # Convert coordinates to radians
    lat1 = np.atleast_1d(c1[0])[...,:,None] * np.pi / 180
    lon1 = np.atleast_1d(c1[1])[...,:,None] * np.pi / 180
    lat2 = np.atleast_1d(c2[0])[...,None,:] * np.pi / 180
    lon2 = np.atleast_1d(c2[1])[...,None,:] * np.pi / 180

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    A = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    C = 2 * np.arctan2(np.sqrt(A), np.sqrt(1-A))
    
    return radius * C

def logsumexp(X, axis=None, keepdims=False):
    """
    Compute log(sum(exp(X)) in a numerically stable way
    """

    X = np.asanyarray(X)
    
    maxX = np.amax(X, axis=axis, keepdims=True)

    if np.ndim(maxX) > 0:
        maxX[~np.isfinite(maxX)] = 0
    elif not np.isfinite(maxX):
        maxX = 0

    X = X - maxX

    if not keepdims:
        maxX = np.squeeze(maxX, axis=axis)

    return np.log(np.sum(np.exp(X), axis=axis, keepdims=keepdims)) + maxX

def mean(X, axis=None, keepdims=False):
    """
    Compute the mean, ignoring NaNs.
    """
    if np.ndim(X) == 0:
        if axis is not None:
            raise ValueError("Axis out of bounds")
        return X
    X = np.asanyarray(X)
    nans = np.isnan(X)
    X = X.copy()
    X[nans] = 0
    m = (np.sum(X, axis=axis, keepdims=keepdims) / 
         np.sum(~nans, axis=axis, keepdims=keepdims))
    return m
