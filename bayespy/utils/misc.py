################################################################################
# Copyright (C) 2011-2013 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
General numerical functions and methods.

"""
import functools
import itertools
import operator

import sys
import getopt

import numpy as np
import scipy as sp
import scipy.linalg as linalg
import scipy.special as special
import scipy.optimize as optimize
import scipy.sparse as sparse

import tempfile as tmp

import unittest
from numpy import testing


def flatten_axes(X, *ndims):
    ndim = sum(ndims)
    if np.ndim(X) < ndim:
        raise ValueError("Not enough ndims in the array")
    if len(ndims) == 0:
        return X
    shape = np.shape(X)
    i = np.ndim(X) - ndim
    plates = shape[:i]
    nd_sums = i + np.cumsum((0,) + ndims)
    sizes = tuple(
        np.prod(shape[i:j])
        for (i, j) in zip(nd_sums[:-1], nd_sums[1:])
    )
    return np.reshape(X, plates + sizes)


def reshape_axes(X, *shapes):
    ndim = len(shapes)
    if np.ndim(X) < ndim:
        raise ValueError("Not enough ndims in the array")
    i = np.ndim(X) - ndim
    sizes = tuple(np.prod(sh) for sh in shapes)
    if np.shape(X)[i:] != sizes:
        raise ValueError("Shapes inconsistent with sizes")
    shape = tuple(i for sh in shapes for i in sh)
    return np.reshape(X, np.shape(X)[:i] + shape)


def find_set_index(index, set_lengths):
    """
    Given set sizes and an index, returns the index of the set

    The given index is for the concatenated list of the sets.
    """
    # Negative indices to positive
    if index < 0:
        index += np.sum(set_lengths)

    # Indices must be on range (0, N-1)
    if index >= np.sum(set_lengths) or index < 0:
        raise Exception("Index out bounds")

    return np.searchsorted(np.cumsum(set_lengths), index, side='right')


def parse_command_line_arguments(mandatory_args, *optional_args_list, argv=None):
    """
    Parse command line arguments of style "--parameter=value".

    Parameter specification is tuple: (name, converter, description).

    Some special handling:

    * If converter is None, the command line does not accept any value
      for it, but instead use either "--option" to enable or
      "--no-option" to disable.

    * If argument name contains hyphens, those are converted to
      underscores in the keys of the returned dictionaries.

    Parameters
    ----------

    mandatory_args : list of tuples
        Specs for mandatory arguments

    optional_args_list : list of lists of tuples
        Specs for each optional arguments set

    argv : list of strings (optional)
        The command line arguments. By default, read sys.argv.

    Returns
    -------

    args : dictionary
        The parsed mandatory arguments

    kwargs : dictionary
        The parsed optional arguments

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from bayespy.utils import misc
    >>> (args, kwargs) = misc.parse_command_line_arguments(
    ...     # Mandatory arguments
    ...     [
    ...         ('name',     str,  "Full name"),
    ...         ('age',      int,  "Age (years)"),
    ...         ('employed', None, "Working"),
    ...     ],
    ...     # Optional arguments
    ...     [
    ...         ('phone',          str, "Phone number"),
    ...         ('favorite-color', str, "Favorite color")
    ...     ],
    ...     argv=['--name=John Doe',
    ...           '--age=42',
    ...           '--no-employed',
    ...           '--favorite-color=pink']
    ... )
    >>> print(args)
    {'age': 42, 'employed': False, 'name': 'John Doe'}
    >>> print(kwargs)
    {'favorite_color': 'pink'}

    It is possible to have several optional argument sets:

    >>> (args, kw_info, kw_fav) = misc.parse_command_line_arguments(
    ...     # Mandatory arguments
    ...     [
    ...         ('name',     str,  "Full name"),
    ...     ],
    ...     # Optional arguments (contact information)
    ...     [
    ...         ('phone', str, "Phone number"),
    ...         ('email', str, "E-mail address")
    ...     ],
    ...     # Optional arguments (preferences)
    ...     [
    ...         ('favorite-color', str, "Favorite color"),
    ...         ('favorite-food',  str, "Favorite food")
    ...     ],
    ...     argv=['--name=John Doe',
    ...           '--favorite-color=pink',
    ...           '--email=john.doe@email.com',
    ...           '--favorite-food=spaghetti']
    ... )
    >>> print(args)
    {'name': 'John Doe'}
    >>> print(kw_info)
    {'email': 'john.doe@email.com'}
    >>> print(kw_fav)
    {'favorite_color': 'pink', 'favorite_food': 'spaghetti'}

    """

    if argv is None:
        argv = sys.argv[1:]

    mandatory_arg_names = [arg[0] for arg in mandatory_args]

    # Sizes of each optional argument list
    optional_args_lengths = [len(opt_args) for opt_args in optional_args_list]

    all_args = mandatory_args + functools.reduce(operator.add, optional_args_list)

    # Create a list of arg names for the getopt parser
    arg_list = []
    for arg in all_args:
        arg_name = arg[0].lower()
        if arg[1] is None:
            arg_list.append(arg_name)
            arg_list.append("no-" + arg_name)
        else:
            arg_list.append(arg_name + "=")

    if len(set(arg_list)) < len(arg_list):
        raise Exception("Argument names are not unique")

    # Use getopt parser
    try:
        (cl_opts, cl_args) = getopt.getopt(argv, "", arg_list)
    except getopt.GetoptError as err:
        print(err)
        print("Usage:")
        for arg in all_args:
            if arg[1] is None:
                print("--{0}\t{1}".format(arg[0].lower(),
                                          arg[2]))
            else:
                print("--{0}=<{1}>\t{2}".format(arg[0].lower(),
                                               str(arg[1].__name__).upper(),
                                               arg[2]))
        sys.exit(2)

    # A list of all valid flag names: ["--first-argument", "--another-argument"]
    valid_flags = []
    valid_flag_arg_indices = []
    for (ind, arg) in enumerate(all_args):
        valid_flags.append("--" + arg[0].lower())
        valid_flag_arg_indices.append(ind)
        if arg[1] is None:
            valid_flags.append("--no-" + arg[0].lower())
            valid_flag_arg_indices.append(ind)

    # Go through all the given command line arguments and store them in the
    # correct dictionaries
    args = dict()
    kwargs_list = [dict() for i in range(len(optional_args_list))]
    handled_arg_names = []
    for (cl_opt, cl_arg) in cl_opts:

        # Get the index of the argument
        try:
            ind = valid_flag_arg_indices[valid_flags.index(cl_opt.lower())]
        except ValueError:
            print("Invalid command line argument: {0}".format(cl_opt))
            raise Exception("Invalid argument given")

        # Check that the argument wasn't already given and then mark the
        # argument as handled
        if all_args[ind][0] in handled_arg_names:
            raise Exception("Same argument given multiple times")
        else:
            handled_arg_names.append(all_args[ind][0])

        # Check whether to add the argument to the mandatory or optional
        # argument dictionary
        if ind < len(mandatory_args):
            dict_to = args
        else:
            dict_index = find_set_index(ind - len(mandatory_args),
                                        optional_args_lengths)
            dict_to = kwargs_list[dict_index]

        # Convert and store the argument
        convert_function = all_args[ind][1]
        arg_name = all_args[ind][0].replace('-', '_')
        if convert_function is None:
            if cl_opt[:5] == "--no-":
                dict_to[arg_name] = False
            else:
                dict_to[arg_name] = True
        else:
            dict_to[arg_name] = convert_function(cl_arg)

    # Check if some mandatory argument was not given
    for arg_name in mandatory_arg_names:
        if arg_name not in handled_arg_names:
            raise Exception("Mandatory argument --{0} not given".format(arg_name))

    return tuple([args] + kwargs_list)


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


def put(x, indices, y, axis=-1, ufunc=np.add):
    """A kind of inverse mapping of `np.take`

    In a simple, the operation can be thought as:

    .. code-block:: python

       x[indices] += y

    with the exception that all entries of `y` are used instead of just the
    first occurence corresponding to a particular element. That is, the results
    are accumulated, and the accumulation function can be changed by providing
    `ufunc`. For instance, `np.multiply` corresponds to:

    .. code-block:: python

       x[indices] *= y

    Whereas `np.take` picks indices along an axis and returns the resulting
    array, `put` similarly picks indices along an axis but accumulates the
    given values to those entries.

    Example
    -------

    .. code-block:: python

       >>> x = np.zeros(3)
       >>> put(x, [2, 2, 0, 2, 2], 1)
       array([ 1.,  0.,  4.])

    `y` must broadcast to the shape of `np.take(x, indices)`:

    .. code-block:: python

       >>> x = np.zeros((3,4))
       >>> put(x, [[2, 2, 0, 2, 2], [1, 2, 1, 2, 1]], np.ones((2,1,4)), axis=0)
       array([[ 1.,  1.,  1.,  1.],
              [ 3.,  3.,  3.,  3.],
              [ 6.,  6.,  6.,  6.]])

    """
    #x = np.copy(x)
    ndim = np.ndim(x)
    if not isinstance(axis, int):
        raise ValueError("Axis must be an integer")

    # Make axis index positive: [0, ..., ndim-1]
    if axis < 0:
        axis = axis + ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("Axis out of bounds")

    indices = axis*(slice(None),) + (indices,) + (ndim-axis-1)*(slice(None),)
    #y = add_trailing_axes(y, ndim-axis-1)
    ufunc.at(x, indices, y)
    return x


def put_simple(y, indices, axis=-1, length=None):
    """An inverse operation of `np.take` with accumulation and broadcasting.

    Compared to `put`, the difference is that the result array is initialized
    with an array of zeros whose shape is determined automatically and `np.add`
    is used as the accumulator.

    """

    if length is None:
        # Try to determine the original length of the axis by finding the
        # largest index. It is more robust to give the length explicitly.
        indices = np.copy(indices)
        indices[indices<0] = np.abs(indices[indices<0]) - 1
        length = np.amax(indices) + 1

    if not isinstance(axis, int):
        raise ValueError("Axis must be an integer")

    # Make axis index negative: [-ndim, ..., -1]
    if axis >= 0:
        raise ValueError("Axis index must be negative")

    y = atleast_nd(y, abs(axis)-1)
    shape_y = np.shape(y)
    end_before = axis - np.ndim(indices) + 1
    start_after = axis + 1
    if end_before == 0:
        shape_x = shape_y + (length,)
    elif start_after == 0:
        shape_x = shape_y[:end_before] + (length,)
    else:
        shape_x = shape_y[:end_before] + (length,) + shape_y[start_after:]

    x = np.zeros(shape_x)

    return put(x, indices, y, axis=axis)


def grid(x1, x2):
    """ Returns meshgrid as a (M*N,2)-shape array. """
    (X1, X2) = np.meshgrid(x1, x2)
    return np.hstack((X1.reshape((-1,1)),X2.reshape((-1,1))))


# class CholeskyDense():

#     def __init__(self, K):
#         self.U = linalg.cho_factor(K)

#     def solve(self, b):
#         if sparse.issparse(b):
#             b = b.toarray()
#         return linalg.cho_solve(self.U, b)

#     def logdet(self):
#         return 2*np.sum(np.log(np.diag(self.U[0])))

#     def trace_solve_gradient(self, dK):
#         return np.trace(self.solve(dK))

# class CholeskySparse():

#     def __init__(self, K):
#         self.LD = cholmod.cholesky(K)

#     def solve(self, b):
#         if sparse.issparse(b):
#             b = b.toarray()
#         return self.LD.solve_A(b)

#     def logdet(self):
#         return self.LD.logdet()
#         #np.sum(np.log(LD.D()))

#     def trace_solve_gradient(self, dK):
#         # WTF?! numpy.multiply doesn't work for two sparse
#         # matrices.. It returns a result but it is incorrect!

#         # Use the identity trace(K\dK)=sum(inv(K).*dK) by computing
#         # the sparse inverse (lower triangular part)
#         iK = self.LD.spinv(form='lower')
#         return (2*iK.multiply(dK).sum()
#                 - iK.diagonal().dot(dK.diagonal()))
#         # Multiply by two because of symmetry (remove diagonal once
#         # because it was taken into account twice)
#         #return np.multiply(self.LD.inv().todense(),dK.todense()).sum()
#         #return self.LD.inv().multiply(dK).sum() # THIS WORKS
#         #return np.multiply(self.LD.inv(),dK).sum() # THIS NOT WORK!! WTF??
#         iK = self.LD.spinv()
#         return iK.multiply(dK).sum()
#         #return (2*iK.multiply(dK).sum()
#         #        - iK.diagonal().dot(dK.diagonal()))
#         #return (2*np.multiply(iK, dK).sum()
#         #        - iK.diagonal().dot(dK.diagonal())) # THIS NOT WORK!!
#         #return np.trace(self.solve(dK))


# def cholesky(K):
#     if isinstance(K, np.ndarray):
#         return CholeskyDense(K)
#     elif sparse.issparse(K):
#         return CholeskySparse(K)
#     else:
#         raise Exception("Unsupported covariance matrix type")

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

def is_scalar_integer(x):
    t = np.asanyarray(x).dtype.type
    return np.ndim(x) == 0 and issubclass(t, np.integer)


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

def multidigamma(a, d):
    """
    Returns the derivative of the log of multivariate gamma.
    """
    return np.sum(special.digamma(a[...,None] - 0.5*np.arange(d)),
                  axis=-1)

m_digamma = multidigamma


def diagonal(A):
    return np.diagonal(A, axis1=-2, axis2=-1)


def make_diag(X, ndim=1, ndim_from=0):
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
    if ndim < 0:
        raise ValueError("Parameter ndim must be non-negative integer")

    if ndim_from < 0:
        raise ValueError("Parameter ndim_to must be non-negative integer")

    if ndim_from > ndim:
        raise ValueError("Parameter ndim_to must not be greater than ndim")

    if ndim == 0:
        return X

    if np.ndim(X) < 2 * ndim_from:
        raise ValueError("The array does not have enough axes")

    if ndim_from > 0:
        if np.shape(X)[-ndim_from:] != np.shape(X)[-2*ndim_from:-ndim_from]:
            raise ValueError("The array X is not square")

    if ndim == ndim_from:
        return X

    X = atleast_nd(X, ndim+ndim_from)

    if ndim > 0:
        if ndim_from > 0:
            I = identity(*(np.shape(X)[-(ndim_from+ndim):-ndim_from]))
        else:
            I = identity(*(np.shape(X)[-ndim:]))
        X = add_axes(X, axis=np.ndim(X)-ndim_from, num=ndim-ndim_from)
        X = I * X
    return X


def get_diag(X, ndim=1, ndim_to=0):
    """
    Get the diagonal of an array.

    If ndim>1, take the diagonal of the last 2*ndim axes.
    """
    if ndim < 0:
        raise ValueError("Parameter ndim must be non-negative integer")

    if ndim_to < 0:
        raise ValueError("Parameter ndim_to must be non-negative integer")

    if ndim_to > ndim:
        raise ValueError("Parameter ndim_to must not be greater than ndim")

    if ndim == 0:
        return X

    if np.ndim(X) < 2*ndim:
        raise ValueError("The array does not have enough axes")

    if np.shape(X)[-ndim:] != np.shape(X)[-2*ndim:-ndim]:
        raise ValueError("The array X is not square")

    if ndim == ndim_to:
        return X

    n_plate_axes = np.ndim(X) - 2 * ndim
    n_diag_axes = ndim - ndim_to

    axes = tuple(range(0, np.ndim(X) - ndim + ndim_to))

    lengths = [0, n_plate_axes, n_diag_axes, ndim_to, ndim_to]
    cutpoints = list(np.cumsum(lengths))

    axes_plates = axes[cutpoints[0]:cutpoints[1]]
    axes_diag= axes[cutpoints[1]:cutpoints[2]]
    axes_dims1 = axes[cutpoints[2]:cutpoints[3]]
    axes_dims2 = axes[cutpoints[3]:cutpoints[4]]

    axes_input = axes_plates + axes_diag + axes_dims1 + axes_diag + axes_dims2
    axes_output = axes_plates + axes_diag + axes_dims1 + axes_dims2

    return np.einsum(X, axes_input, axes_output)


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


def normalized_exp(phi):
    """Compute exp(phi) so that exp(phi) sums to one.

    This is useful for computing probabilities from log evidence.
    """
    logsum_p = logsumexp(phi, axis=-1, keepdims=True)
    logp = phi - logsum_p
    p = np.exp(logp)
    # Because of small numerical inaccuracy, normalize the probabilities
    # again for more accurate results
    return (
        p / np.sum(p, axis=-1, keepdims=True),
        logsum_p
    )


def invpsi(x):
    r"""
    Inverse digamma (psi) function.

    The digamma function is the derivative of the log gamma function.
    This calculates the value Y > 0 for a value X such that digamma(Y) = X.

    For the new version, see Appendix C:
    http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf

    For the previous implementation, see:
    http://www4.ncsu.edu/~pfackler/

    Are there speed/accuracy differences between the methods?
    """
    x = np.asanyarray(x)

    y = np.where(
        x >= -2.22,
        np.exp(x) + 0.5,
        -1/(x - special.psi(1))
    )
    for i in range(5):
        y = y - (special.psi(y) - x) / special.polygamma(1, y)

    return y

    # # Previous implementation. Is it worse? Is there difference?
    # L = 1.0
    # y = np.exp(x)
    # while (L > 1e-10):
    #     y += L*np.sign(x-special.psi(y))
    #     L /= 2
    # # Ad hoc by Jaakko
    # y = np.where(x < -100, -1 / x, y)
    # return y


def invgamma(x):
    r"""
    Inverse gamma function.

    See: http://mathoverflow.net/a/28977
    """
    k = 1.461632
    c = 0.036534
    L = np.log((x+c)/np.sqrt(2*np.pi))
    W = special.lambertw(L/np.exp(1))
    return L/W + 0.5


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


def gradient(f, x, epsilon=1e-6):
    return optimize.approx_fprime(x, f, epsilon)


def broadcast(*arrays, ignore_axis=None):
    """
    Explicitly broadcast arrays to same shapes.

    It is possible ignore some axes so that the arrays are not broadcasted
    along those axes.
    """

    shapes = [np.shape(array) for array in arrays]

    if ignore_axis is None:
        full_shape = broadcasted_shape(*shapes)

    else:
        try:
            ignore_axis = tuple(ignore_axis)
        except TypeError:
            ignore_axis = (ignore_axis,)

        if len(ignore_axis) != len(set(ignore_axis)):
            raise ValueError("Indices must be unique")

        if any(i >= 0 for i in ignore_axis):
            raise ValueError("Indices must be negative")

        # Put lengths of ignored axes to 1
        cut_shapes = [
            tuple(
                1
                if i in ignore_axis else
                shape[i]
                for i in range(-len(shape), 0)
            )
            for shape in shapes
        ]

        full_shape = broadcasted_shape(*cut_shapes)

    return [np.ones(full_shape) * array for array in arrays]


def block_diag(*arrays):
    """
    Form a block diagonal array from the given arrays.

    Compared to SciPy's block_diag, this utilizes broadcasting and accepts more
    than dimensions in the input arrays.

    """

    arrays = broadcast(*arrays, ignore_axis=(-1, -2))

    plates = np.shape(arrays[0])[:-2]

    M = sum(np.shape(array)[-2] for array in arrays)
    N = sum(np.shape(array)[-1] for array in arrays)

    Y = np.zeros(plates + (M, N))

    i_start = 0
    j_start = 0
    for array in arrays:
        i_end = i_start + np.shape(array)[-2]
        j_end = j_start + np.shape(array)[-1]
        Y[...,i_start:i_end,j_start:j_end] = array
        i_start = i_end
        j_start = j_end

    return Y


def concatenate(*arrays, axis=-1):
    """
    Concatenate arrays along a given axis.

    Compared to NumPy's concatenate, this utilizes broadcasting.
    """

    # numpy.concatenate doesn't do broadcasting, so we need to do it explicitly
    return np.concatenate(
        broadcast(*arrays, ignore_axis=axis),
        axis=axis
    )
