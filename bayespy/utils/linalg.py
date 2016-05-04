################################################################################
# Copyright (C) 2011-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
General numerical functions and methods.

"""

import itertools
import numpy as np
import scipy as sp
#import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.optimize as optimize
import scipy.sparse as sparse
#import scikits.sparse.cholmod as cholmod

# THIS IS SOME NEW GENERALIZED UFUNC FOR LINALG FEATURE, NOT IN OFFICIAL NUMPY
# REPO YET
#import numpy.linalg._gufuncs_linalg as gula
#import numpy.core.gufuncs_linalg as gula

from . import misc


def chol(C, ndim=1):
    if sparse.issparse(C):
        if ndim != 1:
            raise NotImplementedError()
        # Sparse Cholesky decomposition (returns a Factor object)
        return cholmod.cholesky(C)
    else:
        # Computes Cholesky decomposition for a collection of matrices.
        # The last 2*ndim axes of C are considered as the matrix.
        if ndim == 0:
            return np.sqrt(C)
        shape_original = np.shape(C)[-ndim:]
        C = misc.flatten_axes(C, ndim, ndim)
        if np.shape(C)[-1] != np.shape(C)[-2]:
            raise ValueError("Not square matrix w.r.t. ndim sense")
        U = np.empty(np.shape(C))
        for i in misc.nested_iterator(np.shape(U)[:-2]):
            try:
                U[i] = linalg.cho_factor(C[i])[0]
            except np.linalg.linalg.LinAlgError:
                raise Exception("Matrix not positive definite")
        return misc.reshape_axes(U, shape_original, shape_original)


def chol_solve(U, b, out=None, matrix=False, ndim=1):
    if isinstance(U, np.ndarray):
        if sparse.issparse(b):
            b = b.toarray()

        if ndim == 0:
            return (b / U) / U

        shape = np.shape(U)[-ndim:]
        U = misc.flatten_axes(U, ndim, ndim)

        if matrix:
            shape_b = np.shape(b)[-ndim:]
            B = misc.flatten_axes(b, ndim, ndim)
            B = transpose(b, ndim=1)
            U = U[...,None,:,:]
        else:
            B = misc.flatten_axes(b, ndim)

        # Allocate memory
        sh_u = U.shape[:-2]
        sh_b = B.shape[:-1]
        l_u = len(sh_u)
        l_b = len(sh_b)

        # Check which axis are iterated over with B along with U
        ind_b = [slice(None)] * l_b
        l_min = min(l_u, l_b)
        jnd_b = tuple(i for i in range(-l_min,0) if sh_b[i]==sh_u[i])

        if out == None:
            # Shape of the result (broadcasting rules)
            sh = misc.broadcasted_shape(sh_u, sh_b)
            #out = np.zeros(np.shape(B))
            out = np.zeros(sh + B.shape[-1:])

        for i in misc.nested_iterator(np.shape(U)[:-2]):

            # The goal is to run Cholesky solver once for all vectors of B
            # for which the matrices of U are the same (according to the
            # broadcasting rules). Thus, we collect all the axes of B for
            # which U is singleton and form them as a 2-D matrix and then
            # run the solver once.

            # Select those axes of B for which U and B are not singleton
            for j in jnd_b:
                ind_b[j] = i[j]

            # Collect all the axes for which U is singleton
            b = B[tuple(ind_b) + (slice(None),)]

            # Reshape it to a 2-D (or 1-D) array
            orig_shape = b.shape
            if b.ndim > 1:
                b = b.reshape((-1, b.shape[-1]))

            # slice(None) to all preceeding axes and ellipsis for the last
            # axis:
            if len(ind_b) < len(sh):
                ind_out = (slice(None),) + tuple(ind_b) + (slice(None),)
            else:
                ind_out = tuple(ind_b) + (slice(None),)

            out[ind_out] = linalg.cho_solve((U[i], False),
                                            b.T).T.reshape(orig_shape)

        if matrix:
            out = transpose(out, ndim=1)
            out = misc.reshape_axes(out, shape, shape_b)
        else:
            out = misc.reshape_axes(out, shape)

        return out

    elif isinstance(U, cholmod.Factor):
        if ndim != 1:
            raise NotImplementedError()
        if matrix:
            raise NotImplementedError()
        if sparse.issparse(b):
            b = b.toarray()
        return U.solve_A(b)
    else:
        raise ValueError("Unknown type of Cholesky factor")


def chol_inv(U, ndim=1):
    if isinstance(U, np.ndarray):
        if ndim == 0:
            return (1 / U) / U
        shape = np.shape(U)[-ndim:]
        U = misc.flatten_axes(U, ndim, ndim)
        # Allocate memory
        V = np.tile(np.identity(np.shape(U)[-1]), np.shape(U)[:-2]+(1,1))
        for i in misc.nested_iterator(np.shape(U)[:-2]):
            V[i] = linalg.cho_solve(
                (U[i], False),
                V[i],
                overwrite_b=True # This would need Fortran order
            )
        V = misc.reshape_axes(V, shape, shape)
        return V

    elif isinstance(U, cholmod.Factor):
        raise NotImplementedError
        if ndim != 1:
            raise NotImplementedError()
    else:
        raise ValueError("Unknown type of Cholesky factor")

def chol_logdet(U, ndim=1):
    if isinstance(U, np.ndarray):
        if ndim == 0:
            return 2 * np.log(U)
        U = misc.flatten_axes(U, ndim, ndim)
        return 2*np.sum(np.log(np.einsum('...ii->...i',U)), axis=-1)
    elif isinstance(U, cholmod.Factor):
        if ndim != 1:
            raise NotImplementedError()
        return np.sum(np.log(U.D()))
    else:
        raise ValueError("Unknown type of Cholesky factor")

def logdet_chol(U):
    if isinstance(U, np.ndarray):
        # Computes Cholesky decomposition for a collection of matrices.
        return 2*np.sum(np.log(np.einsum('...ii->...i', U)), axis=(-1,))
    elif isinstance(U, cholmod.Factor):
        return np.sum(np.log(U.D()))


def logdet_tri(R):
    """
    Logarithm of the absolute value of the determinant of a triangular matrix.
    """
    return np.sum(np.log(np.abs(np.einsum('...ii->...i', R))))


def logdet_cov(C, ndim=1):
    return chol_logdet(chol(C, ndim=ndim), ndim=ndim)


def solve_triangular(U, B, ndim=1, **kwargs):
    if ndim != 1:
        raise NotImplementedError("Not yet implemented for ndim!=1")
    # Allocate memory
    U = np.atleast_2d(U)
    B = np.atleast_1d(B)
    sh_u = U.shape[:-2]
    sh_b = B.shape[:-1]
    l_u = len(sh_u)
    l_b = len(sh_b)

    # Check which axis are iterated over with B along with U
    ind_b = [slice(None)] * l_b
    l_min = min(l_u, l_b)
    jnd_b = tuple(i for i in range(-l_min,0) if sh_b[i]==sh_u[i])

    # Shape of the result (broadcasting rules)
    sh = misc.broadcasted_shape(sh_u, sh_b)
    out = np.zeros(sh + B.shape[-1:])
    for i in misc.nested_iterator(np.shape(U)[:-2]):

        # The goal is to run triangular solver once for all vectors of
        # B for which the matrices of U are the same (according to the
        # broadcasting rules). Thus, we collect all the axes of B for
        # which U is singleton and form them as a 2-D matrix and then
        # run the solver once.
        
        # Select those axes of B for which U and B are not singleton
        for j in jnd_b:
            ind_b[j] = i[j]
            
        # Collect all the axes for which U is singleton
        b = B[tuple(ind_b) + (slice(None),)]

        # Reshape it to a 2-D (or 1-D) array
        orig_shape = b.shape
        if b.ndim > 1:
            b = b.reshape((-1, b.shape[-1]))

        # slice(None) to all preceeding axes and ellipsis for the last
        # axis:
        if len(ind_b) < len(sh):
            ind_out = (slice(None),) + tuple(ind_b) + (slice(None),)
        else:
            ind_out = tuple(ind_b) + (slice(None),)

        out[ind_out] = linalg.solve_triangular(U[i],
                                               b.T,
                                               **kwargs).T.reshape(orig_shape)
        
    return out
    

    

def inner(*args, ndim=1):
    """
    Compute inner product.

    The number of arrays is arbitrary.  The number of dimensions is arbitrary.
    """
    axes = tuple(range(-ndim,0))
    return misc.sum_product(*args, axes_to_sum=axes)


def outer(A, B, ndim=1):
    """
    Computes outer product over the last axes of A and B.

    The other axes are broadcasted. Thus, if A has shape (..., N) and B has
    shape (..., M), then the result has shape (..., N, M).

    Using the argument `ndim` it is possible to change that how many axes
    trailing axes are used for the outer product. For instance, if ndim=3, A and
    B have shapes (...,N1,N2,N3) and (...,M1,M2,M3), the result has shape
    (...,N1,M1,N2,M2,N3,M3).
    """
    if not isinstance(ndim, int) or ndim < 0:
        raise ValueError('ndim must be non-negative integer')
    if ndim > 0:
        if ndim > np.ndim(A):
            raise ValueError('Argument ndim larger than ndim of the first '
                             'parameter')
        if ndim > np.ndim(B):
            raise ValueError('Argument ndim larger than ndim of the second '
                             'parameter')
        shape_A = np.shape(A) + (1,)*ndim
        shape_B = np.shape(B)[:-ndim] + (1,)*ndim + np.shape(B)[-ndim:]
        A = np.reshape(A, shape_A)
        B = np.reshape(B, shape_B)
    return np.asanyarray(A) * np.asanyarray(B)


def _dot(A, B):
    """
    Dot product which handles broadcasting properly.

    Future NumPy will have a better built-in implementation for this.
    """
    A_plates = np.shape(A)[:-2]
    B_plates = np.shape(B)[:-2]
    M = np.shape(A)[-2]
    N = np.shape(B)[-1]
    Y_plates = misc.broadcasted_shape(A_plates, B_plates)
    if Y_plates == ():
        return np.dot(A, B)
    indices = misc.nested_iterator(Y_plates)
    Y_shape = Y_plates + (M, N)
    Y = np.zeros(Y_shape)
    for i in indices:
        Y[i] = np.dot(A[misc.safe_indices(i, A_plates)],
                      B[misc.safe_indices(i, B_plates)])
    return Y

def dot(*arrays):
    """
    Compute matrix-matrix product.

    You can give multiple arrays, the dot product is computed from left to
    right: A1*A2*A3*...*AN. The dot product is computed over the last two axes
    of each arrays. All other axes must be broadcastable.
    """
    if len(arrays) == 0:
        return 0
    else:
        Y = np.asanyarray(arrays[0])
        for X in arrays[1:]:
            X = np.asanyarray(X)
            if np.ndim(Y) < 2 or np.ndim(X) < 2:
                raise ValueError("Must be at least 2-D arrays")
            if np.shape(Y)[-1] != np.shape(X)[-2]:
                raise ValueError("Dimensions do not match")
            # Replace this with numpy.dot when NumPy implements broadcasting in dot
            Y = _dot(Y, X)
            #Y = np.einsum('...ik,...kj->...ij', Y, X)
            #Y = gula.matrix_multiply(Y, X)
        return Y

def tracedot(A, B):
    """
    Computes trace(A*B).
    """
    return np.einsum('...ij,...ji->...', A, B)


def inv(A, ndim=1):
    """
    General array inversion.

    Supports broadcasting and inversion of multidimensional arrays.  For
    instance, an array with shape (4,3,2,3,2) could mean that there are four
    (3*2) x (3*2) matrices to be inverted. This can be done by inv(A, ndim=2).
    For inverting scalars, ndim=0. For inverting matrices, ndim=1.
    """
    A = np.asanyarray(A)
    if ndim == 0:
        return 1 / A
    elif ndim == 1:
        return np.linalg.inv(A)
    else:
        raise NotImplementedError()


def mvdot(A, b, ndim=1):
    """
    Compute matrix-vector product.

    Applies broadcasting.
    """
    # TODO/FIXME: A bug in inner1d:
    # https://github.com/numpy/numpy/issues/3338
    #
    # b = np.asanyarray(b)
    # return gula.inner1d(A, b[...,np.newaxis,:])
    # 
    # Use einsum instead:
    if ndim > 0:
        b = misc.add_axes(b, num=ndim, axis=-1-ndim)

    return inner(A, b, ndim=ndim)
    ## if ndim != 1:
    ##     raise NotImplementedError("mvdot not yet implemented for ndim!=1")

    ## return _dot(A, b[...,None])[...,0]
    ## #return np.einsum('...ik,...k->...i', A, b)

def mmdot(A, B, ndim=1):
    """
    Compute matrix-matrix product.

    Applies broadcasting.
    """
    if ndim == 0:
        return A * B
    elif ndim == 1:
        return _dot(A, B)
    else:
        raise Exception("mmdot not yet implemented for ndim>1")
    #return np.einsum('...ik,...kj->...ij', A, B)

def transpose(X, ndim=1):
    """
    Transpose the matrix.
    """
    for n in range(ndim):
        X = np.swapaxes(X, -1-n, -1-ndim-n)
    return X
    ## if ndim != 1:
    ##     raise Exception("transpose not yet implemented for ndim!=1")
    ## return np.swapaxes(X, -1, -2)

def m_dot(A,b):
    raise DeprecationWarning()
    # Compute matrix-vector product over the last two axes of A and
    # the last axes of b.  Other axes are broadcasted. If A has shape
    # (..., M, N) and b has shape (..., N), then the result has shape
    # (..., M)
    
    #b = reshape(b, shape(b)[:-1] + (1,) + shape(b)[-1:])
    #return np.dot(A, b)
    return np.einsum('...ik,...k->...i', A, b)
    # TODO: Use einsum!!
    #return np.sum(A*b[...,np.newaxis,:], axis=(-1,))

def block_banded_solve(A, B, y):
    """
    Invert symmetric, banded, positive-definite matrix.

    A contains the diagonal blocks.

    B contains the superdiagonal blocks (their transposes are the
    subdiagonal blocks).

    Shapes:
    A: (...,   N, D, D)
    B: (..., N-1, D, D)
    y: (...,   N,    D)

    The algorithm is basically LU decomposition.

    Computes only the diagonal and super-diagonal blocks of the
    inverse. The true inverse is dense, in general.

    Assume each block has the same size.

    Return:
    * inverse blocks
    * solution to the system
    * log-determinant
    """
    
    # Number of time instance and dimensionality
    N = np.shape(y)[-2]
    D = np.shape(y)[-1]

    # Check the shape of the diagonal blocks
    if np.shape(A)[-3] != N:
        raise ValueError("The number of diagonal blocks is incorrect")
    if np.shape(A)[-2:] != (D,D):
        raise ValueError("The diagonal blocks have wrong shape")

    # Check the shape of the super-diagonal blocks
    if np.shape(B)[-3] != N-1:
        raise ValueError("The number of super-diagonal blocks is incorrect")
    if np.shape(B)[-2:] != (D,D):
        raise ValueError("The diagonal blocks have wrong shape")

    plates_VC = misc.broadcasted_shape(np.shape(A)[:-3],
                                       np.shape(B)[:-3])
    plates_y = misc.broadcasted_shape(plates_VC,
                                      np.shape(y)[:-2])
                      
    V = np.empty(plates_VC+(N,D,D))
    C = np.empty(plates_VC+(N-1,D,D))
    x = np.empty(plates_y+(N,D))

    #
    # Forward recursion
    #
    
    # In the forward recursion, store the Cholesky factor in V. So you
    # don't need to recompute them in the backward recursion.

    # TODO: This whole algorithm could be implemented as in-place operation.
    # Might be a nice feature (optional?)

    x[...,0,:] = y[...,0,:]
    V[...,0,:,:] = chol(A[...,0,:,:])
    ldet = chol_logdet(V[...,0,:,:])
    for n in range(N-1):
        # Compute the solution of the system
        x[...,n+1,:] = (y[...,n+1,:] 
                        - mvdot(misc.T(B[...,n,:,:]), 
                                chol_solve(V[...,n,:,:], 
                                           x[...,n,:])))
        # Compute the superdiagonal block of the inverse
        C[...,n,:,:] = chol_solve(V[...,n,:,:], 
                                  B[...,n,:,:],
                                  matrix=True)
        # Compute the diagonal block
        V[...,n+1,:,:] = (A[...,n+1,:,:] 
                        - mmdot(misc.T(B[...,n,:,:]), C[...,n,:,:]))
        # Ensure symmetry by 0.5*(V+V.T)
        V[...,n+1,:,:] = 0.5 * (V[...,n+1,:,:] + misc.T(V[...,n+1,:,:]))
        # Compute and store the Cholesky factor of the diagonal block
        V[...,n+1,:,:] = chol(V[...,n+1,:,:])
        # Compute the log-det term here, too
        ldet += chol_logdet(V[...,n+1,:,:])

    #
    # Backward recursion
    #
    x[...,-1,:] = chol_solve(V[...,-1,:,:], x[...,-1,:])
    V[...,-1,:,:] = chol_inv(V[...,-1,:,:])
    for n in reversed(range(N-1)):
        # Compute the solution of the system
        x[...,n,:] = chol_solve(V[...,n,:,:], 
                                x[...,n,:] - mvdot(B[...,n,:,:], 
                                                   x[...,n+1,:]))
        # Compute the diagonal block of the inverse
        V[...,n,:,:] = (chol_inv(V[...,n,:,:]) 
                        + mmdot(C[...,n,:,:], 
                                mmdot(V[...,n+1,:,:], 
                                misc.T(C[...,n,:,:]))))
        C[...,n,:,:] = - mmdot(C[...,n,:,:], V[...,n+1,:,:])
        # Ensure symmetry by 0.5*(V+V.T)
        V[...,n,:,:] = 0.5 * (V[...,n,:,:] + misc.T(V[...,n,:,:]))

    return (V, C, x, ldet)
    
