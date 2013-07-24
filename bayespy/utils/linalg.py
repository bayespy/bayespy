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

from .utils import nested_iterator

def chol(C):
    if sparse.issparse(C):
        # Sparse Cholesky decomposition (returns a Factor object)
        return cholmod.cholesky(C)
    else:
        # Computes Cholesky decomposition for a collection of matrices.
        # The last two axes of C are considered as the matrix.
        C = np.atleast_2d(C)
        U = np.empty(np.shape(C))
        for i in nested_iterator(np.shape(U)[:-2]):
            try:
                U[i] = linalg.cho_factor(C[i])[0]
            except np.linalg.linalg.LinAlgError:
                print(C[i])
                raise Exception("Matrix not positive definite")
        return U

def chol_solve(U, b):
    if isinstance(U, np.ndarray):
        if sparse.issparse(b):
            b = b.toarray()
            
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

    elif isinstance(U, cholmod.Factor):
        if sparse.issparse(b):
            b = b.toarray()
        return U.solve_A(b)
    else:
        raise ValueError("Unknown type of Cholesky factor")

def chol_inv(U):
    if isinstance(U, np.ndarray):
        # Allocate memory
        V = np.tile(np.identity(np.shape(U)[-1]), np.shape(U)[:-2]+(1,1))
        for i in nested_iterator(np.shape(U)[:-2]):
            V[i] = linalg.cho_solve((U[i], False),
                                    V[i],
                                    overwrite_b=True) # This would need Fortran order

        return V
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
        # Computes Cholesky decomposition for a collection of matrices.
        return 2*np.sum(np.log(np.einsum('...ii->...i', U)), axis=(-1,))
    elif isinstance(U, cholmod.Factor):
        return np.sum(np.log(U.D()))
    
def logdet_tri(R):
    """
    Logarithm of the absolute value of the determinant of a triangular matrix.
    """
    return np.sum(np.log(np.abs(np.einsum('...ii->...i', R))))
    
def logdet_cov(C):
    return logdet_chol(chol(C))

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
    

    


def outer(A,B):
    # Computes outer product over the last axes of A and B. The other
    # axes are broadcasted. Thus, if A has shape (..., N) and B has
    # shape (..., M), then the result has shape (..., N, M)
    return A[...,np.newaxis]*B[...,np.newaxis,:]

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
            Y = np.einsum('...ik,...kj->...ij', Y, X)
            #Y = gula.matrix_multiply(Y, X)
        return Y

def tracedot(A, B):
    """
    Computes trace(A*B).
    """
    return np.einsum('...ij,...ji->...', A, B)

def inv(A):
    if np.ndim(A) == 2:
        return np.linalg.inv(A)
    else:
        raise NotImplementedError()
    # return gula.inv(A)

def mvdot(A, b):
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
    return np.einsum('...ik,...k->...i', A, b)

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
