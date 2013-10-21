# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Copyright (C) 2012 Jaakko Luttinen <jaakko.luttinen@iki.fi>
# Released under the terms of the GNU GPL v3

import warnings
cimport stdlib
cimport python as py
import numpy as np
cimport numpy as np
from scipy import sparse

# Put all relevant distance functions under this namespace
from scipy.spatial.distance import pdist, cdist, squareform

np.import_array()

cdef extern from "numpy/arrayobject.h":
    # Cython 0.12 complains about PyTypeObject being an "incomplete type" on
    # this line:
    #py.PyTypeObject PyArray_Type
    # So use a hack:
    struct MyHackReallyPyTypeObject:
        pass
    MyHackReallyPyTypeObject PyArray_Type
    object PyArray_NewFromDescr(MyHackReallyPyTypeObject * subtype,
                                np.dtype descr,
                                int nd,
                                np.npy_intp * dims,
                                np.npy_intp * strides,
                                void * data,
                                int flags,
                                object obj)
    # This is ridiculous: the description of PyArrayObject in numpy.pxd does
    # not mention the 'base' member, so we need a separate wrapper just to
    # expose it:
    ctypedef struct ndarray_with_base "PyArrayObject":
        void * base

    # In Cython 0.14.1, np.NPY_F_CONTIGUOUS is broken, because numpy.pxd
    # claims that it is of a non-existent type called 'enum requirements', and
    # new versions of Cython attempt to stash it in a temporary variable of
    # this type, which then annoys the C compiler.
    enum:
        NPY_F_CONTIGUOUS

cdef inline np.ndarray set_base(np.ndarray arr, object base):
    cdef ndarray_with_base * hack = <ndarray_with_base *> arr
    py.Py_INCREF(base)
    hack.base = <void *> base
    return arr

cdef extern from "sparse_distance.h":
    cdef enum:
         FULL, LOWER, UPPER, STRICTLY_LOWER, STRICTLY_UPPER
         
    void sppdist_sqeuclidean(double *X,
                             int m,
                             int n,
                             double threshold,
                             int form,
                             double **out_Dx,
                             int **out_Dij,
                             int *out_nzmax)

    void spcdist_sqeuclidean(double *X1,
                             int m1,
                             double *X2,
                             int m2,
                             int n,
                             double threshold,
                             double **out_Dx,
                             int **out_Dij,
                             int *out_nzmax)

cdef object _integer_py_dtype = np.dtype(np.int32)
assert sizeof(int) == _integer_py_dtype.itemsize == 4

cdef object _real_py_dtype = np.dtype(np.float64)
assert sizeof(double) == _real_py_dtype.itemsize == 8

cdef object _complex_py_dtype = np.dtype(np.complex128)
assert _complex_py_dtype.itemsize == 2 * sizeof(double) == 16

cdef class _CleanupCOO(object):
    cdef int * _ij
    cdef double * _x
    def __dealloc__(self):
        stdlib.free(self._ij)
        stdlib.free(self._x)
        
cdef class _CleanupCSC(object):
    cdef int * _p
    cdef int * _i
    cdef double * _x
    def __dealloc__(self):
        stdlib.free(self._p)
        stdlib.free(self._i)
        stdlib.free(self._x)
        
cdef _py_sparse_coo(double *x, int *ij, int m, int n, int nzmax):
    cdef _CleanupCOO cleaner = _CleanupCOO()
    cleaner._ij = ij
    cleaner._x = x
    
#    PyObject *PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)

    shape = (m, n)

    cdef np.npy_intp nz = nzmax
    cdef np.npy_intp nz2[2]
    nz2[0] = 2
    nz2[1] = nzmax

    py.Py_INCREF(_integer_py_dtype)
    ijnd = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                           _integer_py_dtype, 2,
                                           nz2,
                                           NULL,
                                           ij,
                                           NPY_F_CONTIGUOUS, None),
                      cleaner)
    py.Py_INCREF(_real_py_dtype)
    data = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                         _real_py_dtype, 1,
                                         &nz,
                                         NULL,
                                         x,
                                         NPY_F_CONTIGUOUS, None),
                      cleaner)

    return sparse.coo_matrix((data, ijnd), shape=shape)

cdef _py_sparse_csc(int *p, int *i, double *x, int m, int n, int nzmax):
    cdef _CleanupCSC cleaner = _CleanupCSC()
    cleaner._p = p
    cleaner._i = i
    cleaner._x = x
    
#    PyObject *PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)

    shape = (m, n)
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp ncol_plus_1 = n + 1
    indptr = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                           _integer_py_dtype, 1,
                                           &ncol_plus_1,
                                           NULL,
                                           p,
                                           NPY_F_CONTIGUOUS, None),
                      cleaner)
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp nz = nzmax
    indices = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                            _integer_py_dtype, 1,
                                            &nz,
                                            NULL,
                                            i,
                                            NPY_F_CONTIGUOUS, None),
                      cleaner)
    py.Py_INCREF(_real_py_dtype)
    data = set_base(PyArray_NewFromDescr(&PyArray_Type,
                                         _real_py_dtype, 1,
                                         &nz,
                                         NULL,
                                         x,
                                         NPY_F_CONTIGUOUS, None),
                      cleaner)
    return sparse.csc_matrix((data, indices, indptr), shape=shape)

def sparse_pdist(X, threshold, form="strictly_lower", format="csc"):

    # Outputs
    cdef double *Dx = NULL
    cdef int *Dij = NULL
    cdef int nzmax
    cdef int uplo

    if np.ndim(X) == 0:
        X = np.atleast_2d(X)
    elif np.ndim(X) == 1:
        X = X[:,np.newaxis]

    (m,n) = X.shape

    # Check empty cases
    if m == 0:
        return np.empty((m,m))
    #if n == 0:
    #    return sp.csc_matrix((m,m)).asformat(format)

    cdef np.ndarray[np.double_t, ndim=2, mode="c"] X_c
    X_c = np.ascontiguousarray(X, dtype=np.double)

    if form.lower() == "strictly_lower":
        uplo = STRICTLY_LOWER
    elif form.lower() == "lower":
        uplo = LOWER
    elif form.lower() == "strictly_upper":
        uplo = STRICTLY_UPPER
    elif form.lower() == "upper":
        uplo = UPPER
    elif form.lower() == "full":
        uplo = FULL
    else:
        raise Exception, "Unknown form requested"


    sppdist_sqeuclidean(<double*> X_c.data,
                        m,
                        n,
                        threshold,
                        uplo,
                        &Dx,
                        &Dij,
                        &nzmax)

    return _py_sparse_coo(Dx, Dij, m, m, nzmax).asformat(format)

def sparse_cdist(X1, X2, threshold, format="csc"):

    # Outputs
    cdef double *Dx = NULL
    cdef int *Dij = NULL
    cdef int nzmax

    if np.ndim(X1) == 0:
        X1 = np.atleast_2d(X1)
    elif np.ndim(X1) == 1:
        X1 = X1[:,np.newaxis]
    elif np.ndim(X1) > 2:
        raise Exception, "Input matrices must be 0-2 -dimensional"

    if np.ndim(X2) == 0:
        X2 = np.atleast_2d(X2)
    elif np.ndim(X2) == 1:
        X2 = X2[:,np.newaxis]
    elif np.ndim(X2) > 2:
        raise Exception, "Input matrices must be 0-2 -dimensional"

    (m1,n1) = X1.shape
    (m2,n2) = X2.shape

    # Check empty cases
    if m1 == 0 or m2 == 0:
        return np.empty((m1,m2))

    # Check that inputs have the same dimensionality
    if n1 != n2:
        raise Exception, "Matrices must have the same number of columns"
    n = n1

    cdef np.ndarray[np.double_t, ndim=2, mode="c"] X1_c
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] X2_c
    X1_c = np.ascontiguousarray(X1, dtype=np.double)
    X2_c = np.ascontiguousarray(X2, dtype=np.double)

    spcdist_sqeuclidean(<double*> X1_c.data,
                        m1,
                        <double*> X2_c.data,
                        m2,
                        n,
                        threshold,
                        &Dx,
                        &Dij,
                        &nzmax)

    return _py_sparse_coo(Dx, Dij, m1, m2, nzmax).asformat(format)

__all__ = ["pdist", "cdist"]
