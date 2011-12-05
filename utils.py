import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special

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
    if np.isscalar(a) or isinstance(a, list) or isinstance(a, np.ndarray):
        return True


def sum_product(*args, axes_to_keep=None, axes_to_sum=None, keepdims=False):

    # Computes sum(arg[0]*arg[1]*arg[2]*..., axis=axes_to_sum) without
    # explicitly computing the intermediate product

    # Dimensionality of the result
    max_dim = 0
    for k in range(len(args)):
        max_dim = max(max_dim, np.ndim(args[k]))

    axes = list()
    if axes_to_sum == None and axes_to_keep == None:
        # Sum over all axes if none given
        axes = []
    elif axes_to_sum != None:
        if np.isscalar(axes_to_sum):
            axes_to_sum = [axes_to_sum]
        for i in range(max_dim):
            if i not in axes_to_sum and (-max_dim+i) not in axes_to_sum:
                axes.append(i)
    elif axes_to_keep != None:
        if np.isscalar(axes_to_keep):
            axes_to_keep = [axes_to_keep]
        for i in range(max_dim):
            if i in axes_to_keep or (-max_dim+i) in axes_to_keep:
                axes.append(i)
    else:
        raise Exception("You can't give both axes to sum and keep")


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
    y = np.einsum(*pairs)

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


def broadcasted_shape_from_arrays(*args):
    # Computes the resulting shape if shapes a and b are broadcasted
    # together

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
            
def broadcasted_shape(a,b):
    # Computes the resulting shape if shapes a and b are broadcasted
    # together
    l_max = max(len(a), len(b))
    s = ()
    for i in range(-l_max,0):
        if -i > len(b):
            s += (a[i],)
        elif -i > len(a) or a[i] == 1 or a[i] == b[i]:
            s += (b[i],)
        elif b[i] == 1:
            s += (a[i],)
        else:
            raise Exception("Shapes do not broadcast")
    return s
            
    
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

def chol(C):
    return decomp.cho_factor(C)[0]

def chol_solve(U, b):
    return decomp.cho_solve((U, False), b)

def logdet_chol(U):
    return 2*np.sum(np.log(np.diag(U)))
    
def m_chol(C):
    # Computes Cholesky decomposition for a collection of matrices.
    # The last two axes of C are considered as the matrix.
    C = np.atleast_2d(C)
    U = np.empty(np.shape(C))
    for i in nested_iterator(np.shape(U)[:-2]):
        try:
            U[i] = decomp.cho_factor(C[i])[0]
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

        out[ind_out] = decomp.cho_solve((U[i], False),
                                        b.T).T.reshape(orig_shape)

        
    return out
    

def m_chol_inv(U):
    # Allocate memory
    V = np.tile(np.identity(np.shape(U)[-1]), np.shape(U)[:-2]+(1,1))
    for i in nested_iterator(np.shape(U)[:-2]):
        V[i] = decomp.cho_solve((U[i], False),
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

def m_dot(A,b):
    # Compute matrix-vector product over the last two axes of A and
    # the last axes of b.  Other axes are broadcasted. If A has shape
    # (..., M, N) and b has shape (..., N), then the result has shape
    # (..., M)
    
    #b = reshape(b, shape(b)[:-1] + (1,) + shape(b)[-1:])

    # TODO: Use einsum!!
    return np.sum(A*b[...,np.newaxis,:], axis=(-1,))

