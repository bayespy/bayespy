import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.optimize as optimize
import scipy.sparse as sparse
import scikits.sparse.cholmod as cholmod

def nans(size=()):
    return np.tile(np.nan, size)

def array_to_scalar(x):
    # This transforms an N-dimensional array to a scalar. It's most
    # useful when you know that the array has only one element and you
    # want it out as a scalar.
    return np.ravel(x)[0]

def grid(x1, x2):
    """ Returns meshgrid as a (M*N,2)-shape array. """
    (X1, X2) = np.meshgrid(x1, x2)
    return np.hstack((X1.reshape((-1,1)),X2.reshape((-1,1))))


class CholeskyDense():
    
    def __init__(self, K):
        self.U = decomp.cho_factor(K)
    
    def solve(self, b):
        if sparse.issparse(b):
            b = b.toarray()
        return decomp.cho_solve(self.U, b)

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
        #print("HERE 1\n")
        #print(self.LD.L())
        iK = self.LD.spinv(form='lower')
        return (2*iK.multiply(dK).sum()
                - iK.diagonal().dot(dK.diagonal()))
        #print(self.LD.L())
        #print("Compare spinv to inv")
        #print(iK.todense())
        #print(self.LD.inv().todense())
        #print("HERE 2\n")
        #print(iK.todense())
        # Multiply by two because of symmetry (remove diagonal once
        # because it was taken into account twice)
        #print("Try this")
        #print(self.LD.inv().todense())
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
        #print('func')
        set_values(x)
        # Compute lower bound (and gradient terms)
        return -lowerbound()
        #return f

    # Function for computing the gradient of the lower bound
    def funcprime(x):
        # Collect the gradients from the nodes
        #print('funcprime')
        set_values(x)
        # Compute lower bound (and gradient terms)
        #lowerbound()
        return -gradient()
        #return df

    # Optimize
    if gradient != None:
        print('Checking gradient')
        check_gradient(x0, func, funcprime, 1e-6)

        xopt = optimize.fmin_bfgs(func, x0, fprime=funcprime, maxiter=100)
        #xopt = optimize.fmin_ncg(func, x0, fprime=funcprime, maxiter=50)
    else:
        xopt = optimize.fmin_bfgs(func, x0, maxiter=100)
        #xopt = optimize.fmin_ncg(func, x0, maxiter=50)

    # Set optimal values to the nodes
    print(xopt)
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
        #print(x)
        #print(ind_all)
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

        #print('gradient')
        #print(dl)
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

def moveaxis(A, axis_from, axis_to):

    """ Move the axis number axis_from to be the axis number axis_to. """
    #print('moveaxis', np.shape(A), axis_from, axis_to)
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
            raise Exception("Shapes %s and %s do not broadcast" % (a,b))
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

#def spinv_chol(L):
    

def chol(C):
    if sparse.issparse(C):
        # Sparse Cholesky decomposition (returns a Factor object)
        return cholmod.cholesky(C)
    else:
        # Dense Cholesky decomposition
        return decomp.cho_factor(C)[0]

def chol_solve(U, b):
    if isinstance(U, np.ndarray):
        if sparse.issparse(b):
            b = b.toarray()
        return decomp.cho_solve((U, False), b)
    elif isinstance(U, cholmod.Factor):
        if sparse.issparse(b):
            b = b.toarray()
        return U.solve_A(b)

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
            U[i] = decomp.cho_factor(C[i])[0]
        except np.linalg.linalg.LinAlgError:
            print(C[i])
            raise Exception("Matrix not positive definite")
    return U


def m_chol_solve(U, B, out=None):

    #print('m_chol_solve', B)
    
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

