
import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as chol
import scipy.special as special
import matplotlib.pyplot as plt
import time
import profile

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

def m_chol(C):
    # Computes Cholesky decomposition for a collection of matrices.
    # The last two axes of C are considered as the matrix.
    C = np.atleast_2d(C)
    U = np.empty(np.shape(C))
    for i in nested_iterator(np.shape(U)[:-2]):
        try:
            U[i] = chol.cho_factor(C[i])[0]
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

        out[ind_out] = chol.cho_solve((U[i], False),
                                      b.T).T.reshape(orig_shape)

        
    return out
    

def m_chol_inv(U):
    # Allocate memory
    V = np.tile(np.identity(np.shape(U)[-1]), np.shape(U)[:-2]+(1,1))
    for i in nested_iterator(np.shape(U)[:-2]):
        V[i] = chol.cho_solve((U[i], False),
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

# A node must have the following methods in order to communicate with
# other nodes.
#
# message_to_child()
#
# message_to_parent(index)
#
# get_shape(index)
#
# We should have a base class for default implementations. Subclasses
# of this base class should have methods
#
# message_to_child()
#
# message(index, msg_parents)

# A variable node must have the following methods.
#
# update()
#
# We should have a base class for default implementations. This would
# also implement message_to_child and contain natural parameterization
# stuff.  Subclasses should implement methods
#
# message(index, msg_parents)

# A deterministic node just implements child_message and parent_message.

class Node:

    # Proposed functions:
    def logpdf_integrated(self):
        # The log pdf when this node is integrated out (useful for
        # type-2 ML or predictive densities)
        return

    def random(self):
        # Draw a random variable from the node
        return

    def gibbs_sampling(self):
        # Hmm.. For Gibbs and for generating samples from the model?
        return

    def __init__(self, *args, **kwargs):
        try:
            self.plates = kwargs['plates']
        except KeyError:
            self.plates = ()
        try:
            self.name = kwargs['name']
        except KeyError:
            self.name = ""
        try:
            self.dims = kwargs['dims']
        except KeyError:
            raise Exception("You need to specify the dimensionality" \
                            + " of the distribution for class" \
                            + str(self.__class__))
        # Parents
        self.parents = args
        # Inform parent nodes
        for (index,parent) in enumerate(self.parents):
            parent.add_child(self, index)
        # Children
        self.children = list()


    def add_child(self, child, index):
        self.children.append((child, index))

    def get_shape(self, ind):
        return self.plates + self.dims[ind]

    def plate_multiplier(self, *args):
        # Check broadcasting of the shapes
        for arg in args:
            broadcasted_shape(self.plates, arg)
            
        r = 1
        for j in range(-len(self.plates),0):
            mult = True
            for arg in args:
                if not (-j > len(arg) or arg[j] == 1):
                    mult = False
            if mult:
                r *= self.plates[j]
        return r

    def message_from_children(self):
        msg = [np.array(0.0) for i in range(len(self.dims))]
        total_mask = False
        for (child,index) in self.children:
            (m, mask) = child.message_to_parent(index)
            total_mask = np.logical_or(total_mask, mask)
            for i in range(len(self.dims)):
                # Check broadcasting shapes
                sh = broadcasted_shape(self.get_shape(i), np.shape(m[i]))
                try:
                    # Try exploiting broadcasting rules
                    msg[i] += m[i]
                except ValueError:
                    msg[i] = msg[i] + m[i]

        # TODO: Should the mask be returned also?
        return (msg, total_mask)

    def message_from_parent(self):
        pass

    def get_moments(self):
        raise Exception('Not implemented')

    def message_to_child(self):
        return self.get_moments()
        # raise Exception("Not implemented. Subclass should implement this!")

    def message_to_parent(self, index):
        # In principle, a node could have multiple parental roles with
        # respect to a single node, that is, there can be duplicates
        # in self.parents and that's ok. This implementation might not
        # take this kind of exceptional situation properly into
        # account.. E.g., you could do PCA such that Y=X'*X. Or should
        # it be restricted such that a node can have only one parental
        # role?? Maybe there should at least be an ID for each
        # connection.
        if index < len(self.parents):
            # Get moments from parents
            u_parents = list()
            for (i,parent) in enumerate(self.parents):
                if i != index:
                    u_parents.append(parent.message_to_child())
                else:
                    u_parents.append(None)

            # Decompose our own message to parent[index]
            (m, my_mask) = self.get_message(index, u_parents)

            # The parent we're sending the message to
            parent = self.parents[index]

            # Compute mask message
            s = axes_to_collapse(np.shape(my_mask), parent.plates)
            mask = np.any(my_mask, axis=s, keepdims=True)
            mask = squeeze_to_dim(mask, len(parent.plates))
            
            # Compact the message to a proper shape
            for i in range(len(m)):

                # Ignorations (add extra axes to broadcast properly).
                # This sends zero messages to parent from such
                # variables we are ignoring in this node. This is
                # useful for handling missing data.

                # Apply the mask to the message
                # Sum the dimensions of the message matrix to match
                # the dimensionality of the parents natural
                # parameterization (as there may be less plates for
                # parents)

                shape_mask = np.shape(my_mask) + (1,) * len(parent.dims[i])
                my_mask2 = np.reshape(my_mask, shape_mask)

                #try:
                m[i] = m[i] * my_mask2
                #except:

                shape_m = np.shape(m[i])
                # If some dimensions of the message matrix were
                # singleton although the corresponding dimension
                # of the parents natural parameterization is not,
                # multiply the message (i.e., broadcasting)

                # Plates in the message
                dim_parent = len(self.parents[index].dims[i])
                if dim_parent > 0:
                    plates_m = shape_m[:-dim_parent]
                else:
                    plates_m = shape_m

                # Compute the multiplier (multiply by the number of
                # plates for which both the message and parent have
                # single plates)
                r = self.plate_multiplier(plates_m, parent.plates)

                shape_parent = parent.get_shape(i)


                s = axes_to_collapse(shape_m, shape_parent)
                m[i] = np.sum(m[i], axis=s, keepdims=True)

                m[i] = squeeze_to_dim(m[i], len(shape_parent))
                m[i] *= r

            return (m, mask)
        else:
            # Unknown parent
            raise Exception("Unknown parent requesting a message")

    def get_message(self, index, u_parents):
        raise Exception("Not implemented.")
        pass


class NodeVariable(Node):

    def __init__(self, *args, **kwargs):

        Node.__init__(self, *args, **kwargs)

        # Natural parameters and moments. Note that the size of the
        # array self.u or self.phi doesn't actually need to equal
        # self.dims+self.plates, it will be broadcasted automatically
        # (although this might be extremely rare: it would mean that
        # there are identical posterior distributions over
        # plates). Anyway, for this reason (and some others), we need
        # to explicityly have the dimensionalities dims and plates? Do
        # we need dims? Yes, because you need to be able to check what
        # plates are missing.
        self.phi = [np.array(0.0) for i in range(len(self.dims))]
        self.u = [np.array(0.0) for i in range(len(self.dims))]

        # Terms for the lower bound (G for latent and F for observed)
        self.g = 0
        self.f = 0

        # Not observed
        self.observed = False

        # By default, ignore all elements
        self.mask = False

    def get_message(self, index, u_parents):
        return (self.message(index, u_parents),
                self.mask)


    def update(self):
        if not np.all(self.observed):

            # Messages from parents
            u_parents = [parent.message_to_child() for parent in self.parents]
                
            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            # Update natural parameters using children (just add the
            # messages to phi)
            for (child,index) in self.children:
                (m, mask) = child.message_to_parent(index)
                # Combine masks
                #
                # TODO: Maybe you would like to compute a new mask
                # at every update?
                self.mask = np.logical_or(self.mask,mask)
                for i in range(len(self.phi)):
                    try:
                        # Try exploiting broadcasting rules
                        self.phi[i] += m[i]
                    except ValueError:
                        self.phi[i] = self.phi[i] + m[i]

            # Update moments
            self.update_moments_and_g()

    def update_phi_from_parents(self, u_parents):
        # This makes correct broadcasting
        phi = self.compute_phi_from_parents(u_parents)
        for i in range(len(self.phi)):
            self.phi[i].fill(0)
            try:
                # Try exploiting broadcasting rules
                self.phi[i] += phi[i]
            except ValueError:
                self.phi[i] = self.phi[i] + phi[i]
    
    def get_moments(self):
        return self.u

    def message(self, index, u_parents):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

    def compute_phi_from_parents(self, u_parents):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

    def compute_g_from_parents(self, u_parents):
        raise Exception("Not implemented for " + str(self.__class__))

    def update_moments_and_g(self):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

    def lower_bound_contribution(self):
        # Messages from parents
        u_parents = [parent.message_to_child() for parent in self.parents]
        phi = self.compute_phi_from_parents(u_parents)
        # G from parents
        L = self.compute_g_from_parents(u_parents)
        # L = g
        # G for unobserved variables (ignored variables are handled
        # properly automatically)
        latent_mask = np.logical_not(self.observed)
        #latent_mask = np.logical_and(self.mask, np.logical_not(self.observed))
        L = L - self.g * latent_mask
        # F for observed variables
        L = L + self.f * self.observed
        for (phi_p, phi_q, u_q, dims) in zip(phi, self.phi, self.u, self.dims):
            # Form a mask which puts observed variables to zero and
            # broadcasts properly
            latent_mask = add_axes(latent_mask,
                                   len(self.plates) - np.ndim(latent_mask),
                                   len(dims))
            axis_sum = tuple(range(-len(dims),0))

            # Compute the term
            Z = np.sum((phi_p - phi_q*latent_mask) * u_q,
                       axis=axis_sum)

            L = L + Z

        return (np.sum(L*self.mask)
                * self.plate_multiplier(np.shape(L),
                                        np.shape(self.mask)))
        return L
            
    def fix_moments_and_f(self, u, f, mask):
        #print(u)
        for (i,v) in enumerate(u):
            # This is what the dimensionality "should" be
            s = self.plates + self.dims[i]
            t = np.shape(v)
            if s != t:
                    msg = "Dimensionality of the observations incorrect."
                    msg += "\nShape of input: " + str(t)
                    msg += "\nExpected shape: " + str(s)
                    msg += "\nCheck plates."
                    raise Exception(msg)
            #self.phi[i] = 0

        self.observed = mask
        for i in range(len(self.u)):
            obs_mask = add_axes(mask,
                                len(self.plates) - np.ndim(mask),
                                len(self.dims[i]))
            self.u[i] = (obs_mask * u[i]
                         + np.logical_not(obs_mask) * self.u[i])

        self.f = f
        
        # Observed nodes should not be ignored
        self.mask = np.logical_or(self.mask, self.observed)


class NodeConstant(NodeVariable):
    def __init__(self, u, **kwargs):
        NodeVariable.__init__(self, **kwargs)
        self.fix_moments_and_f(u, 0, True)

class NodeConstantScalar(NodeConstant):
    def __init__(self, a, **kwargs):
        NodeConstant.__init__(self,
                              [a],
                              plates=np.shape(a),
                              dims=[()])

class NodeConstantGaussian(NodeConstant):
    def __init__(self, X, **kwargs):
        X = np.atleast_1d(X)
        d = X.shape[-1]
        NodeConstant.__init__(self,
                              [X, m_outer(X, X)],
                              plates=X.shape[:-1],
                              dims=[(d,), (d,d)],
                              **kwargs)
        
class NodeConstantWishart(NodeConstant):
    def __init__(self, Lambda, **kwargs):
        Lambda = np.atleast_2d(Lambda)
        if Lambda.shape[-1] != Lambda.shape[-2]:
            raise Exception("Lambda not a square matrix.")
        NodeConstant.__init__(self,
                              [Lambda, m_chol_logdet(m_chol(Lambda))],
                              plates=Lambda.shape[:-2],
                              dims=[Lambda.shape[-2:], ()],
                              **kwargs)


class NodeGamma(NodeVariable):

    # Gamma(a, b)

    def __init__(self, a, b, plates=(), **kwargs):

        # TODO: USE asarray(a)

        # Check for constant a
        if np.isscalar(a) or isinstance(a, np.ndarray):
            a = NodeConstantScalar(a)

        # Check for constant b
        if np.isscalar(b) or isinstance(b, np.ndarray):
            b = NodeConstant([b, np.log(b)], plates=np.shape(b), dims=[(),()])

        # Construct
        NodeVariable.__init__(self, a, b, plates=plates, dims=[(),()], **kwargs)

    def compute_phi_from_parents(self, u_parents):
        return [-u_parents[1][0],
                1*u_parents[0][0]]

    def compute_g_from_parents(self, u_parents):
        a = u_parents[0][0]
        gammaln_a = special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

        
    def update_moments_and_g(self):
        log_b = np.log(-self.phi[0])
        self.u[0] = self.phi[1] / (-self.phi[0])
        self.u[1] = special.digamma(self.phi[1]) - log_b
        
        self.g = self.phi[1] * log_b - special.gammaln(self.phi[1])

    def message(self, index, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-self.u[0],
                    u_parents[0][0]]

    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("Gamma(" + str(a) + ", " + str(b) + ")")


class NodeNormal(NodeVariable):

    # Normal(mu, 1/tau)

    def __init__(self, mu, tau, plates=(), **kwargs):

        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = NodeConstant([mu, mu**2], plates=np.shape(mu), dims=[(),()])

        # Check for constant tau
        if np.isscalar(tau) or isinstance(tau, np.ndarray):
            tau = NodeConstant([tau, log(tau)], plates=np.shape(tau), dims=[(),()])

        # Construct
        NodeVariable.__init__(self, mu, tau, plates=plates, dims=[(),()], **kwargs)

    def compute_phi_from_parents(self, u_parents):
        return [u_parents[1][0] * u_parents[0][0],
                -u_parents[1][0] / 2]

    def update_moments_and_g(self):
        self.u[0] = -self.phi[0] / (2*self.phi[1])
        self.u[1] = self.u[0]**2 - 1 / (2*self.phi[1])

        self.g = (-0.5 * self.u[0] * self.phi[0]
                  + 0.5 * np.log(-2*self.phi[1]))

    def compute_g_from_parents(self, u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        tau = u_parents[1][0]
        log_tau = u_parents[1][1]
        g = -0.5 * mumu*tau + 0.5 * log_tau

        return g


    def message(self, index, u_parents):
        if index == 0:
            return [u_parents[1][0] * self.u[0],
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            return [-0.5 * (self.u[1] - 2*self.u[0]*u_parents[0][0] + u_parents[0][1]),
                    0.5]

    def observe(self, x, mask):
        f = -0.5 * np.log(2*np.pi)
        self.fix_moments_and_f([x, x**2], f, mask)

    def show(self):
        mu = self.u[0]
        s2 = self.u[1] - mu**2
        print("Normal(" + str(mu) + ", " + str(s2) + ")")


class NodeWishart(NodeVariable):

    # Wishart(n, inv(V))

    def __init__(self, n, V, plates=(), **kwargs):

        # Check for constant n
        if np.isscalar(n) or isinstance(n, np.ndarray):            
            n = NodeConstantScalar(n)
            
        # Check for constant V
        if np.isscalar(V) or isinstance(V, np.ndarray):
            V = NodeConstantWishart(V)

        NodeVariable.__init__(self, n, V, plates=plates, dims=V.dims, **kwargs)
        
    def compute_phi_from_parents(self, u_parents):
        return [-0.5 * u_parents[1][0],
                0.5 * u_parents[0][0]]


    def update_moments(self):
        U = m_chol(-self.phi[0])
        k = U[0].shape[0]
        self.u[0] = self.phi[1][...,np.newaxis,np.newaxis] * m_chol_inv(U)
        self.u[1] = -m_chol_logdet(U) + m_digamma(self.phi[1], k)

    def message(self, index, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-0.5 * self.u[0],
                    0.5 * self.u_parents[0][0]]

    def show(self):
        print("Wishart(n, A)")
        print("  n =")
        print(2*self.phi[1])
        print("  A =")
        print(0.5 * self.u[0] / self.phi[1][...,np.newaxis,np.newaxis])

class NodeGaussian(NodeVariable):

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda, plates=(), **kwargs):

        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = NodeConstantGaussian(mu)

        # Check for constant Lambda
        if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
            Lambda = NodeConstantWishart(Lambda)

        # You could check whether the dimensions of mu and Lambda
        # match (and Lambda is square)
        if Lambda.dims[0][-1] != mu.dims[0][-1]:
            raise Exception("Dimensionalities of mu and Lambda do not match.")

        # Construct
        NodeVariable.__init__(self, mu, Lambda,
                              plates=plates,
                              dims=mu.dims,
                              **kwargs)

    def compute_phi_from_parents(self, u_parents):
        return [m_dot(u_parents[1][0], u_parents[0][0]),
                -0.5 * u_parents[1][0]]


    def update_moments_and_g(self):
        L = m_chol(-self.phi[1])
        # Moments
        self.u[0] = m_chol_solve(L, 0.5*self.phi[0])
        self.u[1] = (m_outer(self.u[0], self.u[0])
                     + 0.5 * m_chol_inv(L))
        # G
        self.g = (-0.5 * np.einsum('...i,...i', self.u[0], self.phi[0])
                  + 0.5 * m_chol_logdet(L)
                  + 0.5 * np.log(2) * self.dims[0][0])


    def compute_g_from_parents(self, u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * np.sum(logdet_Lambda))

        return g


    def message(self, index, u_parents):
        if index == 0:
            return [m_dot(u_parents[1][0], self.u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = m_outer(self.u[0], u_parents[0][0])
            return [-0.5 * (self.u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]


    def random(self):
        U = m_chol(-self.phi[1])
        return (self.u[0]
                + 0.5 * m_chol_solve(U,
                                     np.random.normal(0, 1,
                                                      self.get_shape(0))))

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - m_outer(mu, mu)
        print("Gaussian(mu, Cov)")
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

    def observe(self, x):
        self.fix_moments([x, m_outer(x,x)])



class NodeDot(Node):

    # This node satisfies Normal-protocol to children and
    # Gaussian-protocol to parents

    # This node is deterministic, just handles message processing.

    # y(i0,i1,...,in) = sum_d prod_k xk(i0,i1,...,in,d)

    def __init__(self, *args, **kwargs):
        # For now, do not use plates other than from the parents,
        # although it would be possible (it would mean that you'd
        # create several "datasets" with identical "PCA
        # distribution"). Maybe it is better to create such plates in
        # children nodes?
        plates = []
        for x in args:
            # Convert constant matrices to nodes
            if np.isscalar(x) or isinstance(x, np.ndarray):
                x = NodeConstantGaussian(x)
            # Dimensionality of the Gaussian(s). You should check that
            # all the parents have the same dimensionality!
            self.d = x.dims[0]
            # Check consistency of plates (broadcasting rules!)
            for ind in range(min(len(plates),len(x.plates))):
                if plates[-ind-1] == 1:
                    plates[-ind-1] = x.plates[-ind-1]
                elif x.plates[-ind-1] != 1 and plates[-ind-1] != x.plates[-ind-1]:
                    raise Exception('Plates do not match')
            # Add new extra plates
            plates = list(x.plates[:(len(x.plates)-len(plates))]) + plates

        Node.__init__(self, *args, plates=tuple(plates), dims=[(),()], **kwargs)

            

    def get_moments(self):
        if len(self.parents) == 0:
            return [0, 0]

        str1 = '...i' + ',...i' * (len(self.parents)-1)
        str2 = '...ij' + ',...ij' * (len(self.parents)-1)

        u1 = list()
        u2 = list()
        for parent in self.parents:
            u = parent.message_to_child()
            u1.append(u[0])
            u2.append(u[1])

        x = [np.einsum(str1, *u1),
             np.einsum(str2, *u2)]

        return x
        


    def get_parameters(self):
        # Compute mean and variance
        u = self.get_moments()
        u[1] -= u[0]**2
        return u
        

    def get_message(self, index, u_parents):
        
        (m, mask) = self.message_from_children()

        parent = self.parents[index]

        # Compute both messages
        for i in range(2):

            # Add extra axes to the message from children
            m_shape = np.shape(m[i]) + (1,) * (i+1)
            m[i] = np.reshape(m[i], m_shape)

            # Add extra axes to the mask from children
            mask_shape = np.shape(mask) + (1,) * (i+1)
            mask_i = np.reshape(mask, mask_shape)

            # List of elements to multiply together
            A = [m[i], mask_i]
            for k in range(len(u_parents)):
                if k != index:
                    A.append(u_parents[k][i])

            # Find out which axes are summed over. Also, because
            # we are summing over the dimensions already in this
            # function (for efficiency), we need to cancel the
            # effect of the plate-multiplier applied in the
            # message_to_parent function.
            full_shape = broadcasted_shape_from_arrays(*A)
            axes = axes_to_collapse(full_shape, parent.get_shape(i))
            r = 1
            for j in axes:
                r *= full_shape[j]

            # Compute dot product
            m[i] = sum_product(*A, axes_to_sum=axes, keepdims=True) / r

        # Compute the mask
        s = axes_to_collapse(np.shape(mask), parent.plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = squeeze_to_dim(mask, len(parent.plates))

        return (m, mask)
        


def m_plot(x, Y, style):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.plot(x, Y[i], style)

def m_errorplot(x, Y, L, U):
    Y = np.atleast_2d(Y)
    M = Y.shape[-2]
    for i in range(M):
        plt.subplot(M,1,i+1)
        plt.fill_between(x,
                           Y[i]-L[i],
                           Y[i]+U[i],
                           facecolor=(0.6,0.6,0.6,1),
                           edgecolor=(0,0,0,0))
        plt.plot(x, Y[i], color=(0,0,0,1))
        plt.ylabel(str(i))



def test_pca():

    # Dimensionalities
    dataset = 1
    if dataset == 1:
        M = 10
        N = 100
        D_y = 3
        D = 3
        # Generate data
        w = np.random.normal(0, 1, size=(M,1,D_y))
        x = np.random.normal(0, 1, size=(1,N,D_y))
        f = sum_product(w, x, axes_to_sum=[-1])#np.einsum('...i,...i', w, x)
        y = f + np.random.normal(0, 0.5, size=(M,N))
    elif dataset == 2:
        # Data from matlab comparison
        f = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_f.txt')
        y = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_y.txt')
        D = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_d.txt')
        (M,N) = np.shape(y)

    # Construct the PCA model with ARD

    # alpha = NodeGamma(1e-5, 1e-5, plates=(D,))
    #Lambda = NodeWishart(D, (10**-10) * np.identity(D), plates=(), name='Lambda')
    
    X = NodeGaussian(np.zeros(D), np.identity(D), name="X", plates=(1,N))
    X.update()
    X.u[0] = X.random()

    #W = NodeGaussian(np.zeros(D), Lambda, name="W", plates=(M,1))
    W = NodeGaussian(np.zeros(D), np.identity(D), name="W", plates=(M,1))
    W.update()
    W.u[0] = W.random()

    WX = NodeDot(W,X)

    tau = NodeGamma(1e-5, 1e-5, name="tau")
    tau.update()

    Y = NodeNormal(WX, tau, name="Y", plates=(M,N))
    Y.update()

    # Initialize (from prior)

    # Y.update()
    # mask = True
    # mask = np.ones((M,N), dtype=np.bool)
    mask = np.random.rand(M,N) < 0.4
    mask[:,20:40] = False
    Y.observe(y, mask)

    # Inference
    L_last = -np.inf
    for i in range(200):
        t = time.clock()
        X.update()
        W.update()
        tau.update()

        L_X = X.lower_bound_contribution()
        L_W = W.lower_bound_contribution()
        L_tau = tau.lower_bound_contribution()
        L_Y = Y.lower_bound_contribution()
        #print("X: %f, W: %f, tau: %f, Y: %f" % (L_X, L_W, L_tau, L_Y))
        L = L_X + L_W + L_tau + L_Y
        print("Iteration %d: loglike=%e (%.3f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-12:
            print("Converged.")
            break
        L_last = L

    #return


    #print(shape(yh))
    plt.clf()
    WX_params = WX.get_parameters()
    fh = WX_params[0] * np.ones(y.shape)
    err_fh = 2*np.sqrt(WX_params[1]) * np.ones(y.shape)
    m_errorplot(np.arange(N), fh, err_fh, err_fh)
    m_plot(np.arange(N), f, 'g')
    m_plot(np.arange(N), y, 'r+')
        
    tau.show()



def test_normal():

    M = 10
    N = 5

    # mu
    mu = NodeNormal(0.0, 10**-5, name="mu", plates=())
    print("Prior for mu:")
    mu.update()
    mu.show()

    # tau
    tau = NodeGamma(10**-5, 10**-5, plates=(N,), name="tau")
    print("Prior for tau:")
    tau.update()
    tau.show()

    # x
    x = NodeNormal(mu, tau, plates=(M,N), name="x")
    print("Prior for x:")
    x.update()
    x.show()

    # y (generate data)
    y = NodeNormal(x, 1, plates=(M,N), name="y")
    y.observe(random.normal(loc=10, scale=10, size=(M,N)))

    # Inference
    for i in range(50):
        x.update()
        mu.update()
        tau.update()

    print("Posterior for mu:")
    mu.show()
    print("Posterior for tau:")
    tau.show()
    print("Posterior for x:")
    x.show()
    
    return
    
def test_multivariate():    

    D = 3
    N = 100
    M = 200

    # mu
    mu = NodeGaussian(np.zeros(D), 10**(-10)*np.identity(D), plates=(M,1), name='mu')
    print("Prior for mu:")
    mu.update()
    mu.show()

    # Lambda
    Lambda = NodeWishart(D, (10**-10) * np.identity(D), plates=(1,N), name='Lambda')
    print("Prior for Lambda:")
    Lambda.update()
    Lambda.show()

    #Y = NodeGaussian(mu, 10**(-2)*identity(D), plates=(M,N), name='Y')
    Y = NodeGaussian(mu, Lambda, plates=(M,N), name='Y')
    Y.observe(random.normal(loc=10, scale=10, size=(M,N,D)))

    ## # y (generate data)
    ## for i in range(100):
    ##     y = NodeGaussian(mu, Lambda)
    ##     v = random.normal(0,1, D)
    ##     y.fix(v)

    # Inference
    try:
        for i in range(50):
            mu.update()
            Lambda.update()

        print("Posterior for mu:")
        mu.show()
        print("Posterior for Lambda:")
        Lambda.show()
    except Exception:
        pass
    

if __name__ == '__main__':

    # FOR INTERACTIVE SESSIONS, NON-BLOCKING PLOTTING:
    plt.ion()

    test_pca()
    #profile.run('test_pca()', 'profile.tmp')
    #test_normal()
    #test_multivariate()

