
import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as chol
import scipy.special as special
#from numpy import *
#from scipy.special import digamma, gammaln
#from scipy.linalg.decomp_cholesky import cho_factor, cho_solve
# import pylab
import matplotlib.pyplot as plt
import time
import profile




def nested_iterator(max_inds):
    s = (range(i) for i in max_inds)
    return itertools.product(*s)

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

def broadcasted_shape(a,b):
    # Computes the resulting shape if shapes a and b are broadcasted
    # together
    l_max = max(len(a), len(b))
    s = ()
    ## print('a')
    ## print(a)
    ## print('b')
    ## print(b)
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
            
    #s = [a[i] if -i>len(b) or b[i]==1 else b[i] for i in range(-l_max,0)]
    #return tuple(s)
    

def m_chol_solve(U, B, out=None):
    # Allocate memory
    #V = tile(identity(shape(U)[-1]), shape(U)[:-2]+(1,1))
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

    #l_max = max(l_u, l_b)
    #sh = [sh_u[i] if -i>l_b or sh_b[i]==1 else sh_b[i] for i in range(-l_max,0)]
    
    if out == None:
        # Shape of the result (broadcasting rules)
        sh = broadcasted_shape(sh_u, sh_b)
        #out = np.zeros(np.shape(B))
        out = np.zeros(sh + B.shape[-1:])
    for i in nested_iterator(np.shape(U)[:-2]):

        # The goal is to run Cholesky solver for each vector of B for
        # which the matrices of U are the same (according to the
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

        ## print('B')
        ## print(B.shape)
        ## print('b')
        ## print(b.shape)
        ## print('orig b')
        ## print(orig_shape)
        ## print('ind b')
        ## print(ind_b)
        ## print('U')
        ## print(U.shape)
        ## print('Ui')
        ## print(U[i].shape)
        ## print('out')
        ## print(out.shape)
        ## print('ind out')
        ## print(ind_out)
        ## print('out(ind_out)')
        ## print(out[ind_out].shape)
        
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

## def cho_logdet(L):
##     raise Exception('Do not use this')
##     return 2*sum(log(diag(L[0])))

## def cho_inv(L):
##     # See LAPACK file dpptri.f !!
##     raise Exception('Do not use this')
##     return cho_solve(L, identity(L[0].shape[0]), overwrite_b=True)

## def cov_logdet(C):
##     raise Exception('Do not use this')
##     return cho_logdet(cho_factor(atleast_2d(C))[0])

def m_digamma(a, d):
    y = 0
    for i in range(d):
        y += special.digamma(a + 0.5*(1-i))
    return y

def m_outer(A,B):
    # Computes outer product over the last axes of A and B. The other
    # axes are broadcasted. Thus, if A has shape (..., N) and B has
    # shape (..., M), then the result has shape (..., N, M)
    ## A = reshape(A, shape(A)+(1,))
    ## d = shape(B)
    ## B = reshape(B, d[:-1] + (1,) + d[-1:])
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
            #print(arg)
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
        #msg = [np.zeros(self.get_shape(i)) for i in range(len(self.dims))]
        msg = [np.array(0.0) for i in range(len(self.dims))]
        for (child,index) in self.children:
            m = child.message_to_parent(index)
            for i in range(len(self.dims)):
                # Check broadcasting shapes
                sh = broadcasted_shape(self.get_shape(i), np.shape(m[i]))
                ## print('SHAPE IN NODE MESSAGE FROM CHILDREN')
                ## print(sh)
                try:
                    # Try exploiting broadcasting rules
                    msg[i] += m[i]
                except ValueError:
                    msg[i] = msg[i] + m[i]
                #msg[i] += m[i]
        return msg

    def message_from_parent(self):
        pass

    def message_to_child(self):
        raise Exception("Not implemented. Subclass should implement this!")

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
            m = self.message(index, u_parents)
            # Sum over singleton dimensions
            for i in range(len(m)):

                # If some dimensions of the message matrix were
                # singleton although the corresponding dimension
                # of the parents natural parameterization is not,
                # multiply the message (i.e., broadcasting)

                # Plates in the message
                dim_parent = len(self.parents[index].dims[i])
                if dim_parent > 0:
                    plates_m = np.shape(m[i])[:-dim_parent]
                else:
                    plates_m = np.shape(m[i])
                plates_parent = self.parents[index].plates

                ## print('parent index ' + str(index))
                ## print('parent is ' + self.parents[index].name)
                ## print('self is ' + self.name)
                ## print('plates for self   ' + str(self.plates))
                ## print('plates for parent ' + str(plates_parent))
                ## print('plates for msg    ' + str(plates_m))

                # Compute the multiplier (multiply by the number of
                # plates if both the message and parent have single
                # plates)
                ## r = 1
                ## for j in range(-len(self.plates),0):
                ##     if ((-j > len(plates_m) or plates_m[j] == 1) and
                ##         (-j > len(plates_parent) or plates_parent[j] == 1)):
                ##         r *= self.plates[j]
                m[i] *= self.plate_multiplier(plates_m, plates_parent)

                # Sum the dimensions of the message matrix to
                # match the dimensionality of the parents natural
                # parameterization (i.e., there may be less plates
                # for parents)

                shape_parent = self.parents[index].get_shape(i)
                #shape_self = self.get_shape(i)
                shape_m = np.shape(m[i])

                s = ()
                for j in range(len(shape_m)):
                    if j >= len(shape_parent) or shape_parent[-1-j] == 1:
                        s += (len(shape_m)-j-1,)
                m[i] = np.sum(m[i], axis=tuple(s), keepdims=True)
                    
                s = tuple(range(len(shape_m)-len(shape_parent)))
                m[i] = np.squeeze(m[i], axis=s)

            return m
        else:
            # Unknown parent
            raise Exception("Unknown parent requesting a message")

    def message(self, index, u_parents):
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
        self.phi = [np.array(np.nan) for i in range(len(self.dims))]
        self.u = [np.array(np.nan) for i in range(len(self.dims))]
        #self.u = list()
        #self.u = [None] * len(self.dims)
        
        # Allocate arrays
        #for d in self.dims:
            #self.phi.append(np.zeros(self.plates+d))
            #self.u.append(np.zeros(self.plates+d))

        self.g = np.nan

        # Not observed
        self.fixed = False

    def update(self):
        if not self.fixed:

            # Messages from parents
            u_parents = [parent.message_to_child() for parent in self.parents]
            ## u_parents = list()
            ## for parent in self.parents:
            ##     u_parents.append(parent.message_to_child())
                
            # Allocate arrays
            #for i in range(len(self.phi)):
                #self.phi[i].fill(0)
                #self.u[i].fill(0)
                
            # Update natural parameters using parents
            # self.update_phi(u_parents)
            self.update_phi_from_parents(u_parents)

            # Update natural parameters using children (just add the
            # messages to phi)
            for (child,index) in self.children:
                m = child.message_to_parent(index)
                for i in range(len(self.phi)):
                    try:
                        # Try exploiting broadcasting rules
                        self.phi[i] += m[i]
                    except ValueError:
                        self.phi[i] = self.phi[i] + m[i]
                    #self.phi[i] += m[i]

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

    ## def broadcast(self, V):
    ##     for i in range(len(self.dims)):
    ##         d = self.get_shape(i)
    ##         s = ()
    ##         for j in range(len(d)):
    ##             if j > 
            

    def message_to_child(self):
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
        # TODO: CHECK PLATES!!!
        # Messages from parents
        u_parents = [parent.message_to_child() for parent in self.parents]
        phi = self.compute_phi_from_parents(u_parents)
        # NOTE: compute_g_from_parents and self.g should have been
        # broadcasted properly to plates!!
        L = self.compute_g_from_parents(u_parents)
        L -= self.g
        for (phi_p, phi_q, u_q) in zip(phi, self.phi, self.u):
            # This broadcasts to plates properly
            L += np.sum((phi_p-phi_q) * u_q)
        return L
            
    def fix_moments_and_g(self, u, minus_f):
        #print(u)
        for (i,v) in enumerate(u):
            # This is what the dimensionality "should" be
            s = self.plates + self.dims[i]
            t = np.shape(v)
            ## for j in range(max(len(s),len(t))):
            ##     if j>=len(s) or (j<len(t) and s[j] != t[j] and t[j] != 1):
            if s != t:
                    msg = "Dimensionality of the observations incorrect."
                    msg += "\nShape of input: " + str(t)
                    msg += "\nExpected shape: " + str(s)
                    msg += "\nCheck plates."
                    raise Exception(msg)
            self.phi[i] = 0
            
        self.u = u
        self.fixed = True
        self.g = minus_f


class NodeConstant(NodeVariable):
    def __init__(self, u, **kwargs):
        NodeVariable.__init__(self, **kwargs)
        self.fix_moments_and_g(u, 0)

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

    ## def update_phi(self, u_parents):
    ##     self.phi[0] += -u_parents[1][0]
    ##     self.phi[1] += u_parents[0][0]

    def compute_g_from_parents(self, u_parents):
        a = u_parents[0][0]
        gammaln_a = special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = (np.sum(a*b)
             * self.plate_multiplier(np.shape(a), np.shape(b)))
        g -= (np.sum(gammaln_a)
              * self.plate_multiplier(np.shape(gammaln_a)))
        return g
        ## return np.sum(np.ones(self.plates) *
        ##               (u_parents[0][0] * u_parents[1][1]
        ##                - special.gammaln(u_parents[0][0])))
        
    def update_moments_and_g(self):
        log_b = np.log(-self.phi[0])
        self.u[0] = self.phi[1] / (-self.phi[0])
        self.u[1] = special.digamma(self.phi[1]) - log_b
        ## self.u[0] += self.phi[1] / (-self.phi[0])
        ## self.u[1] += digamma(self.phi[1]) - log(-self.phi[0])
        
        self.g = (np.sum(self.phi[1] * log_b)
                  * self.plate_multiplier(np.shape(self.phi[1]),
                                          np.shape(log_b)))
        self.g -= (np.sum(special.gammaln(self.phi[1]))
                   * self.plate_multiplier(np.shape(self.phi[1])))
        ## self.g = np.sum(np.ones(self.plates) *
        ##                 (self.phi[1] * log_b - special.gammaln(self.phi[1])))

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

    ## def update_phi(self, u_parents):
    ##     self.phi[0] += u_parents[1][0] * u_parents[0][0]
    ##     self.phi[1] += -u_parents[1][0] / 2

    def update_moments_and_g(self):
        self.u[0] = -self.phi[0] / (2*self.phi[1])
        self.u[1] = self.u[0]**2 - 1 / (2*self.phi[1])
        ## self.u[0] += -self.phi[0] / (2*self.phi[1])
        ## self.u[1] += self.u[0]**2 - 1 / (2*self.phi[1])
        self.g = (-0.5*np.sum(self.u[0]*self.phi[0])
                  * self.plate_multiplier(np.shape(self.u[0]),
                                          np.shape(self.phi[0])))
        self.g += (0.5*np.sum(np.log(-2*self.phi[1]))
                   * self.plate_multiplier(np.shape(self.phi[1])))
        ## self.g = (-0.5*np.sum(np.ones(self.plates)*self.u[0]*self.phi[0])
        ##           + 0.5*np.sum(np.ones(self.plates)*np.log(-2*self.phi[1])))

    def compute_g_from_parents(self, u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        tau = u_parents[1][0]
        log_tau = u_parents[1][1]
        g = (-0.5 * np.sum(mumu*tau)
             * self.plate_multiplier(np.shape(mumu), np.shape(tau)))
        g += (0.5 * np.sum(log_tau)
              * self.plate_multiplier(np.shape(log_tau)))
        return g
        ## return np.sum(np.ones(self.plates) *
        ##               (-0.5*u_parents[0][1]*u_parents[1][0]
        ##                + 0.5*u_parents[1][1]))

    def message(self, index, u_parents):
        if index == 0:
            return [u_parents[1][0] * self.u[0],
                    -0.5 * u_parents[1][0]]
            #return [u_parents[1][0] * self.u[0] * np.ones(self.plates),
            #        -0.5 * u_parents[1][0] * np.ones(self.plates)]
        elif index == 1:
            # The second element can be 0.5 or
            # 0.5*ones(self.u[0].shape), how to implement it?
            # Multiply before summing? That'd be broadcasting. :)
            ## if self.u[0] == None:
            ##     print('None u0')
            ## if self.u[1] == None:
            ##     print('None u1')
            ## if u_parents[0][0] == None:
            ##     print('None pu00')
            ## if u_parents[0][1] == None:
            ##     print('None pu01')
                
            return [-0.5 * (self.u[1] - 2*self.u[0]*u_parents[0][0] + u_parents[0][1]),
                    0.5]
            #return [-0.5 * (self.u[1] - 2*self.u[0]*u_parents[0][0] + u_parents[0][1]) * np.ones(self.plates),
            #        0.5 * np.ones(self.plates)]

    def observe(self, x):
        g = 0.5 * np.log(2*np.pi) * self.plate_multiplier()#np.prod(self.plates)
        self.fix_moments_and_g([x, x**2], g)

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
            ## V = atleast_2d(V)
            ## V = NodeConstant([V, m_chol_logdet(m_chol(V))],
            ##                  plates=shape(V)[:-2],
            ##                  dims=[shape(V)[-2:], ()])

        NodeVariable.__init__(self, n, V, plates=plates, dims=V.dims, **kwargs)
        
    def compute_phi_from_parents(self, u_parents):
        return [-0.5 * u_parents[1][0],
                0.5 * u_parents[0][0]]

    ## def update_phi(self, u_parents):
    ##     self.phi[0] += -0.5 * u_parents[1][0]
    ##     self.phi[1] += 0.5 * u_parents[0][0]

    def update_moments(self):
        U = m_chol(-self.phi[0])
        k = U[0].shape[0]
        self.u[0] = self.phi[1][...,np.newaxis,np.newaxis] * m_chol_inv(U)
        self.u[1] = -m_chol_logdet(U) + m_digamma(self.phi[1], k)
        ## self.u[0] += self.phi[1][...,newaxis,newaxis] * m_chol_inv(U)
        ## self.u[1] += -m_chol_logdet(U) + m_digamma(self.phi[1], k)

    def message(self, index, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            # x_mu = dot(self.u[0], u_parents[0][0].T)
            return [-0.5 * self.u[0],
                    0.5 * self.u_parents[0][0]]
            ## return [-0.5 * self.u[0],
            ##         0.5 * self.u_parents[0][0] * np.ones(self.plates)]

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

    ## def update_phi(self, u_parents):
    ##     self.phi[0] += m_dot(u_parents[1][0], u_parents[0][0])
    ##     self.phi[1] += -0.5 * u_parents[1][0]

    def update_moments_and_g(self):
        L = m_chol(-self.phi[1])
        self.u[0] = m_chol_solve(L, 0.5*self.phi[0])
        #print(np.shape(self.u[0]))
        self.u[1] = (m_outer(self.u[0], self.u[0])
                     + 0.5 * m_chol_inv(L))
        ## self.u[1] += (m_outer(self.u[0], self.u[0])
        ##               + 0.5 * m_chol_inv(L))
        self.g = (-0.5 * np.sum(self.u[0]*self.phi[0])
                  * self.plate_multiplier(np.shape(self.u[0])[:-1],
                                          np.shape(self.phi[0])[:-1]))
        self.g += (0.5 * np.sum(m_chol_logdet(L))
                   * self.plate_multiplier(np.shape(L)[:-2]))
        # Cholesky factor L needs to be multiplied by two:
        self.g += (0.5 * np.log(2) * self.dims[0][0]
                   * self.plate_multiplier())
        ## self.g = (-0.5 * np.sum(self.u[0]*self.phi[0])
        ##           + 0.5 * (np.sum(m_chol_logdet(L))
        ##                    + np.log(2) * np.prod(self.plates+self.dims[0])))


    def compute_g_from_parents(self, u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.sum(np.einsum('...ij,...ij',mumu,Lambda))
             * self.plate_multiplier(mumu.shape[:-2],
                                     Lambda.shape[:-2]))
        g += (0.5 * np.sum(logdet_Lambda)
              * self.plate_multiplier(np.shape(logdet_Lambda)))
        return g
        ## return np.sum(np.ones(self.plates) *
        ##               (-0.5*np.einsum('...ij,...ij',u_parents[0][1],u_parents[1][0])
        ##                + 0.5*u_parents[1][1]))

    def message(self, index, u_parents):
        if index == 0:
            return [m_dot(u_parents[1][0], self.u[0]),
                    -0.5 * u_parents[1][0]]
            ## return [m_dot(u_parents[1][0], self.u[0]) * np.ones(self.get_shape(0)),
            ##         -0.5 * u_parents[1][0] * np.ones(self.get_shape(1))]
        elif index == 1:
            xmu = m_outer(self.u[0], u_parents[0][0])
            return [-0.5 * (self.u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]
            ## return [-0.5 * (self.u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]) * np.ones(self.get_shape(1)),
            ##         0.5 * np.ones(self.plates)]


    def random(self):
        U = m_chol(-self.phi[1])
        return self.u[0] + 0.5 * m_chol_solve(U,
                                              np.random.normal(0,
                                                               1,
                                                               np.shape(self.u[0])))

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
        #print(shape(self.u[0]))
        #print(shape(self.u[1]))

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

        ## print('Plates in NodeDot init')
        ## print(self.plates)
        ## raise Exception('debuggin')
            

    def compute_moments(self):
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

        ## print(str1)
        ## print(str2)
        x = [np.einsum(str1, *u1),
             np.einsum(str2, *u2)]

        ## print(np.sum(x[0]))
        ## print(np.sum(x[1]))

        return x
        

        # TODO: Probably you could this more efficiently using einsum
        # directly, without computing the large moment matrices!!!

        # Compute the moments of the products
        x = [np.ones(self.d),
             np.ones(self.d + self.d)]
        ## x = [np.ones(self.get_shape(0) + self.d),
        ##      np.ones(self.get_shape(1) + self.d + self.d)]
        for parent in self.parents:
            u = parent.message_to_child()
            # Product of the means
            try:
                x[0] *= u[0]
            except ValueError:
                x[0] = x[0] * u[0]
            # Product of the second moments
            try:
                x[1] *= u[1]
            except ValueError:
                x[1] = x[1] * u[1]

        # Sum the "latent" dimension
        x[0] = np.sum(x[0], axis=-1)
        x[1] = np.sum(x[1], axis=(-2,-1))

        return x

    def get_parameters(self):
        # Compute mean and variance
        u = self.compute_moments()
        u[1] -= u[0]**2
        return u
        

    def message_to_child(self):
        return self.compute_moments()

            
    def message(self, index, u_parents):
        
        m = self.message_from_children()

        #print(np.shape(m[0]))
        #print(np.shape(m[1]))

        # TODO: You could sum some dimensions so that you don't need
        # to store large m unless m actually is large

        m[0] = np.repeat(m[0][...,np.newaxis], self.d, axis=-1)
        m[1] = np.repeat(np.repeat(m[1][...,np.newaxis,np.newaxis],
                                   self.d,
                                   axis=-1),
                         self.d,
                         axis=-2)
        for (i,u) in enumerate(u_parents):
            if i != index:
                try:
                    m[0] *= u[0]
                except ValueError:
                    m[0] = m[0] * u[0]

                try:
                    m[1] *= u[1]
                except ValueError:
                    m[1] = m[1] * u[1]

        return m
        

        

## # TODO: How to do constant nodes?
## # x = NodeGaussian(array([0, 0]), array([[1, 0.5], [0.5, 1]])
        
## # TODO: How to do observations?
## # y.observe(array([[1,2,3],[4,5,6]]))

# TODO: Compute lower bound!

# TODO: How to do missing values?


# def plot_envelope(x,Y,L,U):


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
        D = 3
        # Generate data
        w = np.random.normal(0, 1, size=(M,1,D))
        x = np.random.normal(0, 1, size=(1,N,D))
        f = np.einsum('...i,...i', w, x)
        y = f + np.random.normal(0, 0.5, size=(M,N))
    elif dataset == 2:
        # Data from matlab comparison
        f = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_f.txt')
        y = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_y.txt')
        D = np.genfromtxt('/home/jluttine/matlab/fa/data_pca_01_d.txt')
        (M,N) = np.shape(y)

    # Construct the PCA model

    X = NodeGaussian(np.zeros(D), np.identity(D), name="X", plates=(1,N))
    X.update()
    X.u[0] = X.random()

    W = NodeGaussian(np.zeros(D), np.identity(D), name="W", plates=(M,1))
    W.update()
    W.u[0] = W.random()

    WX = NodeDot(W,X)

    tau = NodeGamma(1e-5, 1e-5, name="tau")
    tau.update()

    Y = NodeNormal(WX, tau, name="Y", plates=(M,N))

    # Initialize (from prior)

    # Hmm.. I'd like these to obtain the prior because no data is
    # given yet. Needs the missing value handling!
    ## tau.show()
    ## Y.update()
    ## W.update()
    ## X.update()
    ## tau.update()
    ## Y.update()
    ## W.update()
    ## X.update()
    ## tau.update()
    ## tau.show()
    ## return

    Y.observe(y)

    #try:
    # Inference
    L_last = -np.inf
    for i in range(500):
        t = time.clock()
        X.update()
        #print("X")
        #print(X.u[1][0,0,:,:])
        W.update()
        #print("W")
        #print(W.u[1][0,0,:,:])
        tau.update()
        #print("tau")
        #print(tau.u[0])

        L_X = X.lower_bound_contribution()
        L_W = W.lower_bound_contribution()
        L_tau = tau.lower_bound_contribution()
        L_Y = Y.lower_bound_contribution()
        print("X: %f, W: %f, tau: %f, Y: %f" % (L_X, L_W, L_tau, L_Y))
        L = L_X + L_W + L_tau + L_Y
        print("Iteration %d: loglike=%f (%f seconds)" % (i+1, L, time.clock()-t))
        if L_last > L:
            L_diff = (L_last - L)
            raise Exception("Lower bound decreased %e! Bug somewhere or numerical inaccuracy?" % L_diff)
        if L - L_last < 1e-8:
            print("Converged.")
            break
        L_last = L

    #return

    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(np.arange(N), y.T)

    plt.subplot(2,1,2)
    yh = np.einsum('...i,...i', W.u[0], X.u[0])
    plt.plot(np.arange(N), yh.T)

    #print(shape(yh))
    plt.clf()
    WX_params = WX.get_parameters()
    m_errorplot(np.arange(N), WX_params[0], 2*np.sqrt(WX_params[1]), 2*np.sqrt(WX_params[1]))
    m_plot(np.arange(N), yh, 'k')
    m_plot(np.arange(N), f, 'g')
    m_plot(np.arange(N), y, 'r+')
    #except Exception:
     #   print('WAT IS TIS ERROR??')
      #  pass
        
    tau.show()
    #print(tau.u[0])

    print(X.phi[0].shape)
    print(X.phi[1].shape)
    #print(X.phi)
    
    #X.show()
    #W.show()

    

    #print(shape(y))
    #print(y)


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
    #profile.run('test_pca()')
    #test_normal()
    #test_multivariate()

