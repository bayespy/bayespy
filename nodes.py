
import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.spatial.distance as distance
import scipy.optimize as optimize

from utils import *

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

    def get_all_vb_terms(self):
        vb_terms = self.get_vb_term()
        for (child,index) in self.children:
            vb_terms |= child.get_vb_term()
        return vb_terms

    def get_vb_term(self):
        vb_terms = set()
        for (child,index) in self.children:
            vb_terms |= child.get_vb_term()
        return vb_terms


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
        #self.phi = [np.array([]) for i in range(len(self.dims))]
        #self.u = [np.array([]) for i in range(len(self.dims))]
        self.phi = [np.array(0.0) for i in range(len(self.dims))]
        self.u = [np.array(0.0) for i in range(len(self.dims))]

        # Terms for the lower bound (G for latent and F for observed)
        self.g = 0
        self.f = 0

        # Not observed
        self.observed = False

        # By default, ignore all elements
        self.mask = False

    def get_vb_term(self):
        return {self.lower_bound_contribution}

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

    def get_children_vb_bound(self):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

    def start_optimization(self):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

    def stop_optimization(self):
        raise Exception("Not implemented for " + str(self.__class__))
        pass

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

    def lower_bound_contribution(self, gradient=False):
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
    def lower_bound_contribution(self, gradient=False):
        return 0

class NodeConstantScalar(NodeConstant):
    def __init__(self, a, **kwargs):
        NodeConstant.__init__(self,
                              [a],
                              plates=np.shape(a),
                              dims=[()],
                              **kwargs)

    def start_optimization(self):
        # FIXME: Set the plate sizes appropriately!!
        x0 = self.u[0]
        #self.gradient = np.zeros(np.shape(x0))
        def transform(x):
            # E.g., for positive scalars you could have exp here.
            self.gradient = np.zeros(np.shape(x0))
            self.u[0] = x
        def gradient():
            # This would need to apply the gradient of the
            # transformation to the computed gradient
            return self.gradient
            
        return (x0, transform, gradient)

    def add_to_gradient(self, d):
        #print('added to gradient in node')
        self.gradient += d
        #print(d)
        #print('self:')
        #print(self.gradient)

    def message_to_child(self, gradient=False):
        if gradient:
            #print('node sending gradient')
            return (self.u, [ [np.ones(np.shape(self.u)),
                               #self.gradient] ])
                               self.add_to_gradient] ])
        else:
            return self.u


    def stop_optimization(self):
        #raise Exception("Not implemented for " + str(self.__class__))
        pass

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
        
    #def update_moments_and_g(self):

    def compute_g_from_parents(self, u_parents):
        n = u_parents[0][0]
        V = u_parents[1][0]
        logdet_V = u_parents[1][1]
        k = self.dims[0][0]
        g = 0.5*n*logdet_V - 0.5*k*n*log(2) - multigammaln(n/2)
        return g


    #def message(self, index, u_parents):

    def compute_phi_from_parents(self, u_parents):
        return [-0.5 * u_parents[1][0],
                0.5 * u_parents[0][0]]


    def update_moments_and_g(self):
        U = m_chol(-self.phi[0])
        k = self.dims[0][0]
        #k = U[0].shape[0]
        logdet_phi0 = m_chol_logdet(U)
        self.u[0] = self.phi[1][...,np.newaxis,np.newaxis] * m_chol_inv(U)
        self.u[1] = -logdet_phi0 + m_digamma(self.phi[1], k)

        ## self.u[0] = self.phi[1][...,np.newaxis,np.newaxis] * m_chol_inv(U)
        ## self.u[1] = -m_chol_logdet(U) + m_digamma(self.phi[1], k)

        self.g = self.phi[1] * logdet_phi0 - special.multigammaln(self.phi[1], k)

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

    ## # Pseudo for GPFA:
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k3 = gp_cov_rq(magnitude=.., lengthscale=.., alpha=..)
    ## f = NodeGPSet(0, [k1,k2,k3]) # assumes block diagonality
    ## # f = NodeGPSet(0, [[k11,k12,k13],[k21,k22,k23],[k31,k32,k33]])
    ## X = GaussianFromGP(f, [ [[t0,0],[t0,1],[t0,2]], [t1,0],[t1,1],[t1,2], ..])
    ## ...
    

    ## # Construct a sum of GPs if interested only in the sum term
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k = gp_cov_sum(k1, k2)
    ## f = NodeGP(0, k)
    ## f.observe(x, y)
    ## f.update()
    ## (mp, kp) = f.get_parameters()
    
    

    ## # Construct a sum of GPs when interested also in the individual
    ## # GPs:
    ## k1 = gp_cov_se(magnitude=theta1, lengthscale=theta2)
    ## k2 = gp_cov_periodic(magnitude=.., lengthscale=.., period=..)
    ## k3 = gp_cov_delta(magnitude=theta3)
    ## f = NodeGPSum(0, [k1,k2,k3])
    ## x = np.array([1,2,3,4,5,6,7,8,9,10])
    ## y = np.sin(x[0]) + np.random.normal(0, 0.1, (10,))
    ## # Observe the sum (index 0)
    ## f.observe((0,x), y)
    ## # Inference
    ## f.update()
    ## (mp, kp) = f.get_parameters()
    ## # Mean of the sum
    ## mp[0](...)
    ## # Mean of the individual terms
    ## mp[1](...)
    ## mp[2](...)
    ## mp[3](...)
    ## # Covariance of the sum
    ## kp[0][0](..., ...)
    ## # Other covariances
    ## kp[1][1](..., ...)
    ## kp[2][2](..., ...)
    ## kp[3][3](..., ...)
    ## kp[1][2](..., ...)
    ## kp[1][3](..., ...)
    ## kp[2][3](..., ...)

class NodeWishartFromGamma(Node):
    
    def __init__(self, alpha, **kwargs):

        # Check for constant n
        if np.isscalar(alpha) or isinstance(alpha, np.ndarray):            
            alpha = NodeConstantGamma(alpha)

        #NodeVariable.__init__(self, n, V, plates=plates, dims=V.dims, **kwargs)
        k = alpha.plates[-1]
        Node.__init__(self,
                      alpha,
                      plates=alpha.plates[:-1],
                      dims=[(k,k),()],
                      **kwargs)
        
    def get_moments(self):
        u = self.parents[0].message_to_child()

        return [np.identity(self.dims[0][0]) * u[0][...,np.newaxis],
                np.sum(u[1], axis=(-1))]

    def get_message(self, index, u_parents):
        
        (m, mask) = self.message_from_children()

        # Take the diagonal
        m[0] = np.einsum('...ii->...i', m[0])
        m[1] = np.reshape(m[1], np.shape(m[1]) + (1,))
        # m[1] is ok

        mask = mask[...,np.newaxis]

        return (m, mask)
        


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


