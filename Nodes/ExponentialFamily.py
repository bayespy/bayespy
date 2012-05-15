
import itertools
import numpy as np
import scipy as sp
import scipy.linalg.decomp_cholesky as decomp
import scipy.linalg as linalg
import scipy.special as special
import scipy.spatial.distance as distance

import imp

import utils
imp.reload(utils)

## # Differentiate model and inference (and data).
## #
## # Pseudo code:

## # Model:
## mu = Gaussian(0, 1, plates=(10,))
## tau = Gamma(0.1, 0.1, plates=(10,))
## Y = Gaussian(X, tau, plates=(5,10))

## # Data;
## Y.observe(rand(5,10))

## # Inference engine
## Q = VB(X,tau,Y) # or Q = Gibbs(X,tau,Y) or Q = EP(..), Q = MAP(..)
## # or
## Q = VB()
## Q.add(VBGaussian(X))
## Q.add(VBGamma(tau))
## Q.add(VBGaussian(Y))
## # Inference algorithm
## Q.inference(maxiter=100)
## # or
## for i in range(100):
##     Q(X).update()
##     Q(tau).update()



# Gradients:
#
# ( (x1,...,xn), [ (dx1,...,dxn,callback), ..., (dx1,...,dxn,callback) ] )
    



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

    def __init__(self, *args, dims=None, plates=(), name=""):

        if dims is None:
            raise Exception("You need to specify the dimensionality" \
                            + " of the distribution for class" \
                            + str(self.__class__))
        self.dims = dims
        self.plates = plates
        self.name = name

        # Parents
        self.parents = args
        # Inform parent nodes
        for (index,parent) in enumerate(self.parents):
            if parent:
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
            utils.broadcasted_shape(self.plates, arg)
            
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
                sh = utils.broadcasted_shape(self.get_shape(i), np.shape(m[i]))
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
        raise NotImplementedError()

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
            s = utils.axes_to_collapse(np.shape(my_mask), parent.plates)
            mask = np.any(my_mask, axis=s, keepdims=True)
            mask = utils.squeeze_to_dim(mask, len(parent.plates))
            
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


                s = utils.axes_to_collapse(shape_m, shape_parent)
                m[i] = np.sum(m[i], axis=s, keepdims=True)

                m[i] = utils.squeeze_to_dim(m[i], len(shape_parent))
                m[i] *= r

            return (m, mask)
        else:
            # Unknown parent
            raise Exception("Unknown parent requesting a message")

    def get_message(self, index, u_parents):
        raise Exception("Not implemented.")
        pass


class ExponentialFamily(Node):

    # Overwrite this
    ndims = None

    @staticmethod
    def compute_logpdf(u, phi, g, f):
        """ Compute E[log p(X)] given E[u], E[phi], E[g] and
        E[f]. Does not sum over plates."""
        L = g + f
        for (phi_i, u_i, len_dims_i) in zip(phi, u, len_dims):
            # Axes to sum (dimensions of the variable, not the plates)
            axis_sum = tuple(range(-len_dims_i,0))
            # Compute the term
            # TODO/FIXME: Use einsum!
            L = L + np.sum(phi_i * u_i, axis=axis_sum)
        return L

    @staticmethod
    def compute_phi_from_parents(u_parents):
        """ Compute E[phi] over q(parents) """
        raise NotImplementedError()

    @staticmethod
    def compute_g_from_parents(u_parents):
        """ Compute E[g(phi)] over q(parents) """
        raise NotImplementedError()

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        """ Compute E[u] and g(phi) for given phi """
        raise NotImplementedError()

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        raise NotImplementedError()

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        raise NotImplementedError()

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi and u. """
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):

        super().__init__(*args,
                         dims=self.compute_dims(*args),
                         **kwargs)

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
        return (self.compute_message(index, self.u, u_parents),
                self.mask)
        ## return (self.message(index, u_parents),
        ##         self.mask)

    def initialize(self):
        if not np.all(self.observed):

            # Messages from parents
            u_parents = [parent.message_to_child() for parent in self.parents]
                
            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            # Update moments
            self.update_u_and_g()


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
            self.update_u_and_g()

    def update_phi_from_parents(self, u_parents):
        # This makes correct broadcasting
        self.phi = self.compute_phi_from_parents(u_parents)
        # Make sure phi has the correct number of axes. It makes life
        # a bit easier elsewhere.
        for i in range(len(self.phi)):
            axes = len(self.plates) + self.ndims[i] - np.ndim(self.phi[i])
            print('update_phi_from_parents:')
            print(np.shape(self.phi[i]))
            print(self.plates)
            print(self.ndims[i])
            self.phi[i] = utils.add_leading_axes(self.phi[i], axes)
                                                 
        ## for i in range(len(self.phi)):
        ##     self.phi[i].fill(0)
        ##     try:
        ##         # Try exploiting broadcasting rules
        ##         self.phi[i] += phi[i]
        ##     except ValueError:
        ##         self.phi[i] = self.phi[i] + phi[i]
    
    def get_moments(self):
        return self.u

    def get_children_vb_bound(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def start_optimization(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def stop_optimization(self):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def message(self, index, u_parents):
        raise NotImplementedError("Not implemented for " + str(self.__class__))

    def update_u_and_g(self):

        # Mask for plates to update (i.e., unobserved plates)
        update_mask = np.logical_not(self.observed)
        # Compute u and g
        (u, g) = self.compute_u_and_g(self.phi, mask=update_mask)

        # Store the computed moments u but do not change moments for
        # observations, i.e., utilize the mask.
        for ind in range(len(u)):
            # Add axes to the mask for the variable dimensions (mask
            # contains only axes for the plates).
            u_mask = utils.add_trailing_axes(update_mask, self.ndims[ind])

            # Enlarge self.u[ind] as necessary so that it can store the
            # broadcasted result.
            sh = utils.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
            self.u[ind] = utils.repeat_to_shape(self.u[ind], sh)

            # Use mask to update only unobserved plates and keep the
            # observed as before
            np.copyto(self.u[ind],
                      u[ind],
                      where=u_mask)

    def lower_bound_contribution(self, gradient=False):
        # Compute E[ log p(X|parents) - log q(X) ] over q(X)q(parents)
        
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
            latent_mask = utils.add_axes(latent_mask,
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
        #return L
            
    def fix_u_and_f(self, u, f, mask=1):
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
            obs_mask = utils.add_axes(mask,
                                len(self.plates) - np.ndim(mask),
                                len(self.dims[i]))
            self.u[i] = (obs_mask * u[i]
                         + np.logical_not(obs_mask) * self.u[i])

        self.f = f
        
        # Observed nodes should not be ignored
        self.mask = np.logical_or(self.mask, self.observed)

    def observe(self, x, mask=1):
        (u, f) = self.compute_fixed_u_and_f(x)
        #print(u)
        self.fix_u_and_f(u, f, mask=mask)
        ## print(x)
        ## print(mask)
        ## print(u)

class NodeConstant(Node):
    def __init__(self, u, **kwargs):
        self.u = u
        Node.__init__(self, **kwargs)

    def message_to_child(self, gradient=False):
        if gradient:
            return (self.u, [])
        else:
            return self.u

        #self.fix_u_and_f(u, 0, True)
    #def lower_bound_contribution(self, gradient=False):
        #return 0

## class NodeConstant(ExponentialFamily):
##     def __init__(self, u, **kwargs):
##         ExponentialFamily.__init__(self, **kwargs)
##         self.fix_u_and_f(u, 0, True)
##     def lower_bound_contribution(self, gradient=False):
##         return 0

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
            #print('NodeConstantScalar.transform')
            self.gradient = np.zeros(np.shape(x0))
            self.u[0] = x
        def gradient():
            # This would need to apply the gradient of the
            # transformation to the computed gradient
            return self.gradient
            
        return (x0, transform, gradient)

    def add_to_gradient(self, d):
        #print('added to gradient in node')
        #print('NodeConstantScalar.add_to_gradient')
        self.gradient += d
        #print(self.gradient)
        #print(d)
        #print('self:')
        #print(self.gradient)

    def message_to_child(self, gradient=False):
        if gradient:
            #print('node sending gradient', np.shape(self.u))
            return (self.u, [ [np.ones(np.shape(self.u[0])),
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
                              [X, utils.m_outer(X, X)],
                              plates=X.shape[:-1],
                              dims=[(d,), (d,d)],
                              **kwargs)
        
class NodeConstantWishart(NodeConstant):
    def __init__(self, Lambda, **kwargs):
        Lambda = np.atleast_2d(Lambda)
        if Lambda.shape[-1] != Lambda.shape[-2]:
            raise Exception("Lambda not a square matrix.")
        NodeConstant.__init__(self,
                              [Lambda, utils.m_chol_logdet(utils.m_chol(Lambda))],
                              plates=Lambda.shape[:-2],
                              dims=[Lambda.shape[-2:], ()],
                              **kwargs)


# Gamma(a, b)
class NodeGamma(ExponentialFamily):

    ndims = (0, 0)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [-u_parents[1][0],
                1*u_parents[0][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        a = u_parents[0][0]
        gammaln_a = special.gammaln(a)
        b = u_parents[1][0]
        log_b = u_parents[1][1]
        g = a * log_b - gammaln_a
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        log_b = np.log(-phi[0])
        u0 = phi[1] / (-phi[0])
        u1 = special.digamma(phi[1]) - log_b
        u = [u0, u1]
        g = phi[1] * log_b - special.gammaln(phi[1])
        return (u, g)
        

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return [-u[0],
                    u_parents[0][0]]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        return [(), ()]

    def __init__(self, a, b, plates=(), **kwargs):

        # TODO: USE asarray(a)

        # Check for constant a
        if np.isscalar(a) or isinstance(a, np.ndarray):
            a = NodeConstantScalar(a)

        # Check for constant b
        if np.isscalar(b) or isinstance(b, np.ndarray):
            b = NodeConstant([b, np.log(b)], plates=np.shape(b), dims=[(),()])

        # Construct
        super().__init__(a, b, plates=plates, **kwargs)

    def show(self):
        a = self.phi[1]
        b = -self.phi[0]
        print("Gamma(" + str(a) + ", " + str(b) + ")")


class NodeNormal(ExponentialFamily):

    ndims = (0, 0)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        phi = [u_parents[1][0] * u_parents[0][0],
               -u_parents[1][0] / 2]
        return phi

    @staticmethod
    def compute_g_from_parents(u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        tau = u_parents[1][0]
        log_tau = u_parents[1][1]
        g = -0.5 * mumu*tau + 0.5 * log_tau
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        u0 = -phi[0] / (2*phi[1])
        u1 = u0**2 - 1 / (2*phi[1])
        u = [u0, u1]
        g = (-0.5 * u[0] * phi[0] + 0.5 * np.log(-2*phi[1]))
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        u = [x, x**2]
        f = -np.log(2*np.pi)/2
        return (u, f)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [u_parents[1][0] * u[0],
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            return [-0.5 * (u[1] - 2*u[0]*u_parents[0][0] + u_parents[0][1]),
                    0.5]
        raise NotImplementedError()

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        return [(), ()]

    # Normal(mu, 1/tau)

    def __init__(self, mu, tau, plates=(), **kwargs):

        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = NodeConstant([mu, mu**2], plates=np.shape(mu), dims=[(),()])

        # Check for constant tau
        if np.isscalar(tau) or isinstance(tau, np.ndarray):
            tau = NodeConstant([tau, log(tau)], plates=np.shape(tau), dims=[(),()])

        # Construct
        super().__init__(mu, tau, plates=plates, **kwargs)


    def show(self):
        mu = self.u[0]
        s2 = self.u[1] - mu**2
        print("Normal(" + str(mu) + ", " + str(s2) + ")")


class Wishart(ExponentialFamily):

    ndims = (2, 0)

    @staticmethod
    def compute_g_from_parents(u_parents):
        n = u_parents[0][0]
        V = u_parents[1][0]
        logdet_V = u_parents[1][1]
        k = np.shape(V)[-1]
        #k = self.dims[0][0]
        g = 0.5*n*logdet_V - 0.5*k*n*log(2) - multigammaln(n/2)
        return g

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [-0.5 * u_parents[1][0],
                0.5 * u_parents[0][0]]

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        U = utils.m_chol(-phi[0])
        k = np.shape(phi[0])[-1]
        #k = self.dims[0][0]
        logdet_phi0 = utils.m_chol_logdet(U)
        u0 = phi[1][...,np.newaxis,np.newaxis] * utils.m_chol_inv(U)
        u1 = -logdet_phi0 + utils.m_digamma(phi[1], k)
        u = [u0, u1]
        g = phi[1] * logdet_phi0 - special.multigammaln(phi[1], k)
        return (u, g)

    @staticmethod
    def message(index, u, u_parents):
        if index == 0:
            raise Exception("No analytic solution exists")
        elif index == 1:
            return (-0.5 * u[0],
                    0.5 * u_parents[0][0])

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        # Has the same dimensionality as the second parent.
        return parents[1].dims

    # Wishart(n, inv(V))

    def __init__(self, n, V, plates=(), **kwargs):

        # Check for constant n
        if np.isscalar(n) or isinstance(n, np.ndarray):            
            n = NodeConstantScalar(n)
            
        # Check for constant V
        if np.isscalar(V) or isinstance(V, np.ndarray):
            V = NodeConstantWishart(V)

        ExponentialFamily.__init__(self, n, V, plates=plates, **kwargs)
        
    def show(self):
        print("Wishart(n, A)")
        print("  n =")
        print(2*self.phi[1])
        print("  A =")
        print(0.5 * self.u[0] / self.phi[1][...,np.newaxis,np.newaxis])

class Gaussian(ExponentialFamily):

    ndims = (1, 2)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        print('in Gaussian.compute_phi_from_parents')
        print(u_parents)
        print(np.shape(u_parents[1][0]))
        print(np.shape(u_parents[0][0]))
        return [utils.m_dot(u_parents[1][0], u_parents[0][0]),
                -0.5 * u_parents[1][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * np.sum(logdet_Lambda))
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        L = utils.m_chol(-phi[1])
        k = np.shape(phi[0])[-1]
        # Moments
        u0 = utils.m_chol_solve(L, 0.5*phi[0])
        u1 = utils.m_outer(u0, u0) + 0.5 * utils.m_chol_inv(L)
        u = [u0, u1]
        # G
        g = (-0.5 * np.einsum('...i,...i', u[0], phi[0])
             + 0.5 * utils.m_chol_logdet(L)
             + 0.5 * np.log(2) * k)
             #+ 0.5 * np.log(2) * self.dims[0][0])
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(x)[-1]
        u = [x, utils.m_outer(x,x)]
        f = -k/2*np.log(2*np.pi)
        return (u, f)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [utils.m_dot(u_parents[1][0], u[0]),
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            xmu = utils.m_outer(u[0], u_parents[0][0])
            return [-0.5 * (u[1] - xmu - xmu.swapaxes(-1,-2) + u_parents[0][1]),
                    0.5]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi and u. """
        # Has the same dimensionality as the first parent.
        print('in gaussian compute dims: parent.dims:', parents[0].dims)
        print('in gaussian compute dims: parent.u:', parents[0].u)
        return parents[0].dims

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
        super().__init__(mu, Lambda,
                         plates=plates,
                         **kwargs)

    def random(self):
        U = utils.m_chol(-self.phi[1])
        return (self.u[0]
                + 0.5 * utils.m_chol_solve(U,
                                     np.random.normal(0, 1,
                                                      self.get_shape(0))))

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - utils.m_outer(mu, mu)
        print("Gaussian(mu, Cov)")
        print("  mu = ")
        print(mu)
        print("  Cov = ")
        print(str(Cov))

    ## def observe(self, x):
    ##     self.fix_moments([x, utils.m_outer(x,x)])

class ConstantDirichlet(NodeConstant):
    def __init__(self, x, **kwargs):
        x = np.atleast_1d(X)
        d = x.shape[-1]
        super().__init__([np.log(x)],
                         plates=x.shape[:-1],
                         dims=[(d,)],
                         **kwargs)


class Dirichlet(ExponentialFamily):

    ndims = (1,)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [u_parents[0][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        return u_parents[0][1]

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        sum_gammaln = np.sum(special.gammaln(phi[0]), axis=-1)
        gammaln_sum = special.gammaln(np.sum(phi[0], axis=-1))
        psi_sum = special.psi(np.sum(phi[0], axis=-1, keepdims=True))
        
        # Moments <log x>
        u0 = special.psi(phi[0]) - psi_sum
        u = [u0]
        # G
        g = gammaln_sum - sum_gammaln
        return (u, g)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [u[0], 1]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        # Has the same dimensionality as the parent for its first
        # moment.
        return parents[0].dims[:1]

    def __init__(self, alpha, plates=(), **kwargs):

        # Check for constant alpha
        if np.isscalar(alpha) or isinstance(alpha, np.ndarray):
            gammaln_sum = special.gammaln(np.sum(alpha, axis=-1))
            sum_gammaln = np.sum(special.gammaln(alpha), axis=-1)
            z = gammaln_sum - sum_gammaln
            d = np.shape(alpha)[-1]
            alpha = NodeConstant([alpha, z],
                                 plates=np.shape(alpha)[:-1],
                                 dims=((d,), ()))

        # Construct
        super().__init__(alpha,
                         plates=plates,
                         **kwargs)

    def random(self):
        raise NotImplementedError()

    def show(self):
        alpha = self.phi[0]
        print("Dirichlet(alpha)")
        print("  alpha = ")
        print(alpha)



def Categorical(p, **kwargs):

    # Get the number of categories (static methods may need this)
    if np.isscalar(p) or isinstance(p, np.ndarray):
        n_categories = np.shape(p)[-1]
    else:
        n_categories = p.dims[0][0]

    # The actual categorical distribution node
    class _Categorical(ExponentialFamily):

        ndims = (1,)

        @staticmethod
        def compute_logpdf(u, u_parents):
            return np.einsum(u[0], u_parents[0][0], axis=-1)

        @staticmethod
        def compute_phi_from_parents(u_parents):
            return [u_parents[0][0]]

        @staticmethod
        def compute_g_from_parents(u_parents):
            return 0 #-np.log(np.sum(np.exp(u_parents[0][0]), axis=-1))

        @staticmethod
        def compute_u_and_g(phi, mask=True):
            p = np.exp(phi[0])
            sum_p = np.sum(p, axis=-1, keepdims=True)
            # Moments
            u0 = p / sum_p
            u = [u0]
            # G
            g = -np.log(sum_p)
            return (u, g)

        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """

            # TODO: You could check that x has proper dimensions
            x = np.array(x, dtype=np.int)

            u0 = np.zeros((np.size(x), n_categories))
            u0[[np.arange(np.size(x)), x]] = 1
            f = 0
            return ([u0], f)

        @staticmethod
        def compute_message(index, u, u_parents):
            """ . """
            if index == 0:
                return (u[0],)

        @staticmethod
        def compute_dims(*parents):
            """ Compute the dimensions of phi/u. """
            # Has the same dimensionality as the parent.
            return parents[0].dims

        def __init__(self, p, **kwargs):

            # Check for constant mu
            if np.isscalar(p) or isinstance(p, np.ndarray):
                p = ConstantDirichlet(p)

            # Construct
            super().__init__(p,
                             **kwargs)


        def random(self):
            raise NotImplementedError()

        def show(self):
            p = self.u[0] #np.exp(self.phi[0])
            #p /= np.sum(p, axis=-1, keepdims=True)
            print("Categorical(p)")
            print("  p = ")
            print(p)

    return _Categorical(p, **kwargs)

    ## def observe(self, x):
    ##     self.fix_u_and_f(self.u, 0)

# Pseudo:
# Mixture(Gaussian)(z, mu, Lambda)

def Mixture(distribution, cluster_plate=-1):

    class _Mixture(ExponentialFamily):

        ndims = distribution.ndims

        @staticmethod
        def compute_phi_from_parents(u_parents):
            # Compute weighted average of the parameters

            #print('Mixture.compute_phi_from_parents', u_parents)

            # Cluster parameters
            phi = distribution.compute_phi_from_parents(u_parents[1:])
            # Contributions/weights/probabilities
            p = u_parents[0][0]
            
            for ind in range(len(phi)):
                # Compute element-wise product and then sum over K clusters.
                # Note that the dimensions aren't perfectly aligned because
                # the cluster dimension (K) may be arbitrary for phi, and phi
                # also has dimensions (Dd,..,D0) of the parameters.
                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                # Shape(p)      = [Nn,..,N0,K]
                # Shape(result) = [Nn,..,N0,Dd,..,D0]
                # General broadcasting rules apply for Nn,..,N0, that is,
                # preceding dimensions may be missing or dimension may be
                # equal to one. Probably, shape(phi) has lots of missing
                # dimensions and/or dimensions that are one.

                # The number of dimensions for the phi parameter
                # dimensions, for instance, phi for Gaussian has one
                # (1) axis for mean vector and two (2) axes for
                # precision matrix, thus ndims=(1,2).
                axes_phi = distribution.ndims[ind]
                #phi[ind] = utils.add_leading_axes(phi[ind], np.ndim(phi[ind]))
                # Move cluster axis to be the last
                if cluster_plate < 0:
                    cluster_axis = cluster_plate - axes_phi
                else:
                    cluster_axis = cluster_plate
                phi[ind] = utils.moveaxis(phi[ind], cluster_axis, -1)
                # For broadcasting, add new axes to p
                p = utils.add_trailing_axes(p, axes_phi)
                # Move cluster axis to be the last
                p = utils.moveaxis(p, -axes_phi-1, -1)
                # Product and then sum over the clusters (last axis)
                phi[ind] = utils.sum_product(p, phi[ind], axes_to_sum=-1)
                ## phi[ind] = np.einsum(phi, [Ellipsis,0],
                ##                      p, [Ellipsis,0],
                ##                      [Ellipsis])
            return phi

        @staticmethod
        def compute_g_from_parents(u_parents):
            # Compute g for clusters
            g = distribution.compute_g_from_parents(u_parents[1:])
            # Move cluster axis to last
            g = utils.moveaxis(g, cluster_axis, -1)
            # Cluster contributions/probabilities/weights
            p = u_parents[0][0]
            # Weighted average of g over the clusters
            g = utils.sum_product(p, g, axis_to_sum=-1)
            return g

        @staticmethod
        def compute_u_and_g(phi, mask=True):
            return distribution.compute_u_and_g(phi, mask=mask)

        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            return distribution.compute_fixed_u_and_f(x)
            #raise NotImplementedError()

        @staticmethod
        def compute_message(index, u, u_parents):
            """ . """
            if index == 0:
                # Compute log pdf for each element
                print('Mixture.message, u_parents:', u_parents)
                print('Mixture.distribution:', distribution)
                print(self.parents[0].__class__)
                print(self.parents[1].__class__)
                print(self.parents[2].__class__)
                phi = distribution.compute_phi_from_parents(u_parents[1:])
                g = distribution.compute_g_from_parents(u_parents[1:])
                L = distribution.compute_logpdf(u, phi, g, 0)
                # Sum over other than the cluster dimensions? No!
                # Hmm.. I think the message passing method will do
                # that automatically
                #L = np.sum(L, ...)
                return [L]

            elif index >= 1:
                # Weigh the messages with the responsibilities
                #
                # FIXME: This isn't this simple because there is an
                # axis for the clusters..
                m = distribution.compute_message(index-1, u, u_parents[1:])
                for i in range(len(m)):
                    # Responsibility for cluster i is the first
                    # parent's first moment's i-th element
                    #
                    # TODO: You could do summing here already so that
                    # you wouldn't compute huge matrices as
                    # intermediate result. Use einsum.
                    print('Mixture.compute_message:')
                    print(np.shape(m[i]))
                    print(np.shape(u_parents[0][0][i]))
                    m[i] = m[i] * u_parents[0][0][i]
                return m

        @staticmethod
        def compute_dims(*parents):
            """ Compute the dimensions of phi and u. """
            return distribution.compute_dims(*parents[1:])

        def __init__(self, z, *args, **kwargs):
            # Check for constant mu
            if np.isscalar(z) or isinstance(z, np.ndarray):
                z = ConstantCategorical(z)
            # Construct
            super().__init__(z, *args,
                             **kwargs)

    return _Mixture

    ## def show(self):
    ##     p = self.u[0] #np.exp(self.phi[0])
    ##     #p /= np.sum(p, axis=-1, keepdims=True)
    ##     print("Categorical(p)")
    ##     print("  p = ")
    ##     print(p)

    ## def observe(self, x):
    ##     # TODO: You could check that x has proper dimensions
    ##     x = np.array(x, dtype=np.int)
        
    ##     # Initial array of zeros
    ##     d = self.dims[0][0]
    ##     self.u[0] = np.zeros(np.shape(x)+(d,))
        
    ##     # Compute indices
    ##     x += d*np.arange(np.size(x),dtype=int).reshape(np.shape(x))
    ##     x = x[...,np.newaxis]
    ##     # Set 1 to elements corresponding to the observations
    ##     np.put(self.u[0], x, 1)
    ##     self.show()
        
    ##     self.fix_u_and_f(self.u, 0)




class NodeWishartFromGamma(Node):
    
    def __init__(self, alpha, **kwargs):

        # Check for constant n
        if np.isscalar(alpha) or isinstance(alpha, np.ndarray):            
            alpha = NodeConstantGamma(alpha)

        #ExponentialFamily.__init__(self, n, V, plates=plates, dims=V.dims, **kwargs)
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
            full_shape = utils.broadcasted_shape_from_arrays(*A)
            axes = utils.axes_to_collapse(full_shape, parent.get_shape(i))
            r = 1
            for j in axes:
                r *= full_shape[j]

            # Compute dot product
            m[i] = utils.sum_product(*A, axes_to_sum=axes, keepdims=True) / r

        # Compute the mask
        s = utils.axes_to_collapse(np.shape(mask), parent.plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = utils.squeeze_to_dim(mask, len(parent.plates))

        return (m, mask)


