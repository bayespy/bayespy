
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

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        raise NotImplementedError()

    # Proposed functions:
    def logpdf_integrated(self):
        # The log pdf when this node is integrated out (useful for
        # type-2 ML or predictive densities)
        return

    def random(self):
        # Draw a random variable from the node
        raise NotImplementedError()

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

    def plates_to_parent(self, index):
        return self.plates

    def get_shape(self, ind):
        return self.plates + self.dims[ind]

    @staticmethod
    def plate_multiplier(plates, *args):
        # Check broadcasting of the shapes
        for arg in args:
            utils.broadcasted_shape(plates, arg)
            
        r = 1
        for j in range(-len(plates),0):
            mult = True
            for arg in args:
                if not (-j > len(arg) or arg[j] == 1):
                    mult = False
            if mult:
                r *= plates[j]
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

    def get_moments(self):
        raise NotImplementedError()

    def message_to_child(self):
        return self.get_moments()
        # raise Exception("Not implemented. Subclass should implement this!")

    def moments_from_parents(self, exclude=()):
        u_parents = list()
        for (i,parent) in enumerate(self.parents):
            if not i in exclude:
                u_parents.append(parent.message_to_child())
            else:
                u_parents.append(None)
        return u_parents

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

            # Get parents' moments
            u_parents = self.moments_from_parents(exclude=(index,))

            # Decompose our own message to parent[index]
            (m, my_mask) = self.get_message(index, u_parents)

            #print('message_to_parent', self.name, index, m)

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
                ## print('ExpFam.msg_to_parent')
                ## print(m.__class__)
                ## print(my_mask2.__class__)
                #print(my_mask2)
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
                plates_self = self.plates_to_parent(index)
                r = self.plate_multiplier(plates_self, plates_m, parent.plates)

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
    ndims_parents = None

    @classmethod
    def compute_logpdf(cls, u, phi, g, f):
        """ Compute E[log p(X)] given E[u], E[phi], E[g] and
        E[f]. Does not sum over plates."""

        # TODO/FIXME: Should I take into account what is latent or
        # observed, or what is even totally ignored (by the mask).
        L = g + f
        for (phi_i, u_i, ndims_i) in zip(phi, u, cls.ndims):
        #for (phi_i, u_i, len_dims_i) in zip(phi, u, len_dims):
            # Axes to sum (dimensions of the variable, not the plates)
            axis_sum = tuple(range(-ndims_i,0))
            #axis_sum = tuple(range(-len_dims_i,0))
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

    @staticmethod
    def compute_dims_from_values(x):
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
        #self.phi = [np.array(0.0) for i in range(len(self.dims))]
        #self.u = [np.array(0.0) for i in range(len(self.dims))]
        axes = len(self.plates)*(1,)
        self.phi = [utils.nans(axes+dim) for dim in self.dims]
        self.u = [utils.nans(axes+dim) for dim in self.dims]
#        self.u = [np.array(0.0) for i in range(len(self.dims))]

        # Terms for the lower bound (G for latent and F for observed)
        self.g = 0
        self.f = 0

        # Not observed
        self.observed = False

        # By default, ignore all plates
        self.mask = False

    def get_vb_term(self):
        return {self.lower_bound_contribution}

    def get_message(self, index, u_parents):
        return (self.compute_message(index, self.u, u_parents),
                self.mask)
        ## return (self.message(index, u_parents),
        ##         self.mask)

    def initialize_from_prior(self):
        if not np.all(self.observed):

            # Messages from parents
            #u_parents = [parent.message_to_child() for parent in self.parents]
            u_parents = self.moments_from_parents()

            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            # Update moments
            (u, g) = self.compute_u_and_g(self.phi, mask=True)
            self.update_u_and_g(u, g, mask=True)


    def initialize_from_parameters(self, *args):
        # Get the moments of the parameters if they were fixed to the
        # given values.
        u_parents = list()
        for (ind, x) in enumerate(args):
            (u, _) = self.parents[ind].compute_fixed_u_and_f(x)
            u_parents.append(u)
        # Update natural parameters
        self.update_phi_from_parents(u_parents)
        # Update moments
        # TODO/FIXME: Use the mask of observations!
        (u, g) = self.compute_u_and_g(self.phi, mask=True)
        self.update_u_and_g(u, g, mask=True)

    ## def initialize_from_value(self, x):
    ##     # Update moments from value
    ##     (u, f) = self.compute_fixed_u_and_f(x)
    ##     self.update_u_and_g(u, np.nan, mask=True)

    ## def initialize_from_random(self):
    ##     self.initialize_from_prior()
    ##     self.initialize_from_value(self.random())

    def update(self):
        if not np.all(self.observed):

            # Messages from parents
            u_parents = [parent.message_to_child() for parent in self.parents]
                
            # Update natural parameters using parents
            self.update_phi_from_parents(u_parents)

            #print('update, phi', self.name, self.phi)
            # Update natural parameters using children (just add the
            # messages to phi)
            for (child,index) in self.children:
                #print('ExpFam.update:', self.name)

                # TODO/FIXME: These m are nans..
                (m, mask) = child.message_to_parent(index)

                # Combine masks
                #
                # TODO: Maybe you would like to compute a new mask
                # at every update?
                self.mask = np.logical_or(self.mask,mask)
                for i in range(len(self.phi)):
                    ## try:
                    ##     # Try exploiting broadcasting rules
                    ##     #
                    ##     # TODO/FIXME: This has the problem that if phi
                    ##     # is updated from parents such that phi is a
                    ##     # view into parent's moments thus modifying
                    ##     # phi would modify parents moments.. Maybe one
                    ##     # should check that nobody else is viewing
                    ##     # phi?
                    ##     self.phi[i] += m[i]
                    ## except ValueError:
                    ##     self.phi[i] = self.phi[i] + m[i]
                    #print('update2, phi', i, self.name, self.phi, m[i])
                    self.phi[i] = self.phi[i] + m[i]

            # Mask for plates to update (i.e., unobserved plates)
            update_mask = np.logical_not(self.observed)
            # Compute u and g
            (u, g) = self.compute_u_and_g(self.phi, mask=update_mask)
            # Update moments
            self.update_u_and_g(u, g, mask=update_mask)

    def update_phi_from_parents(self, u_parents):
        # This makes correct broadcasting
        self.phi = self.compute_phi_from_parents(u_parents)
        #print('update_phi, phi', self.phi)
        # Make sure phi has the correct number of axes. It makes life
        # a bit easier elsewhere.
        for i in range(len(self.phi)):
            axes = len(self.plates) + self.ndims[i] - np.ndim(self.phi[i])
            ## print('update_phi_from_parents:')
            ## print(np.shape(self.phi[i]))
            ## print(self.plates)
            ## print(self.dims[i])
            if axes > 0:
                # Add axes
                self.phi[i] = utils.add_leading_axes(self.phi[i], axes)
            elif axes < 0:
                # Remove extra leading axes
                first = -(len(self.plates)+self.ndims[i])
                sh = np.shape(self.phi[i])[first:]
                self.phi[i] = np.reshape(self.phi[i], sh)
            #print(np.shape(self.phi[i]))
                                                 
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

    def update_u(self, u, mask=True):
        # Store the computed moments u but do not change moments for
        # observations, i.e., utilize the mask.
        for ind in range(len(u)):
            # Add axes to the mask for the variable dimensions (mask
            # contains only axes for the plates).
            u_mask = utils.add_trailing_axes(mask, self.ndims[ind])

            # Enlarge self.u[ind] as necessary so that it can store the
            # broadcasted result.
            sh = utils.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
            self.u[ind] = utils.repeat_to_shape(self.u[ind], sh)

            # Use mask to update only unobserved plates and keep the
            # observed as before
            np.copyto(self.u[ind],
                      u[ind],
                      where=u_mask)

            # Make sure u has the correct number of dimensions:
            shape = self.get_shape(ind)
            ndim = len(shape)
            ndim_u = np.ndim(self.u[ind])
            if ndim > ndim_u:
                self.u[ind] = utils.add_leading_axes(u[ind], ndim - ndim_u)
            elif ndim < np.ndim(self.u[ind]):
                raise Exception("Weird, this shouldn't happen.. :)")


    def update_u_and_g(self, u, g, mask=True):

        self.update_u(u, mask=mask)
        # TODO/FIXME: Apply mask to g too!!
        self.g = g
        

        ## # Store the computed moments u but do not change moments for
        ## # observations, i.e., utilize the mask.
        ## for ind in range(len(u)):
        ##     # Add axes to the mask for the variable dimensions (mask
        ##     # contains only axes for the plates).
        ##     u_mask = utils.add_trailing_axes(mask, self.ndims[ind])

        ##     # Enlarge self.u[ind] as necessary so that it can store the
        ##     # broadcasted result.
        ##     sh = utils.broadcasted_shape_from_arrays(self.u[ind], u[ind], u_mask)
        ##     self.u[ind] = utils.repeat_to_shape(self.u[ind], sh)

        ##     # Use mask to update only unobserved plates and keep the
        ##     # observed as before
        ##     np.copyto(self.u[ind],
        ##               u[ind],
        ##               where=u_mask)

        ##     # TODO/FIXME: Apply mask to g too!!
        ##     self.g = g

        ##     # Make sure u has the correct number of dimensions:
        ##     shape = self.get_shape(ind)
        ##     ndim = len(shape)
        ##     ndim_u = np.ndim(self.u[ind])
        ##     if ndim > ndim_u:
        ##         self.u[ind] = utils.add_leading_axes(u[ind], ndim - ndim_u)
        ##     elif ndim < np.ndim(self.u[ind]):
        ##         raise Exception("Weird, this shouldn't happen.. :)")

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
            phi_q = np.where(latent_mask, phi_q, 0)
            # TODO/FIXME: Use einsum here?
            Z = np.sum((phi_p-phi_q) * u_q, axis=axis_sum)
            ## Z = np.sum((phi_p - phi_q*latent_mask) * u_q,
            ##            axis=axis_sum)

            L = L + Z

        return (np.sum(L*self.mask)
                * self.plate_multiplier(self.plates,
                                        np.shape(L),
                                        np.shape(self.mask)))
        #return L
            
    def fix_u_and_f(self, u, f, mask=True):
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


        self.update_u(u, mask=mask)
        
        ## for i in range(len(self.u)):
        ##     obs_mask = utils.add_axes(mask,
        ##                         len(self.plates) - np.ndim(mask),
        ##                         len(self.dims[i]))

        ##     # TODO/FIXME: Use copyto!
        ##     self.u[i] = (obs_mask * u[i]
        ##                  + np.logical_not(obs_mask) * self.u[i])

        # TODO/FIXME: Use the mask?
        self.f = f
        
    def observe(self, x, mask=True):
        (u, f) = self.compute_fixed_u_and_f(x)
        #print(u)
        self.fix_u_and_f(u, f, mask=mask)

        # Observed nodes should not be ignored
        self.observed = mask
        self.mask = np.logical_or(self.mask, self.observed)

        #print('observe', self.name, self.u)
        ## print(x)
        ## print(mask)
        ## print(u)

    def integrated_logpdf_from_parents(self, index):

        """ Approximates the posterior predictive pdf \int
        p(x|parents) q(parents) dparents in log-scale as \int
        q(parents_i) exp( \int q(parents_\i) \log p(x|parents)
        dparents_\i ) dparents_i."""

        raise NotImplementedError()

def Constant(distribution):
    class _Constant(Node):

        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            return distribution.compute_fixed_u_and_f(x)
        
        def __init__(self, x, **kwargs):
            self.u = distribution.compute_fixed_moments(x)
            dims = distribution.compute_dims_from_values(x)
            super().__init__(dims=dims, **kwargs)
            
        def get_moments(self):
            return self.u
        
    return _Constant
        

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
    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        return ([x], 0)

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

## class NodeConstantGaussian(NodeConstant):
##     def __init__(self, X, **kwargs):
##         X = np.atleast_1d(X)
##         d = X.shape[-1]
##         NodeConstant.__init__(self,
##                               [X, utils.m_outer(X, X)],
##                               plates=X.shape[:-1],
##                               dims=[(d,), (d,d)],
##                               **kwargs)
        
## class NodeConstantWishart(NodeConstant):
##     @staticmethod
##     def compute_fixed_u_and_f(Lambda):
##         """ Compute u(x) and f(x) for given x. """
##         u = [Lambda,
##              utils.m_chol_logdet(utils.m_chol(Lambda))]
##         f = 0
##         return (u, f)

##     def __init__(self, Lambda, **kwargs):
##         Lambda = np.atleast_2d(Lambda)
##         if Lambda.shape[-1] != Lambda.shape[-2]:
##             raise Exception("Lambda not a square matrix.")
##         NodeConstant.__init__(self,
##                               [Lambda, utils.m_chol_logdet(utils.m_chol(Lambda))],
##                               plates=Lambda.shape[:-2],
##                               dims=[Lambda.shape[-2:], ()],
##                               **kwargs)


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
    ndims_parents = [None, (2, 0)]

    @staticmethod
    def compute_fixed_moments(Lambda):
        """ Compute moments for fixed x. """
        ldet = utils.m_chol_logdet(utils.m_chol(Lambda))
        u = [Lambda,
             ldet]
        return u

    @staticmethod
    def compute_g_from_parents(u_parents):
        n = u_parents[0][0]
        V = u_parents[1][0]
        logdet_V = u_parents[1][1]
        k = np.shape(V)[-1]
        #k = self.dims[0][0]
        # TODO: Check whether this is correct:
        #g = 0.5*n*logdet_V - special.multigammaln(n/2, k)
        g = 0.5*n*logdet_V - 0.5*k*n*np.log(2) - special.multigammaln(n/2, k)
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
    def compute_fixed_u_and_f(Lambda):
        """ Compute u(x) and f(x) for given x. """
        k = np.shape(Lambda)[-1]
        ldet = utils.m_chol_logdet(utils.m_chol(Lambda))
        u = [Lambda,
             ldet]
        f = -(k+1)/2 * ldet
        return (u, f)

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

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        d = np.shape(x)[-1]
        return [(d,d), ()]

    # Wishart(n, inv(V))

    def __init__(self, n, V, plates=(), **kwargs):

        # Check for constant n
        if np.isscalar(n) or isinstance(n, np.ndarray):            
            n = NodeConstantScalar(n)
            
        # Check for constant V
        if np.isscalar(V) or isinstance(V, np.ndarray):
            V = Constant(Wishart)(V)

        ExponentialFamily.__init__(self, n, V, plates=plates, **kwargs)
        
    def show(self):
        print("%s ~ Wishart(n, A)" % self.name)
        print("  n =")
        print(2*self.phi[1])
        print("  A =")
        print(0.5 * self.u[0] / self.phi[1][...,np.newaxis,np.newaxis])

class Gaussian(ExponentialFamily):

    ndims = (1, 2)
    ndims_parents = [(1, 2), (2, 0)]
    # Observations are vectors (1-D):
    ndim_observations = 1

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments for fixed x. """
        return [x, utils.m_outer(x,x)]

    @staticmethod
    def compute_phi_from_parents(u_parents):
        ## print('in Gaussian.compute_phi_from_parents')
        ## print(u_parents)
        ## print(np.shape(u_parents[1][0]))
        ## print(np.shape(u_parents[0][0]))
        return [utils.m_dot(u_parents[1][0], u_parents[0][0]),
                -0.5 * u_parents[1][0]]

    @staticmethod
    def compute_g_from_parents(u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        Lambda = u_parents[1][0]
        logdet_Lambda = u_parents[1][1]
        g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
             + 0.5 * logdet_Lambda)
        ## g = (-0.5 * np.einsum('...ij,...ij',mumu,Lambda)
        ##      + 0.5 * np.sum(logdet_Lambda))
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        #print(-phi[1])
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
        ## print('in gaussian compute dims: parent.dims:', parents[0].dims)
        ## print('in gaussian compute dims: parent.u:', parents[0].u)
        return parents[0].dims

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        d = np.shape(x)[-1]
        return [(d,), (d,d)]

    # Gaussian(mu, inv(Lambda))

    def __init__(self, mu, Lambda, plates=(), **kwargs):

        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = Constant(Gaussian)(mu)

        # Check for constant Lambda
        if np.isscalar(Lambda) or isinstance(Lambda, np.ndarray):
            Lambda = Constant(Wishart)(Lambda)

        # You could check whether the dimensions of mu and Lambda
        # match (and Lambda is square)
        if Lambda.dims[0][-1] != mu.dims[0][-1]:
            raise Exception("Dimensionalities of mu and Lambda do not match.")

        # Construct
        super().__init__(mu, Lambda,
                         plates=plates,
                         **kwargs)

    def random(self):
        # TODO/FIXME: You shouldn't draw random values for
        # observed/fixed elements!

        # Note that phi[1] is -0.5*inv(Cov)
        U = utils.m_chol(-2*self.phi[1])
        mu = self.u[0]
        z = np.random.normal(0, 1, self.get_shape(0))
        # Compute mu + U'*z
        #return mu + np.einsum('...ij,...i->...j', U, z)
        #scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=False)
        #print('gaussian.random', np.shape(mu), np.shape(z))
        z = utils.m_solve_triangular(U, z, trans='T', lower=False)
        return mu + z
        #return self.u[0] + utils.m_chol_solve(U, z)

    ## def initialize_random_mean(self):
    ##     # First, initialize the distribution from prior?
    ##     self.initialize_from_prior()
        
    ##     if not np.all(self.observed):
    ##         # Draw a random sample
    ##         x = self.random()

    ##         # Update parameter for the mean using the sample
    ##         self.phi[0] = -2*utils.m_dot(self.phi[1], x)

    ##         # Update moments
    ##         (u, g) = self.compute_u_and_g(self.phi, mask=True)
    ##         self.update_u_and_g(u, g, mask=True)
            

    def show(self):
        mu = self.u[0]
        Cov = self.u[1] - utils.m_outer(mu, mu)
        print("%s ~ Gaussian(mu, Cov)" % self.name)
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
        #return [u_parents[0][0].copy()]

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
        print("%s ~ Dirichlet(alpha)" % self.name)
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
        def compute_phi_from_parents(u_parents):
            return [u_parents[0][0]]

        @staticmethod
        def compute_g_from_parents(u_parents):
            return 0

        @staticmethod
        def compute_u_and_g(phi, mask=True):
            # For numerical reasons, scale contributions closer to
            # one, i.e., subtract the maximum of the log-contributions.
            max_phi = np.max(phi[0], axis=-1, keepdims=True)
            p = np.exp(phi[0]-max_phi)
            sum_p = np.sum(p, axis=-1, keepdims=True)
            # Moments
            u0 = p / sum_p
            u = [u0]
            # G
            g = -np.log(sum_p) - max_phi
            g = np.squeeze(g, axis=-1)
            #print('Categorical.compute_u_and_g, g:', np.sum(g), np.shape(g), np.sum(max_phi))
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
            #print('message in categorical:', u[0])
            if index == 0:
                return [ u[0].copy() ]

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
            print("%s ~ Categorical(p)" % self.name)
            print("  p = ")
            print(p)

    return _Categorical(p, **kwargs)

    ## def observe(self, x):
    ##     self.fix_u_and_f(self.u, 0)

# Pseudo:
# Mixture(Gaussian)(z, mu, Lambda)

def Mixture(distribution, cluster_plate=-1):

    if cluster_plate >= 0:
        raise Exception("Give negative value for axis index cluster_plates")

    class _Mixture(ExponentialFamily):

        ndims = distribution.ndims

        @staticmethod
        def compute_phi_from_parents(u_parents):

            # Compute weighted average of the parameters

            # Cluster parameters
            Phi = distribution.compute_phi_from_parents(u_parents[1:])
            # Contributions/weights/probabilities
            P = u_parents[0][0]

            phi = list()
            
            for ind in range(len(Phi)):
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

                if cluster_plate < 0:
                    cluster_axis = cluster_plate - distribution.ndims[ind]
                #else:
                #    cluster_axis = cluster_plate

                # Move cluster axis to the last:
                # Shape(phi)    = [Nn,..,N0,Dd,..,D0,K]
                phi.append(utils.moveaxis(Phi[ind], cluster_axis, -1))

                # Add axes to p:
                # Shape(p)      = [Nn,..,N0,K,1,..,1]
                p = utils.add_trailing_axes(P, distribution.ndims[ind])
                # Move cluster axis to the last:
                # Shape(p)      = [Nn,..,N0,1,..,1,K]
                p = utils.moveaxis(p, -(distribution.ndims[ind]+1), -1)
                #print('Mixture.compute_phi, p:', np.sum(p, axis=-1))
                #print('mixture.compute_phi shapes:')
                #print(np.shape(p))
                #print(np.shape(phi[ind]))
                
                # Now the shapes broadcast perfectly and we can sum
                # p*phi over the last axis:
                # Shape(result) = [Nn,..,N0,Dd,..,D0]
                phi[ind] = utils.sum_product(p, phi[ind], axes_to_sum=-1)
                
            return phi

        @staticmethod
        def compute_g_from_parents(u_parents):

            # Compute weighted average of g over the clusters.

            # Shape(g)      = [Nn,..,K,..,N0]
            # Shape(p)      = [Nn,..,N0,K]
            # Shape(result) = [Nn,..,N0]

            # Compute g for clusters:
            # Shape(g)      = [Nn,..,K,..,N0]
            g = distribution.compute_g_from_parents(u_parents[1:])
            
            # Move cluster axis to last:
            # Shape(g)      = [Nn,..,N0,K]
            g = utils.moveaxis(g, cluster_plate, -1)

            # Cluster assignments/contributions/probabilities/weights:
            # Shape(p)      = [Nn,..,N0,K]
            p = u_parents[0][0]
            
            # Weighted average of g over the clusters. As p and g are
            # properly aligned, you can just sum p*g over the last
            # axis and utilize broadcasting:
            # Shape(result) = [Nn,..,N0]
            #print('mixture.compute_g_from_parents p and g:', np.shape(p), np.shape(g))
            g = utils.sum_product(p, g, axes_to_sum=-1)

            #print('mixture.compute_g_from_parents g:', np.sum(g), np.shape(g))

            return g

        @staticmethod
        def compute_u_and_g(phi, mask=True):
            return distribution.compute_u_and_g(phi, mask=mask)

        @staticmethod
        def compute_fixed_u_and_f(x):
            """ Compute u(x) and f(x) for given x. """
            return distribution.compute_fixed_u_and_f(x)

        @staticmethod
        def compute_message(index, u, u_parents):
            """ . """

            #print('Mixture.compute_message:')
            
            if index == 0:

                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                # Shape(L)      = [Nn,..,K,..,N0]
                # Shape(u)      = [Nn,..,N0,Dd,..,D0]
                # Shape(result) = [Nn,..,N0,K]

                # Compute g:
                # Shape(g)      = [Nn,..,K,..,N0]
                g = distribution.compute_g_from_parents(u_parents[1:])
                # Reshape(g):
                # Shape(g)      = [Nn,..,N0,K]
                g = utils.moveaxis(g, cluster_plate, -1)

                # Compute phi:
                # Shape(phi)    = [Nn,..,K,..,N0,Dd,..,D0]
                phi = distribution.compute_phi_from_parents(u_parents[1:])
                # Reshape phi:
                # Shape(phi)    = [Nn,..,N0,K,Dd,..,D0]
                for ind in range(len(phi)):
                    phi[ind] = utils.moveaxis(phi[ind],
                                              cluster_plate-distribution.ndims[ind],
                                              -1-distribution.ndims[ind])

                # Reshape u:
                # Shape(u)      = [Nn,..,N0,1,Dd,..,D0]
                u_self = list()
                for ind in range(len(u)):
                    u_self.append(np.expand_dims(u[ind],
                                                 axis=(-1-distribution.ndims[ind])))
                    
                # Compute logpdf:
                # Shape(L)      = [Nn,..,N0,K]
                L = distribution.compute_logpdf(u_self, phi, g, 0)
                
                # Sum over other than the cluster dimensions? No!
                # Hmm.. I think the message passing method will do
                # that automatically

                ## print(np.shape(phi[0]))
                ## print(np.shape(u_self[0]))
                ## print(np.shape(g))
                ## print(np.shape(L))
                
                return [L]

            elif index >= 1:

                # Parent index for the distribution used for the
                # mixture.
                index = index - 1

                # Reshape u:
                # Shape(u)      = [Nn,..1,..,N0,Dd,..,D0]
                u_self = list()
                for ind in range(len(u)):
                    if cluster_plate < 0:
                        cluster_axis = cluster_plate - distribution.ndims[ind]
                    else:
                        cluster_axis = cluster_plate
                    u_self.append(np.expand_dims(u[ind], axis=cluster_axis))
                    
                # Message from the mixed distribution
                m = distribution.compute_message(index, u_self, u_parents[1:])

                # Weigh the messages with the responsibilities
                for i in range(len(m)):

                    # Shape(m)      = [Nn,..,K,..,N0,Dd,..,D0]
                    # Shape(p)      = [Nn,..,N0,K]
                    # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
                    
                    # Number of axes for the variable dimensions for
                    # the parent message.
                    D = distribution.ndims_parents[index][i]

                    # Responsibilities for clusters are the first
                    # parent's first moment:
                    # Shape(p)      = [Nn,..,N0,K]
                    p = u_parents[0][0]
                    # Move the cluster axis to the proper place:
                    # Shape(p)      = [Nn,..,K,..,N0]
                    p = utils.moveaxis(p, -1, cluster_plate)
                    # Add axes for variable dimensions to the contributions
                    # Shape(p)      = [Nn,..,K,..,N0,1,..,1]
                    p = utils.add_trailing_axes(p, D)

                    if cluster_plate < 0:
                        # Add the variable dimensions
                        cluster_axis = cluster_plate - D

                    # Add axis for clusters:
                    # Shape(m)      = [Nn,..,1,..,N0,Dd,..,D0]
                    #m[i] = np.expand_dims(m[i], axis=cluster_axis)
                        
                    #
                    # TODO: You could do summing here already so that
                    # you wouldn't compute huge matrices as
                    # intermediate result. Use einsum.

                    ## print(np.shape(m[i]))
                    ## print(np.shape(p))

                    # Compute the message contributions for each
                    # cluster:
                    # Shape(result) = [Nn,..,K,..,N0,Dd,..,D0]
                    m[i] = m[i] * p

                    #print(np.shape(m[i]))
                    
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

        def plates_to_parent(self, index):
            if index == 0:
                return self.plates
            else:
                if cluster_plate < 0:
                    plates = list(self.plates)
                    if cluster_plate < 0:
                        k = len(self.plates) + cluster_plate + 1
                    else:
                        k = cluster_plate
                    plates.insert(k, self.parents[0].dims[0][0])
                    plates = tuple(plates)
                    #print('plates_to_parent', cluster_plate,  plates)
                    ## plates = (self.plates[:cluster_plate] +
                    ##           self.parents[0].dims[0] +
                    ##           self.plates[cluster_plate:])
                return plates
            
        def integrated_logpdf_from_parents(self, x, index):

            """ Approximates the posterior predictive pdf \int
            p(x|parents) q(parents) dparents in log-scale as \int
            q(parents_i) exp( \int q(parents_\i) \log p(x|parents)
            dparents_\i ) dparents_i."""

            if index == 0:
                # Integrate out the cluster assignments

                # First, integrate the cluster parameters in log-scale
                
                # compute_logpdf(cls, u, phi, g, f):

                # Shape(x) = [M1,..,Mm,N1,..,Nn,D1,..,Dd]
                # Add the cluster axis to x:
                # Shape(x) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
                cluster_axis = cluster_plate - distribution.ndim_observations
                x = np.expand_dims(x, axis=cluster_axis)

                u_parents = self.moments_from_parents()
                
                # Shape(u) = [M1,..,Mm,N1,..,1,..,Nn,D1,..,Dd]
                # Shape(f) = [M1,..,Mm,N1,..,1,..,Nn]
                (u, f) = distribution.compute_fixed_u_and_f(x)
                # Shape(phi) = [N1,..,K,..,Nn,D1,..,Dd]
                phi = distribution.compute_phi_from_parents(u_parents[1:])
                # Shape(g) = [N1,..,K,..,Nn]
                g = distribution.compute_g_from_parents(u_parents[1:])
                # Shape(lpdf) = [M1,..,Mm,N1,..,K,..,Nn]
                lpdf = distribution.compute_logpdf(u, phi, g, f)

                # From logpdf to pdf, but avoid over/underflow
                lpdf_max = np.max(lpdf, axis=cluster_plate, keepdims=True)
                pdf = np.exp(lpdf-lpdf_max)

                # Move cluster axis to be the last:
                # Shape(pdf) = [M1,..,Mm,N1,..,Nn,K]
                pdf = utils.moveaxis(pdf, cluster_plate, -1)

                #print('integrated_logpdf', pdf)
                
                # Cluster assignments/probabilities/weights
                # Shape(p) = [N1,..,Nn,K]
                p = u_parents[0][0]

                #self.parents[0].show()
                #print('integrated_logpdf, p:', p)
                
                # Weighted average. TODO/FIXME: Use einsum!
                # Shape(pdf) = [M1,..,Mm,N1,..,Nn]
                pdf = np.sum(pdf * p, axis=cluster_plate)

                #print('integrated_logpdf', pdf)
                
                # Back to log-scale (add the overflow fix!)
                lpdf_max = np.squeeze(lpdf_max, axis=cluster_plate)
                lpdf = np.log(pdf) + lpdf_max

                return lpdf

            raise NotImplementedError()
        
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


