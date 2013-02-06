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
Base class for a node in a variational Bayesian network.

The node can be stochastic or deterministic.
"""

import numpy as np

from bayespy.utils import utils
#import utils

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
# callback(f(dx1),...,f(dx2))
    



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

    def __init__(self, *parents, dims=None, plates=None, name=""):

        if dims is None:
            raise Exception("You need to specify the dimensionality" \
                            + " of the distribution for class" \
                            + str(self.__class__))

        self.dims = dims
        self.name = name

        # Parents
        self.parents = parents
        # Inform parent nodes
        for (index,parent) in enumerate(self.parents):
            if parent:
                parent.add_child(self, index)

        # Check plates
        parent_plates = [self.plates_from_parent(index) 
                         for index in range(len(self.parents))]
        if plates is None:
            # By default, use the minimum number of plates determined
            # from the parent nodes
            try:
                self.plates = utils.broadcasted_shape(*parent_plates)
            except ValueError:
                raise ValueError("The plates of the parents do not broadcast.")
        else:
            # Use custom plates
            self.plates = plates
            # TODO/FIXME: Check that these plates are consistent with parents.
            # This is not a good test yet.. You need to check that the
            # parent_plates are a subset of plates.
            try:
                plates_broadcasted = utils.broadcasted_shape(plates, *parent_plates)
            except ValueError:
                raise ValueError("The given plates and the plates of the "
                                 "parents do not broadcast.")
            

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
        """
        Add a child node.

        Parameters
        ----------
        child : node
        index : int
           The parent index of this node for the child node.  
           The child node recognizes its parents by their index 
           number.
        """
        self.children.append((child, index))

    def plates_to_parent(self, index):
        return self.plates

    def plates_from_parent(self, index):
        return self.parents[index].plates

    def get_shape(self, ind):
        return self.plates + self.dims[ind]

    @staticmethod
    def plate_multiplier(plates, *args):
        # Check broadcasting of the shapes
        for arg in args:
            utils.broadcasted_shape(plates, arg)

        # Check that each arg-plates are a subset of plates?
        for arg in args:
            if not utils.is_shape_subset(arg, plates):
                raise ValueError("The shapes in args are not a sub-shape of "
                                 "plates.")
            
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

            # The parent we're sending the message to
            parent = self.parents[index]

            # Compute mask message:
            # "Sum" (i.e., logical or) over the plates that have unit length in 
            # the parent node.
            s = utils.axes_to_collapse(np.shape(my_mask), parent.plates)
            mask = np.any(my_mask, axis=s, keepdims=True)
            mask = utils.squeeze_to_dim(mask, len(parent.plates))
            
            # Compact the message to a proper shape
            for i in range(len(m)):

                # Empty messages are given as None. We can ignore those.
                if m[i] is not None:

                    # Ignorations (add extra axes to broadcast properly).
                    # This sends zero messages to parent from such
                    # variables we are ignoring in this node. This is
                    # useful for handling missing data.

                    # Apply the mask to the message
                    # Sum the dimensions of the message matrix to match
                    # the dimensionality of the parents natural
                    # parameterization (as there may be less plates for
                    # parents)

                    # Extend the plate mask to include unit axes for variable
                    # dimensions.
                    shape_mask = np.shape(my_mask) + (1,) * len(parent.dims[i])
                    my_mask2 = np.reshape(my_mask, shape_mask)

                    # Apply the mask to the message
                    m[i] = np.where(my_mask2, m[i], 0)

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
                    try:
                        ## print('node.msg', self.name, plates_self, 
                        ##                           plates_m,
                        ##                           parent.plates)
                        r = self.plate_multiplier(plates_self, 
                                                  plates_m,
                                                  parent.plates)
                    except ValueError:
                        raise ValueError("The plates of the message and "
                                         "parent[%d] node (%s) are not a "
                                         "broadcastable subset of the plates "
                                         "of this node (%s).  The message has "
                                         "shape %s, meaning plates %s. This "
                                         "node has plates %s with respect to "
                                         "the parent[%d], which has plates %s."
                                         % (index,
                                            parent.name,
                                            self.name,
                                            np.shape(m[i]), 
                                            plates_m, 
                                            plates_self, 
                                            index,
                                            parent.plates))

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
        """
        Get the message and the mask for a parent.

        Parameters:
        -----------
        index : int
            Index of the parent requesting the message.
        u_parents : list of list of ndarrays
            List of parents' moments.

        Returns:
        --------
        m : list of ndarrays
            Message as a list of arrays.
        mask : boolean ndarray
            Mask telling which plates should be taken into account.

        Notes:
        ------
        Sub-classes should implement this method.
        """
        raise NotImplementedError()

    def move_plates(self, from_plate, to_plate):
        return _MovePlate(self, 
                          from_plate,
                          to_plate,
                          name=self.name + ".move_plates")

    def add_plate_axis(self, to_plate):
        return _AddPlateAxis(self,
                             to_plate,
                             name=self.name+".add_plate_axis")

class _AddPlateAxis(Node):

    def __init__(self, X, to_plate, plates=None, **kwargs):
        
        if plates is not None:
            raise ValueError("Do not specify plates.")

        plates = X.plates

        N = len(plates) + 1

        # Check the parameters
        if to_plate >= N or to_plate < -N:
            raise ValueError("Invalid plate position to add.")

        # Use positive indexing only
        if to_plate < 0:
            to_plate += N
        self.to_plate = to_plate

        # Move the plate
        plates = list(plates)
        plates.insert(self.to_plate, 1)

        super().__init__(X, 
                         plates=tuple(plates),
                         dims=X.dims,
                         **kwargs)

    def plates_to_parent(self, index):
        return self.plates[:self.to_plate] + self.plates[(self.to_plate+1):]

        
    def get_moments(self):
        """
        Get the moments with an added plate axis.
        """

        # Get parents' moments
        u = self.parents[0].message_to_child()

        # Move a plate axis
        u = list(u)
        for i in range(len(u)):
            # Make sure the moments have all the axes
            diff = len(self.plates) + len(self.dims[i]) - np.ndim(u[i]) - 1
            u[i] = utils.add_leading_axes(u[i], diff)
            # Add one axes to the correct position
            sh_u = list(np.shape(u[i]))
            sh_u.insert(self.to_plate, 1)
            u[i] = np.reshape(u[i], sh_u)

        return tuple(u)

    def get_message(self, index, u_parents):
        """
        Compute the message to a parent node.
        """

        # Get the message from children
        (m, mask) = self.message_from_children()

        # Remove the added message plate
        for i in range(len(m)):
            # Make sure the message has all the axes
            diff = len(self.plates) + len(self.dims[i]) - np.ndim(m[i])
            m[i] = utils.add_leading_axes(m[i], diff)
            # Remove the axis
            sh_m = list(np.shape(m[i]))
            sh_m.pop(self.to_plate)
            m[i] = np.reshape(m[i], sh_m)

        # Remove the added mask plate
        diff = len(self.plates) - np.ndim(mask)
        mask = utils.add_leading_axes(mask, diff)
        sh_mask = list(np.shape(mask))
        sh_mask.pop(self.to_plate)
        mask = np.reshape(mask, sh_mask)

        return (m, mask)


class _MovePlate(Node):
    """
    Move a plate to a given position.

    NOTE: This has NOT been tested yet..
    """

    def __init__(self, X, from_plate, to_plate, plates=None, **kwargs):

        if plates is not None:
            raise ValueError("Do not specify plates.")

        plates = X.plates

        # Check the parameters
        if from_plate >= len(plates) or from_plate < -len(plates):
            raise ValueError("Invalid plate to move from.")
        if to_plate >= len(plates) or to_plate < -len(plates):
            raise ValueError("Invalid plate to move to.")

        # Use positive indexing only
        if from_plate < 0:
            from_plate = len(plates) + from_plate
        if to_plate < 0:
            to_plate = len(plates) + to_plate + 1

        # Move the plate
        plates = list(plates)
        plates.insert(to_plate, plates.pop(from_plate))

        super().__init__(X, 
                         plates=tuple(plates),
                         dims=X.dims,
                         **kwargs)

    def get_moments(self):
        """
        Get the moments with moved plates.
        """

        # Get parents' moments
        u = self.parents[0].message_to_child()

        # Move a plate axis
        u = list(u)
        for i in range(len(u)):
            u[i] = utils.moveaxis(u[i], self.from_plate, self.to_plate)

        return tuple(u)

    def get_message(self, index, u_parents):
        """
        Compute the message to a parent node.
        """

        # Get the message from children
        (m, mask) = self.message_from_children()

        # Move message plates
        for i in range(len(m)):
            diff = len(self.plates) + len(self.dims[i]) - self.np.ndim(m[i])
            m[i] = utils.add_leading_axes(m[i], diff)
            m[i] = utils.moveaxis(m[i], self.to_plate, self.from_plate)

        # Move mask plates
        mask = utils.add_leading_axes(mask, len(self.plates) - np.ndim(mask))
        mask = utils.moveaxis(mask, self.to_plate, self.from_plate)

        return (m, mask)

    
class NodeConstant(Node):
    def __init__(self, u, **kwargs):
        self.u = u
        Node.__init__(self, **kwargs)

    def message_to_child(self, gradient=False):
        if gradient:
            return (self.u, [])
        else:
            return self.u


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
            self.gradient = np.zeros(np.shape(x0))
            self.u[0] = x
        def gradient():
            # This would need to apply the gradient of the
            # transformation to the computed gradient
            return self.gradient
            
        return (x0, transform, gradient)

    def add_to_gradient(self, d):
        self.gradient += d

    def message_to_child(self, gradient=False):
        if gradient:
            return (self.u, [ [np.ones(np.shape(self.u[0])),
                               #self.gradient] ])
                               self.add_to_gradient] ])
        else:
            return self.u


    def stop_optimization(self):
        #raise Exception("Not implemented for " + str(self.__class__))
        pass


    
