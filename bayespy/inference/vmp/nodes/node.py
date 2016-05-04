################################################################################
# Copyright (C) 2013-2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
import matplotlib.pyplot as plt
import functools

from bayespy.utils import misc

"""
This module contains a sketch of a new implementation of the framework.
"""

def message_sum_multiply(plates_parent, dims_parent, *arrays):
    """
    Compute message to parent and sum over plates.

    Divide by the plate multiplier.
    """
    # The shape of the full message
    shapes = [np.shape(array) for array in arrays]
    shape_full = misc.broadcasted_shape(*shapes)
    # Find axes that should be summed
    shape_parent = plates_parent + dims_parent
    sum_axes = misc.axes_to_collapse(shape_full, shape_parent)
    # Compute the multiplier for cancelling the
    # plate-multiplier.  Because we are summing over the
    # dimensions already in this function (for efficiency), we
    # need to cancel the effect of the plate-multiplier
    # applied in the message_to_parent function.
    r = 1
    for j in sum_axes:
        if j >= 0 and j < len(plates_parent):
            r *= shape_full[j]
        elif j < 0 and j < -len(dims_parent):
            r *= shape_full[j]
    # Compute the sum-product
    m = misc.sum_multiply(*arrays,
                          axis=sum_axes,
                          sumaxis=True,
                          keepdims=True) / r
    # Remove extra axes
    m = misc.squeeze_to_dim(m, len(shape_parent))
    return m


class Moments():
    """
    Base class for defining the expectation of the sufficient statistics.

    The benefits:

      * Write statistic-specific features in one place only. For instance,
        covariance from Gaussian message.
    
      * Different nodes may have identically defined statistic so you need to
        implement related features only once. For instance, Gaussian and
        GaussianARD differ on the prior but the moments are the same.

      * General processing nodes which do not change the type of the moments may
        "inherit" the features from the parent node. For instance, slicing
        operator.

      * Conversions can be done easily in both of the above cases if the message
        conversion is defined in the moments class. For instance,
        GaussianMarkovChain to Gaussian and VaryingGaussianMarkovChain to
        Gaussian.
    """

    _converters = {}


    class NoConverterError(Exception):
        pass


    def get_instance_converter(self, **kwargs):
        """Default converter within a moments class is an identity.

        Override this method when moment class instances are not identical if
        they have different attributes.

        """
        if len(kwargs) > 0:
            raise NotImplementedError(
                "get_instance_converter not implemented for class {0}"
                .format(self.__class__.__name__)
            )
        return None


    def get_instance_conversion_kwargs(self):
        """
        Override this method when moment class instances are not identical if
        they have different attributes.
        """
        return {}


    @classmethod
    def add_converter(cls, moments_to, converter):
        cls._converters = cls._converters.copy()
        cls._converters[moments_to] = converter
        return


    def get_converter(self, moments_to):
        """
        Finds conversion to another moments type if possible.

        Note that a conversion from moments A to moments B may require
        intermediate conversions.  For instance: A->C->D->B.  This method finds
        the path which uses the least amount of conversions and returns that
        path as a single conversion.  If no conversion path is available, an
        error is raised.

        The search algorithm starts from the original moments class and applies
        all possible converters to get a new list of moments classes. This list
        is extended by adding recursively all parent classes because their
        converters are applicable. Then, all possible converters are applied to
        this list to get a new list of current moments classes. This is iterated
        until the algorithm hits the target moments class or its subclass.
        """

        # Check if there is no need for a conversion
        #
        # TODO/FIXME: This isn't sufficient. Moments can have attributes that
        # make them incompatible (e.g., ndim in GaussianMoments).
        if isinstance(self, moments_to):
            return lambda X: X

        # Initialize variables
        visited = set()
        visited.add(self.__class__)
        converted_list = [(self.__class__, [])]

        # Each iteration step consists of two parts:
        # 1) form a set of the current classes and all their parent classes 
        #    recursively
        # 2) from the current set, apply possible conversions to get a new set 
        #    of classes
        # Repeat these two steps until in step (1) you hit the target class.
        
        while len(converted_list) > 0:
            # Go through all parents recursively so we can then use all
            # converters that are available
            current_list = []
            for (moments_class, converter_path) in converted_list:
                if issubclass(moments_class, moments_to):
                    # Shortest conversion path found, return the resulting total
                    # conversion function
                    return misc.composite_function(converter_path)
                current_list.append((moments_class, converter_path))
                parents = list(moments_class.__bases__)
                for parent in parents:
                    # Recursively add parents
                    for p in parent.__bases__:
                        if isinstance(p, Moments):
                            parents.append(p)
                    # Add un-visited parents
                    if issubclass(parent, Moments) and parent not in visited:
                        visited.add(parent)
                        current_list.append((parent, converter_path))

            # Find all converters and extend the converter paths
            converted_list = []
            for (moments_class, converter_path) in current_list:
                for (conv_mom_cls, conv) in moments_class._converters.items():
                    if conv_mom_cls not in visited:
                        visited.add(conv_mom_cls)
                        converted_list.append((conv_mom_cls,
                                               converter_path + [conv])) 

        raise self.NoConverterError("No conversion defined from %s to %s"
                                    % (self.__class__.__name__,
                                       moments_to.__name__))


    def compute_fixed_moments(self, x):
        # This method can't be static because the computation of the moments may
        # depend on, for instance, ndim in Gaussian arrays.
        raise NotImplementedError("compute_fixed_moments not implemented for "
                                  "%s" 
                                  % (self.__class__.__name__))


    @classmethod
    def from_values(cls, x):
        raise NotImplementedError("from_values not implemented "
                                  "for %s"
                                  % (cls.__name__))

def ensureparents(func):
    @functools.wraps(func)
    def wrapper(self, *parents, **kwargs):
        # Convert parents to proper nodes
        if self._parent_moments is None:
            raise ValueError(
                "Parent moments must be defined for {0}"
                .format(self.__class__.__name__)
            )
        parents = [
            Node._ensure_moments(
                parent,
                moments.__class__,
                **moments.get_instance_conversion_kwargs()
            )
            for (parent, moments) in zip(parents, self._parent_moments)
        ]
        # parents = list(parents)
        # for (ind, parent) in enumerate(parents):
        #     parents[ind] = self._ensure_moments(parent, 
        #                                         self._parent_moments[ind])
        # Run the function
        return func(self, *parents, **kwargs)

    return wrapper


class Node():
    """
    Base class for all nodes.

    mask
    dims
    plates
    parents
    children
    name

    Sub-classes must implement:
    1. For computing the message to children:
       get_moments(self):
    2. For computing the message to parents:
       _get_message_and_mask_to_parent(self, index)

    Sub-classes may need to re-implement:
    1. If they manipulate plates:
       _compute_weights_to_parent(index, weights)
       _plates_to_parent(self, index)
       _plates_from_parent(self, index)
    """

    # These are objects of the _parent_moments_class. If the default way of
    # creating them is not correct, write your own creation code.
    _moments = None
    _parent_moments = None
    plates = None

    _id_counter = 0

    @ensureparents
    def __init__(self, *parents, dims=None, plates=None, name="",
                 notify_parents=True, plotter=None, plates_multiplier=None,
                 allow_dependent_parents=False):

        self.parents = parents
        self.dims = dims
        self.name = name
        self._plotter = plotter

        if not allow_dependent_parents:
            parent_id_list = []
            for parent in parents:
                parent_id_list = parent_id_list + list(parent._get_id_list())
            if len(parent_id_list) != len(set(parent_id_list)):
                raise ValueError("Parent nodes are not independent")

        # Inform parent nodes
        if notify_parents:
            for (index,parent) in enumerate(self.parents):
                parent._add_child(self, index)

        # Check plates
        parent_plates = [self._plates_from_parent(index) 
                         for index in range(len(self.parents))]
        if any(p is None for p in parent_plates):
            raise ValueError("Method _plates_from_parent returned None")

        # Get and validate the plates for this node
        plates = self._total_plates(plates, *parent_plates)
        if self.plates is None:
            self.plates = plates

        # By default, ignore all plates
        self.mask = np.array(False)

        # Children
        self.children = set()

        # Get and validate the plate multiplier
        parent_plates_multiplier = [self._plates_multiplier_from_parent(index) 
                                   for index in range(len(self.parents))]
        #if plates_multiplier is None:
        #    plates_multiplier = parent_plates_multiplier
        plates_multiplier = self._total_plates(plates_multiplier,
                                              *parent_plates_multiplier)
        self.plates_multiplier = plates_multiplier


    def _get_id_list(self):
        """
        Returns the stochastic ID list.

        This method is used to check that same stochastic nodes are not direct
        parents of a node several times. It is only valid if there are
        intermediate stochastic nodes.

        To put it another way: each ID corresponds to one factor q(..) in the
        posterior approximation. Different IDs mean different factors, thus they
        mean independence. The parents must have independent factors.

        Stochastic nodes should return their unique ID. Deterministic nodes
        should return the IDs of their parents. Constant nodes should return
        empty list of IDs.
        """
        raise NotImplementedError()


    @classmethod
    def _total_plates(cls, plates, *parent_plates):
        if plates is None:
            # By default, use the minimum number of plates determined
            # from the parent nodes
            try:
                return misc.broadcasted_shape(*parent_plates)
            except ValueError:
                raise ValueError(
                    "The plates of the parents do not broadcast: {0}".format(
                        parent_plates
                    )
                )
        else:
            # Check that the parent_plates are a subset of plates.
            for (ind, p) in enumerate(parent_plates):
                if not misc.is_shape_subset(p, plates):
                    raise ValueError("The plates %s of the parents "
                                     "are not broadcastable to the given "
                                     "plates %s."
                                     % (p,
                                        plates))
            return plates


    @staticmethod
    def _ensure_moments(node, moments_class, **kwargs):
        try:
            converter = node._moments.get_converter(moments_class)
        except AttributeError:
            from .constant import Constant
            return Constant(
                moments_class.from_values(node, **kwargs),
                node
            )
        else:
            node = converter(node)
            converter = node._moments.get_instance_converter(**kwargs)
            if converter is not None:
                from .converters import NodeConverter
                return NodeConverter(converter, node)
            return node


    def _compute_plates_to_parent(self, index, plates):
        # Sub-classes may want to overwrite this if they manipulate plates
        return plates


    def _compute_plates_from_parent(self, index, plates):
        # Sub-classes may want to overwrite this if they manipulate plates
        return plates


    def _compute_plates_multiplier_from_parent(self, index, plates_multiplier):
        return self._compute_plates_from_parent(index, plates_multiplier)


    def _plates_to_parent(self, index):
        return self._compute_plates_to_parent(index, self.plates)


    def _plates_from_parent(self, index):
        return self._compute_plates_from_parent(index,
                                                self.parents[index].plates)


    def _plates_multiplier_from_parent(self, index):
        return self._compute_plates_multiplier_from_parent(
            index,
            self.parents[index].plates_multiplier
        )


    @property
    def plates_multiplier(self):
        """ Plate multiplier is applied to messages to parents """
        return self.__plates_multiplier


    @plates_multiplier.setter
    def plates_multiplier(self, value):
        # TODO/FIXME: Check that multiplier is consistent with plates
        self.__plates_multiplier = value
        return


    def get_shape(self, ind):
        return self.plates + self.dims[ind]

    def _add_child(self, child, index):
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
        self.children.add((child, index))

    def _remove_child(self, child, index):
        """
        Remove a child node.
        """
        self.children.remove((child, index))

    def get_mask(self):
        return self.mask
    
    ## def _get_message_mask(self):
    ##     return self.mask
    
    def _set_mask(self, mask):
        # Sub-classes may overwrite this method if they have some other masks to
        # be combined (for instance, observation mask)
        self.mask = mask
    
    def _update_mask(self):
        # Combine masks from children
        mask = np.array(False)
        for (child, index) in self.children:
            mask = np.logical_or(mask, child._mask_to_parent(index))
        # Set the mask of this node
        self._set_mask(mask)
        if not misc.is_shape_subset(np.shape(self.mask), self.plates):

            raise ValueError("The mask of the node %s has updated "
                             "incorrectly. The plates in the mask %s are not a "
                             "subset of the plates of the node %s."
                             % (self.name,
                                np.shape(self.mask),
                                self.plates))
        
        # Tell parents to update their masks
        for parent in self.parents:
            parent._update_mask()


    def _compute_weights_to_parent(self, index, weights):
        """Compute the mask used for messages sent to parent[index].

        The mask tells which plates in the messages are active. This method is
        used for obtaining the mask which is used to set plates in the messages
        to parent to zero.

        Sub-classes may want to overwrite this method if they do something to
        plates so that the mask is somehow altered.

        """
        return weights


    def _mask_to_parent(self, index):
        """
        Get the mask with respect to parent[index].

        The mask tells which plate connections are active. The mask is "summed"
        (logical or) and reshaped into the plate shape of the parent. Thus, it
        can't be used for masking messages, because some plates have been summed
        already. This method is used for propagating the mask to parents.
        """
        mask = self._compute_weights_to_parent(index, self.mask) != 0

        # Check the shape of the mask
        plates_to_parent = self._plates_to_parent(index)
        if not misc.is_shape_subset(np.shape(mask), plates_to_parent):
            raise ValueError("In node %s, the mask being sent to "
                             "parent[%d] (%s) has invalid shape: The shape of "
                             "the mask %s is not a sub-shape of the plates of "
                             "the node with respect to the parent %s. It could "
                             "be that this node (%s) is manipulating plates "
                             "but has not overwritten the method "
                             "_compute_weights_to_parent."
                             % (self.name,
                                index,
                                self.parents[index].name,
                                np.shape(mask),
                                plates_to_parent,
                                self.__class__.__name__))

        # "Sum" (i.e., logical or) over the plates that have unit length in 
        # the parent node.
        parent_plates = self.parents[index].plates
        s = misc.axes_to_collapse(np.shape(mask), parent_plates)
        mask = np.any(mask, axis=s, keepdims=True)
        mask = misc.squeeze_to_dim(mask, len(parent_plates))
        return mask

    def _message_to_child(self):

        u = self.get_moments()

        # Debug: Check that the message has appropriate shape
        for (ui, dim) in zip(u, self.dims):
            ndim = len(dim)
            if ndim > 0:
                if np.shape(ui)[-ndim:] != dim:
                    raise RuntimeError(
                        "A bug found by _message_to_child for %s: "
                        "The variable axes of the moments %s are not equal to "
                        "the axes %s defined by the node %s. A possible reason "
                        "is that the plates of the node are inferred "
                        "incorrectly from the parents, and the method "
                        "_plates_from_parents should be implemented."
                        % (self.__class__.__name__,
                           np.shape(ui)[-ndim:],
                           dim,
                           self.name))
                if not misc.is_shape_subset(np.shape(ui)[:-ndim],
                                            self.plates):
                    raise RuntimeError(
                        "A bug found by _message_to_child for %s: "
                        "The plate axes of the moments %s are not a subset of "
                        "the plate axes %s defined by the node %s."
                        % (self.__class__.__name__,
                           np.shape(ui)[:-ndim],
                           self.plates,
                           self.name))
            else:
                if not misc.is_shape_subset(np.shape(ui), self.plates):
                    raise RuntimeError(
                        "A bug found by _message_to_child for %s: "
                        "The plate axes of the moments %s are not a subset of "
                        "the plate axes %s defined by the node %s."
                        % (self.__class__.__name__,
                           np.shape(ui),
                           self.plates,
                           self.name))
        return u
                
    def _message_to_parent(self, index, u_parent=None):

        # Compute the message, check plates, apply mask and sum over some plates
        if index >= len(self.parents):
            raise ValueError("Parent index larger than the number of parents")

        # Compute the message and mask
        (m, mask) = self._get_message_and_mask_to_parent(index, u_parent=u_parent)
        mask = misc.squeeze(mask)

        # Plates in the mask
        plates_mask = np.shape(mask)

        # The parent we're sending the message to
        parent = self.parents[index]

        # Plates with respect to the parent
        plates_self = self._plates_to_parent(index)

        # Plate multiplier of the parent
        multiplier_parent = self._plates_multiplier_from_parent(index)

        # Check if m is a logpdf function (for black-box variational inference)
        if callable(m):
            return m

            def m_function(*args):
                lpdf = m(*args)
                # Log pdf only contains plate axes!
                plates_m = np.shape(lpdf)
                r = (self.broadcasting_multiplier(plates_self,
                                                  plates_m,
                                                  plates_mask,
                                                  parent.plates) *
                     self.broadcasting_multiplier(self.plates_multiplier,
                                                  multiplier_parent))
                axes_msg = misc.axes_to_collapse(plates_m, parent.plates)
                m[i] = misc.sum_multiply(mask_i, m[i], r,
                                         axis=axes_msg,
                                         keepdims=True)

                # Remove leading singular plates if the parent does not have
                # those plate axes.
                m[i] = misc.squeeze_to_dim(m[i], len(shape_parent))

            return m_function
            raise NotImplementedError()

        # Compact the message to a proper shape
        for i in range(len(m)):

            # Empty messages are given as None. We can ignore those.
            if m[i] is not None:

                try:
                    r = self.broadcasting_multiplier(self.plates_multiplier,
                                                     multiplier_parent)
                except:
                    raise ValueError("The plate multipliers are incompatible. "
                                     "This node (%s) has %s and parent[%d] "
                                     "(%s) has %s"
                                     % (self.name,
                                        self.plates_multiplier,
                                        index,
                                        parent.name,
                                        multiplier_parent))

                ndim = len(parent.dims[i])
                # Source and target shapes
                if ndim > 0:
                    dims = misc.broadcasted_shape(np.shape(m[i])[-ndim:],
                                                  parent.dims[i])
                    from_shape = plates_self + dims
                else:
                    from_shape = plates_self
                to_shape = parent.get_shape(i)
                # Add variable axes to the mask
                mask_i = misc.add_trailing_axes(mask, ndim)
                # Apply mask and sum plate axes as necessary (and apply plate
                # multiplier)
                m[i] = r * misc.sum_multiply_to_plates(np.where(mask_i, m[i], 0),
                                                       to_plates=to_shape,
                                                       from_plates=from_shape,
                                                       ndim=0)

        return m

    def _message_from_children(self, u_self=None):
        msg = [np.zeros(shape) for shape in self.dims]
        #msg = [np.array(0.0) for i in range(len(self.dims))]
        isfunction = None
        for (child,index) in self.children:
            m = child._message_to_parent(index, u_parent=u_self)
            if callable(m):
                if isfunction is False:
                    raise NotImplementedError()
                elif isfunction is None:
                    msg = m
                else:
                    def join(m1, m2):
                        return (m1[0] + m2[0], m1[1] + m2[1])
                    msg = lambda x: join(m(x), msg(x))
                    isfunction = True
            else:
                if isfunction is True:
                    raise NotImplementedError()
                else:
                    isfunction = False
                    for i in range(len(self.dims)):
                        if m[i] is not None:
                            # Check broadcasting shapes
                            sh = misc.broadcasted_shape(self.get_shape(i), np.shape(m[i]))
                            try:
                                # Try exploiting broadcasting rules
                                msg[i] += m[i]
                            except ValueError:
                                msg[i] = msg[i] + m[i]

        return msg

    def _message_from_parents(self, exclude=None):
        return [list(parent._message_to_child())
                if ind != exclude else
                None
                for (ind,parent) in enumerate(self.parents)]

    def get_moments(self):
        raise NotImplementedError()

    def delete(self):
        """
        Delete this node and the children
        """
        for (ind, parent) in enumerate(self.parents):
            parent._remove_child(self, ind)
        for (child, _) in self.children:
            child.delete()

    @staticmethod
    def broadcasting_multiplier(plates, *args):
        return misc.broadcasting_multiplier(plates, *args)
        ## """
        ## Compute the plate multiplier for given shapes.

        ## The first shape is compared to all other shapes (using NumPy
        ## broadcasting rules). All the elements which are non-unit in the first
        ## shape but 1 in all other shapes are multiplied together.

        ## This method is used, for instance, for computing a correction factor for
        ## messages to parents: If this node has non-unit plates that are unit
        ## plates in the parent, those plates are summed. However, if the message
        ## has unit axis for that plate, it should be first broadcasted to the
        ## plates of this node and then summed to the plates of the parent. In
        ## order to avoid this broadcasting and summing, it is more efficient to
        ## just multiply by the correct factor. This method computes that
        ## factor. The first argument is the full plate shape of this node (with
        ## respect to the parent). The other arguments are the shape of the message
        ## array and the plates of the parent (with respect to this node).
        ## """
        
        ## # Check broadcasting of the shapes
        ## for arg in args:
        ##     misc.broadcasted_shape(plates, arg)

        ## # Check that each arg-plates are a subset of plates?
        ## for arg in args:
        ##     if not misc.is_shape_subset(arg, plates):
        ##         raise ValueError("The shapes in args are not a sub-shape of "
        ##                          "plates.")
            
        ## r = 1
        ## for j in range(-len(plates),0):
        ##     mult = True
        ##     for arg in args:
        ##         # if -j <= len(arg) and arg[j] != 1:
        ##         if not (-j > len(arg) or arg[j] == 1):
        ##             mult = False
        ##     if mult:
        ##         r *= plates[j]
        ## return r

    def move_plates(self, from_plate, to_plate):
        return _MovePlate(self, 
                          from_plate,
                          to_plate,
                          name=self.name + ".move_plates")

    def add_plate_axis(self, to_plate):
        return AddPlateAxis(to_plate)(self,
                                      name=self.name+".add_plate_axis")

    def __getitem__(self, index):
        return Slice(self, index,
                     name=(self.name+".__getitem__"))

    def has_plotter(self):
        """
        Return True if the node has a plotter
        """
        return callable(self._plotter)

    def set_plotter(self, plotter):
        self._plotter = plotter
    
    def plot(self, fig=None, **kwargs):
        """
        Plot the node distribution using the plotter of the node

        Because the distributions are in general very difficult to plot, the
        user must specify some functions which performs the plotting as
        wanted. See, for instance, bayespy.plot.plotting for available plotters,
        that is, functions that perform plotting for a node.
        """
        if fig is None:
            fig = plt.gcf()
        if callable(self._plotter):
            ax = self._plotter(self, fig=fig, **kwargs)
            fig.suptitle('q(%s)' % self.name)
            return ax
        else:
            raise Exception("No plotter defined, can not plot")


    @staticmethod
    def _compute_message(*arrays, plates_from=(), plates_to=(), ndim=0):
        """
        A general function for computing messages by sum-multiply

        The function computes the product of the input arrays and then sums to
        the requested plates.
        """

        # Check that the plates broadcast properly
        if not misc.is_shape_subset(plates_to, plates_from):
            raise ValueError("plates_to must be broadcastable to plates_from")

        # Compute the explicit shape of the product
        shapes = [np.shape(array) for array in arrays]
        arrays_shape = misc.broadcasted_shape(*shapes)

        # Compute plates and dims that are present
        if ndim == 0:
            arrays_plates = arrays_shape
            dims = ()
        else:
            arrays_plates = arrays_shape[:-ndim]
            dims = arrays_shape[-ndim:]

        # Compute the correction term.  If some of the plates that should be
        # summed are actually broadcasted, one must multiply by the size of the
        # corresponding plate
        r = Node.broadcasting_multiplier(plates_from, arrays_plates, plates_to)

        # For simplicity, make the arrays equal ndim
        arrays = misc.make_equal_ndim(*arrays)

        # Keys for the input plates: (N-1, N-2, ..., 0)
        nplates = len(arrays_plates)
        in_plate_keys = list(range(nplates-1, -1, -1))

        # Keys for the output plates
        out_plate_keys = [key
                          for key in in_plate_keys
                          if key < len(plates_to) and plates_to[-key-1] != 1]

        # Keys for the dims
        dim_keys = list(range(nplates, nplates+ndim))

        # Total input and output keys
        in_keys = len(arrays) * [in_plate_keys + dim_keys]
        out_keys = out_plate_keys + dim_keys

        # Compute the sum-product with correction
        einsum_args = misc.zipper_merge(arrays, in_keys) + [out_keys]
        y = r * np.einsum(*einsum_args)

        # Reshape the result and apply correction
        nplates_result = min(len(plates_to), len(arrays_plates))
        if nplates_result == 0:
            plates_result = []
        else:
            plates_result = [min(plates_to[ind], arrays_plates[ind])
                             for ind in range(-nplates_result, 0)]

        y = np.reshape(y, plates_result + list(dims))

        return y


from .deterministic import Deterministic


def slicelen(s, length=None):
    if length is not None:
        s = slice(*(s.indices(length)))
    return max(0, misc.ceildiv(s.stop - s.start, s.step))

class Slice(Deterministic):

    """
    Basic slicing for plates.
    
    Slicing occurs when index is a slice object (constructed by start:stop:step
    notation inside of brackets), an integer, or a tuple of slice objects and
    integers.

    Currently, accept slices, newaxis, ellipsis and integers. For instance, does
    not accept lists/tuples to pick multiple indices of the same axis.

    Ellipsis expand to the number of : objects needed to make a selection tuple
    of the same length as x.ndim. Only the first ellipsis is expanded, any
    others are interpreted as :.

    Similar to:
    http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing
    """


    def __init__(self, X, slices, **kwargs):

        self._moments = X._moments
        self._parent_moments = (X._moments,)

        # Force a list
        if not isinstance(slices, tuple):
            slices = [slices]
        else:
            slices = list(slices)

        #
        # Expand Ellipsis
        #

        # Compute the number of required axes and how Ellipsis is expanded
        num_axis = 0
        ellipsis_index = None
        for (k, s) in enumerate(slices):

            if misc.is_scalar_integer(s) or isinstance(s, slice):
                num_axis += 1

            elif s is None:
                pass
                
            elif s is Ellipsis:
                # Index is an ellipsis, e.g., [...]
                
                if ellipsis_index is None:
                    # Expand ...
                    ellipsis_index = k
                else:
                    # Interpret ... as :
                    num_axis += 1
                    slices[k] = slice(None)
                    
            else:
                raise TypeError("Invalid argument type: {0}".format(s.__class__))

        if num_axis > len(X.plates):
            raise IndexError("Too many indices")

        # The number of plates that were not given explicit slicing (either
        # Ellipsis was used or the number of slices was smaller than the number
        # of plate axes)
        expand_len = len(X.plates) - num_axis
        
        if ellipsis_index is not None:
            # Replace Ellipsis with correct number of :
            k = ellipsis_index
            del slices[k]
            slices = slices[:k] + [slice(None)] * expand_len + slices[k:]
        else:
            # Add trailing : so that each plate has explicit slicing
            slices = slices + [slice(None)] * expand_len

        #
        # Preprocess indexing:
        # - integer indices to non-negative values
        # - slice start/stop values to non-negative
        # - slice start/stop values based on the size of the plate
        #

        # Index for parent plates
        j = 0
        
        for (k, s) in enumerate(slices):

            if misc.is_scalar_integer(s):
                # Index is an integer, e.g., [3]
                
                if s < 0:
                    # Handle negative index
                    s += X.plates[j]
                if s < 0 or s >= X.plates[j]:
                    raise IndexError("Index out of range")
                # Store the preprocessed integer index
                slices[k] = s
                j += 1

            elif isinstance(s, slice):
                # Index is a slice, e.g., [2:6]
                
                # Normalize the slice
                s = slice(*(s.indices(X.plates[j])))
                if slicelen(s) <= 0:
                    raise IndexError("Slicing leads to empty plates")
                slices[k] = s
                j += 1

        self.slices = slices

        super().__init__(X,
                         dims=X.dims,
                         **kwargs)

    def _plates_to_parent(self, index):
        return self.parents[index].plates

    def _plates_from_parent(self, index):

        plates = list(self.parents[index].plates)

        # Compute the plates. Note that Ellipsis has already been preprocessed
        # to a proper number of :
        k = 0
        for s in self.slices:
            # Then, each case separately: slice, newaxis, integer
            
            if isinstance(s, slice):
                # Slice, e.g., [2:5]
                N = slicelen(s)
                if N <= 0:
                    raise IndexError("Slicing leads to empty plates")
                plates[k] = N
                k += 1
                
            elif s is None:
                # [np.newaxis]
                plates = plates[:k] + [1] + plates[k:]
                k += 1
                
            elif misc.is_scalar_integer(s):
                # Integer, e.g., [3]
                del plates[k]
            else:
                raise RuntimeError("BUG: Unknown index type. Should capture earlier.")

        return tuple(plates)

    @staticmethod
    def __reverse_indexing(slices, m_child, plates, dims):
        """
        A helpful function for performing reverse indexing/slicing
        """

        j = -1 # plate index for parent
        i = -1 # plate index for child
        child_slices = ()
        parent_slices = ()
        msg_plates = ()

        # Compute plate axes in the message from children
        ndim = len(dims)
        if ndim > 0:
            m_plates = np.shape(m_child)[:-ndim]
        else:
            m_plates = np.shape(m_child)

        for s in reversed(slices):

            if misc.is_scalar_integer(s):
                # Case: integer
                parent_slices = (s,) + parent_slices
                msg_plates = (plates[j],) + msg_plates
                j -= 1
            elif s is None:
                # Case: newaxis
                if -i <= len(m_plates):
                    child_slices = (0,) + child_slices
                i -= 1
            elif isinstance(s, slice):
                # Case: slice
                if -i <= len(m_plates):
                    child_slices = (slice(None),) + child_slices
                parent_slices = (s,) + parent_slices
                if ((-i > len(m_plates) or m_plates[i] == 1)
                    and slicelen(s) == plates[j]):
                    # Broadcasting can be applied. The message does not need
                    # to be explicitly shaped to the full size
                    msg_plates = (1,) + msg_plates
                else:
                    # No broadcasting. Must explicitly form the full size
                    # axis
                    msg_plates = (plates[j],) + msg_plates
                j -= 1
                i -= 1
            else:
                raise RuntimeError("BUG: Unknown index type. Should capture earlier.")

        # Set the elements of the message
        m_parent = np.zeros(msg_plates + dims)
        if np.ndim(m_parent) == 0 and np.ndim(m_child) == 0:
            m_parent = m_child
        elif np.ndim(m_parent) == 0:
            m_parent = m_child[child_slices]
        elif np.ndim(m_child) == 0:
            m_parent[parent_slices] = m_child
        else:
            m_parent[parent_slices] = m_child[child_slices]

        return m_parent


    def _compute_weights_to_parent(self, index, weights):
        """
        Compute the mask to the parent node.
        """
        if index != 0:
            raise ValueError("Invalid index")
        parent = self.parents[0]

        return self.__reverse_indexing(self.slices,
                                       weights,
                                       parent.plates,
                                       ())


    def _compute_message_to_parent(self, index, m, u):
        """
        Compute the message to a parent node.
        """

        if index != 0:
            raise ValueError("Invalid index")
        parent = self.parents[0]

        # Apply reverse indexing for the message arrays
        msg = [self.__reverse_indexing(self.slices, 
                                       m_child,
                                       parent.plates, 
                                       dims)
               for (m_child, dims) in zip(m, parent.dims)]

        return msg

    def _compute_moments(self, u):
        """
        Get the moments with an added plate axis.
        """

        # Process each moment
        for n in range(len(u)):

            # Compute the effective plates in the message/moment
            ndim = len(self.dims[n])
            if ndim > 0:
                shape = np.shape(u[n])[:-ndim]
            else:
                shape = np.shape(u[n])

            # Construct a list of slice objects
            u_slices = []

            # Index for the shape
            j = -len(self.parents[0].plates)

            for (k, s) in enumerate(self.slices):

                if s is None:
                    # [np.newaxis]
                    if -j < len(shape):
                        # Only add newaxis if there are some axes before
                        # this. It does not make any difference if you added
                        # leading unit axes
                        u_slices.append(s)
                    
                else:
                    # slice or integer index

                    if -j <= len(shape):
                        # The moment has this axis, so it is not broadcasting it
                        if shape[j] != 1:
                            # Use the slice as it is
                            u_slices.append(s)
                        elif isinstance(s, slice):
                            # Slice.
                            # The moment is using broadcasting, just pick the
                            # first element but use slice in order to keep the
                            # axis
                            u_slices.append(slice(0,1,1))
                        else:
                            # Integer.
                            # The moment is using broadcasting, just pick the
                            # first element
                            u_slices.append(0)
                            
                    j += 1

            # Slice the message/moment
            u[n] = u[n][tuple(u_slices)]
        
        return u

def AddPlateAxis(to_plate):
    
    if to_plate >= 0:
        raise Exception("Give negative value for axis index to_plate.")

    class _AddPlateAxis(Deterministic):

        def __init__(self, X, **kwargs):

            nonlocal to_plate

            N = len(X.plates) + 1

            # Check the parameters
            if to_plate >= 0 or to_plate < -N:
                raise ValueError("Invalid plate position to add.")

            # Use positive indexing only
            ## if to_plate < 0:
            ##     to_plate += N
            # Use negative indexing only
            if to_plate >= 0:
                to_plate -= N
                #self.to_plate = to_plate

            super().__init__(X, 
                             dims=X.dims,
                             **kwargs)

        def _plates_to_parent(self, index):
            plates = list(self.plates)
            plates.pop(to_plate)
            return tuple(plates)
        #return self.plates[:to_plate] + self.plates[(to_plate+1):]

        def _plates_from_parent(self, index):
            plates = list(self.parents[index].plates)
            plates.insert(len(plates)-to_plate+1, 1)
            return tuple(plates)


        def _compute_weights_to_parent(self, index, weights):
            # Remove the added mask plate
            if abs(to_plate) <= np.ndim(weights):
                sh_weighs = list(np.shape(weights))
                sh_weights.pop(to_plate)
                weights = np.reshape(weights, sh_weights)
            return weights


        def _compute_message_to_parent(self, index, m, *u_parents):
            """
            Compute the message to a parent node.
            """

            # Remove the added message plate
            for i in range(len(m)):
                # Remove the axis
                if np.ndim(m[i]) >= abs(to_plate) + len(self.dims[i]):
                    axis = to_plate - len(self.dims[i])
                    sh_m = list(np.shape(m[i]))
                    sh_m.pop(axis)
                    m[i] = np.reshape(m[i], sh_m)

            return m

        def _compute_moments(self, u):
            """
            Get the moments with an added plate axis.
            """

            # Get parents' moments
            #u = self.parents[0].message_to_child()

            # Move a plate axis
            u = list(u)
            for i in range(len(u)):
                # Make sure the moments have all the axes
                #diff = len(self.plates) + len(self.dims[i]) - np.ndim(u[i]) - 1
                #u[i] = misc.add_leading_axes(u[i], diff)
                
                # The location of the new axis/plate:
                axis = np.ndim(u[i]) - abs(to_plate) - len(self.dims[i]) + 1
                if axis > 0:
                    # Add one axes to the correct position
                    sh_u = list(np.shape(u[i]))
                    sh_u.insert(axis, 1)
                    u[i] = np.reshape(u[i], sh_u)

            return u

    return _AddPlateAxis
        

class NodeConstantScalar(Node):
    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        return ([x], 0)

    def __init__(self, a, **kwargs):
        self.u = [a]
        super().__init__(self,
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


    
