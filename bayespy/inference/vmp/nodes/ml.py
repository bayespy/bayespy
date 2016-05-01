################################################################################
# Copyright (C) 2016 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################

import numpy as np

from .node import Moments
from .deterministic import Deterministic
from .stochastic import Stochastic


class DeltaMoments(Moments):
    r"""
    Class for the moments of constants or delta distributed variables
    """


    def __init__(self, shape):
        self.shape = shape
        return super().__init__()


    def get_converter(self, moments_to):
        if issubclass(DeltaMoments, moments_to):
            return lambda x: x
        return get_delta_moments_class_converter(moments_to)


    def compute_fixed_moments(self, x):
        r"""
        Compute the moments for a fixed value
        """
        return [x]


    def compute_dims_from_values(self, x):
        r"""
        Return the shape of the moments for a fixed value.
        """
        return ((),)


    def get_instance_conversion_kwargs(self):
        return dict(shape=self.shape)


    def get_instance_converter(self, shape):
        if shape != self.shape:
            raise ValueError()
        return None


class DeltaClassConverterMoments(Moments):


    def __init__(self, x, moments_class):
        self.x = x
        self.moments_class = moments_class
        return


    def get_instance_conversion_kwargs(self):
        return dict(i_am_delta=True)


    def get_instance_converter(self, **kwargs):
        if kwargs.get('i_am_delta'):
            return None
        moments = self.moments_class.from_values(
            self.x.get_moments()[0],
            **kwargs
        )
        return DeltaInstanceConverter(moments)


def get_delta_moments_class_converter(moments_class):


    class DeltaClassConverter(Deterministic):


        def __init__(self, node):
            self._parent_moments = (node._moments,)
            self._moments = DeltaClassConverterMoments(node, moments_class)
            return super().__init__(node, dims=((),))


        def _compute_moments(self, u):
            return u


        def _compute_message_to_parent(self, index, m, u):
            return m


    return DeltaClassConverter


class DeltaInstanceConverter():


    def __init__(self, moments):
        self.moments = moments
        return


    def compute_moments(self, u):
        return self.moments.compute_fixed_moments(u[0])


    def compute_message_to_parent(self, m, u_parent):
        x = u_parent[0]
        (u, du) = self.moments.compute_fixed_moments(x, gradient=m)
        return [du]


    def compute_weights_to_parent(self, weights):
        return 1


    def plates_multiplier_from_parent(self, plates_multiplier):
        return ()


    def plates_from_parent(self, plates):
        return self.moments.plates_from_shape(plates)


    def plates_to_parent(self, plates):
        return self.moments.shape_from_plates(plates)


class MaximumLikelihood(Stochastic):


    _parent_moments = ()


    def __init__(self, array, regularization=None, **kwargs):
        self._x = array
        self._moments = DeltaMoments(np.shape(array))
        self._regularization = regularization
        return super().__init__(
            plates=np.shape(array),
            dims=( (), ),
            initialize=False,
            **kwargs
        )


    def _get_id_list(self):
        return []


    def get_moments(self):
        return [self._x]


    def lower_bound_contribution(self, ignore_masked=None):
        if self._regularization is None:
            return 0

        return -np.sum(self._regularization(self._x))


    def get_riemannian_gradient(self):
        m_children = self._message_from_children(u_self=self.get_moments())
        g = m_children
        # TODO/FIXME: REGULARIZATION GRADIENT!!
        return g


    def get_gradient(self, rg):
        return rg


    def get_parameters(self):
        return [self._x]


    def set_parameters(self, x):
        if len(x) != 1:
            raise Exception("Wrong number of parameters. Should be 1, is {0}".format(len(x)))
        self._x = x[0]
        return


    def _update_distribution_and_lowerbound(self, m):
        raise NotImplementedError()


class Function(Deterministic):


    def __init__(self, function, *nodes_gradients, shape=None, **kwargs):
        self._function = function
        (nodes, gradients) = zip(*nodes_gradients)
        self._parent_moments = tuple(node._moments for node in nodes)
        self._gradients = gradients
        if shape is None:
            # Shape wasn't given explicitly. Computes the output value once to
            # determine the shape.
            y = self._compute_moments(
                *[
                    node.get_moments()
                    for node in nodes
                ]
            )
            shape = np.shape(y[0])
        self._moments = DeltaMoments(shape)
        return super().__init__(*nodes, dims=((),), **kwargs)


    def _compute_moments(self, *u_nodes):
        x = [u[0] for u in u_nodes]
        return [self._function(*x)]


    def _compute_message_to_parent(self, index, m, *u_nodes):
        x = [u[0] for u in u_nodes]
        return [self._gradients[index](m[0], *x)]


    def _compute_weights_to_parent(self, index, mask):
        return 1


    def _compute_plates_from_parent(self, index, plates):
        return self._moments.shape


    def _compute_plates_to_parent(self, index, plates):
        return self.parents[index].plates
