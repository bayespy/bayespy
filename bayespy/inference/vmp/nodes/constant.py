################################################################################
# Copyright (C) 2011-2012,2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np
from bayespy.utils import misc

from .node import Node, Moments

class Constant(Node):
    r"""
    Node for presenting constant values.

    The node wraps arrays into proper node type.
    """

    def __init__(self, moments, x, **kwargs):
        if not isinstance(moments, Moments) and issubclass(moments, Moments):
            raise ValueError("Give moments as an object instance instead of a class")
        self._moments = moments
        x = np.asanyarray(x)
        # Compute moments
        self.u = self._moments.compute_fixed_moments(x)
        # Dimensions of the moments
        dims = self._moments.dims
        # Resolve plates
        D = len(dims[0])
        if D > 0:
            plates = np.shape(self.u[0])[:-D]
        else:
            plates = np.shape(self.u[0])
        kwargs.setdefault('plates', plates)
        self._parent_moments = ()
        # Parent constructor
        super().__init__(dims=dims, **kwargs)


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
        return []


    def get_moments(self):
        return self.u


    def set_value(self, x):
        x = np.asanyarray(x)
        #shapes = [np.shape(ui) for ui in self.u]
        self.u = self._moments.compute_fixed_moments(x)
        for (i, dimsi) in enumerate(self.dims):
            correct_shape = tuple(self.plates) + tuple(dimsi)
            given_shape = np.shape(self.u[i])
            if not misc.is_shape_subset(given_shape, correct_shape):
                raise ValueError(
                    "Incorrect shape {0} for the array, expected {1}"
                    .format(given_shape, correct_shape)
                )
        return


    def lower_bound_contribution(self, gradient=False, **kwargs):
        # Deterministic functions are delta distributions so the lower bound
        # contribuion is zero.
        return 0
