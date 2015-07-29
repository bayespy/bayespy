################################################################################
# Copyright (C) 2011-2012,2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

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
        dims = self._moments.compute_dims_from_values(x)
        # Resolve plates
        D = len(dims[0])
        if D > 0:
            plates = np.shape(self.u[0])[:-D]
        else:
            plates = np.shape(self.u[0])
        # Parent constructor
        super().__init__(dims=dims, plates=plates, **kwargs)


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
        shapes = [np.shape(ui) for ui in self.u]
        self.u = self._moments.compute_fixed_moments(x)
        for (i, shape) in enumerate(shapes):
            if np.shape(self.u[i]) != shape:
                raise ValueError("Incorrect shape for the array")


    def lower_bound_contribution(self, gradient=False, **kwargs):
        # Deterministic functions are delta distributions so the lower bound
        # contribuion is zero.
        return 0
