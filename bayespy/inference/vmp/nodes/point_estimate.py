################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


import numpy as np

from .node import Node, Moments
from .stochastic import Stochastic
from .deterministic import Deterministic


class DeltaMoments(Moments):
    r"""
    Class for the moments of constants or delta distributed variables
    """


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


class DeltaToAny(Deterministic):
    r"""
    Special converter of delta moments to any moments
    """


    def __init__(self, X, moments):
        r"""
        """
        self._moments = moments
        self._parent_moments = [DeltaMoments()]
        #(plates, dims) = moments.compute_plates_and_dims(X.get_shape(0))
        dims = moments.compute_dims_from_shape(X.get_shape(0))
        super().__init__(X, dims=dims, **kwargs)
            

    def _compute_moments(self, u_X):
        r"""
        """
        x = u_X[0]
        return self._moments.compute_fixed_moments(x)
    

    def _compute_message_to_parent(self, index, m_child, u_X):
        r"""
        """
        # Convert child message array to a gradient function
        raise NotImplementedError()
        if index == 0:
            m = m_child[:2]
            return m
        else:
            raise ValueError("Invalid parent index")


    def _compute_weights_to_parent(self, index, weights):
        r"""
        """
        raise NotImplementedError()
        if index == 0:
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_to_parent(self, index):
        r"""
        """
        
        raise NotImplementedError()
        if index == 0:
            self.get_shape(0)
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")


    def _plates_from_parent(self, index):
        r"""
        """
        raise NotImplementedError()
        if index == 0:
            return self.__cached_plates
            raise NotImplementedError()
        else:
            raise ValueError("Invalid parent index")



class Scalar(Stochastic):


    def __init__(self, plates=None):
        dims = [()]
        raise NotImplementedError()


    def get_riemannian_gradient(self):
        m_children = self._message_from_children()
        g = self.annealing * m_children[0]
        return g


    def get_gradient(self, rg):
        return rg


    def get_parameters(self):
        return self.u


    def set_parameters(self, x):
        if len(x) != 1:
            raise Exception("Wrong number of parameters. Should be 1, is {0}".format(len(x)))
        self.u = [x[0]]
        return


class PositiveScalar(Stochastic):
    pass


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
