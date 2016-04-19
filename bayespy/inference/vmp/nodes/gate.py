################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
"""

import numpy as np

from bayespy.utils import misc

from .node import Node, Moments
from .deterministic import Deterministic
from .categorical import CategoricalMoments
from .concatenate import Concatenate


class Gate(Deterministic):
    """
    Deterministic gating of one node.

    Gating is performed over one plate axis.

    Note: You should not use gating for several variables which parents of a
    same node if the gates use the same gate assignments.  In such case, the
    results will be wrong.  The reason is a general one: A stochastic node may
    not be a parent of another node via several paths unless at most one path
    has no other stochastic nodes between them.
    """

    def __init__(self, Z, X, gated_plate=-1, moments=None, **kwargs):
        """
        Constructor for the gating node.

        Parameters
        ----------

        Z : Categorical-like node
           A variable which chooses the index along the gated plate axis

        X : node
           The node whose plate axis is gated

        gated_plate : int (optional)
           The index of the plate axis to be gated (by default, -1, that is,
           the last axis).
        """

        if gated_plate >= 0:
            raise ValueError("Cluster plate must be negative integer")
        self.gated_plate = gated_plate

        if moments is not None:
            X = self._ensure_moments(
                X,
                moments.__class__,
                **moments.get_instance_conversion_kwargs()
            )

        if not isinstance(X, Node):
            raise ValueError("X must be a node or moments should be provided")

        X_moments = X._moments
        self._moments = X_moments
        dims = X.dims
        if len(X.plates) < abs(gated_plate):
            raise ValueError("The gated node does not have a plate axis is "
                             "gated")
        K = X.plates[gated_plate]

        Z = self._ensure_moments(Z, CategoricalMoments, categories=K)

        self._parent_moments = (Z._moments, X_moments)

        if Z.dims != ( (K,), ):
            raise ValueError("Inconsistent number of clusters")
        
        self.K = K

        super().__init__(Z, X, dims=dims, **kwargs)
            

    def _compute_moments(self, u_Z, u_X):
        """
        """

        u = []
        for i in range(len(u_X)):
            # Make the moments of Z and X broadcastable and move the gated plate
            # to be the last axis in the moments, then sum-product over that
            # axis
            ndim = len(self.dims[i])
            z = misc.add_trailing_axes(u_Z[0], ndim)
            z = misc.moveaxis(z, -ndim-1, -1)
            gated_axis = self.gated_plate - ndim
            if np.ndim(u_X[i]) < abs(gated_axis):
                x = misc.add_trailing_axes(u_X[i], 1)
            else:
                x = misc.moveaxis(u_X[i], gated_axis, -1)
            ui = misc.sum_product(z, x, axes_to_sum=-1)
            u.append(ui)
        return u
    

    def _compute_message_to_parent(self, index, m_child, u_Z, u_X):
        """
        """
        if index == 0:
            m0 = 0
            # Compute Child * X, sum over variable axes and move the gated axis
            # to be the last.  Need to do some shape changing in order to make
            # Child and X to broadcast properly.
            for i in range(len(m_child)):
                ndim = len(self.dims[i])
                c = m_child[i][...,None]
                c = misc.moveaxis(c, -1, -ndim-1)
                gated_axis = self.gated_plate - ndim
                x = u_X[i]
                if np.ndim(x) < abs(gated_axis):
                    x = np.expand_dims(x, -ndim-1)
                else:
                    x = misc.moveaxis(x, gated_axis, -ndim-1)
                axes = tuple(range(-ndim, 0))
                m0 = m0 + misc.sum_product(c, x, axes_to_sum=axes)

            # Make sure the variable axis does not use broadcasting
            m0 = m0 * np.ones(self.K)

            # Send the message
            m = [m0]
            return m

        elif index == 1:
            
            m = []
            for i in range(len(m_child)):
                # Make the moments of Z and the message from children
                # broadcastable. The gated plate is handled as the last axis in
                # the arrays and moved to the correct position at the end.

                # Add variable axes to Z moments
                ndim = len(self.dims[i])
                z = misc.add_trailing_axes(u_Z[0], ndim)
                z = misc.moveaxis(z, -ndim-1, -1)
                # Axis index of the gated plate
                gated_axis = self.gated_plate - ndim
                # Add the gate axis to the message from the children
                c = misc.add_trailing_axes(m_child[i], 1)
                # Compute the message to parent
                mi = z * c
                # Add extra axes if necessary
                if np.ndim(mi) < abs(gated_axis):
                    mi = misc.add_leading_axes(mi,
                                                abs(gated_axis) - np.ndim(mi))
                # Move the axis to the correct position
                mi = misc.moveaxis(mi, -1, gated_axis)
                m.append(mi)
                
            return m
        
        else:
            raise ValueError("Invalid parent index")


    def _compute_weights_to_parent(self, index, weights):
        """
        """
        if index == 0:
            return weights
        elif index == 1:
            if self.gated_plate >= 0:
                raise ValueError("Gated plate axis must be negative")
            if np.ndim(weights) >= abs(self.gated_plate):
                mask = np.expand_dims(weights, axis=self.gated_plate)
            return weights
        else:
            raise ValueError("Invalid parent index")


    def _compute_plates_to_parent(self, index, plates):
        """
        """
        if index == 0:
            return plates
        elif index == 1:
            plates = list(plates)
            # Add the cluster plate axis
            if self.gated_plate < 0:
                knd = len(plates) + self.gated_plate + 1
            else:
                raise RuntimeError("Cluster plate axis must be negative")
            plates.insert(knd, self.K)

            return tuple(plates)
        else:
            raise ValueError("Invalid parent index")


    def _compute_plates_from_parent(self, index, plates):
        """
        """
        if index == 0:
            return plates
        elif index == 1:
            plates = list(plates)
            # Remove the cluster plate, if the parent has it
            if len(plates) >= abs(self.gated_plate):
                plates.pop(self.gated_plate)
            return tuple(plates)
        else:
            raise ValueError("Invalid parent index")


def Choose(z, *nodes):
    """Choose plate elements from nodes based on a categorical variable.

    For instance:

    .. testsetup::

       from bayespy.nodes import *

    .. code-block:: python

       >>> import bayespy as bp
       >>> z = [0, 0, 2, 1]
       >>> x0 = bp.nodes.GaussianARD(0, 1)
       >>> x1 = bp.nodes.GaussianARD(10, 1)
       >>> x2 = bp.nodes.GaussianARD(20, 1)
       >>> x = bp.nodes.Choose(z, x0, x1, x2)
       >>> print(x.get_moments()[0])
       [  0.   0.  20.  10.]

    This is basically just a thin wrapper over applying Gate node over the
    concatenation of the nodes.
    """
    categories = len(nodes)
    z = Deterministic._ensure_moments(
        z,
        CategoricalMoments,
        categories=categories
    )
    nodes = [node[...,None] for node in nodes]
    combined = Concatenate(*nodes)
    return Gate(z, combined)
