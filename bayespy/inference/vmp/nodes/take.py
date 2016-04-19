################################################################################
# Copyright (C) 2015 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################

import numpy as np

from .deterministic import Deterministic
from .node import Moments
from bayespy.utils import misc


class Take(Deterministic):
    """
    Choose elements/sub-arrays along a plate axis

    Basically, applies `np.take` on a plate axis. Allows advanced mapping of
    plates.

    Parameters
    ----------
    node : Node
        A node to apply the take operation on.
    indices : array of integers
        Plate elements to pick along a plate axis.
    plate_axis : int (negative)
        The plate axis to pick elements from (default: -1).

    See also
    --------
    numpy.take

    Examples
    --------

    >>> from bayespy.nodes import Gamma, Take
    >>> alpha = Gamma([1, 2, 3], [1, 1, 1])
    >>> x = Take(alpha, [1, 1, 2, 2, 1, 0])
    >>> x.get_moments()[0]
    array([ 2.,  2.,  3.,  3.,  2.,  1.])
    """


    def __init__(self, node, indices, plate_axis=-1, **kwargs):
        self._moments = node._moments
        self._parent_moments = (node._moments,)
        self._indices = np.array(indices)
        self._plate_axis = plate_axis
        self._original_length = node.plates[plate_axis]

        # Validate arguments
        if not misc.is_scalar_integer(plate_axis):
            raise ValueError("Plate axis must be integer")
        if plate_axis >= 0:
            raise ValueError("plate_axis must be negative index")
        if plate_axis < -len(node.plates):
            raise ValueError("plate_axis out of bounds")
        if not issubclass(self._indices.dtype.type, np.integer):
            raise ValueError("Indices must be integers")
        if (np.any(self._indices < -self._original_length) or
            np.any(self._indices >= self._original_length)):
            raise ValueError("Index out of bounds")

        super().__init__(node, dims=node.dims, **kwargs)


    def _compute_moments(self, u_parent):
        u = []
        for (ui, dimi) in zip(u_parent, self.dims):
            axis = self._plate_axis - len(dimi)
            # Just in case the taken axis is using broadcasting and has unit
            # length in u_parent, force it to have the correct length along the
            # axis in order to avoid errors in np.take.
            broadcaster = np.ones((self._original_length,) + (-axis-1)*(1,))
            u.append(np.take(ui*broadcaster, self._indices, axis=axis))
        return u


    def _compute_message_to_parent(self, index, m_child, u_parent):

        m = [
            misc.put_simple(
                mi,
                self._indices,
                axis=self._plate_axis-len(dimi),
                length=self._original_length,
            )
            for (mi, dimi) in zip(m_child, self.dims)
        ]
        return m


    def _compute_weights_to_parent(self, index, weights):

        return misc.put_simple(
            weights,
            self._indices,
            axis=self._plate_axis,
            length=self._original_length,
        )


    def _compute_plates_to_parent(self, index, plates):

        # Number of axes created by take operation
        N = np.ndim(self._indices)

        if self._plate_axis >= 0:
            raise RuntimeError("Plate axis should be negative")

        end_before = self._plate_axis - N + 1
        start_after = self._plate_axis + 1

        if end_before == 0:
            return plates + (self._original_length,)
        elif start_after == 0:
            return plates[:end_before] + (self._original_length,)

        return (plates[:end_before]
                + (self._original_length,)
                + plates[start_after:])


    def _compute_plates_from_parent(self, index, parent_plates):

        plates = parent_plates[:self._plate_axis] + np.shape(self._indices)
        if self._plate_axis != -1:
            plates = plates + parent_plates[(self._plate_axis+1):]
        return plates
