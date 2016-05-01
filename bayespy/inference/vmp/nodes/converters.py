# Copyright (c) 2016 Jaakko Luttinen
# MIT License

from .deterministic import Deterministic


class NodeConverter(Deterministic):
    """
    Simple wrapper to transform moment converters into nodes
    """


    def __init__(self, moments_converter, node):
        self.moments_converter = moments_converter
        self._parent_moments = (node._moments,)
        self._moments = moments_converter.moments
        super().__init__(node, dims=self._moments.dims)


    def _compute_moments(self, u_node):
        return self.moments_converter.compute_moments(u_node)


    def _compute_message_to_parent(self, index, m_child, u_node):
        if index != 0:
            raise IndexError()
        return self.moments_converter.compute_message_to_parent(m_child, u_node)


    def _compute_weights_to_parent(self, index, weights):
        if index != 0:
            raise IndexError()
        return self.moments_converter.compute_weights_to_parent(weights)


    def _compute_plates_to_parent(self, index, plates):
        if index != 0:
            raise IndexError()
        return self.moments_converter.plates_to_parent(plates)


    def _compute_plates_from_parent(self, index, plates):
        if index != 0:
            raise IndexError()
        return self.moments_converter.plates_from_parent(plates)


    def _compute_plates_multiplier_from_parent(self, index, plates_multiplier):
        if index != 0:
            raise IndexError()
        return self.moments_converter.plates_multiplier_from_parent(
            plates_multiplier
        )
