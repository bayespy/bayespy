
"""
This module contains a sketch of a new implementation of the framework.
"""

class Node():
    """
    Base class for all nodes.
    """
    
    def _message_to_child(self):
        # Sub-classes should implement this
        raise NotImplementedError()
    
    def _message_to_parent(self, index):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _message_from_children(self):
        # Get messages from children and sum them together.
        pass

    def _message_from_parents(self, ignore=None):
        return [parent._message_to_child() for parent in self.parents]

    def get_moments(self):
        # or get_distribution? not for all nodes, only stochastic. hmm..
        return self._message_to_child()
    
class Stochastic(Node):
    """
    Base class for nodes that are stochastic.

    u
    mask
    """
    
    def _message_to_child(self):
        return self.u

    def _message_to_parent(self, index):
        u_parents = self._message_from_parents(ignore=index)
        return self._compute_message(index, self.u, *u_parents)
    
    def update(self):
        u_parents = self._message_from_parents()
        m_children = self._message_from_children()
        self._update_distribution_and_lowerbound(m_children, *u_parents)

    def observe(y, mask=True):
        # Fix moments, compute f and propagate mask.
        pass

    def lowerbound():
        # Sub-class should implement this
        raise NotImplementedError()

    @staticmethod
    def _compute_message(index, u_self, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _update_distribution_and_lowerbound(self, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

class Deterministic(Node):
    """
    Base class for nodes that are deterministic.
    """
    
    def _message_to_child(self):
        u_parents = [parent._message_to_child() for parent in self.parents]
        return self._compute_moments(*u_parents)

    def _message_to_parent(self, index):
        u_parents = self._message_from_parents(ignore=index)
        m_children = self._message_from_children()
        return self._compute_message(index, m_children, *u_parents)
        
    
    def _compute_moments(*u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

    def _compute_message(index, m_children, *u_parents):
        # Sub-classes should implement this
        raise NotImplementedError()

class ExponentialFamily(Stochastic):
    """
    A base class for nodes using natural parameterization `phi`.

    phi
    """
    pass
