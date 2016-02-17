################################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under the MIT License.
################################################################################


"""
A module for the Bernoulli distribution node
"""

import numpy as np

from .binomial import (BinomialMoments,
                       BinomialDistribution)
from .expfamily import ExponentialFamily
from .beta import BetaMoments
from .node import Moments


class BernoulliMoments(BinomialMoments):
    """
    Class for the moments of Bernoulli variables.
    """


    def __init__(self):
        super().__init__(1)


class BernoulliDistribution(BinomialDistribution):
    """
    Class for the VMP formulas of Bernoulli variables.
    """


    def __init__(self):
        super().__init__(1)


class Bernoulli(ExponentialFamily):
    r"""
    Node for Bernoulli random variables.

    The node models a binary random variable :math:`z \in \{0,1\}` with prior
    probability :math:`p \in [0,1]` for value one:

    .. math::

        z \sim \mathrm{Bernoulli}(p).

    Parameters
    ----------
    p : beta-like node
        Probability of a successful trial

    Examples
    --------

    >>> import warnings
    >>> warnings.filterwarnings('ignore', category=RuntimeWarning)
    >>> from bayespy.nodes import Bernoulli, Beta
    >>> p = Beta([1e-3, 1e-3])
    >>> z = Bernoulli(p, plates=(10,))
    >>> z.observe([0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    >>> p.update()
    >>> import bayespy.plot as bpplt
    >>> import numpy as np
    >>> bpplt.pdf(p, np.linspace(0, 1, num=100))
    [<matplotlib.lines.Line2D object at 0x...>]
    """

    _moments = BernoulliMoments()
    _distribution = BernoulliDistribution()


    def __init__(self, p, **kwargs):
        """
        Create Bernoulli node.
        """
        super().__init__(p, **kwargs)


    @classmethod
    def _constructor(cls, p, **kwargs):
        """
        Constructs distribution and moments objects.
        """
        p = cls._ensure_moments(p, BetaMoments)
        parent_moments = (p._moments,)
        parents = [p]
        return ( parents,
                 kwargs,
                 ( (), ),
                 cls._total_plates(kwargs.get('plates'),
                                   cls._distribution.plates_from_parent(0, p.plates)),
                 cls._distribution,
                 cls._moments,
                 parent_moments)


    def __str__(self):
        """
        Print the distribution using standard parameterization.
        """
        p = 1 / (1 + np.exp(-self.phi[0]))
        return ("%s ~ Bernoulli(p)\n"
                "  p = \n"
                "%s\n"
                % (self.name, p))


from .deterministic import Deterministic
from .categorical import Categorical, CategoricalMoments

class CategoricalToBernoulli(Deterministic):
    """
    A node for converting 2-class categorical moments to Bernoulli moments.
    """

    
    def __init__(self, Z, **kwargs):
        """
        Create a categorical MC moments to categorical moments conversion node.
        """
        # Convert parent to proper type. Z must be a node.
        if not isinstance(Z._moments, CategoricalMoments):
            raise ValueError("Input node must be categorical")
        K = Z.dims[0][-1]
        if K != 2:
            raise Moments.NoConverterError("Only 2-class categorical can be converted to "
                                           "Bernoulli")
        dims = ( (), )
        self._moments = BernoulliMoments()
        self._parent_moments = (CategoricalMoments(2),)
        super().__init__(Z, dims=dims, **kwargs)

        
    def _compute_moments(self, u_Z):
        """
        Compute the moments given the moments of the parents.
        """
        u0 = u_Z[0][...,0]
        u = [u0]
        return u


    def _compute_message_to_parent(self, index, m, u_Z):
        """
        Compute the message to a parent.
        """
        if index == 0:
            m0 = np.concatenate([m[0][...,None],
                                 np.zeros(np.shape(m[0]))[...,None]],
                                axis=-1)
            return [m0]
        else:
            raise ValueError("Incorrect parent index")
    

# Make use of the conversion node
CategoricalMoments.add_converter(BernoulliMoments,
                                 CategoricalToBernoulli)
