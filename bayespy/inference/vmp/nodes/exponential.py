######################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################


"""
Module for the exponential distribution node.
"""

from .gamma import (GammaMoments,
                    Gamma)
from .expfamily import ExponentialFamily


ExponentialMoments = GammaMoments


class Exponential(Gamma):
    r"""
    Node for exponential random variables.

    .. warning::

        Use :class:`Gamma` instead of this. `Exponential(l)` is equivalent to
        `Gamma(1, l)`.

    Parameters
    ----------

    l : gamma-like node or scalar or array

        Rate parameter

    See also
    --------

    Gamma, Poisson

    Notes
    -----
    
    For simplicity, this is just a gamma node with the first parent fixed to
    one.  Note that this is a bit inconsistent with the BayesPy philosophy which
    states that the node does not only define the form of the prior distribution
    but more importantly the form of the posterior approximation.  Thus, one
    might expect that this node would have exponential posterior distribution
    approximation.  However, it has a gamma distribution.  Also, the moments are
    gamma moments although only E[x] would be the moment of a exponential random
    variable.  All this was done because: a) gamma was already implemented, so
    there was no need to implement anything, and b) people might easily use
    Exponential node as a prior definition and expect to get gamma posterior
    (which is what happens now).  Maybe some day a pure Exponential node is
    implemented and the users are advised to use Gamma(1,b) if they want to use
    an exponential prior distribution but gamma posterior approximation.
    """


    def __init__(self, l, **kwargs):
        raise NotImplementedError("Not yet implemented. Use Gamma(1, lambda)")
        super().__init__(1, l, **kwargs)


    @classmethod
    def _constructor(cls, l, **kwargs):
        raise NotImplementedError("Not yet implemented. Use Gamma(1, lambda)")
