######################################################################
# Copyright (C) 2011,2012 Jaakko Luttinen
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

import numpy as np
import scipy.special as special

from .variable import Variable
from .constant import Constant



class Dirichlet(Variable):

    class ConjugatePrior:

        """ Conjugate prior for Dirichlet distribution. This isn't a
        distribution but can be used to compute the correct fixed
        moments, e.g., for Constant node."""
        

        # Number of trailing axes for variable dimensions in
        # observations. The other axes correspond to plates.
        ndim_observations = 1
        
        @staticmethod
        def compute_fixed_moments(alpha):
            """ Compute moments for fixed x. """
            gammaln_sum = special.gammaln(np.sum(alpha, axis=-1))
            sum_gammaln = np.sum(special.gammaln(alpha), axis=-1)
            z = gammaln_sum - sum_gammaln
            return [alpha, z]

        @staticmethod
        def compute_dims_from_values(alpha):
            """ Compute the dimensions of phi and u. """
            d = np.shape(alpha)[-1]
            return [(d,), ()]


    ndims = (1,)

    @staticmethod
    def compute_phi_from_parents(u_parents):
        return [u_parents[0][0]]
        #return [u_parents[0][0].copy()]

    @staticmethod
    def compute_g_from_parents(u_parents):
        return u_parents[0][1]

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        sum_gammaln = np.sum(special.gammaln(phi[0]), axis=-1)
        gammaln_sum = special.gammaln(np.sum(phi[0], axis=-1))
        psi_sum = special.psi(np.sum(phi[0], axis=-1, keepdims=True))
        
        # Moments <log x>
        u0 = special.psi(phi[0]) - psi_sum
        u = [u0]
        # G
        g = gammaln_sum - sum_gammaln

        return (u, g)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [u[0], 1]

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        # Has the same dimensionality as the parent for its first
        # moment.
        return parents[0].dims[:1]

    def __init__(self, alpha, plates=(), **kwargs):

        # Check for constant alpha
        if np.isscalar(alpha) or isinstance(alpha, np.ndarray):
            alpha = Constant(self.ConjugatePrior)(alpha)
            ## gammaln_sum = special.gammaln(np.sum(alpha, axis=-1))
            ## sum_gammaln = np.sum(special.gammaln(alpha), axis=-1)
            ## z = gammaln_sum - sum_gammaln
            ## d = np.shape(alpha)[-1]
            ## alpha = NodeConstant([alpha, z],
            ##                      plates=np.shape(alpha)[:-1],
            ##                      dims=((d,), ()))

        # Construct
        super().__init__(alpha,
                         plates=plates,
                         **kwargs)

    def random(self):
        raise NotImplementedError()

    def show(self):
        alpha = self.phi[0]
        print("%s ~ Dirichlet(alpha)" % self.name)
        print("  alpha = ")
        print(alpha)
