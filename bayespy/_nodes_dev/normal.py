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

from .variable import Variable
from .constant import Constant
from .gamma import Gamma

class Normal(Variable):

    ndims = (0, 0)
    ndims_parents = [(0, 0), (0, 0)]
    # Observations are scalars (0-D):
    ndim_observations = 0

    @staticmethod
    def compute_fixed_moments(x):
        """ Compute moments u(x) for given x. """
        return [x, x**2]

    @staticmethod
    def compute_phi_from_parents(u_parents):
        phi = [u_parents[1][0] * u_parents[0][0],
               -u_parents[1][0] / 2]
        return phi

    @staticmethod
    def compute_g_from_parents(u_parents):
        mu = u_parents[0][0]
        mumu = u_parents[0][1]
        tau = u_parents[1][0]
        log_tau = u_parents[1][1]
        g = -0.5 * mumu*tau + 0.5 * log_tau
        return g

    @staticmethod
    def compute_u_and_g(phi, mask=True):
        u0 = -phi[0] / (2*phi[1])
        u1 = u0**2 - 1 / (2*phi[1])
        u = [u0, u1]
        g = (-0.5 * u[0] * phi[0] + 0.5 * np.log(-2*phi[1]))
        return (u, g)

    @staticmethod
    def compute_fixed_u_and_f(x):
        """ Compute u(x) and f(x) for given x. """
        u = [x, x**2]
        f = -np.log(2*np.pi)/2
        return (u, f)

    @staticmethod
    def compute_message(index, u, u_parents):
        """ . """
        if index == 0:
            return [u_parents[1][0] * u[0],
                    -0.5 * u_parents[1][0]]
        elif index == 1:
            return [-0.5 * (u[1] - 2*u[0]*u_parents[0][0] + u_parents[0][1]),
                    0.5]
        raise NotImplementedError()

    @staticmethod
    def compute_dims(*parents):
        """ Compute the dimensions of phi/u. """
        # Both moments are scalars, thus, shapes are ()
        return [(), ()]

    @staticmethod
    def compute_dims_from_values(x):
        """ Compute the dimensions of phi and u. """
        return [(), ()]

    # Normal(mu, 1/tau)

    def __init__(self, mu, tau, plates=(), **kwargs):

        # Check for constant mu
        if np.isscalar(mu) or isinstance(mu, np.ndarray):
            mu = Constant(Normal)(mu)

        # Check for constant tau
        if np.isscalar(tau) or isinstance(tau, np.ndarray):
            tau = Constant(Gamma)(tau)

        # Construct
        super().__init__(mu, tau, plates=plates, **kwargs)


    def show(self, parameters=True, mean=True, mode=True, median=True):
        mu = self.u[0]
        tau = self.phi[1]
        #s2 = self.u[1] - mu**2
        print("%s ~ Normal(mu, tau)" % self.name)
        print("  mu =", mu)
        #print(mu)
        print("  tau =", tau)
        #print(tau)
        #print("Normal(" + str(mu) + ", " + str(s2) + ")")

    def plot(self):
        pass
