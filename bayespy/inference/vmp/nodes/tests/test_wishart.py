######################################################################
# Copyright (C) 2015 Jaakko Luttinen
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
Unit tests for `wishart` module.
"""

import numpy as np

from scipy import special

from .. import gaussian
from bayespy.nodes import (Gaussian, 
                           Wishart)

from ...vmp import VB

from bayespy.utils import misc
from bayespy.utils import linalg
from bayespy.utils import random

from bayespy.utils.misc import TestCase

def _student_logpdf(y, mu, Cov, nu):
    D = np.shape(y)[-1]
    return (special.gammaln((nu+D)/2)
            - special.gammaln(nu/2)
            - 0.5 * D * np.log(nu)
            - 0.5 * D * np.log(np.pi)
            - 0.5 * np.linalg.slogdet(Cov)[1]
            - 0.5 * (nu+D) * np.log(1+1/nu*np.einsum('...i,...ij,...j->...',
                                                     y-mu,
                                                     np.linalg.inv(Cov),
                                                     y-mu)))


class TestWishart(TestCase):

    def test_lower_bound(self):
        """
        Test the Wishart VB lower bound
        """

        #
        # By having the Wishart node as the only latent node, VB will give exact
        # results, thus the VB lower bound is the true marginal log likelihood.
        # Thus, check that they are equal. The true marginal likelihood is the
        # multivariate Student-t distribution.
        #

        np.random.seed(42)

        D = 3
        n = (D-1) + np.random.uniform(0.1, 0.5)
        V = random.covariance(D)
        Lambda = Wishart(n, V)
        mu = np.random.randn(D)
        Y = Gaussian(mu, Lambda)
        y = np.random.randn(D)
        Y.observe(y)
        Lambda.update()
        L = Y.lower_bound_contribution() + Lambda.lower_bound_contribution()
        mu = mu
        nu = n + 1 - D
        Cov = V / nu
        self.assertAllClose(L,
                            _student_logpdf(y,
                                            mu,
                                            Cov,
                                            nu))

        pass


    def test_moments(self):
        """
        Test the moments of Wishart node
        """

        np.random.seed(42)

        # Test prior moments
        D = 3
        n = (D-1) + np.random.uniform(0.1,2)
        V = random.covariance(D)
        Lambda = Wishart(n, V)
        Lambda.update()
        u = Lambda.get_moments()
        self.assertAllClose(u[0],
                            n*np.linalg.inv(V),
                            msg='Mean incorrect')
        self.assertAllClose(u[1],
                            (np.sum(special.digamma((n - np.arange(D))/2))
                             + D*np.log(2)
                             - np.linalg.slogdet(V)[1]),
                             msg='Log determinant incorrect')

        # Test posterior moments
        D = 3
        n = (D-1) + np.random.uniform(0.1,2)
        V = random.covariance(D)
        Lambda = Wishart(n, V)
        mu = np.random.randn(D)
        Y = Gaussian(mu, Lambda)
        y = np.random.randn(D)
        Y.observe(y)
        Lambda.update()
        u = Lambda.get_moments()
        n = n + 1
        V = V + np.outer(y-mu, y-mu) 
        self.assertAllClose(u[0],
                            n*np.linalg.inv(V),
                            msg='Mean incorrect')
        self.assertAllClose(u[1],
                            (np.sum(special.digamma((n - np.arange(D))/2))
                             + D*np.log(2)
                             - np.linalg.slogdet(V)[1]),
                             msg='Log determinant incorrect')

        

        pass
