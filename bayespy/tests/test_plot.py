######################################################################
# Copyright (C) 2015 Hannu Hartikainen, Jaakko Luttinen
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
Tests for the module bayespy.plot.

This file mostly contains functional tests. Since testing the plotting
capabilities relies on image comparisons, it's difficult to create
strict unit tests.
"""

import numpy as np
from matplotlib.testing.decorators import image_comparison

import bayespy.plot as bpplt

@image_comparison(baseline_images=['gaussian_mixture'], extensions=['png'])
def test_gaussian_mixture_plot():
    """
    Test the gaussian_mixture plotting function.

    The code is from http://www.bayespy.org/examples/gmm.html
    """
    np.random.seed(1)
    y0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 0.02]], size=50)
    y1 = np.random.multivariate_normal([0, 0], [[0.02, 0], [0, 1]], size=50)
    y2 = np.random.multivariate_normal([2, 2], [[1, -0.9], [-0.9, 1]], size=50)
    y3 = np.random.multivariate_normal([-2, -2], [[0.1, 0], [0, 0.1]], size=50)
    y = np.vstack([y0, y1, y2, y3])

    bpplt.pyplot.plot(y[:,0], y[:,1], 'rx')

    N = 200
    D = 2
    K = 10

    from bayespy.nodes import Dirichlet, Categorical
    alpha = Dirichlet(1e-5*np.ones(K),
                      name='alpha')
    Z = Categorical(alpha,
                    plates=(N,),
                    name='z')

    from bayespy.nodes import Gaussian, Wishart
    mu = Gaussian(np.zeros(D), 1e-5*np.identity(D),
                  plates=(K,),
                  name='mu')
    Lambda = Wishart(D, 1e-5*np.identity(D),
                     plates=(K,),
                     name='Lambda')

    from bayespy.nodes import Mixture
    Y = Mixture(Z, Gaussian, mu, Lambda,
                name='Y')
    Z.initialize_from_random()

    from bayespy.inference import VB
    Q = VB(Y, mu, Lambda, Z, alpha)
    Y.observe(y)
    Q.update(repeat=1000)

    bpplt.gaussian_mixture(Y, scale=2)
    bpplt.pyplot.show()
