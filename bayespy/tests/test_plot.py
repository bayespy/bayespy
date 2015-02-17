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

from nose.tools import nottest

import numpy as np
from matplotlib.testing.decorators import image_comparison

import bayespy.plot as bpplt
from bayespy.nodes import Bernoulli, Beta, Categorical, Dirichlet, \
    Gamma, Gaussian, GaussianARD, Mixture, SumMultiply, Wishart
from bayespy.inference import VB
from bayespy.utils import random

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

    alpha = Dirichlet(1e-5*np.ones(K),
                      name='alpha')
    Z = Categorical(alpha,
                    plates=(N,),
                    name='z')

    mu = Gaussian(np.zeros(D), 1e-5*np.identity(D),
                  plates=(K,),
                  name='mu')
    Lambda = Wishart(D, 1e-5*np.identity(D),
                     plates=(K,),
                     name='Lambda')

    Y = Mixture(Z, Gaussian, mu, Lambda,
                name='Y')
    Z.initialize_from_random()

    Q = VB(Y, mu, Lambda, Z, alpha)
    Y.observe(y)
    Q.update(repeat=1000)

    bpplt.gaussian_mixture(Y, scale=2)
    bpplt.pyplot.show()


@image_comparison(baseline_images=['hinton_r'], extensions=['png'])
def test_hinton_plot_dirichlet():
    (R,P,Z) = _setup_bernoulli_mixture()
    bpplt.hinton(R)

@image_comparison(baseline_images=['hinton_p'], extensions=['png'])
def test_hinton_plot_beta():
    (R,P,Z) = _setup_bernoulli_mixture()
    bpplt.hinton(P)

@image_comparison(baseline_images=['hinton_z'], extensions=['png'])
def test_hinton_plot_categorical():
    (R,P,Z) = _setup_bernoulli_mixture()
    bpplt.hinton(Z)


@image_comparison(baseline_images=['pdf'], extensions=['png'])
def test_pdf_plot():
    data = _setup_linear_regression()
    bpplt.pdf(data['tau'], np.linspace(1e-6,1,100), color='k')
    bpplt.pyplot.axvline(data['s']**(-2), color='r')

@nottest
@image_comparison(baseline_images=['contour'], extensions=['png'])
def test_contour_plot():
    """
    Test the `contour` plotting function.

    This test fails in some environments, so it's disabled for now.
    see https://travis-ci.org/bayespy/bayespy/builds/50995827
    """
    data = _setup_linear_regression()
    bpplt.contour(data['B'], np.linspace(1,3,1000), np.linspace(1,9,1000),
                  n=10, colors='k')
    bpplt.plot(data['c'], x=data['k'], color='r', marker='x', linestyle='None',
               markersize=10, markeredgewidth=2)


def _setup_bernoulli_mixture():
    """
    Setup code for the hinton tests.

    This code is from http://www.bayespy.org/examples/bmm.html
    """
    np.random.seed(1)
    p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    p2 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    p = np.array([p0, p1, p2])

    z = random.categorical([1/3, 1/3, 1/3], size=100)
    x = random.bernoulli(p[z])
    N = 100
    D = 10
    K = 10

    R = Dirichlet(K*[1e-5],
                  name='R')
    Z = Categorical(R,
                    plates=(N,1),
                    name='Z')

    P = Beta([0.5, 0.5],
             plates=(D,K),
             name='P')

    X = Mixture(Z, Bernoulli, P)

    Q = VB(Z, R, X, P)
    P.initialize_from_random()
    X.observe(x)
    Q.update(repeat=1000)

    return (R,P,Z)

def _setup_linear_regression():
    """
    Setup code for the pdf and contour tests.

    This code is from http://www.bayespy.org/examples/regression.html
    """
    np.random.seed(1)
    k = 2 # slope
    c = 5 # bias
    s = 2 # noise standard deviation

    x = np.arange(10)
    y = k*x + c + s*np.random.randn(10)
    X = np.vstack([x, np.ones(len(x))]).T

    B = GaussianARD(0, 1e-6, shape=(2,))

    F = SumMultiply('i,i', B, X)

    tau = Gamma(1e-3, 1e-3)
    Y = GaussianARD(F, tau)
    Y.observe(y)

    Q = VB(Y, B, tau)
    Q.update(repeat=1000)
    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)

    return locals()
