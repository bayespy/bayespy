..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under Version 3.0 of the GNU General Public
   License. See LICENSE for a text of the license.

   This file is part of BayesPy.

   BayesPy is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 3 as
   published by the Free Software Foundation.

   BayesPy is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.


.. testsetup::

    # This is the PCA model from the previous sections
    import numpy as np
    np.random.seed(1)
    from bayespy.nodes import GaussianARD, Gamma, Dot
    D = 3
    X = GaussianARD(0, 1,
                    shape=(D,),
                    plates=(1,100),
                    name='X')
    alpha = Gamma(1e-3, 1e-3,
                  plates=(D,),
                  name='alpha')
    C = GaussianARD(0, alpha,
                    shape=(D,),
                    plates=(10,1),
                    name='C')
    F = Dot(C, X)
    tau = Gamma(1e-3, 1e-3, name='tau')
    Y = GaussianARD(F, tau)
    c = np.random.randn(10, 2)
    x = np.random.randn(2, 100)
    data = np.dot(c, x) + 0.1*np.random.randn(10, 100)
    Y.observe(data)
    Y.observe(data, mask=[[True], [False], [False], [True], [True],
                          [False], [True], [True], [True], [False]])
    from bayespy.inference import VB
    Q = VB(Y, C, X, alpha, tau)
    X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
    from bayespy.inference.vmp import transformations
    rotX = transformations.RotateGaussianARD(X)
    rotC = transformations.RotateGaussianARD(C, alpha)
    R = transformations.RotationOptimizer(rotC, rotX, D)
    Q = VB(Y, C, X, alpha, tau)
    Q.callback = R.rotate
    Q.update(repeat=1000, tol=1e-6, verbose=False)
    Q.update(repeat=50, tol=np.nan, verbose=False)

    from bayespy.nodes import Gaussian

Examining the results
=====================

After the results have been obtained, it is important to be able to examine the
results easily.  The results can be examined either numerically by inspecting
numerical arrays or visually by plotting distributions of the nodes.  In
addition, the posterior distributions can be visualized during the learning
algorithm and the results can saved into a file.


Plotting the results
--------------------

.. currentmodule:: bayespy

The module :mod:`plot` offers some plotting basic functionality:

>>> import bayespy.plot as bpplt

.. currentmodule:: bayespy.plot

The module contains ``matplotlib.pyplot`` module if the user needs that.  For
instance, interactive plotting can be enabled as:

>>> bpplt.pyplot.ion()

The :mod:`plot` module contains some functions but it is not a very
comprehensive collection, thus the user may need to write some problem- or
model-specific plotting functions.  The probability density function of a scalar
random variable can be plotted using the function :func:`pdf`:

>>> bpplt.pdf(Q['tau'], np.linspace(60, 140, num=100))
[<matplotlib.lines.Line2D object at 0x...>]

.. plot::

    # This is the PCA model from the previous sections
    import numpy as np
    np.random.seed(1)
    from bayespy.nodes import GaussianARD, Gamma, Dot
    D = 3
    X = GaussianARD(0, 1,
                    shape=(D,),
                    plates=(1,100),
                    name='X')
    alpha = Gamma(1e-3, 1e-3,
                  plates=(D,),
                  name='alpha')
    C = GaussianARD(0, alpha,
                    shape=(D,),
                    plates=(10,1),
                    name='C')
    F = Dot(C, X)
    tau = Gamma(1e-3, 1e-3, name='tau')
    Y = GaussianARD(F, tau)
    c = np.random.randn(10, 2)
    x = np.random.randn(2, 100)
    data = np.dot(c, x) + 0.1*np.random.randn(10, 100)
    Y.observe(data)
    Y.observe(data, mask=[[True], [False], [False], [True], [True],
                          [False], [True], [True], [True], [False]])
    from bayespy.inference import VB
    Q = VB(Y, C, X, alpha, tau)
    X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
    from bayespy.inference.vmp import transformations
    rotX = transformations.RotateGaussianARD(X)
    rotC = transformations.RotateGaussianARD(C, alpha)
    R = transformations.RotationOptimizer(rotC, rotX, D)
    Q = VB(Y, C, X, alpha, tau)
    Q.callback = R.rotate
    Q.update(repeat=1000, tol=1e-6)
    Q.update(repeat=50, tol=np.nan)

    import bayespy.plot as bpplt
    bpplt.pdf(Q['tau'], np.linspace(60, 140, num=100))

The variable ``tau`` models the inverse variance of the noise, for which the
true value is :math:`0.1^{-2}=100`.  Thus, the posterior captures the true value
quite accurately.  Similarly, the function :func:`contour` can be used to plot
the probability density function of a 2-dimensional variable, for instance:

>>> V = Gaussian([3, 5], [[4, 2], [2, 5]])
>>> bpplt.contour(V, np.linspace(1, 5, num=100), np.linspace(3, 7, num=100))
<matplotlib.contour.QuadContourSet object at 0x...>

.. plot::

    # This is the PCA model from the previous sections
    import numpy as np
    np.random.seed(1)
    from bayespy.nodes import GaussianARD, Gamma, Dot
    D = 3
    X = GaussianARD(0, 1,
                    shape=(D,),
                    plates=(1,100),
                    name='X')
    alpha = Gamma(1e-3, 1e-3,
                  plates=(D,),
                  name='alpha')
    C = GaussianARD(0, alpha,
                    shape=(D,),
                    plates=(10,1),
                    name='C')
    F = Dot(C, X)
    tau = Gamma(1e-3, 1e-3, name='tau')
    Y = GaussianARD(F, tau)
    c = np.random.randn(10, 2)
    x = np.random.randn(2, 100)
    data = np.dot(c, x) + 0.1*np.random.randn(10, 100)
    Y.observe(data)
    Y.observe(data, mask=[[True], [False], [False], [True], [True],
                          [False], [True], [True], [True], [False]])
    from bayespy.inference import VB
    Q = VB(Y, C, X, alpha, tau)
    X.initialize_from_parameters(np.random.randn(1, 100, D), 10)
    from bayespy.inference.vmp import transformations
    rotX = transformations.RotateGaussianARD(X)
    rotC = transformations.RotateGaussianARD(C, alpha)
    R = transformations.RotationOptimizer(rotC, rotX, D)
    Q = VB(Y, C, X, alpha, tau)
    Q.callback = R.rotate
    Q.update(repeat=1000, tol=1e-6)
    Q.update(repeat=50, tol=np.nan)

    import bayespy.plot as bpplt
    from bayespy.nodes import Gaussian
    V = Gaussian([3, 5], [[4, 2], [2, 5]])
    bpplt.contour(V, np.linspace(1, 5, num=100), np.linspace(3, 7, num=100))

Both :func:`pdf` and :func:`contour` require that the user provides the grid on
which the probability density function is computed.  They also support several
keyword arguments for modifying the output, similarly as ``plot`` and
``contour`` in ``matplotlib.pyplot``.

Monitoring during the inference algorithm
-----------------------------------------



Posterior parameters and moments
--------------------------------

blah blah


Saving and loading results
--------------------------

blah blah
