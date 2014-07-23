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

   import numpy
   numpy.random.seed(1)


Linear regression
=================

Data
----

The true parameters of the linear regression:

>>> k = 2 # slope
>>> c = 5 # bias
>>> s = 2 # noise standard deviation

Generate data:

>>> import numpy as np
>>> x = np.arange(10)
>>> y = k*x + c + s*np.random.randn(10)
    
Model
-----

The regressors, that is, the input data:

>>> X = np.vstack([x, np.ones(len(x))]).T

Note that we added a column of ones to the regressor matrix for the bias term.
We model the slope and the bias term in the same node so we do not factorize
between them:

>>> from bayespy.nodes import GaussianARD
>>> B = GaussianARD(0, 1e-6, shape=(2,))

The first element is the slope which multiplies ``x`` and the second element is
the bias term which multiplies the constant ones.  Now we compute the dot
product of ``X`` and ``B``:

>>> from bayespy.nodes import SumMultiply
>>> F = SumMultiply('i,i', B, X)

The noise parameter:

>>> from bayespy.nodes import Gamma
>>> tau = Gamma(1e-3, 1e-3)

The noisy observations:

>>> Y = GaussianARD(F, tau)

Inference
---------

Observe the data:

>>> Y.observe(y)

Construct the variational Bayesian (VB) inference engine by giving all
stochastic nodes:
    
>>> from bayespy.inference import VB
>>> Q = VB(Y, B, tau)

Iterate until convergence:

>>> Q.update(repeat=1000)
Iteration 1: loglike=-4.595948e+01 (... seconds)
...
Iteration 5: loglike=-4.495017e+01 (... seconds)
Converged at iteration 5.


Results
-------

Create a simple predictive model for new inputs:
    
>>> xh = np.linspace(-5, 15, 100)
>>> Xh = np.vstack([xh, np.ones(len(xh))]).T
>>> Fh = SumMultiply('i,i', B, Xh)

Note that we use the learned node ``B`` but create a new regressor array for
predictions.  Plot the predictive distribution of noiseless function values:

>>> import bayespy.plot as bpplt
>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.plot(Fh, x=xh, scale=2)
>>> bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
>>> bpplt.plot(k*xh+c, x=xh, color='r');

.. plot::

   import numpy
   numpy.random.seed(1)
   k = 2 # slope
   c = 5 # bias
   s = 2 # noise standard deviation
   import numpy as np
   x = np.arange(10)
   y = k*x + c + s*np.random.randn(10)
   X = np.vstack([x, np.ones(len(x))]).T
   from bayespy.nodes import GaussianARD
   B = GaussianARD(0, 1e-6, shape=(2,))
   from bayespy.nodes import SumMultiply
   F = SumMultiply('i,i', B, X)
   from bayespy.nodes import Gamma
   tau = Gamma(1e-3, 1e-3)
   Y = GaussianARD(F, tau)
   Y.observe(y)
   from bayespy.inference import VB
   Q = VB(Y, B, tau)
   Q.update(repeat=1000)
   xh = np.linspace(-5, 15, 100)
   Xh = np.vstack([xh, np.ones(len(xh))]).T
   Fh = SumMultiply('i,i', B, Xh)
   import bayespy.plot as bpplt
   bpplt.plot(Fh, x=xh, scale=2)
   bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
   bpplt.plot(k*xh+c, x=xh, color='r');
   bpplt.pyplot.show()

Note that the above plot shows two standard deviation of the posterior of the
noiseless function, thus the data points may lie well outside this range.  The
red line shows the true linear function.  Next, plot the distribution of the
noise parameter and the true value, :math:`2^{-2}=0.25`:

>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.pdf(tau, np.linspace(1e-6,1,100), color='k')
[<matplotlib.lines.Line2D object at 0x...>]
>>> bpplt.pyplot.axvline(s**(-2), color='r')
<matplotlib.lines.Line2D object at 0x...>

.. plot::

   import numpy
   numpy.random.seed(1)
   k = 2 # slope
   c = 5 # bias
   s = 2 # noise standard deviation
   import numpy as np
   x = np.arange(10)
   y = k*x + c + s*np.random.randn(10)
   X = np.vstack([x, np.ones(len(x))]).T
   from bayespy.nodes import GaussianARD
   B = GaussianARD(0, 1e-6, shape=(2,))
   from bayespy.nodes import SumMultiply
   F = SumMultiply('i,i', B, X)
   from bayespy.nodes import Gamma
   tau = Gamma(1e-3, 1e-3)
   Y = GaussianARD(F, tau)
   Y.observe(y)
   from bayespy.inference import VB
   Q = VB(Y, B, tau)
   Q.update(repeat=1000)
   xh = np.linspace(-5, 15, 100)
   Xh = np.vstack([xh, np.ones(len(xh))]).T
   Fh = SumMultiply('i,i', B, Xh)
   import bayespy.plot as bpplt
   bpplt.pdf(tau, np.linspace(1e-6,1,100), color='k')
   bpplt.pyplot.axvline(s**(-2), color='r')
   bpplt.pyplot.show()

The noise level is captured quite well, although the posterior has more mass on
larger noise levels (smaller precision parameter values).  Finally, plot the
distribution of the regression parameters and mark the true value:

>>> bpplt.pyplot.figure()
<matplotlib.figure.Figure object at 0x...>
>>> bpplt.contour(B, np.linspace(1,3,1000), np.linspace(1,9,1000), 
...               n=10, colors='k')
<matplotlib.contour.QuadContourSet object at 0x...>
>>> bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', 
...            markersize=10, markeredgewidth=2)
>>> bpplt.pyplot.xlabel(r'$k$')
<matplotlib.text.Text object at 0x...>
>>> bpplt.pyplot.ylabel(r'$c$');
<matplotlib.text.Text object at 0x...>

.. plot::

   import numpy
   numpy.random.seed(1)
   k = 2 # slope
   c = 5 # bias
   s = 2 # noise standard deviation
   import numpy as np
   x = np.arange(10)
   y = k*x + c + s*np.random.randn(10)
   X = np.vstack([x, np.ones(len(x))]).T
   from bayespy.nodes import GaussianARD
   B = GaussianARD(0, 1e-6, shape=(2,))
   from bayespy.nodes import SumMultiply
   F = SumMultiply('i,i', B, X)
   from bayespy.nodes import Gamma
   tau = Gamma(1e-3, 1e-3)
   Y = GaussianARD(F, tau)
   Y.observe(y)
   from bayespy.inference import VB
   Q = VB(Y, B, tau)
   Q.update(repeat=1000)
   xh = np.linspace(-5, 15, 100)
   Xh = np.vstack([xh, np.ones(len(xh))]).T
   Fh = SumMultiply('i,i', B, Xh)
   import bayespy.plot as bpplt
   bpplt.contour(B, np.linspace(1,3,1000), np.linspace(1,9,1000), 
                 n=10, colors='k')
   bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', 
              markersize=10, markeredgewidth=2)
   bpplt.pyplot.xlabel(r'$k$')
   bpplt.pyplot.ylabel(r'$c$');
   bpplt.pyplot.show()

In this case, the true parameters are captured well by the posterior
distribution.

Improving accuracy
------------------

.. currentmodule:: bayespy.nodes

The model can be improved by not factorizing between ``B`` and ``tau`` but
learning their joint posterior distribution.  This requires a slight
modification to the model by using :class:`GaussianGammaISO` node:

>>> from bayespy.nodes import GaussianGammaISO
>>> B_tau = GaussianGammaISO(np.zeros(2), 1e-6*np.identity(2), 1e-3, 1e-3)

This node contains both the regression parameter vector and the noise parameter.
We compute the dot product similarly as before:
    
>>> F_tau = SumMultiply('i,i', B_tau, X)

However, ``Y`` is constructed as follows:

>>> Y = GaussianARD(F_tau, 1)

Because the noise parameter is already in ``F_tau`` we can give a constant one
as the second argument.  The total noise parameter for ``Y`` is the product of
the noise parameter in ``F_tau`` and one.  Now, inference is run similarly as
before:

>>> Y.observe(y)
>>> Q = VB(Y, B_tau)
>>> Q.update(repeat=1000)
Iteration 1: loglike=-4.678478e+01 (... seconds)
Iteration 2: loglike=-4.678478e+01 (... seconds)
Converged at iteration 2.

Note that the method converges immediately.  This happens because there is only
one unobserved stochastic node so there is no need for iteration and the result
is actually the exact true posterior distribution, not an approximation.
Currently, the main drawback of using this approach is that BayesPy does not yet
contain any plotting utilities for nodes that contain both Gaussian and gamma
variables jointly.

Further extensions
------------------

The approach discussed in this example can easily be extended to non-linear
regression and multivariate regression.  For non-linear regression, the inputs
are first transformed by some known non-linear functions and then linear
regression is applied to this transformed data.  For multivariate regression,
``X`` and ``B`` are concatenated appropriately: If there are more regressors,
add more columns to both ``X`` and ``B``.  If there are more output dimensions,
add plates to ``B``.



