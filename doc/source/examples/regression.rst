
Regression
==========

Linear regression
-----------------

.. code:: python

    import numpy as np
    k = 2
    c = 5
    s = 2
    
    x = np.arange(10)
    y = k*x + c + s*np.random.randn(10)
    
    from bayespy.nodes import GaussianARD
    B = GaussianARD(0, 1e-6, shape=(2,))
    
    X = np.vstack([x, np.ones(len(x))]).T
    
    from bayespy.nodes import SumMultiply
    F = SumMultiply('i,i', B, X)
    
    from bayespy.nodes import Gamma
    tau = Gamma(1e-3, 1e-3)
    
    Y = GaussianARD(F, tau)
    Y.observe(y)
    
    from bayespy.inference import VB
    Q = VB(Y, B, tau)
    
    Q.update(repeat=100)
    
    import bayespy.plot as bpplt
    # These two lines are needed to enable inline plotting IPython Notebooks
    %matplotlib inline
    bpplt.pyplot.plot([])
    
    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)
    bpplt.plot(Fh, x=xh, scale=2)
    bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
    bpplt.plot(k*xh+c, x=xh, color='r');

.. parsed-literal::

    Iteration 1: loglike=-4.515537e+01 (0.000 seconds)
    Iteration 2: loglike=-4.429472e+01 (0.010 seconds)
    Iteration 3: loglike=-4.428241e+01 (0.000 seconds)
    Iteration 4: loglike=-4.428197e+01 (0.000 seconds)
    Iteration 5: loglike=-4.428195e+01 (0.010 seconds)
    Converged.





.. code:: python

    bpplt.pdf(tau, np.linspace(1e-6,1,100), color='k')
    bpplt.pyplot.axvline(s**(-2), color='r')
    # Add labels
    bpplt.pyplot.title(r'$q(\tau)$')
    bpplt.pyplot.xlabel(r'$\tau$');




.. code:: python

    bpplt.contour(B, np.linspace(1,3,1000), np.linspace(1,9,1000), n=10, colors='k')
    bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2)
    # Add labels
    bpplt.pyplot.title(r'$q(k,c)$')
    bpplt.pyplot.xlabel(r'$k$')
    bpplt.pyplot.ylabel(r'$c$');




Improving accuracy
------------------

.. code:: python

    from bayespy.nodes import GaussianGammaISO
    B_tau = GaussianGammaISO(np.zeros(2), 1e-6*np.identity(2), 1e-3, 1e-3)
    
    F_tau = SumMultiply('i,i', B_tau, X)
    
    Y = GaussianARD(F_tau, 1)
    Y.observe(y)
    
    from bayespy.inference import VB
    Q = VB(Y, B_tau)
    
    Q.update(repeat=10)

.. parsed-literal::

    Iteration 1: loglike=-4.594957e+01 (0.000 seconds)
    Iteration 2: loglike=-4.594957e+01 (0.000 seconds)
    Converged.


.. code:: python

    bpplt.pdf(B_tau.get_marginal_logpdf(gaussian=None, gamma=True),
              np.linspace(1e-6,1,100), color='k')
    bpplt.pyplot.axvline(s**(-2), color='r')
    # Add labels
    bpplt.pyplot.title(r'$q(\tau)$')
    bpplt.pyplot.xlabel(r'$\tau$');




.. code:: python

    bpplt.contour(B_tau.get_marginal_logpdf(gaussian=[0,1], gamma=False),
                  np.linspace(1,3,100), np.linspace(1,9,100),
                  n=10, colors='k')
    # Plot the true value
    bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2)
    # Add labels
    bpplt.pyplot.title(r'$q(k,c)$')
    bpplt.pyplot.xlabel(r'$k$')
    bpplt.pyplot.ylabel(r'$c$');




.. code:: python

    bpplt.contour(B_tau.get_marginal_logpdf(gaussian=[0], gamma=True),
                  np.linspace(1,3,100), np.linspace(1e-6,1,100),
                  n=10, colors='k')
    bpplt.plot(s**(-2), x=k, color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2)
    bpplt.pyplot.title(r'$q(k,\tau)$')
    bpplt.pyplot.xlabel(r'$k$')
    bpplt.pyplot.ylabel(r'$\tau$');




.. code:: python

    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh_tau = SumMultiply('i,i', B_tau, Xh)
    bpplt.plot(Fh_tau, x=xh, scale=2)
    bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
    bpplt.plot(k*xh+c, x=xh, color='r')

::


    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)

    <ipython-input-8-bad1c68bbf3d> in <module>()
          2 Xh = np.vstack([xh, np.ones(len(xh))]).T
          3 Fh_tau = SumMultiply('i,i', B_tau, Xh)
    ----> 4 bpplt.plot(Fh_tau, x=xh, scale=2)
          5 bpplt.plot(y, x=x, color='r', marker='x', linestyle='None')
          6 bpplt.plot(k*xh+c, x=xh, color='r')


    /home/jluttine/workspace/bayespy/bayespy/plot.py in plot(Y, axis, scale, center, **kwargs)
        125             return plot_gaussian(Y, axis=axis, scale=scale, center=center, **kwargs)
        126 
    --> 127     (mu, var) = Y.get_mean_and_variance()
        128     std = np.sqrt(var)
        129 


    AttributeError: 'SumMultiply' object has no attribute 'get_mean_and_variance'


Multivariate regression
-----------------------

Non-linear regression
---------------------


