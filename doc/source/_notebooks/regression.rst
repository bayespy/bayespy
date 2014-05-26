
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
    bpplt.plot(k*xh+c, x=xh, color='r')

.. parsed-literal::

    Iteration 1: loglike=-4.348575e+01 (0.010 seconds)
    Iteration 2: loglike=-4.288957e+01 (0.000 seconds)
    Iteration 3: loglike=-4.287939e+01 (0.000 seconds)
    Iteration 4: loglike=-4.287903e+01 (0.000 seconds)
    Iteration 5: loglike=-4.287902e+01 (0.010 seconds)
    Converged.



.. image:: regression_files/regression_2_1.png


.. code:: python

    bpplt.pdf(tau, np.linspace(0,1,100), color='k')
    bpplt.pyplot.axvline(s**(-2), color='r')

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/gamma.py:151: RuntimeWarning: divide by zero encountered in log
      logx = np.log(x)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/expfamily.py:368: RuntimeWarning: invalid value encountered in add
      return (self.g + f + Z)




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7f21f49435d0>




.. image:: regression_files/regression_3_2.png


.. code:: python

    bpplt.contour(B, np.linspace(1,3,100), np.linspace(1,9,100), n=10, colors='k')
    bpplt.plot(c, x=k, color='r', marker='x', linestyle='None', markersize=10, markeredgewidth=2)


.. image:: regression_files/regression_4_0.png


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

    Iteration 1: loglike=-4.419598e+01 (0.000 seconds)
    Iteration 2: loglike=-4.419598e+01 (0.000 seconds)
    Converged.


.. code:: python

    #import bayespy.plot as bpplt
    # These two lines are needed to enable inline plotting IPython Notebooks
    #%matplotlib inline
    #bpplt.plt.plot([])
    
    bpplt.plotmatrix(B_tau)

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/plot.py:747: RuntimeWarning: divide by zero encountered in log
      logx = np.log(x)
    /home/jluttine/workspace/bayespy/bayespy/utils/random.py:190: RuntimeWarning: invalid value encountered in subtract
      return a_logb - gammaln_a + a_logx - logx - bx




.. parsed-literal::

    array([[<matplotlib.axes.AxesSubplot object at 0x7f21f4a17b10>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f49e4310>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f4889890>],
           [<matplotlib.axes.AxesSubplot object at 0x7f21f4848250>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f4813710>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f47ccd90>],
           [<matplotlib.axes.AxesSubplot object at 0x7f21f479e190>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f47313d0>,
            <matplotlib.axes.AxesSubplot object at 0x7f21f46ad2d0>]], dtype=object)




.. image:: regression_files/regression_7_2.png


.. code:: python

    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)
    bpplt.timeseries(Fh, x=xh, scale=2)
    bpplt.plt.plot(x, y, 'rx')
    bpplt.plt.plot(xh, k*xh+c, 'r')



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f21f453ab50>]




.. image:: regression_files/regression_8_1.png


Multivariate regression
-----------------------

Non-linear regression
---------------------


