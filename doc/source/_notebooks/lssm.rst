
Linear state-space model
========================

This example is also available as `an IPython notebook <lssm.ipynb>`_ or
`a Python script <lssm.py>`_.

In linear state-space models a sequence of :math:`M`\ -dimensional
observations :math:`\mathbf{Y}=(\mathbf{y}_1,\ldots,\mathbf{y}_N)` is
assumed to be generated from latent :math:`D`\ -dimensional states
:math:`\mathbf{X}=(\mathbf{x}_1,\ldots,\mathbf{x}_N)` which follow a
first-order Markov process:

.. math::


   \mathbf{x}_{n} &= \mathbf{A}\mathbf{x}_{n-1} + \text{noise} \,,
   \\
   \mathbf{y}_{n} &= \mathbf{C}\mathbf{x}_{n} + \text{noise} \,,

where the noise is Gaussian, :math:`\mathbf{A}` is the :math:`D\times D`
state dynamics matrix and :math:`\mathbf{C}` is the :math:`M\times D`
loading matrix. Usually, the latent space dimensionality :math:`D` is
assumed to be much smaller than the observation space dimensionality
:math:`M` in order to model the dependencies of high-dimensional
observations efficiently.

First, let us generate some toy data:

.. code:: python

    import numpy as np
    
    M = 30
    N = 400
    
    w = 0.3
    a = np.array([[np.cos(w), -np.sin(w), 0, 0], 
                  [np.sin(w), np.cos(w),  0, 0], 
                  [0,         0,          1, 0],
                  [0,         0,          0, 0]])
    c = np.random.randn(M,4)
    x = np.empty((N,4))
    f = np.empty((M,N))
    y = np.empty((M,N))
    x[0] = 10*np.random.randn(4)
    f[:,0] = np.dot(c,x[0])
    y[:,0] = f[:,0] + 3*np.random.randn(M)
    for n in range(N-1):
        x[n+1] = np.dot(a,x[n]) + np.random.randn(4)
        f[:,n+1] = np.dot(c,x[n+1])
        y[:,n+1] = f[:,n+1] + 3*np.random.randn(M)
        
The linear state-space model can be constructed as follows:

.. code:: python

    from bayespy.inference.vmp.nodes.gaussian_markov_chain import GaussianMarkovChain
    from bayespy.inference.vmp.nodes.gaussian import GaussianARD
    from bayespy.inference.vmp.nodes.gamma import Gamma
    from bayespy.inference.vmp.nodes.dot import SumMultiply
    
    D = 10
    
    # Dynamics matrix with ARD
    alpha = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='alpha')
    A = GaussianARD(0,
                    alpha,
                    shape=(D,),
                    plates=(D,),
                    name='A')
    
    # Latent states with dynamics
    X = GaussianMarkovChain(np.zeros(D),         # mean of x0
                            1e-3*np.identity(D), # prec of x0
                            A,                   # dynamics
                            np.ones(D),          # innovation
                            n=N,                 # time instances
                            name='X',
                            initialize=False)
    X.initialize_from_value(np.zeros((N,D))) # just some empty values, X is
                                             # updated first anyway
    
    # Mixing matrix from latent space to observation space using ARD
    gamma = Gamma(1e-5,
                  1e-5,
                  plates=(D,),
                  name='gamma')
    C = GaussianARD(0,
                    gamma,
                    shape=(D,),
                    plates=(M,1),
                    name='C')
    # Initialize nodes (must use some randomness for C, and update X before C)
    C.initialize_from_random()
    
    # Observation noise
    tau = Gamma(1e-5,
                1e-5,
                name='tau')
    
    # Observations
    F = SumMultiply('i,i',
                    C, 
                    X,
                    name='F')
    Y = GaussianARD(F,
                    tau,
                    name='Y')
An inference machine using variational Bayesian inference with
variational message passing is then construced as

.. code:: python

    from bayespy.inference.vmp.vmp import VB
    Q = VB(X, C, gamma, A, alpha, tau, Y)
Observe the data partially (80% is marked missing):

.. code:: python

    from bayespy.utils import random
    
    # Add missing values randomly (keep only 20%)
    mask = random.mask(M, N, p=0.2)
    Y.observe(y, mask=mask)
Then inference (100 iterations) can be run simply as

.. code:: python

    Q.update(repeat=10)

.. parsed-literal::

    Iteration 1: loglike=-3.118644e+04 (0.210 seconds)
    Iteration 2: loglike=-1.129540e+04 (0.210 seconds)
    Iteration 3: loglike=-9.139376e+03 (0.210 seconds)
    Iteration 4: loglike=-8.704676e+03 (0.220 seconds)
    Iteration 5: loglike=-8.531889e+03 (0.200 seconds)
    Iteration 6: loglike=-8.386198e+03 (0.210 seconds)
    Iteration 7: loglike=-8.255826e+03 (0.210 seconds)
    Iteration 8: loglike=-8.176274e+03 (0.210 seconds)
    Iteration 9: loglike=-8.139579e+03 (0.210 seconds)
    Iteration 10: loglike=-8.117779e+03 (0.210 seconds)


Speeding up with parameter expansion
------------------------------------

VB inference can converge extremely slowly if the variables are strongly
coupled. Because VMP updates one variable at a time, it may lead to slow
zigzagging. This can be solved by using parameter expansion which
reduces the coupling. In state-space models, the states
:math:`\mathbf{x}_n` and the loadings :math:`\mathbf{C}` are coupled
through a dot product :math:`\mathbf{Cx}_n`\ , which is unaltered if the
latent space is rotated arbitrarily:

.. math::


   \mathbf{y}_n &= \mathbf{C}\mathbf{x}_n = \mathbf{C}\mathbf{R}^{-1}\mathbf{R}\mathbf{x}_n \,.

Thus, one intuitive transformation would be
:math:`\mathbf{C}\rightarrow\mathbf{C}\mathbf{R}^{-1}` and
:math:`\mathbf{X}\rightarrow\mathbf{R}\mathbf{X}`\ . In order to keep
the dynamics of the latent states unaffected by the transformation, the
state dynamics matrix :math:`\mathbf{A}` must be transformed
accordingly:

.. math::


   \mathbf{R}\mathbf{x}_n &= \mathbf{R}\mathbf{A}\mathbf{R}^{-1} \mathbf{R}\mathbf{x}_{n-1} \,,

resulting in a transformation
:math:`\mathbf{A}\rightarrow\mathbf{R}\mathbf{A}\mathbf{R}^{-1}`\ . For
more details, refer to \*Fast Variational Bayesian Linear State-Space
Model (Luttinen, 2013).

In BayesPy, the transformations can be used as follows:

.. code:: python

    # Import the parameter expansion module
    from bayespy.inference.vmp import transformations
    
    # Rotator of the state dynamics matrix
    rotA = transformations.RotateGaussianARD(Q['A'], Q['alpha'])
    # Rotator of the states (includes rotation of the state dynamics matrix)
    rotX = transformations.RotateGaussianMarkovChain(Q['X'], rotA)
    # Rotator of the loading matrix
    rotC = transformations.RotateGaussianARD(Q['C'], Q['gamma'])
    # Rotation optimizer
    R = transformations.RotationOptimizer(rotX, rotC, D)
Note that it is crucial to select the correct rotation class which
corresponds to the particular model block exactly. The rotation can be
performed after each full VB update:

.. code:: python

    for ind in range(10):
        Q.update()
        R.rotate()

.. parsed-literal::

    Iteration 11: loglike=-8.100983e+03 (0.210 seconds)
    Iteration 12: loglike=-7.622913e+03 (0.210 seconds)
    Iteration 13: loglike=-7.452057e+03 (0.200 seconds)
    Iteration 14: loglike=-7.385975e+03 (0.200 seconds)
    Iteration 15: loglike=-7.351449e+03 (0.210 seconds)
    Iteration 16: loglike=-7.331026e+03 (0.210 seconds)
    Iteration 17: loglike=-7.317997e+03 (0.200 seconds)
    Iteration 18: loglike=-7.309212e+03 (0.200 seconds)
    Iteration 19: loglike=-7.303074e+03 (0.210 seconds)
    Iteration 20: loglike=-7.298661e+03 (0.210 seconds)


If you want to implement your own rotations or check the existing ones,
you may use debugging utilities:

.. code:: python

    for ind in range(10):
        Q.update()
        R.rotate(check_bound=True,
                 check_gradient=True)

.. parsed-literal::

    Iteration 21: loglike=-7.295401e+03 (0.210 seconds)
    Norm of numerical gradient: 3905.05
    Norm of function gradient:  3905.05
    Gradient relative error = 6.39002e-05 and absolute error = 0.249533
    Iteration 22: loglike=-7.292861e+03 (0.210 seconds)
    Norm of numerical gradient: 6245.37

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 6.39002e-05
      warnings.warn("Rotation gradient has relative error %g" % err)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 7.56396e-05
      warnings.warn("Rotation gradient has relative error %g" % err)


.. parsed-literal::

    
    Norm of function gradient:  6245.43
    Gradient relative error = 7.56396e-05 and absolute error = 0.472397
    Iteration 23: loglike=-7.290841e+03 (0.210 seconds)
    Norm of numerical gradient: 3984.43
    Norm of function gradient:  3984.43
    Gradient relative error = 6.78117e-05 and absolute error = 0.270191
    Iteration 24: loglike=-7.289243e+03 (0.210 seconds)
    Norm of numerical gradient: 13053.7

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 6.78117e-05
      warnings.warn("Rotation gradient has relative error %g" % err)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 2.65118e-05
      warnings.warn("Rotation gradient has relative error %g" % err)


.. parsed-literal::

    
    Norm of function gradient:  13053.8
    Gradient relative error = 2.65118e-05 and absolute error = 0.346078
    Iteration 25: loglike=-7.287794e+03 (0.200 seconds)
    Norm of numerical gradient: 4144.61
    Norm of function gradient:  4144.59
    Gradient relative error = 7.02612e-05 and absolute error = 0.291205
    Iteration 26: loglike=-7.286531e+03 (0.210 seconds)
    Norm of numerical gradient: 5821.72

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 7.02612e-05
      warnings.warn("Rotation gradient has relative error %g" % err)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 4.57892e-05
      warnings.warn("Rotation gradient has relative error %g" % err)


.. parsed-literal::

    
    Norm of function gradient:  5821.73
    Gradient relative error = 4.57892e-05 and absolute error = 0.266572
    Iteration 27: loglike=-7.285469e+03 (0.210 seconds)
    Norm of numerical gradient: 15766.4
    Norm of function gradient:  15766.4
    Gradient relative error = 3.5184e-05 and absolute error = 0.554724
    Iteration 28: loglike=-7.284584e+03 (0.200 seconds)
    Norm of numerical gradient: 5782.51

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 3.5184e-05
      warnings.warn("Rotation gradient has relative error %g" % err)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 5.61705e-05
      warnings.warn("Rotation gradient has relative error %g" % err)


.. parsed-literal::

    
    Norm of function gradient:  5782.51
    Gradient relative error = 5.61705e-05 and absolute error = 0.324807
    Iteration 29: loglike=-7.283818e+03 (0.210 seconds)
    Norm of numerical gradient: 9067.22
    Norm of function gradient:  9067.21
    Gradient relative error = 2.4973e-05 and absolute error = 0.226435
    Iteration 30: loglike=-7.283121e+03 (0.200 seconds)
    Norm of numerical gradient: 9594.54

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 2.4973e-05
      warnings.warn("Rotation gradient has relative error %g" % err)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/transformations.py:142: UserWarning: Rotation gradient has relative error 5.43175e-05
      warnings.warn("Rotation gradient has relative error %g" % err)


.. parsed-literal::

    
    Norm of function gradient:  9594.62
    Gradient relative error = 5.43175e-05 and absolute error = 0.521151

