
Bernoulli mixture model
=======================

blaa blaa blaa

.. code:: python

    import numpy as np
    D = 10
    p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    p1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    p = np.array([p0, p1, p2])
    from bayespy.utils import random
    N = 100
    z = random.categorical([1/3, 1/3, 1/3], size=N)
    x = random.bernoulli(p[z])
.. code:: python

    from bayespy.nodes import Categorical, Dirichlet
    K = 5
    R = Dirichlet(K*[1e-3],
                  name='R')
    Z = Categorical(R,
                    plates=(N,1),
                    name='Z')
    
    from bayespy.nodes import Mixture, Bernoulli, Beta
    P = Beta([1e-1, 1e-1],
             plates=(D,K),
             name='P')
    X = Mixture(Z, Bernoulli, P)
    
    X.observe(x)
    
    from bayespy.inference import VB
    Q = VB(Z, R, X, P)
    P.initialize_from_random()
    
    Q.update(repeat=10)

.. parsed-literal::

    Iteration 1: loglike=nan (0.005 seconds)
    Iteration 2: loglike=nan (0.003 seconds)
    Iteration 3: loglike=nan (0.003 seconds)
    Iteration 4: loglike=nan (0.003 seconds)
    Iteration 5: loglike=nan (0.003 seconds)
    Iteration 6: loglike=nan (0.003 seconds)
    Iteration 7: loglike=nan (0.003 seconds)
    Iteration 8: loglike=nan (0.003 seconds)
    Iteration 9: loglike=nan (0.003 seconds)
    Iteration 10: loglike=nan (0.003 seconds)


.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/dirichlet.py:91: RuntimeWarning: divide by zero encountered in log
      logp = np.log(p)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/expfamily.py:71: RuntimeWarning: invalid value encountered in multiply
      L = L + np.sum(phi_i * u_i, axis=axis_sum)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/expfamily.py:71: RuntimeWarning: invalid value encountered in add
      L = L + np.sum(phi_i * u_i, axis=axis_sum)
    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/mixture.py:229: UserWarning: The natural parameters of mixture distribution contain nans. This may happen if you use fixed parameters in your model. Technically, one possible reason is that the cluster assignment probability for some element is zero (p=0) and the natural parameter of that cluster is -inf, thus 0*(-inf)=nan. Solution: Use parameters that assign non-zero probabilities for the whole domain.
      warnings.warn("The natural parameters of mixture distribution "


.. code:: python

    import bayespy.plot.plotting as bpplt
    bpplt.beta_hinton(P)
    import matplotlib.pyplot as plt
    plt.show()

.. parsed-literal::

    /home/jluttine/workspace/bayespy/bayespy/plot/plotting.py:204: RuntimeWarning: invalid value encountered in absolute
      _w = np.abs(w)


