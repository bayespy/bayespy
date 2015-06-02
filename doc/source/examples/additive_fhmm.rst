..
   Copyright (C) 2014 Jaakko Luttinen

   This file is licensed under the MIT License. See LICENSE for a text of the
   license.


Additive factorial hidden Markov model
======================================

.. code:: python

    import numpy as np
    from bayespy.nodes import (Dirichlet, 
                                CategoricalMarkovChain,
                                GaussianARD,
                                Gate,
                                SumMultiply,
                                Gamma)
    
    D = 4   # dimensionality of the data vectors (and mu vectors)
    N = 5   # number of chains
    K = 3   # number of states in each chain
    T = 100 # length of each chain
    
    # Markov chain parameters.
    # Use known values
    p0 = np.ones(K) / K
    P = np.ones((K,K)) / K
    # Or set Dirichlet prior.
    p0 = Dirichlet(np.ones(K), plates=(N,))
    P = Dirichlet(np.ones(K), plates=(N,1,K))
    
    # N Markov chains with K possible states, and length T
    X = CategoricalMarkovChain(p0, P, states=T, plates=(N,))
    
    # For each of the N chains, have K different D-dimensional mu's
    # Unknown mu's
    mu = GaussianARD(0, 1e-3, plates=(D,1,1,K), shape=(N,))
    
    # Gate/select mu's
    print(mu.plates, mu.dims[0], X.plates, X.dims[1])
    Z = Gate(X, mu)
    
    # Sum the mu's of different chains
    F = SumMultiply('i->', Z)
    print(mu.plates, mu.dims[0], X.plates, X.dims[0], Z.plates, Z.dims[0], F.plates, F.dims[0])
    
    # Known observation noise inverse covariance
    tau = np.ones(D)
    # or unknown observation noise inverse covariance
    tau = Gamma(1e-3, 1e-3, plates=(D,))
    
    # Observed process
    Y = GaussianARD(F, tau)
    
    # Data
    data = np.random.randn(T, D)
    Y.observe(data)
    
    from bayespy.inference import VB
    Q = VB(Y, X, p0, P, mu, tau)
    
    Q.update(repeat=10)

.. parsed-literal::

    (4, 1, 1, 3) (5,) (5,) (99, 3, 3)
    (4, 1, 1, 3) (5,) (5,) (3,) (4, 5, 100) (5,) (4, 5, 100) ()


::


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-5-1bb7a2c9abcf> in <module>()
         41 
         42 # Observed process
    ---> 43 Y = GaussianARD(F, tau)
         44 
         45 # Data


    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/expfamily.py in constructor_decorator(self, *args, **kwargs)
         81 
         82             (args, kwargs, dims, plates, dist, stats, pstats) = \
    ---> 83               self._constructor(*args, **kwargs)
         84 
         85             self.dims = dims


    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/gaussian.py in _constructor(cls, mu, alpha, ndim, shape, **kwargs)
        890         plates = cls._total_plates(kwargs.get('plates'),
        891                                    distribution.plates_from_parent(0, mu.plates),
    --> 892                                    distribution.plates_from_parent(1, alpha.plates))
        893 
        894         parents = [mu, alpha]


    /home/jluttine/workspace/bayespy/bayespy/inference/vmp/nodes/node.py in _total_plates(cls, plates, *parent_plates)
        244                 return utils.broadcasted_shape(*parent_plates)
        245             except ValueError:
    --> 246                 raise ValueError("The plates of the parents do not broadcast.")
        247         else:
        248             # Check that the parent_plates are a subset of plates.


    ValueError: The plates of the parents do not broadcast.



